"""Ready-made scenes.

Each function returns a Preset: the bodies in SI units plus hints for the
viewer (colours, exaggerated display sizes, starting camera distance and
playback speed). To add your own scene, write a function like these and list
it in PRESETS at the bottom.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from . import horizons
from .physics import BlackHole, Inspiral, System
from .units import AU, C_LIGHT, DAY, G, M_SUN, R_EARTH, R_SUN, circular_velocity, schwarzschild_radius

Vec3 = tuple[float, float, float]


@dataclass
class Body:
    name: str
    mass: float                    # kg
    position: Vec3                 # m
    velocity: Vec3                 # m/s
    colour: tuple[int, int, int]
    display_radius: float          # m. Real sizes are invisible at these scales, so this is exaggerated.
    is_star: bool = False          # stars glow and light up the planets
    radius: float = 0.0            # real physical radius [m], used for collisions
    white_dwarf: bool = False      # two white dwarfs merging past 1.44 Suns explode as a supernova
    black_hole: bool = False       # drawn as a black shadow; ``radius`` is its event horizon


@dataclass
class Preset:
    name: str
    bodies: list[Body]
    dt_max: float                  # largest timestep the viewer may use [s]
    days_per_second: float         # default playback speed
    view_distance: float           # starting camera distance from the centre [m]
    softening: float = 0.0         # [m], see physics.accelerations
    description: str = ""
    # Optional gravitational-wave inspiral: (first body, second body, merge time [s], merge distance [m])
    inspiral: tuple[str, str, float, float] | None = None
    black_hole: str | None = None  # name of a body whose relativistic pull (and tides) everything feels
    trail_length: int = 600        # trail points kept per body (one per frame)

    def build_inspiral(self, system: System) -> Inspiral | None:
        if self.inspiral is None:
            return None
        first, second, merge_time, merge_distance = self.inspiral
        start = math.dist(system.x[system.names.index(first)], system.x[system.names.index(second)])
        return Inspiral(first, second, merge_time, start, merge_distance)

    def build_system(self) -> System:
        system = System(
            names=[b.name for b in self.bodies],
            m=[b.mass for b in self.bodies],
            x=[b.position for b in self.bodies],
            v=[b.velocity for b in self.bodies],
            radius=[b.radius for b in self.bodies],
            softening=self.softening,
        )
        if self.black_hole is not None:
            system.fields.append(BlackHole(self.black_hole))
            system.update_accelerations()
        system.recenter()
        return system


# --------------------------------------------------------------------------- solar system

# name, colour, display radius [AU], G*M of the planet plus its moons [km^3/s^2] (JPL DE440),
# mean radius [km]
SOLAR_SYSTEM = [
    ("Sun", (255, 210, 110), 0.15, 132712440041.279419, 695700.0),
    ("Mercury", (170, 160, 150), 0.030, 22031.868551, 2439.7),
    ("Venus", (232, 205, 150), 0.045, 324858.592, 6051.8),
    ("Earth", (100, 155, 255), 0.048, 403503.235625, 6371.0),
    ("Mars", (225, 110, 75), 0.036, 42828.375816, 3389.5),
    ("Jupiter", (215, 175, 135), 0.13, 126712764.1, 69911.0),
    ("Saturn", (230, 205, 140), 0.11, 37940584.8418, 58232.0),
    ("Uranus", (150, 215, 230), 0.08, 5794556.4, 25362.0),
    ("Neptune", (90, 130, 240), 0.08, 6836527.10058, 24622.0),
]


def load_snapshot(path: Path = horizons.SNAPSHOT_FILE) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def solar_system(snapshot: dict | None = None) -> Preset:
    """The Sun and eight planets, starting from real JPL Horizons positions and velocities.

    Uses the snapshot bundled in data/ unless one is passed in (see solar_system_live).
    Each planet's moons are folded into the planet, so "Earth" is really Earth + Moon.
    """
    snapshot = snapshot if snapshot is not None else load_snapshot()
    bodies = []
    for name, colour, radius_au, gm_km3, radius_km in SOLAR_SYSTEM:
        state = snapshot["bodies"][name]
        position = tuple(c * 1e3 for c in state["position_km"])
        velocity = tuple(c * 1e3 for c in state["velocity_km_s"])
        bodies.append(Body(name, gm_km3 * 1e9 / G, position, velocity, colour, radius_au * AU, name == "Sun",
                           radius_km * 1e3))
    date = snapshot["epoch"].split()[0]
    return Preset(
        name=f"Solar system (JPL Horizons, {date})",
        bodies=bodies,
        dt_max=0.5 * DAY,
        days_per_second=40.0,
        view_distance=10.0 * AU,
        description="Real positions and velocities from NASA/JPL Horizons. Press F to follow a planet.",
    )


def solar_system_live() -> Preset:
    """Like solar_system(), but downloads today's positions from JPL Horizons (needs internet)."""
    return solar_system(horizons.fetch_solar_system())


# --------------------------------------------------------------------------- figure-8

# Figure-8 solution of the three-body problem (Moore 1993; Chenciner & Montgomery 2000)
# in units where G = 1 and every mass = 1.
FIGURE_EIGHT_POSITION = (0.97000436, -0.24308753)     # body 1; body 2 is at minus this, body 3 at the origin
FIGURE_EIGHT_VELOCITY = (-0.93240737, -0.86473146)    # body 3; bodies 1 and 2 each move at minus half of it
FIGURE_EIGHT_PERIOD = 6.32591398


def figure_eight_period(length: float = AU, mass: float = M_SUN) -> float:
    """Period [s] of the figure-8 scaled to the given length [m] and mass [kg]."""
    return FIGURE_EIGHT_PERIOD * math.sqrt(length**3 / (G * mass))


def figure_eight(length: float = AU, mass: float = M_SUN) -> Preset:
    """Three equal stars chasing each other around a figure-8.

    Scaled to 1 AU and one solar mass, a lap takes about 368 days. It makes a good
    integrator test: switch to Explicit Euler (press I) and watch it gain energy and
    swell out of shape, while leapfrog keeps it on track.
    """
    speed = math.sqrt(G * mass / length)
    (px, py), (vx, vy) = FIGURE_EIGHT_POSITION, FIGURE_EIGHT_VELOCITY
    outer_velocity = (-0.5 * vx * speed, -0.5 * vy * speed, 0.0)
    radius = 0.06 * AU
    bodies = [
        Body("Star 1", mass, (px * length, py * length, 0.0), outer_velocity, (255, 200, 120), radius, True, R_SUN),
        Body("Star 2", mass, (-px * length, -py * length, 0.0), outer_velocity, (140, 190, 255), radius, True, R_SUN),
        Body("Star 3", mass, (0.0, 0.0, 0.0), (vx * speed, vy * speed, 0.0), (255, 140, 180), radius, True, R_SUN),
    ]
    return Preset(
        name="Figure-8 three-body orbit",
        bodies=bodies,
        dt_max=0.25 * DAY,
        days_per_second=60.0,
        view_distance=3.0 * AU,
        description="A famous periodic solution of the three-body problem. Press I to try other integrators.",
    )


# --------------------------------------------------------------------------- supernova finale

def supernova_finale() -> Preset:
    """Two white dwarfs spiral together and explode as a Type Ia supernova.

    White dwarfs are the leftover cores of Sun-like stars: about the size of the
    Earth but as heavy as the Sun. None can be heavier than 1.44 Suns, so when
    these two (1.0 + 0.8 = 1.8 Suns) merge, the merged star blows itself apart
    completely. (Sun-like stars that collide just merge into a bigger star;
    white dwarfs are how stellar collisions really make supernovae.)

    Their orbit shrinks because they give off gravitational waves. From this
    distance that would really take about 10^16 years, so it is sped up to three
    years. Five planets orbit the pair. When it explodes, its gravity is gone
    and the planets fly off in straight lines.
    """
    m_a, m_b = 1.0 * M_SUN, 0.8 * M_SUN
    total = m_a + m_b
    separation = 0.6 * AU
    v_rel = circular_velocity(total, separation)            # relative speed for a circular orbit
    star_size = 0.06 * AU
    bodies = [
        Body("White dwarf A", m_a, (-separation * m_b / total, 0.0, 0.0), (0.0, -v_rel * m_b / total, 0.0),
             (200, 222, 255), star_size, True, 5.45e6, white_dwarf=True),
        Body("White dwarf B", m_b, (separation * m_a / total, 0.0, 0.0), (0.0, v_rel * m_a / total, 0.0),
             (238, 242, 255), star_size, True, 6.97e6, white_dwarf=True),
    ]
    # name, orbit radius [AU], tilt [deg], starting angle [deg], colour, display radius [AU]
    planets = [
        ("Aqua", 1.9, 0, 0, (110, 220, 220), 0.045),
        ("Amber", 2.6, 8, 140, (255, 180, 90), 0.05),
        ("Violet", 3.4, -14, 250, (190, 140, 255), 0.05),
        ("Rose", 4.3, 22, 60, (255, 140, 170), 0.055),
        ("Ice", 5.4, -6, 200, (180, 215, 255), 0.06),
    ]
    for name, r_au, tilt_deg, angle_deg, colour, size_au in planets:
        r, v = r_au * AU, circular_velocity(total, r_au * AU)
        angle, incline = math.radians(angle_deg), math.radians(tilt_deg)
        position = (r * math.cos(angle), r * math.sin(angle), 0.0)
        velocity = (-v * math.sin(angle) * math.cos(incline), v * math.cos(angle) * math.cos(incline),
                    v * math.sin(incline))                  # circular orbit tilted about the starting direction
        bodies.append(Body(name, 5.9722e24, position, velocity, colour, size_au * AU, radius=R_EARTH))

    return Preset(
        name="Supernova finale",
        bodies=bodies,
        dt_max=0.05 * DAY,
        days_per_second=30.0,
        view_distance=13.0 * AU,
        description="Two white dwarfs spiral together, faster and faster... watch the countdown.",
        # Merge when the drawn stars touch; by then the real pair would plunge together within hours.
        inspiral=("White dwarf A", "White dwarf B", 1100 * DAY, 2 * star_size),
    )


# --------------------------------------------------------------------------- black hole

def apoapsis_start(gm: float, r_peri: float, r_apo: float, angle_deg: float, tilt_deg: float):
    """Position and velocity at the far point of an orbit around a black hole that comes in to
    ``r_peri`` and goes out to ``r_apo``. Uses the turning points of the relativistic orbit
    equation, so the orbit really does turn at r_peri, however close that is."""
    up, ua, k = 1.0 / r_peri, 1.0 / r_apo, gm / C_LIGHT**2
    h = math.sqrt(2.0 * gm / ((up + ua) - 2.0 * k * (up * up + up * ua + ua * ua)))   # angular momentum
    v = h / r_apo
    angle, tilt = math.radians(angle_deg), math.radians(tilt_deg)
    position = (r_apo * math.cos(angle), r_apo * math.sin(angle), 0.0)
    velocity = (-v * math.sin(angle) * math.cos(tilt), v * math.cos(angle) * math.cos(tilt), v * math.sin(tilt))
    return position, velocity


def black_hole() -> Preset:
    """Sagittarius A*, the 4.3-million-Sun black hole at the centre of our galaxy, and a few
    unlucky neighbours, all on relativistic orbits (see physics.BlackHole):

      Pulsar              a neutron star on a tight orbit whose closest point swings round
                          by about 70 degrees each lap, tracing a rosette.
      Stellar black hole  on a "zoom-whirl" orbit: it whips round the hole several times
                          close in, then zooms out, again and again.
      S-star              a Sun-like star on a gentler, slowly precessing orbit.
      White dwarf         falls almost straight in and is swallowed whole (it is too small
                          for the tides to tear apart before it crosses the event horizon).
      Sun-like star       dives in from far away, gets torn apart by tides (spaghettified),
                          and part of it falls in while the rest is flung away.
    """
    m_hole = 4.3e6 * M_SUN
    gm = G * m_hole
    r_s = schwarzschild_radius(m_hole)                     # about 0.085 AU
    h_capture = 2.0 * r_s * C_LIGHT                        # less angular momentum than this and you fall in
    bodies = [Body("Sagittarius A*", m_hole, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (255, 170, 90), r_s,
                   radius=r_s, black_hole=True)]

    position, velocity = apoapsis_start(gm, 8 * r_s, 40 * r_s, 0, 15)
    bodies.append(Body("Pulsar", 1.4 * M_SUN, position, velocity, (170, 210, 255), 0.5 * r_s, True, 1.2e4))
    position, velocity = apoapsis_start(gm, 2.07 * r_s, 45 * r_s, 120, -25)     # just outside the separatrix
    bodies.append(Body("Stellar black hole", 10 * M_SUN, position, velocity, (185, 130, 255), 0.5 * r_s,
                       radius=schwarzschild_radius(10 * M_SUN), black_hole=True))
    position, velocity = apoapsis_start(gm, 20 * r_s, 60 * r_s, 230, 35)
    bodies.append(Body("S-star", M_SUN, position, velocity, (255, 210, 150), 0.5 * r_s, True, R_SUN))

    r0, angle = 150 * r_s, math.radians(300)                # falls in with 60% of the angular momentum needed
    bodies.append(Body("White dwarf", 0.6 * M_SUN, (r0 * math.cos(angle), r0 * math.sin(angle), 0.0),
                       (-0.6 * h_capture / r0 * math.sin(angle), 0.6 * h_capture / r0 * math.cos(angle), 0.0),
                       (235, 240, 255), 0.4 * r_s, True, 8.7e6))

    # Comes in on a parabolic orbit with just over the capture angular momentum, so once it is torn apart
    # some of its debris has a little less (and falls in) and some a little more (and whirls round and out).
    r0, angle, tilt, h = 520 * r_s, math.radians(60), math.radians(10), 1.003 * h_capture
    speed = math.sqrt(2 * gm / r0 + 2 * gm * h * h / (C_LIGHT**2 * r0**3))   # zero total energy
    v_t, v_r = h / r0, -math.sqrt(speed**2 - (h / r0) ** 2)
    radial = (math.cos(angle), math.sin(angle), 0.0)
    across = (-math.sin(angle) * math.cos(tilt), math.cos(angle) * math.cos(tilt), math.sin(tilt))
    bodies.append(Body("Sun-like star", M_SUN, tuple(r0 * c for c in radial),
                       tuple(v_r * a + v_t * b for a, b in zip(radial, across)), (255, 228, 140), 0.5 * r_s,
                       True, R_SUN))

    return Preset(
        name="Black hole: Sagittarius A*",
        bodies=bodies,
        dt_max=120.0,
        days_per_second=4 / 24,                            # four hours a second
        view_distance=7.0 * AU,
        description="Relativistic orbits round a 4.3-million-Sun black hole. Watch the Sun-like star come in...",
        black_hole="Sagittarius A*",
        trail_length=1800,
    )


PRESETS = {
    "supernova": supernova_finale,
    "solar": solar_system,
    "solar-live": solar_system_live,
    "figure8": figure_eight,
    "blackhole": black_hole,
}
