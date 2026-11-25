"""3D viewer for the gravity sandbox, drawn with pygame-ce.

There's no 3D engine underneath. A small orbit camera turns 3D positions into
screen positions (the perspective divide), bodies are drawn far-to-near so
nearer ones cover farther ones, and each planet is shaded pixel by pixel from
the direction of its star(s), so you can see its day and night sides.

Start it with ``python main.py`` and press H in the window for the controls.
"""
from __future__ import annotations

import math
import os
import queue
import threading
from dataclasses import dataclass, field

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np  # noqa: E402
import pygame  # noqa: E402

from . import physics, presets  # noqa: E402
from .units import AU, DAY, M_CHANDRASEKHAR, M_SUN, YEAR  # noqa: E402

FPS = 60
MAX_STEPS_PER_FRAME = 250        # physics steps per frame before the sim runs slower than asked
TRAIL_LENGTH = 600               # trail points kept per body (one per frame, so 10 seconds)
MIN_PLANET_PX = 2.5              # bodies never shrink below this many pixels...
MIN_STAR_PX = 4.0
MAX_BODY_PX = 260                # ...or grow beyond this
AMBIENT = 0.07                   # brightness of a planet's night side
DEFAULT_LIGHT = np.array([-0.45, 0.55, -0.70]) / np.linalg.norm([-0.45, 0.55, -0.70])

BACKGROUND = (6, 7, 14)
GRID_COLOUR = (24, 28, 44)
TEXT_COLOUR = (226, 229, 240)
DIM_COLOUR = (140, 146, 172)
GOOD_COLOUR = (130, 220, 150)
WARN_COLOUR = (255, 196, 110)
BAD_COLOUR = (255, 120, 110)

PRESET_KEYS = {
    pygame.K_2: "solar",
    pygame.K_3: "solar-live",
    pygame.K_4: "figure8",
    pygame.K_6: "supernova",
    pygame.K_7: "blackhole",
}

DEBRIS_FRAGMENTS = 400           # pieces a star is torn into by a black hole's tides
PHOTON_RING = 2.6                # apparent edge of a black hole's shadow, in horizon radii (3 sqrt(3) / 2)

SUPERNOVA_ENERGY = 1.5e44        # kinetic energy of the debris [J] (about 10^51 erg, typical of Type Ia)
SUPERNOVA_FRAGMENTS = 2000
SUPERNOVA_CLUMPS = 90
SUPERNOVA_DAYS = 330             # how long the glowing debris is drawn [days]
BLAST_SECONDS = 12.0             # slow motion: real seconds for the blast to sweep past the planets
SCREEN_FLASH_FRAMES = 22
STARFIELD_STARS = 2000
# Star-temperature colours from hot to cool: pale blue, white, yellow, orange, red, dark red.
# (The colour scheme and the background stars are ideas from github.com/peteraraujo/supernova, Apache-2.0.)
TEMPERATURE_COLOURS = [
    (15000.0, (170, 221, 255)),
    (8000.0, (255, 255, 255)),
    (5500.0, (255, 255, 0)),
    (4000.0, (255, 136, 0)),
    (3000.0, (255, 0, 0)),
    (2000.0, (85, 0, 0)),
]

COLLISION_MODES = ("real size", "drawn size", "off")
COLLISION_HELP = {
    "real size": "Collisions at real sizes: bodies merge only if their actual surfaces touch (rare).",
    "drawn size": "Collisions at drawn sizes: bodies merge when the spheres on screen touch (much more often).",
    "off": "Collisions off: bodies pass through each other.",
}
FLASH_FRAMES = 50                # how long a collision flash lasts

HELP_LINES = [
    ("drag mouse", "rotate the view"),
    ("mouse wheel", "zoom"),
    ("2 3 4 6 7", "solar system / today's sky (JPL) / figure-8 / supernova / black hole"),
    ("space", "pause"),
    ("up / down", "double / halve the speed"),
    ("R", "restart the preset"),
    ("I", "switch integrator"),
    ("K", "collisions: real size / drawn size / off"),
    ("F / shift+F", "follow next / previous body"),
    ("T / C", "trails on/off / clear trails"),
    ("L / G", "labels / grid on/off"),
    ("[ / ]", "smaller / bigger bodies"),
    ("H", "hide this help"),
    ("esc", "quit"),
]


# --------------------------------------------------------------------------- small helpers

def dim(colour, factor: float) -> tuple[int, int, int]:
    return tuple(int(c * factor) for c in colour)


def mix(a, b, t: float) -> tuple[int, int, int]:
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


def nice_step(x: float) -> float:
    """Largest of 1, 2 or 5 times a power of ten that is <= x (for grid spacing)."""
    power = 10.0 ** math.floor(math.log10(x))
    for mantissa in (5, 2, 1):
        if mantissa * power <= x:
            return mantissa * power
    return power


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60:.0f} min"
    if seconds < 2 * DAY:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / DAY:.1f} d"


def format_time(seconds: float) -> str:
    if seconds < 10 * DAY:
        return f"{seconds / 3600:,.1f} hours ({seconds / DAY:.2f} days)"
    return f"{seconds / DAY:,.1f} days ({seconds / YEAR:.2f} years)"


def format_speed(days_per_second: float) -> str:
    if days_per_second < 1:
        return f"{days_per_second * 24:.3g} hours/s"
    return f"{days_per_second:.3g} days/s"


def to_surface(pixels: np.ndarray, fmt: str) -> pygame.Surface:
    height, width = pixels.shape[:2]
    return pygame.image.frombytes(np.ascontiguousarray(pixels).tobytes(), (width, height), fmt)


# --------------------------------------------------------------------------- supernova look

def temperature_colour(kelvin: np.ndarray) -> np.ndarray:
    """RGB (0-255 floats) for each temperature, blended along TEMPERATURE_COLOURS in log T."""
    stops = sorted(TEMPERATURE_COLOURS)
    log_t = np.log([t for t, _ in stops])
    rgb = np.array([c for _, c in stops], dtype=float)
    x = np.log(np.clip(kelvin, stops[0][0], stops[-1][0]))
    return np.stack([np.interp(x, log_t, rgb[:, k]) for k in range(3)], axis=-1)


def debris_temperature(age_days: float) -> float:
    """Rough temperature [K] of supernova debris: about 15,000 K at first, cooling to 4,000 K over months."""
    return 4000.0 + 11000.0 * math.exp(-age_days / 35.0)


def supernova_light(age_days: float) -> float:
    """Brightness from 0 to 1: a first glow from the shock-heated gas, then the main peak about
    18 days later, powered by radioactive nickel and cobalt decaying, then a slow fade."""
    early = 0.9 * math.exp(-age_days / 3.0)
    rise = 1.0 - math.exp(-(age_days / 12.0) ** 2)
    decay = 0.75 * math.exp(-age_days / 25.0) + 0.25 * math.exp(-age_days / 111.0)
    return min(1.0, max(early, rise * decay / 0.517))


def supernova_fade(age_days: float) -> float:
    """1, then fading to 0 near SUPERNOVA_DAYS so the debris disappears smoothly."""
    late = max(age_days - (SUPERNOVA_DAYS - 110.0), 0.0)
    return math.exp(-(late / 60.0) ** 2)


# --------------------------------------------------------------------------- camera

@dataclass
class Camera:
    """Orbits around ``target``. Angles in radians, distances in AU."""

    yaw: float = math.radians(-90.0)
    pitch: float = math.radians(40.0)
    distance: float = 5.0
    target: np.ndarray = field(default_factory=lambda: np.zeros(3))
    fov: float = math.radians(50.0)

    def rotate(self, dx: float, dy: float) -> None:
        self.yaw -= dx * 0.006
        limit = math.radians(89.0)
        self.pitch = min(max(self.pitch + dy * 0.006, -limit), limit)

    def zoom(self, clicks: float) -> None:
        self.distance = min(max(self.distance * 0.88**clicks, 1e-4), 5e3)


class View:
    """The camera transform for one frame: world (AU) -> camera space -> pixels.

    Camera space has x to the right, y up and z = depth in front of the camera.
    """

    def __init__(self, camera: Camera, size: tuple[int, int]):
        width, height = size
        back = np.array([
            math.cos(camera.pitch) * math.cos(camera.yaw),
            math.cos(camera.pitch) * math.sin(camera.yaw),
            math.sin(camera.pitch),
        ])
        forward = -back
        right = np.cross(forward, (0.0, 0.0, 1.0))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        self.eye = camera.target + camera.distance * back
        self.rotation = np.stack([right, up, forward])
        self.focal = 0.5 * height / math.tan(0.5 * camera.fov)     # pixels per unit at depth 1
        self.centre = np.array([0.5 * width, 0.5 * height])
        self.near = 0.01 * camera.distance

    def to_camera(self, points: np.ndarray) -> np.ndarray:
        return (points - self.eye) @ self.rotation.T

    def to_screen(self, camera_points: np.ndarray) -> np.ndarray:
        """Perspective divide. Only meaningful for points in front of the camera (z > near)."""
        xy = camera_points[..., :2] * (self.focal / camera_points[..., 2:3])
        return self.centre + xy * np.array([1.0, -1.0])            # screen y points down


def draw_segments(surface, view: View, starts: np.ndarray, ends: np.ndarray, colours) -> None:
    """Draw many straight 3D lines at once, each clipped where it passes behind the camera.
    (Straight lines stay straight in perspective, so the two end points are all we need.)"""
    a, b = view.to_camera(starts), view.to_camera(ends)
    za, zb = a[:, 2], b[:, 2]
    keep = (za > view.near) | (zb > view.near)
    with np.errstate(divide="ignore", invalid="ignore"):          # rows that divide by zero aren't used
        a = np.where((za <= view.near)[:, None], a + (b - a) * ((view.near - za) / (zb - za))[:, None], a)
        b = np.where((zb <= view.near)[:, None], b + (a - b) * ((view.near - zb) / (a[:, 2] - zb))[:, None], b)
    pa = np.clip(view.to_screen(a[keep]), -1e5, 1e5).tolist()
    pb = np.clip(view.to_screen(b[keep]), -1e5, 1e5).tolist()
    colours = [colours] * len(pa) if isinstance(colours, tuple) else [c for c, k in zip(colours, keep) if k]
    for p, q, colour in zip(pa, pb, colours):
        pygame.draw.line(surface, colour, p, q)


def draw_polyline(surface, view: View, points: np.ndarray, colour, antialias: bool = False) -> None:
    """Draw a 3D polyline, leaving out the parts behind the camera."""
    if len(points) < 2:
        return
    cam = view.to_camera(points)
    visible = np.flatnonzero(cam[:, 2] > view.near)
    if len(visible) < 2:
        return
    screen = np.clip(view.to_screen(cam[visible]), -1e5, 1e5)
    draw = pygame.draw.aalines if antialias else pygame.draw.lines
    start = 0
    for end in [*(np.flatnonzero(np.diff(visible) > 1) + 1), len(visible)]:
        if end - start >= 2:
            draw(surface, colour, False, screen[start:end].tolist())
        start = end


# --------------------------------------------------------------------------- trails and sprites

class Trails:
    """Ring buffer of recent positions (AU) for every body."""

    def __init__(self, bodies: int, length: int = TRAIL_LENGTH):
        self.points = np.zeros((bodies, length, 3))
        self.length = length
        self.count = 0
        self.head = 0

    def add(self, positions: np.ndarray) -> None:
        self.points[:, self.head] = positions
        self.head = (self.head + 1) % self.length
        self.count = min(self.count + 1, self.length)

    def clear(self) -> None:
        self.count = 0
        self.head = 0

    def remove(self, index: int) -> None:
        """Forget one body's trail (after it was absorbed in a collision)."""
        self.points = np.delete(self.points, index, axis=0)

    def ordered(self) -> np.ndarray:
        """Array of shape (bodies, count, 3), oldest point first."""
        order = (np.arange(self.count) + self.head - self.count) % self.length
        return self.points[:, order]


@dataclass
class Flash:
    """A burst drawn where two bodies collided."""

    position: np.ndarray             # AU
    colour: tuple[int, int, int]
    age: int = 0                     # frames since the collision


@dataclass
class Supernova:
    """How a supernova is drawn. Its debris (physics.Ejecta) is also part of the physics."""

    ejecta: physics.Ejecta
    heat: np.ndarray                 # (P,) temperature factor per fragment: the slow inner debris is hottest
    front_speed: float               # speed of the leading edge of the debris [m/s]
    frames: int = 0                  # real frames since the explosion, for the first flash

    @classmethod
    def create(cls, ejecta: physics.Ejecta) -> "Supernova":
        # Speeds are sorted, so a fragment's rank says how far out it is: the slow inner debris stays
        # hottest, the fast outer layers thin out and cool first (blue-white core, yellow-orange edge).
        rank = np.arange(len(ejecta.speeds)) / max(len(ejecta.speeds) - 1, 1)
        return cls(ejecta, 1.45 - 1.1 * rank, float(np.quantile(ejecta.speeds, 0.95)))


class Starfield:
    """Faint background stars, infinitely far away, so they turn with the camera but never shift."""

    def __init__(self, rng: np.random.Generator, count: int = STARFIELD_STARS):
        self.directions = rng.normal(size=(count, 3))
        self.directions /= np.linalg.norm(self.directions, axis=1, keepdims=True)
        level = 25 + 200 * rng.random(count) ** 3            # mostly faint, a few bright
        tint = np.array([(1.0, 1.0, 1.0), (0.82, 0.9, 1.0), (1.0, 0.9, 0.78)])[rng.integers(0, 3, count)]
        self.colours = (level[:, None] * tint).astype(int)
        self.big = level > 170

    def draw(self, surface: pygame.Surface, view: "View", lens: tuple[np.ndarray, float] | None = None) -> None:
        """``lens`` = (black hole position [AU], its horizon radius [AU]) bends the starlight round it."""
        cam = self.directions @ view.rotation.T              # rotation only: the stars are at infinity
        colours, big = self.colours.astype(float), self.big
        if lens is not None:
            cam, colours, big = self.lensed(cam, colours, big, view, *lens)
        ahead = cam[:, 2] > 0.05
        xy = view.centre + cam[ahead, :2] / cam[ahead, 2:3] * view.focal * np.array([1.0, -1.0])
        width, height = surface.get_size()
        inside = (xy[:, 0] >= 0) & (xy[:, 0] < width - 1) & (xy[:, 1] >= 0) & (xy[:, 1] < height - 1)
        shown = np.clip(colours[ahead][inside], 0, 255).astype(int)
        for (x, y), colour, large in zip(xy[inside].astype(int).tolist(), shown.tolist(), big[ahead][inside].tolist()):
            surface.fill(colour, (x, y, 2 if large else 1, 2 if large else 1))

    @staticmethod
    def lensed(directions, colours, big, view: "View", centre_au: np.ndarray, r_s_au: float):
        """Gravitational lensing by a black hole, using the point-lens equation.

        A star at angle b from the hole appears as two images, at (b +- sqrt(b^2 + 4 tE^2)) / 2:
        one pushed outwards, one on the opposite side inside the Einstein ring (radius tE =
        sqrt(2 r_s / D) for stars far behind a hole at distance D), each brightened by the lens's
        magnification. No starlight shows inside the shadow. (This weak-field formula is only
        roughly right this close to a hole, but it shows the effect.)
        """
        hole = view.to_camera(centre_au[np.newaxis, :])[0]
        distance = float(np.linalg.norm(hole))
        if hole[2] <= 0 or distance < 4 * r_s_au:
            return directions, colours, big
        axis = hole / distance
        theta_e = math.sqrt(2.0 * r_s_au / distance)
        theta_shadow = PHOTON_RING * r_s_au / distance
        cos_b = np.clip(directions @ axis, -1.0, 1.0)
        beta = np.arccos(cos_b)
        near = (beta < 20 * theta_e) & (beta > 1e-9)
        if not near.any():
            return directions, colours, big
        side = directions[near] - cos_b[near, None] * axis
        side /= np.linalg.norm(side, axis=1, keepdims=True)
        b = beta[near]
        root = np.sqrt(b * b + 4.0 * theta_e**2)
        outer, inner = 0.5 * (b + root), 0.5 * (root - b)
        u = b / theta_e
        mu = (u * u + 2.0) / (2.0 * u * np.sqrt(u * u + 4.0))
        directions, colours, big = directions.copy(), colours.copy(), big.copy()
        directions[near] = np.cos(outer)[:, None] * axis + np.sin(outer)[:, None] * side
        colours[near] *= np.minimum(mu + 0.5, 6.0)[:, None]
        hidden = np.flatnonzero(near)[outer < theta_shadow]
        directions[hidden] = -axis                           # swallowed by the shadow: point it behind us
        echo = inner > theta_shadow                           # the second image, on the far side of the ring
        mirror = np.cos(inner[echo])[:, None] * axis - np.sin(inner[echo])[:, None] * side[echo]
        faint = colours[np.flatnonzero(near)[echo]] / np.minimum(mu[echo] + 0.5, 6.0)[:, None] \
            * np.minimum(mu[echo] - 0.5, 6.0)[:, None]
        return (np.vstack([directions, mirror]), np.vstack([colours, faint]),
                np.concatenate([big, big[np.flatnonzero(near)[echo]]]))


class Sprites:
    """Makes (and caches) the bitmaps for planets and stars."""

    def __init__(self) -> None:
        self._spheres: dict[float, tuple[int, np.ndarray, np.ndarray]] = {}
        self._stars: dict[tuple, tuple] = {}

    @staticmethod
    def _disc(radius: float):
        n = int(math.ceil(radius)) + 1
        coords = np.arange(-n, n + 1) / radius
        u, v = np.meshgrid(coords, coords)                 # u across to the right, v down
        rho = np.hypot(u, v)
        alpha = (np.clip((1.0 - rho) * radius + 0.5, 0.0, 1.0) * 255).astype(np.uint8)   # soft 1 px edge
        return n, u, v, rho, alpha

    def _sphere(self, radius: float):
        key = round(radius * 2) / 2
        if key not in self._spheres:
            if len(self._spheres) > 200:
                self._spheres.clear()
            n, u, v, rho, alpha = self._disc(key)
            # Surface normal of the visible half of the sphere, in camera space.
            normal = np.dstack([u, -v, -np.sqrt(np.clip(1.0 - rho**2, 0.0, 1.0))])
            self._spheres[key] = (n, normal, alpha)
        return self._spheres[key]

    def planet(self, radius: float, colour, lights) -> tuple[pygame.Surface, int]:
        """Sphere lit by ``lights``: a list of (unit direction in camera space, weight)."""
        n, normal, alpha = self._sphere(radius)
        light = np.zeros(alpha.shape)
        for direction, weight in lights:
            light += weight * np.clip(normal @ direction, 0.0, None)
        shade = AMBIENT + (1.0 - AMBIENT) * np.clip(light, 0.0, 1.0)
        rgb = (shade[..., None] * np.asarray(colour, dtype=float)).astype(np.uint8)
        return to_surface(np.dstack([rgb, alpha]), "RGBA"), n

    def star(self, radius: float, colour):
        """(core sprite, core half-size, glow sprite, glow half-size). The glow is drawn additively."""
        key = (round(radius * 2) / 2, tuple(colour))
        if key not in self._stars:
            if len(self._stars) > 200:
                self._stars.clear()
            r = key[0]
            n, _, _, rho, alpha = self._disc(r)
            whiteness = (np.clip(1.0 - rho, 0.0, 1.0) ** 0.6)[..., None]      # white-hot centre
            rgb = (np.asarray(colour, dtype=float) * (1.0 - whiteness) + 255.0 * whiteness).astype(np.uint8)
            core = to_surface(np.dstack([rgb, alpha]), "RGBA")
            glow_radius = min(4.0 * r + 8.0, 420.0)
            m, _, _, rho_glow, _ = self._disc(glow_radius)
            intensity = 0.5 * np.clip(1.0 - rho_glow, 0.0, 1.0) ** 2.5
            glow = to_surface((intensity[..., None] * np.asarray(colour, dtype=float)).astype(np.uint8), "RGB")
            self._stars[key] = (core, n, glow, m)
        return self._stars[key]

    def burst(self, radius: float, colour, strength: float) -> tuple[pygame.Surface, int]:
        """Soft glow (drawn additively) for collision flashes."""
        key = ("burst", round(radius), tuple(colour), round(strength, 1))
        if key not in self._stars:
            if len(self._stars) > 200:
                self._stars.clear()
            m, _, _, rho, _ = self._disc(max(round(radius), 2))
            intensity = strength * np.clip(1.0 - rho, 0.0, 1.0) ** 1.5
            rgb = (intensity[..., None] * np.asarray(colour, dtype=float)).astype(np.uint8)
            self._stars[key] = (to_surface(rgb, "RGB"), m)
        return self._stars[key]

    def ring(self, radius: float, colour) -> tuple[pygame.Surface, int]:
        """A thin glowing ring (drawn additively): a black hole's photon ring."""
        key = ("ring", round(radius), tuple(colour))
        if key not in self._stars:
            if len(self._stars) > 200:
                self._stars.clear()
            r = max(round(radius), 3)
            m, _, _, rho, _ = self._disc(r + 6.0)
            width = 1.6 / r                                    # about 1.6 px either side
            intensity = np.exp(-(((rho * (r + 6.0) / r) - 1.0) / width) ** 2)
            intensity += 0.25 * np.exp(-(((rho * (r + 6.0) / r) - 1.0) / (4 * width)) ** 2)   # soft halo
            rgb = (np.clip(intensity, 0, 1)[..., None] * np.asarray(colour, dtype=float)).astype(np.uint8)
            self._stars[key] = (to_surface(rgb, "RGB"), m)
        return self._stars[key]

    def nova_glow(self, strength: float, radius: float) -> pygame.Surface:
        """The supernova's blinding core glow (drawn additively), scaled to ``radius`` pixels."""
        level = max(1, min(12, round(strength * 12)))
        key = ("nova", level)
        if key not in self._stars:
            _, _, _, rho, _ = self._disc(160.0)
            intensity = (level / 12) * np.clip(1.0 - rho, 0.0, 1.0) ** 1.6
            self._stars[key] = to_surface((intensity[..., None] * np.array([255.0, 238.0, 215.0])).astype(np.uint8),
                                          "RGB")
        size = max(2, int(2 * radius))
        return pygame.transform.smoothscale(self._stars[key], (size, size))


# --------------------------------------------------------------------------- the app

class App:
    def __init__(self, preset: str = "supernova", size: tuple[int, int] = (1100, 700)):
        pygame.init()
        pygame.display.set_caption("3D Gravity Sandbox")
        self.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas,dejavusansmono,menlo,couriernew", 15)
        self.label_font = pygame.font.SysFont("segoeui,dejavusans,helvetica,arial", 13)
        self.sprites = Sprites()
        self.camera = Camera()
        self.integrators = list(physics.INTEGRATORS)
        self.integrator = 0
        self.paused = False
        self.dragging = False
        self.show_trails = self.show_labels = self.show_grid = self.show_help = True
        self.size_factor = 1.0
        self.follow = -1                      # -1 = centre of mass, otherwise a body index
        self.collision_mode = 0               # index into COLLISION_MODES: real sizes to start with
        self.collisions = 0
        self.flashes: list[Flash] = []
        self.supernovae: list[Supernova] = []
        self.screen_flash = 0                 # frames of white-out left after a supernova
        self.anchor: np.ndarray | None = None # fixed camera target (the supernova site), if any
        self.auto_frame = False               # zoom to follow the debris, then the departing planets
        self.speed_cap: float | None = None   # slow-motion limit on days per second, if any
        self.blast_until = 0.0                # sim time when the blast has passed the planets
        self.blast_settled = True             # False while planets still feel the debris' pull
        self.hole: physics.BlackHole | None = None   # the black hole everything orbits, if any
        self.debris: physics.Debris | None = None    # a star torn apart by its tides
        self.flare = 0.0                      # glow from debris falling into the hole
        self.rng = np.random.default_rng()
        self.starfield = Starfield(np.random.default_rng(7))
        self._glow_layer: pygame.Surface | None = None
        self.status, self.status_frames = "", 0
        self.last_steps, self.last_dt, self.effective_speed, self.capped = 0, 0.0, 0.0, False
        self._labels: dict = {}
        self._fetch_results: queue.Queue = queue.Queue()
        self._fetching = False
        self.load(preset)

    # ----- presets

    def load(self, key: str) -> None:
        if key == "solar-live":
            if not hasattr(self, "system"):
                self.apply(presets.solar_system())     # something to look at while downloading
            self.fetch_live()
        else:
            self.apply(presets.PRESETS[key]())

    def apply(self, preset: presets.Preset) -> None:
        self.preset = preset
        self.system = preset.build_system()
        self.colours = [b.colour for b in preset.bodies]
        self.radii = np.array([b.display_radius for b in preset.bodies]) / AU
        self.is_star = [b.is_star for b in preset.bodies]
        self.white_dwarf = [b.white_dwarf for b in preset.bodies]
        self.is_hole = [b.black_hole for b in preset.bodies]
        self.refresh_stars()
        self.trails = Trails(self.system.n, preset.trail_length)
        self.hole = next((f for f in self.system.fields if isinstance(f, physics.BlackHole)), None)
        self.debris = None
        self.flare = 0.0
        self.inspiral = preset.build_inspiral(self.system)
        self.collisions = 0
        self.flashes = []
        self.supernovae = []
        self.screen_flash = 0
        self.anchor = None
        self.auto_frame = False
        self.speed_cap = None
        self.blast_settled = True
        self.reset_baselines()
        self.days_per_second = preset.days_per_second
        self.camera.distance = preset.view_distance / AU
        self.follow = -1
        self.paused = False
        self.set_status(preset.description, seconds=6)

    def refresh_stars(self) -> None:
        self.stars = [i for i, star in enumerate(self.is_star) if star]
        self.star_set = set(self.stars)

    def reset_baselines(self) -> None:
        """Reference values for the energy and momentum drift readouts."""
        self.energy0 = self.system.energy()
        self.momentum0 = self.system.momentum()
        self.momentum_scale = float(np.sum(self.system.m * np.linalg.norm(self.system.v, axis=1))) or 1.0

    def fetch_live(self) -> None:
        """Download today's solar system in the background so the window keeps responding."""
        if self._fetching:
            return
        self._fetching = True
        self.set_status("Downloading today's planet positions from JPL Horizons...", seconds=120)

        def worker() -> None:
            try:
                self._fetch_results.put(presets.solar_system_live())
            except Exception as exc:          # offline, API down, ...
                self._fetch_results.put(exc)

        threading.Thread(target=worker, daemon=True).start()

    def poll_fetch(self) -> None:
        try:
            result = self._fetch_results.get_nowait()
        except queue.Empty:
            return
        self._fetching = False
        if isinstance(result, Exception):
            self.apply(presets.solar_system())
            self.set_status(f"Couldn't reach JPL Horizons ({str(result)[:60]}). Showing the bundled snapshot.", 10)
        else:
            self.apply(result)
            self.set_status("Loaded today's positions from JPL Horizons.", seconds=6)

    def set_status(self, text: str, seconds: float) -> None:
        self.status, self.status_frames = text, int(seconds * FPS)

    # ----- input

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Returns False when the app should quit."""
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False
            self.handle_key(event.key, event.mod)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.camera.rotate(*event.rel)
        elif event.type == pygame.MOUSEWHEEL:
            self.camera.zoom(event.y)
            self.auto_frame = False              # the user has taken over the zoom
        return True

    def handle_key(self, key: int, mods: int = 0) -> None:
        if key in PRESET_KEYS:
            self.load(PRESET_KEYS[key])
        elif key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key == pygame.K_r:
            self.apply(self.preset)
        elif key in (pygame.K_UP, pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            self.take_speed_control()
            self.days_per_second = min(self.days_per_second * 2, 1e5)
        elif key in (pygame.K_DOWN, pygame.K_MINUS, pygame.K_KP_MINUS):
            self.take_speed_control()
            self.days_per_second = max(self.days_per_second / 2, 1e-3)
        elif key == pygame.K_i:
            self.integrator = (self.integrator + 1) % len(self.integrators)
            self.set_status(f"Integrator: {self.integrators[self.integrator]}", seconds=3)
        elif key == pygame.K_k:
            self.collision_mode = (self.collision_mode + 1) % len(COLLISION_MODES)
            self.set_status(COLLISION_HELP[COLLISION_MODES[self.collision_mode]], seconds=5)
        elif key == pygame.K_f:
            step = -1 if mods & pygame.KMOD_SHIFT else 1
            self.follow = (self.follow + 1 + step) % (self.system.n + 1) - 1
        elif key == pygame.K_t:
            self.show_trails = not self.show_trails
        elif key == pygame.K_c:
            self.trails.clear()
        elif key == pygame.K_l:
            self.show_labels = not self.show_labels
        elif key == pygame.K_g:
            self.show_grid = not self.show_grid
        elif key == pygame.K_LEFTBRACKET:
            self.size_factor = max(self.size_factor / 1.5, 0.05)
        elif key == pygame.K_RIGHTBRACKET:
            self.size_factor = min(self.size_factor * 1.5, 50.0)
        elif key == pygame.K_h:
            self.show_help = not self.show_help

    # ----- simulation

    def take_speed_control(self) -> None:
        """Up/Down during slow motion start from the current (slowed) speed and end the slow motion."""
        if self.speed_cap is not None:
            self.days_per_second, self.speed_cap = self.speed_cap, None

    def playback_speed(self) -> float:
        """Days per second for this frame, including any supernova slow motion."""
        if self.speed_cap is None:
            return self.days_per_second
        if self.system.t > self.blast_until:
            self.speed_cap *= 1.025                         # the blast has passed: ease back to normal speed
        if self.speed_cap >= self.days_per_second:
            self.speed_cap = None
            return self.days_per_second
        return self.speed_cap

    def advance(self) -> None:
        """Move the simulation forward by one frame's worth of time."""
        wanted = self.playback_speed() * DAY / FPS            # simulated seconds this frame
        steps = max(1, math.ceil(wanted / self.preset.dt_max))
        uniform_dt = self.preset.dt_max if steps > MAX_STEPS_PER_FRAME else wanted / steps
        step = physics.INTEGRATORS[self.integrators[self.integrator]]
        supernovae_before, done, elapsed = len(self.supernovae), 0, 0.0
        while done < MAX_STEPS_PER_FRAME and elapsed < wanted * (1 - 1e-9):
            if self.hole is None:
                dt = uniform_dt
            else:   # near a black hole, shrink the step for whatever is closest to it
                extra = self.debris.x if self.debris is not None else None
                limit = max(self.hole.step_limit(self.system, extra), 0.5)
                dt = min(self.preset.dt_max, limit, wanted - elapsed)
            self.physics_step(dt, step)
            done, elapsed = done + 1, elapsed + dt
            if len(self.supernovae) > supernovae_before:
                break               # a supernova just went off: the rest happens in slow motion, from the next frame
        self.capped = elapsed < wanted * (1 - 1e-6) and len(self.supernovae) == supernovae_before
        self.last_steps, self.last_dt = done, elapsed / max(done, 1)
        self.effective_speed = elapsed * FPS / DAY
        self.flare *= 0.97
        for flash in self.flashes:
            flash.age += 1
        self.flashes = [flash for flash in self.flashes if flash.age < FLASH_FRAMES]
        for nova in self.supernovae:
            nova.frames += 1
        self.supernovae = [nova for nova in self.supernovae
                           if nova.ejecta.age(self.system.t) < SUPERNOVA_DAYS * DAY]
        self.screen_flash = max(self.screen_flash - 1, 0)
        blasts = [f for f in self.system.fields if isinstance(f, physics.Ejecta)]
        if not self.blast_settled and self.system.n and blasts:
            # Once no planet feels the debris any more, energy is conserved again: restart the readout.
            felt = max(float(np.max(f.enclosed_mass(np.linalg.norm(self.system.x - f.centre_at(self.system.t), axis=1),
                                                    self.system.t) / f.mass)) for f in blasts)
            if felt < 1e-3:
                self.blast_settled = True
                self.reset_baselines()
        if not np.all(np.isfinite(self.system.x)):
            self.paused = True
            self.set_status("The simulation blew up (a very close encounter?). Press R to restart.", seconds=30)
            return
        self.trails.add(self.system.x / AU)

    def physics_step(self, dt: float, step) -> None:
        """One timestep: bodies (with collisions), any inspiral, and anything a black hole is doing."""
        for event in physics.step_with_collisions(self.system, dt, step, self.collision_radii()):
            self.on_collision(event)
        if self.inspiral is not None:
            pair = physics.apply_inspiral(self.system, self.inspiral, dt)
            if pair is not None:
                i, j = pair
                speed = float(np.linalg.norm(self.system.v[j] - self.system.v[i]))
                self.inspiral = None
                self.on_collision(physics.merge(self.system, i, j, impact_speed=speed))
        if self.hole is not None:
            k = self.system.names.index(self.hole.name)
            r_s = self.hole.schwarzschild_radius(self.system)
            if self.debris is not None:
                self.debris.step(dt, self.system.x[k], self.system.v[k], self.system.G * self.system.m[k])
                # Inside 2 r_s nothing on a near-parabolic orbit can climb back out: it's falling in.
                fell = self.debris.clear(self.system.x[k], 2.0 * r_s, 4000.0 * r_s)
                self.flare = min(self.flare + fell / 25.0, 3.0)
            self.check_tides(k, r_s)

    def check_tides(self, k: int, r_s: float) -> None:
        """Tear apart any star that has come inside its tidal radius (if that is outside the horizon)."""
        s = self.system
        for i in range(s.n):
            if i == k or self.is_hole[i] or s.radius[i] <= 0:
                continue
            reach = physics.tidal_radius(s.radius[i], s.m[i], s.m[k])
            if reach > r_s and np.linalg.norm(s.x[i] - s.x[k]) < reach:
                self.tear_apart(i)
                return

    def tear_apart(self, index: int) -> None:
        """A tidal disruption: the star becomes a stream of debris (physics.tidal_disruption)."""
        name = self.system.names[index]
        self.debris = physics.tidal_disruption(self.system, index, self.hole.name, DEBRIS_FRAGMENTS, self.rng)
        self.radii = np.delete(self.radii, index)
        del self.colours[index], self.is_star[index], self.white_dwarf[index], self.is_hole[index]
        self.trails.remove(index)
        self.refresh_stars()
        if self.follow == index:
            self.follow = -1
        elif self.follow > index:
            self.follow -= 1
        self.reset_baselines()
        self.set_status(f"{name} has been torn apart by tides (spaghettified)! About "
                        f"{self.debris.bound_fraction:.0%} of it is still bound to the hole.", seconds=10)

    def collision_radii(self) -> np.ndarray | None:
        """Sizes [m] used for collisions in the current mode, or None when collisions are off."""
        mode = COLLISION_MODES[self.collision_mode]
        if mode == "off":
            # Even with collisions off, anything crossing an event horizon is swallowed.
            return np.where(self.is_hole, self.system.radius, 0.0) if any(self.is_hole) else None
        if mode == "real size":
            return self.system.radius
        return self.radii * (AU * self.size_factor)          # the exaggerated sizes drawn on screen

    def on_collision(self, event: physics.Collision) -> None:
        """Keep the viewer's per-body lists in step with a merger, and show it happening."""
        keep, gone = event.keep, event.removed
        hit_star = self.is_star[keep] or self.is_star[gone]
        two_white_dwarfs = self.white_dwarf[keep] and self.white_dwarf[gone]
        into_hole = self.is_hole[keep]
        self.radii[keep] *= 1.0 if into_hole else event.growth
        flash_colour = mix(self.colours[keep], (255, 255, 255), 0.5)
        self.radii = np.delete(self.radii, gone)
        del self.colours[gone], self.is_star[gone], self.white_dwarf[gone], self.is_hole[gone]
        self.trails.remove(gone)
        survivor = keep - 1 if keep > gone else keep
        self.is_star[survivor] = hit_star and not into_hole
        self.refresh_stars()
        if self.follow == gone:
            self.follow = survivor
        elif self.follow > gone:
            self.follow -= 1
        self.collisions += 1
        self.reset_baselines()          # a merger turns kinetic energy into heat, so restart the drift readout
        if into_hole:                   # nothing comes back out to make a flash
            self.set_status(f"{event.absorbed} crossed the event horizon of {event.survivor}: swallowed whole.", 6)
            return
        self.flashes.append(Flash(event.position / AU, flash_colour))
        if two_white_dwarfs and self.system.m[survivor] > M_CHANDRASEKHAR:
            self.explode(survivor, event.position)
            return
        speed = event.impact_speed / 1e3
        if hit_star:
            self.set_status(f"{event.absorbed} crashed into {event.survivor} at {speed:,.0f} km/s!", seconds=5)
        else:
            self.set_status(f"{event.absorbed} and {event.survivor} collided at {speed:,.1f} km/s and merged.", 5)

    def explode(self, index: int, position: np.ndarray) -> None:
        """Type Ia supernova: the merged white dwarf blows itself apart and leaves nothing behind.

        Its whole mass becomes debris coasting outward at thousands of km/s
        (physics.Ejecta). The planets keep feeling the debris still inside their
        orbits, so each is set free only when the blast passes it. The blast is
        timed for real, which takes under a day, so playback drops into slow motion
        while it sweeps past the planets and then eases back to normal speed.
        """
        mass, velocity = self.system.m[index], self.system.v[index].copy()
        physics.remove_body(self.system, index)
        self.radii = np.delete(self.radii, index)
        del self.colours[index], self.is_star[index], self.white_dwarf[index], self.is_hole[index]
        self.trails.remove(index)
        self.refresh_stars()
        if self.follow == index:
            self.follow = -1
        elif self.follow > index:
            self.follow -= 1
        self.inspiral = None

        ejecta = physics.Ejecta.from_explosion(position, velocity, self.system.t, mass, SUPERNOVA_ENERGY,
                                               SUPERNOVA_FRAGMENTS, self.rng, clumps=SUPERNOVA_CLUMPS)
        self.system.fields.append(ejecta)
        self.system.update_accelerations()
        nova = Supernova.create(ejecta)
        self.supernovae.append(nova)
        reach = float(np.max(np.linalg.norm(self.system.x - position, axis=1))) if self.system.n else 5 * AU
        crossing = reach / nova.front_speed                 # seconds for the blast to pass the farthest planet
        self.speed_cap = crossing / DAY / BLAST_SECONDS
        self.blast_until = self.system.t + 1.1 * crossing
        self.blast_settled = False

        self.anchor = position / AU                         # keep the camera on the explosion site
        self.screen_flash = SCREEN_FLASH_FRAMES
        self.auto_frame = True
        self.reset_baselines()
        self.set_status(f"TYPE Ia SUPERNOVA! {mass / M_SUN:.1f} Suns of white dwarf went past the 1.44-Sun limit "
                        "and blew apart. Slow motion...", seconds=10)

    def black_hole_report(self) -> list[str]:
        state = self.hole_state()
        if state is None:
            return []
        k = self.system.names.index(self.hole.name)
        r_s = state[1] * AU
        lines = [f"{self.hole.name}: {self.system.m[k] / M_SUN / 1e6:.1f} million Suns, event horizon "
                 f"{r_s / 1e9:.1f} million km ({state[1]:.3f} AU)"]
        if self.debris is not None:
            total = len(self.debris.x) + self.debris.swallowed + self.debris.escaped
            lines.append(f"star debris: {self.debris.swallowed / total:.0%} fallen in, "
                         f"{self.debris.bound_fraction:.0%} bound to the hole, the rest flying away")
        return lines

    def supernova_report(self) -> str:
        if not self.supernovae:
            return ""
        nova, t = self.supernovae[-1], self.system.t
        age = nova.ejecta.age(t)
        age_text = f"{age / 3600:.1f} hours" if age < 2 * DAY else f"{age / DAY:,.0f} days"
        return (f"supernova {age_text} old: blast front {nova.ejecta.front(t) / AU:,.1f} AU out "
                f"at {nova.front_speed / 1e3:,.0f} km/s, debris ~{debris_temperature(age / DAY):,.0f} K")

    def inspiral_countdown(self) -> str:
        if self.inspiral is None:
            return ""
        names = self.system.names
        if self.inspiral.first not in names or self.inspiral.second not in names:
            return ""
        i, j = names.index(self.inspiral.first), names.index(self.inspiral.second)
        distance = float(np.linalg.norm(self.system.x[j] - self.system.x[i]))
        days = max(self.inspiral.time_left(distance), 0.0) / DAY
        return f"the stars are {distance / AU:.3f} AU apart - merging in ~{days:,.0f} days"

    def step_frame(self) -> None:
        self.poll_fetch()
        if not self.paused:
            self.advance()
        self.draw()

    # ----- drawing

    def draw(self) -> None:
        if self.follow >= 0:
            target = self.system.x[self.follow] / AU
        elif self.anchor is not None:
            target = self.anchor
        else:
            target = self.system.center_of_mass() / AU
        if np.all(np.isfinite(target)):
            self.camera.target = target
        positions = self.system.x / AU
        if self.auto_frame and self.system.n:
            # Keep the planets in view; while the debris is still glowing, pull back to follow it out too.
            spread = float(np.max(np.linalg.norm(positions - self.camera.target, axis=1)))
            goal = max(self.preset.view_distance / AU, 2.6 * spread)
            for nova in self.supernovae:
                if supernova_fade(nova.ejecta.age(self.system.t) / DAY) > 0.5:
                    goal = max(goal, min(2.2 * nova.ejecta.front(self.system.t, 0.5) / AU, 3000.0))
            self.camera.distance += (goal - self.camera.distance) * 0.03
        self.screen.fill(BACKGROUND)
        view = View(self.camera, self.screen.get_size())
        self.starfield.draw(self.screen, view, self.lens())
        if self.show_grid:
            self.draw_grid(view, positions)
            self.draw_hole_guides(view)
        if self.show_trails:
            self.draw_trails(view)
        self.draw_bodies(view, positions)
        self.draw_debris(view)
        self.draw_flashes(view)
        self.draw_supernovae(view)
        if self.screen_flash > 0:
            overlay = pygame.Surface(self.screen.get_size())
            overlay.fill((255, 250, 240))
            overlay.set_alpha(int(230 * (self.screen_flash / SCREEN_FLASH_FRAMES) ** 2))
            self.screen.blit(overlay, (0, 0))
        self.draw_hud()
        if self.show_help:
            self.draw_help()

    def draw_grid(self, view: View, positions: np.ndarray) -> None:
        """A grid on the z = 0 plane, plus a line from each body down to it to show its height."""
        spacing = nice_step(self.camera.distance / 4)
        half = 10
        cx, cy = (round(c / spacing) * spacing for c in self.camera.target[:2])
        ticks = np.arange(-half, half + 1) * spacing
        low, high, zeros = np.full_like(ticks, -half * spacing), np.full_like(ticks, half * spacing), np.zeros_like(ticks)
        starts = np.vstack([np.column_stack([cx + low, cy + ticks, zeros]), np.column_stack([cx + ticks, cy + low, zeros])])
        ends = np.vstack([np.column_stack([cx + high, cy + ticks, zeros]), np.column_stack([cx + ticks, cy + high, zeros])])
        draw_segments(self.screen, view, starts, ends, GRID_COLOUR)
        lifted = np.flatnonzero(np.all(np.isfinite(positions), axis=1) & (np.abs(positions[:, 2]) > 1e-9))
        if len(lifted):
            feet = positions[lifted] * np.array([1.0, 1.0, 0.0])
            draw_segments(self.screen, view, positions[lifted], feet, [dim(self.colours[i], 0.35) for i in lifted])

    def draw_trails(self, view: View) -> None:
        if self.trails.count < 2:
            return
        history = self.trails.ordered()
        if history.shape[1] > 900:                     # long trails: every other point looks the same, costs half
            history = np.concatenate([history[:, :-1:2], history[:, -1:]], axis=1)
        n = history.shape[1]
        for i in range(self.system.n):
            base = dim(self.colours[i], 0.6)
            for k in range(4):                         # older quarters are darker, so trails fade out
                a = k * (n - 1) // 4
                b = (k + 1) * (n - 1) // 4 + 1
                draw_polyline(self.screen, view, history[i, a:b], dim(base, 0.3 + 0.7 * (k + 1) / 4), antialias=True)

    def hole_state(self) -> tuple[np.ndarray, float] | None:
        """(position [AU], horizon radius [AU]) of the scene's black hole, if there is one."""
        if self.hole is None or self.hole.name not in self.system.names:
            return None
        k = self.system.names.index(self.hole.name)
        return self.system.x[k] / AU, self.hole.schwarzschild_radius(self.system) / AU

    def lens(self) -> tuple[np.ndarray, float] | None:
        return self.hole_state()

    def draw_hole_guides(self, view: View) -> None:
        """Faint rings in the hole's plane: the innermost stable orbit (3 r_s) and the distance at
        which a Sun-like star is torn apart by tides."""
        state = self.hole_state()
        if state is None:
            return
        centre, r_s = state
        k = self.system.names.index(self.hole.name)
        circle = np.linspace(0.0, 2.0 * math.pi, 121)
        rings = [(3.0 * r_s, (60, 120, 140), "innermost stable orbit (3 r_s)", -140.0),
                 (physics.tidal_radius(6.957e8, 1.989e30, self.system.m[k]) / AU, (120, 60, 60),
                  "Sun-like stars are torn apart inside this", -40.0)]
        hole = view.to_camera(centre[np.newaxis, :])[0]
        for radius, colour, text, angle in rings:
            ring = centre + radius * np.column_stack([np.cos(circle), np.sin(circle), np.zeros_like(circle)])
            draw_polyline(self.screen, view, ring, colour, antialias=True)
            big_enough = hole[2] > view.near and radius * view.focal / hole[2] > 60    # label only when zoomed in
            if self.show_labels and big_enough:
                a = math.radians(angle)
                point = view.to_camera((centre + radius * np.array([math.cos(a), math.sin(a), 0.0]))[np.newaxis, :])[0]
                if point[2] > view.near:
                    x, y = view.to_screen(point)
                    self.screen.blit(self.label(text, colour), (int(x + 4), int(y - 8)))

    def draw_debris(self, view: View) -> None:
        """A star's remains after a tidal disruption: hot near the hole, cooling as they fly out,
        plus a glow from the fragments falling in."""
        state = self.hole_state()
        if state is None:
            return
        centre, r_s = state
        layer = self.glow_layer()
        hole = view.to_camera(centre[np.newaxis, :])[0]
        if self.flare > 0.02 and hole[2] > view.near:
            hx, hy = view.to_screen(hole)
            glow = self.sprites.nova_glow(min(self.flare, 1.0), max(30.0, 7.0 * r_s * view.focal / hole[2]))
            layer.blit(glow, (int(hx - glow.get_width() / 2), int(hy - glow.get_height() / 2)),
                       special_flags=pygame.BLEND_RGB_ADD)
        if self.debris is not None and len(self.debris.x):
            points = self.debris.x / AU
            r = np.linalg.norm(points - centre, axis=1) / r_s
            cam = view.to_camera(points)
            ahead = cam[:, 2] > view.near
            xy = np.clip(view.to_screen(cam[ahead]), -1e5, 1e5).astype(int)
            kelvin = np.clip(3000.0 + 14000.0 * 6.0 / np.maximum(r[ahead], 6.0), 2500.0, 16000.0)
            redshift = np.sqrt(np.clip(1.0 - 1.0 / np.maximum(r[ahead], 1.0), 0.0, 1.0))
            colours = (temperature_colour(kelvin) * redshift[:, None]).astype(int)
            for (x, y), colour in zip(xy.tolist(), colours.tolist()):
                pygame.draw.circle(layer, colour, (x, y), 1)
        self.screen.blit(layer, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

    def draw_bodies(self, view: View, positions: np.ndarray) -> None:
        cam = view.to_camera(positions)
        finite = np.all(np.isfinite(cam), axis=1)
        depth = np.where(finite, cam[:, 2], -np.inf)
        screen = view.to_screen(np.where(finite[:, None], cam, 1.0))
        state = self.hole_state()
        for i in np.argsort(-depth):                   # farthest first
            if depth[i] <= view.near:
                continue
            x, y = screen[i]
            is_star = i in self.star_set
            radius = self.radii[i] * self.size_factor * view.focal / depth[i]
            if self.is_hole[i]:                        # a black disk (the horizon) inside its photon ring
                horizon = max(self.radii[i] * view.focal / depth[i], 2.0)
                pygame.draw.circle(self.screen, (0, 0, 0), (int(x), int(y)), int(horizon))
                ring, m = self.sprites.ring(max(PHOTON_RING * horizon, 5.0), self.colours[i])
                self.screen.blit(ring, (int(x - m), int(y - m)), special_flags=pygame.BLEND_RGB_ADD)
                radius = PHOTON_RING * horizon
            else:
                colour = self.colours[i]
                if state is not None:                  # light climbing out of the hole's gravity loses energy
                    distance = float(np.linalg.norm(positions[i] - state[0])) / state[1]
                    colour = dim(colour, math.sqrt(max(0.0, 1.0 - 1.0 / max(distance, 1.0))))
                radius = min(max(radius, MIN_STAR_PX if is_star else MIN_PLANET_PX), MAX_BODY_PX)
                if is_star:
                    core, n, glow, m = self.sprites.star(radius, colour)
                    self.screen.blit(glow, (int(x - m), int(y - m)), special_flags=pygame.BLEND_RGB_ADD)
                    self.screen.blit(core, (int(x - n), int(y - n)))
                else:
                    sprite, n = self.sprites.planet(radius, colour, self.lights_for(i, cam))
                    self.screen.blit(sprite, (int(x - n), int(y - n)))
            if self.show_labels:
                label = self.label(self.system.names[i], self.colours[i])
                reach = 0.75 * radius if self.is_hole[i] else radius     # tuck labels in against a big ring
                self.screen.blit(label, (int(x + reach + 4), int(y - reach - label.get_height() + 4)))

    def draw_flashes(self, view: View) -> None:
        """An expanding glow and shock ring where each recent collision happened."""
        for flash in self.flashes:
            cam = view.to_camera(flash.position[np.newaxis, :])[0]
            if cam[2] <= view.near:
                continue
            x, y = view.to_screen(cam)
            progress = flash.age / FLASH_FRAMES
            fade = 1.0 - progress
            glow, m = self.sprites.burst(12 + 60 * progress, flash.colour, fade)
            self.screen.blit(glow, (int(x - m), int(y - m)), special_flags=pygame.BLEND_RGB_ADD)
            pygame.draw.circle(self.screen, dim(flash.colour, fade), (int(x), int(y)), int(8 + 110 * progress), 2)

    def glow_layer(self) -> pygame.Surface:
        """A black, screen-sized surface for drawing things that add light (blitted with BLEND_RGB_ADD)."""
        if self._glow_layer is None or self._glow_layer.get_size() != self.screen.get_size():
            self._glow_layer = pygame.Surface(self.screen.get_size())
        self._glow_layer.fill((0, 0, 0))
        return self._glow_layer

    def draw_supernovae(self, view: View) -> None:
        """The debris itself: every fragment where the physics puts it, coloured by temperature
        (blue-white when hot, through yellow and orange to red as it cools), plus a glow that
        follows the supernova's brightness and a ring marking the blast front."""
        t = self.system.t
        for nova in self.supernovae:
            age_days = nova.ejecta.age(t) / DAY
            light, fade = supernova_light(age_days), supernova_fade(age_days)
            layer = self.glow_layer()
            centre = view.to_camera(nova.ejecta.centre_at(t)[np.newaxis, :] / AU)[0]
            if centre[2] > view.near:
                cx, cy = view.to_screen(centre)
                px_per_au = view.focal / centre[2]
                strength = fade * max(light, math.exp(-nova.frames / 20.0))
                if strength > 0.02:
                    radius = min(max(45.0, 0.6 * nova.ejecta.front(t, 0.5) / AU * px_per_au), 650.0)
                    glow = self.sprites.nova_glow(min(strength, 1.0), radius)
                    layer.blit(glow, (int(cx - glow.get_width() / 2), int(cy - glow.get_height() / 2)),
                               special_flags=pygame.BLEND_RGB_ADD)
                front = nova.ejecta.front(t) / AU * px_per_au
                if 3 < front < 4000:
                    pygame.draw.circle(layer, dim((120, 170, 230), 0.25 + 0.5 * light * fade), (int(cx), int(cy)),
                                       int(front), 2)
            fragments = view.to_camera(nova.ejecta.positions(t) / AU)
            ahead = fragments[:, 2] > view.near
            points = np.clip(view.to_screen(fragments[ahead]), -1e5, 1e5).astype(int)
            kelvin = debris_temperature(age_days) * nova.heat[ahead]
            colours = (temperature_colour(kelvin) * (fade * (0.3 + 0.7 * light))).astype(int)
            sizes = 1 + (kelvin > 9000)
            for (x, y), colour, size in zip(points.tolist(), colours.tolist(), sizes.tolist()):
                pygame.draw.circle(layer, colour, (x, y), size)
            self.screen.blit(layer, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

    def lights_for(self, i: int, cam: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """Direction to each star (camera space), weighted by 1/distance^2 and normalised."""
        lights = []
        for s in self.stars:
            offset = cam[s] - cam[i]
            dist2 = float(offset @ offset)
            if s != i and math.isfinite(dist2) and dist2 > 0:
                lights.append((offset / math.sqrt(dist2), 1.0 / dist2))
        if not lights:
            return [(DEFAULT_LIGHT, 1.0)]
        total = sum(weight for _, weight in lights)
        return [(direction, weight / total) for direction, weight in lights]

    def label(self, text: str, colour) -> pygame.Surface:
        key = (text, tuple(colour))
        if key not in self._labels:
            self._labels[key] = self.label_font.render(text, True, mix(colour, TEXT_COLOUR, 0.5))
        return self._labels[key]

    def draw_hud(self) -> None:
        s = self.system
        drift = (s.energy() - self.energy0) / abs(self.energy0) if self.energy0 else 0.0
        momentum_drift = float(np.linalg.norm(s.momentum() - self.momentum0)) / self.momentum_scale
        drift_colour = GOOD_COLOUR if abs(drift) < 1e-4 else WARN_COLOUR if abs(drift) < 1e-2 else BAD_COLOUR
        if self.follow >= 0:
            following = s.names[self.follow]
        elif self.anchor is not None:
            following = "the supernova site"
        else:
            following = "centre of mass"
        if self.inspiral is not None:      # the pair is meant to lose energy, so show how much has gone
            energy_line = (f"gravitational waves have carried off {self.energy0 - s.energy():.2e} J", DIM_COLOUR)
        elif not self.blast_settled:       # the debris' pull is changing, so energy isn't conserved yet
            energy_line = ("energy: changing while the blast carries the star's mass past the planets", DIM_COLOUR)
        else:
            since = "  (since the blast passed)" if self.anchor is not None else "  (since last crash)" if self.collisions else ""
            energy_line = (f"energy drift   dE/E0 = {drift:+.2e}{since}", drift_colour)
        if self.blast_settled:
            momentum_line = (f"momentum drift       = {momentum_drift:.1e}", DIM_COLOUR)
        else:
            momentum_line = ("momentum: changing too, while the debris pulls on the planets", DIM_COLOUR)
        capped = ", max" if self.capped else ""
        slow = "  SLOW MOTION" if self.speed_cap is not None else ""
        lines = [
            (self.preset.name, TEXT_COLOUR),
            (f"t = {format_time(s.t)}", TEXT_COLOUR),
            (f"speed {format_speed(self.effective_speed)}  "
             f"(dt {format_duration(self.last_dt)}, {self.last_steps} steps/frame{capped}){slow}",
             WARN_COLOUR if slow else DIM_COLOUR),
            (f"integrator: {self.integrators[self.integrator]}", DIM_COLOUR),
            (f"collisions: {COLLISION_MODES[self.collision_mode]} (K)"
             + (f", {self.collisions} so far" if self.collisions else ""), DIM_COLOUR),
            energy_line,
            momentum_line,
            (f"camera on {following}, {self.camera.distance:.3g} AU away   {self.clock.get_fps():.0f} fps",
             DIM_COLOUR),
        ]
        countdown = self.inspiral_countdown()
        if countdown:
            lines.append((countdown, WARN_COLOUR))
        report = self.supernova_report()
        if report:
            lines.append((report, WARN_COLOUR))
        lines.extend((text, WARN_COLOUR) for text in self.black_hole_report())
        if not self.show_help:
            lines.append(("press H for help", DIM_COLOUR))
        if self.paused:
            lines.append(("PAUSED (space to resume)", WARN_COLOUR))
        if self.status_frames > 0:
            lines.append((self.status, WARN_COLOUR))
            self.status_frames -= 1
        self.draw_panel(lines, top=12)

    def draw_help(self) -> None:
        self.draw_panel([(f"{key:<13}{action}", DIM_COLOUR) for key, action in HELP_LINES], bottom=12)

    def draw_panel(self, lines, top: int | None = None, bottom: int | None = None) -> None:
        rendered = [self.font.render(text, True, colour) for text, colour in lines]
        width = max(r.get_width() for r in rendered) + 20
        height = sum(r.get_height() for r in rendered) + 14
        y = top if top is not None else self.screen.get_height() - height - (bottom or 0)
        panel = pygame.Surface((width, height), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 150))
        self.screen.blit(panel, (12, y))
        y += 7
        for r in rendered:
            self.screen.blit(r, (22, y))
            y += r.get_height()

    # ----- main loop

    def run(self, frames: int = 0, screenshot: str | None = None) -> None:
        count, running = 0, True
        while running:
            for event in pygame.event.get():
                running = self.handle_event(event) and running
            self.step_frame()
            pygame.display.flip()
            self.clock.tick(FPS)
            count += 1
            if frames and count >= frames:
                running = False
        if screenshot:
            pygame.image.save(self.screen, screenshot)
        pygame.quit()


def run(preset: str = "supernova", size: tuple[int, int] = (1100, 700), frames: int = 0,
        screenshot: str | None = None) -> None:
    App(preset, size).run(frames=frames, screenshot=screenshot)
