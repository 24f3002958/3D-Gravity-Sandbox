"""N-body gravity engine: state, forces, integrators and conservation checks.

State lives in numpy arrays, so every body's acceleration is computed in one
vectorised pass instead of a Python loop over pairs:

    m : (N,)   masses         [kg]
    x : (N, 3) positions      [m]
    v : (N, 3) velocities     [m/s]
    a : (N, 3) accelerations  [m/s^2]  (always kept in step with x)

Every integrator here costs one force evaluation per step, so they can be
compared fairly using the energy-drift readout in the viewer.

Collisions are optional: give step_with_collisions() a radius for every body
and any two bodies that touch during a step are merged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

import numpy as np

from .units import C_LIGHT, G as G_SI


def accelerations(x: np.ndarray, m: np.ndarray, G: float = G_SI, softening: float = 0.0) -> np.ndarray:
    """Acceleration of every body caused by all the others.

        a_i = G * sum_j  m_j (x_j - x_i) / (|x_j - x_i|^2 + eps^2)^(3/2)

    Every pair is evaluated from the same positions, so the forces come out
    equal and opposite and total momentum is conserved. The softening length
    ``eps`` stops the force blowing up when two bodies pass very close.
    """
    d = x[np.newaxis, :, :] - x[:, np.newaxis, :]      # d[i, j] = x_j - x_i
    r2 = np.einsum("ijk,ijk->ij", d, d) + softening**2
    np.fill_diagonal(r2, np.inf)                       # a body doesn't pull on itself
    weights = m[np.newaxis, :] * r2**-1.5              # m_j / r_ij^3
    return G * np.einsum("ij,ijk->ik", weights, d)


@dataclass
class System:
    """A set of gravitating bodies plus the simulation clock ``t`` [s].

    ``radius`` holds each body's physical radius [m]; 0 means a point mass.
    ``fields`` are extra influences that aren't plain bodies, such as supernova
    debris or a black hole's relativistic correction: objects with a
    ``field_acceleration(system)`` method and, optionally, ``field_energy(system)``.
    """

    names: list[str]
    m: np.ndarray
    x: np.ndarray
    v: np.ndarray
    radius: np.ndarray | None = None
    softening: float = 0.0
    G: float = G_SI
    t: float = 0.0
    fields: list = field(default_factory=list)
    a: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.names = list(self.names)
        self.m = np.array(self.m, dtype=float)
        self.x = np.array(self.x, dtype=float).reshape(-1, 3)
        self.v = np.array(self.v, dtype=float).reshape(-1, 3)
        self.radius = np.zeros(len(self.m)) if self.radius is None else np.array(self.radius, dtype=float)
        if not len(self.names) == len(self.m) == len(self.x) == len(self.v) == len(self.radius):
            raise ValueError("names, m, x, v and radius must describe the same number of bodies")
        self.update_accelerations()

    @property
    def n(self) -> int:
        return len(self.m)

    def update_accelerations(self) -> None:
        self.a = accelerations(self.x, self.m, self.G, self.softening)
        for source in self.fields:
            self.a += source.field_acceleration(self)

    def recenter(self) -> None:
        """Switch to the centre-of-mass frame: centre of mass at the origin and at rest."""
        self.x -= self.center_of_mass()
        self.v -= self.momentum() / self.m.sum()
        # Shifting every position by the same amount leaves the accelerations unchanged.

    def center_of_mass(self) -> np.ndarray:
        return (self.m[:, None] * self.x).sum(axis=0) / self.m.sum()

    def momentum(self) -> np.ndarray:
        return (self.m[:, None] * self.v).sum(axis=0)

    def kinetic_energy(self) -> float:
        return 0.5 * float(np.sum(self.m * np.einsum("ij,ij->i", self.v, self.v)))

    def potential_energy(self) -> float:
        """Matches the softened force: U = -G m_i m_j / sqrt(r^2 + eps^2) for each pair."""
        i, j = np.triu_indices(self.n, k=1)
        r = np.sqrt(np.sum((self.x[i] - self.x[j]) ** 2, axis=1) + self.softening**2)
        return -self.G * float(np.sum(self.m[i] * self.m[j] / r))

    def energy(self) -> float:
        extra = sum(source.field_energy(self) for source in self.fields if hasattr(source, "field_energy"))
        return self.kinetic_energy() + self.potential_energy() + extra


def leapfrog(s: System, dt: float) -> None:
    """Kick-drift-kick leapfrog (velocity Verlet).

    Half a velocity kick, a full position drift, new forces, the other half
    kick. It is second order and symplectic: the energy error stays bounded
    instead of building up, which is why orbits stay closed.
    """
    s.v += 0.5 * dt * s.a
    s.x += dt * s.v
    s.t += dt
    s.update_accelerations()
    s.v += 0.5 * dt * s.a


def semi_implicit_euler(s: System, dt: float) -> None:
    """Velocity first, then position using the *new* velocity.

    This is the update rule the 2D sandbox used. It is symplectic as well
    (energy stays bounded) but only first order, so its errors are much larger
    than leapfrog's for the same timestep.
    """
    s.v += dt * s.a
    s.x += dt * s.v
    s.t += dt
    s.update_accelerations()


def explicit_euler(s: System, dt: float) -> None:
    """Textbook Euler: position from the *old* velocity.

    Energy grows on every orbit, so planets slowly spiral outwards. It is here
    to show why the choice of integrator matters.
    """
    s.x += dt * s.v
    s.v += dt * s.a
    s.t += dt
    s.update_accelerations()


INTEGRATORS: dict[str, Callable[[System, float], None]] = {
    "Leapfrog": leapfrog,
    "Semi-implicit Euler (2D version)": semi_implicit_euler,
    "Explicit Euler": explicit_euler,
}


# --------------------------------------------------------------------------- collisions

@dataclass
class Collision:
    """What happened when two bodies hit each other."""

    survivor: str            # the heavier body, which absorbs the other
    absorbed: str
    keep: int                # survivor's index before the absorbed body was removed
    removed: int             # absorbed body's index (that body no longer exists)
    position: np.ndarray     # where it happened [m]
    impact_speed: float      # relative speed as they met [m/s]
    time: float              # simulation time [s]
    growth: float            # survivor's new radius divided by its old radius


@lru_cache(maxsize=64)
def _pairs(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n, k=1)


def might_touch(x0: np.ndarray, x1: np.ndarray, radii: np.ndarray) -> bool:
    """Quick check before first_contact(): False means no pair can have touched this step.

    During a step two bodies can't get closer than their final distance minus
    how far each of them moved, so if that is still more than their combined
    radii for every pair, nothing touched.
    """
    moved = x1 - x0
    moved = np.sqrt(np.einsum("ij,ij->i", moved, moved))
    d = x1[np.newaxis, :, :] - x1[:, np.newaxis, :]
    gap = np.sqrt(np.einsum("ijk,ijk->ij", d, d)) - (moved + radii)[:, None] - (moved + radii)[None, :]
    np.fill_diagonal(gap, np.inf)
    return bool(gap.min() <= 0.0)


def first_contact(x0: np.ndarray, x1: np.ndarray, radii: np.ndarray) -> tuple[int, int, float] | None:
    """Earliest pair of bodies whose spheres touch while moving from ``x0`` to ``x1``.

    Each body is treated as moving in a straight line during the step and the
    whole path is checked, not just the end points, so a fast planet can't skip
    straight through a star between two steps. Returns (i, j, fraction of the
    step at first contact), or None if nothing touched.
    """
    i, j = _pairs(len(radii))
    reach = radii[i] + radii[j]
    start = x0[j] - x0[i]
    move = (x1[j] - x1[i]) - start
    # |start + t*move|^2 - reach^2 = a t^2 + b t + c
    a = np.einsum("ij,ij->i", move, move)
    b = 2.0 * np.einsum("ij,ij->i", start, move)
    c = np.einsum("ij,ij->i", start, start) - reach**2
    t_closest = np.clip(np.divide(-b, 2.0 * a, out=np.zeros_like(a), where=a > 0), 0.0, 1.0)
    hit = (reach > 0) & (c + t_closest * (b + a * t_closest) <= 0.0)
    if not hit.any():
        return None
    a, b, c = a[hit], b[hit], c[hit]
    root = np.sqrt(np.maximum(b * b - 4.0 * a * c, 0.0))
    t_first = np.where(c <= 0.0, 0.0, np.divide(-b - root, 2.0 * a, out=np.zeros_like(a), where=a > 0))
    k = int(np.argmin(t_first))
    return int(i[hit][k]), int(j[hit][k]), float(t_first[k])


def merge(s: System, i: int, j: int, impact_speed: float | None = None) -> Collision:
    """Perfectly inelastic collision: the lighter body is absorbed by the heavier one.

    Mass and momentum are conserved, the survivor ends up at the pair's centre
    of mass, and their volumes add up. Kinetic energy is not conserved: in a
    real impact it turns into heat.
    """
    if s.m[j] > s.m[i]:
        i, j = j, i
    mi, mj = s.m[i], s.m[j]
    total = mi + mj
    old_radius = s.radius[i]
    new_radius = float(np.cbrt(s.radius[i] ** 3 + s.radius[j] ** 3))
    event = Collision(
        survivor=s.names[i],
        absorbed=s.names[j],
        keep=i,
        removed=j,
        position=(mi * s.x[i] + mj * s.x[j]) / total,
        impact_speed=float(np.linalg.norm(s.v[j] - s.v[i])) if impact_speed is None else impact_speed,
        time=s.t,
        growth=new_radius / old_radius if old_radius > 0 else 1.0,
    )
    s.x[i] = event.position
    s.v[i] = (mi * s.v[i] + mj * s.v[j]) / total
    s.m[i] = total
    s.radius[i] = new_radius
    s.m = np.delete(s.m, j)
    s.x = np.delete(s.x, j, axis=0)
    s.v = np.delete(s.v, j, axis=0)
    s.radius = np.delete(s.radius, j)
    del s.names[j]
    s.update_accelerations()
    return event


def step_with_collisions(s: System, dt: float, integrator: Callable[[System, float], None] = leapfrog,
                         radii: np.ndarray | None = None) -> list[Collision]:
    """One integrator step; then, if ``radii`` [m] are given, merge every pair that touched.

    ``radii`` is usually ``s.radius`` (real sizes) but any per-body sizes work,
    e.g. the exaggerated sizes the viewer draws. Pass None to switch collisions off.
    """
    if radii is None or s.n < 2:
        integrator(s, dt)
        return []
    radii = np.array(radii, dtype=float)
    x0, v0 = s.x.copy(), s.v.copy()
    integrator(s, dt)
    if not might_touch(x0, s.x, radii):
        return []
    events = []
    while s.n > 1 and (contact := first_contact(x0, s.x, radii)) is not None:
        i, j, _ = contact
        mi, mj = s.m[i], s.m[j]
        # Speed from the start of the step: the end of the step may already be deep
        # inside the star, where the (point-mass) pull is unrealistically strong.
        event = merge(s, i, j, impact_speed=float(np.linalg.norm(v0[j] - v0[i])))
        keep, gone = event.keep, event.removed
        x0[keep] = (mi * x0[i] + mj * x0[j]) / (mi + mj)
        v0[keep] = (mi * v0[i] + mj * v0[j]) / (mi + mj)
        radii[keep] = np.cbrt(radii[i] ** 3 + radii[j] ** 3)
        x0, v0, radii = np.delete(x0, gone, axis=0), np.delete(v0, gone, axis=0), np.delete(radii, gone)
        events.append(event)
    return events


def remove_body(s: System, index: int) -> None:
    """Delete a body completely, e.g. a star destroyed in a supernova."""
    s.m = np.delete(s.m, index)
    s.x = np.delete(s.x, index, axis=0)
    s.v = np.delete(s.v, index, axis=0)
    s.radius = np.delete(s.radius, index)
    del s.names[index]
    s.update_accelerations()


# --------------------------------------------------------------------------- gravitational-wave inspiral

@dataclass
class Inspiral:
    """Two bodies spiralling together by giving off gravitational waves, sped up.

    Real gravitational waves drain a circular orbit's energy so that its size
    shrinks as r = r0 * (1 - t/T)^(1/4): slowly at first, then faster and
    faster (the "chirp" LIGO detects), reaching zero at time T. Here that is
    done with a drag on the pair's relative motion of strength
    k = (r0/r)^4 / (8T), which removes energy and angular momentum in exactly
    the same proportions, so the orbit follows the same law but with T chosen
    freely. The pair's total momentum is untouched.
    """

    first: str
    second: str
    merge_time: float          # T [s]
    start_distance: float      # r0 [m]
    merge_distance: float      # stop following the orbit and merge them below this [m]

    def time_left(self, distance: float) -> float:
        """Time [s] until the pair reaches merge_distance, for a circular orbit."""
        return self.merge_time * ((distance / self.start_distance) ** 4
                                  - (self.merge_distance / self.start_distance) ** 4)


def apply_inspiral(s: System, inspiral: Inspiral, dt: float) -> tuple[int, int] | None:
    """Apply one step's worth of inspiral. Returns the pair's indices once they are close enough to merge."""
    try:
        i, j = s.names.index(inspiral.first), s.names.index(inspiral.second)
    except ValueError:
        return None                                      # one of them is already gone
    distance = float(np.linalg.norm(s.x[j] - s.x[i]))
    if distance <= inspiral.merge_distance:
        return i, j
    k = (inspiral.start_distance / distance) ** 4 / (8.0 * inspiral.merge_time)
    change = (s.v[j] - s.v[i]) * np.expm1(-k * dt)     # relative velocity shrinks by exp(-k dt)
    total = s.m[i] + s.m[j]
    s.v[i] -= (s.m[j] / total) * change
    s.v[j] += (s.m[i] / total) * change
    return None


# --------------------------------------------------------------------------- supernova debris

@dataclass
class Ejecta:
    """Debris from a supernova coasting outward, and the pull it still has on everything else.

    A few seconds after the explosion the gas is moving far faster than its own
    pressure or gravity can change, so each fragment flies in a straight line at
    constant speed and the cloud expands homologously (speed proportional to
    distance). Speeds follow the exponential density profile used for Type Ia
    supernovae (Dwarkadas & Chevalier 1998), density ~ exp(-v / v_e) with
    v_e = sqrt(E / 6M), so the cloud carries kinetic energy E.

    By Newton's shell theorem a body at distance r from the centre feels only the
    debris still inside r, pulling as if it were all at the centre. So a planet
    keeps orbiting until the blast passes it, then carries on in a straight line.
    """

    centre: np.ndarray           # explosion point [m]
    drift: np.ndarray            # velocity of the cloud's centre [m/s]
    start: float                 # explosion time [s]
    mass: float                  # total debris mass [kg]
    directions: np.ndarray       # (P, 3) unit vectors, one per fragment
    speeds: np.ndarray           # (P,) fragment speeds [m/s], sorted

    @classmethod
    def from_explosion(cls, centre, drift, start: float, mass: float, energy: float, fragments: int,
                       rng: np.random.Generator, clumps: int = 0) -> "Ejecta":
        """Debris with kinetic energy ``energy`` [J]. With ``clumps`` > 0, most fragments are
        bunched into that many knots, like the clumpy debris seen in real supernova remnants."""
        v_e = math.sqrt(energy / (6.0 * mass))
        speeds = np.sort(rng.gamma(3.0, v_e, fragments))      # mass per speed ~ v^2 exp(-v / v_e)
        directions = rng.normal(size=(fragments, 3))
        if clumps:
            knots = rng.normal(size=(clumps, 3))
            knots /= np.linalg.norm(knots, axis=1, keepdims=True)
            bunched = rng.random(fragments) < 0.7
            directions[bunched] = (knots[rng.integers(0, clumps, bunched.sum())]
                                   + 0.18 * rng.normal(size=(bunched.sum(), 3)))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        return cls(np.array(centre, dtype=float), np.array(drift, dtype=float), float(start), float(mass),
                   directions, speeds)

    def age(self, t: float) -> float:
        return max(t - self.start, 0.0)

    def centre_at(self, t: float) -> np.ndarray:
        return self.centre + self.drift * (t - self.start)

    def positions(self, t: float) -> np.ndarray:
        return self.centre_at(t) + self.directions * (self.speeds * self.age(t))[:, None]

    def front(self, t: float, fraction: float = 0.95) -> float:
        """Radius [m] containing ``fraction`` of the debris."""
        return float(np.quantile(self.speeds, fraction)) * self.age(t)

    def enclosed_mass(self, r: np.ndarray, t: float) -> np.ndarray:
        """Debris mass [kg] still inside distance ``r`` [m] of the centre."""
        age = self.age(t)
        if age <= 0.0:
            return np.full(np.shape(r), self.mass)
        return self.mass * np.searchsorted(self.speeds, np.asarray(r) / age) / len(self.speeds)

    def acceleration(self, x: np.ndarray, t: float, G: float) -> np.ndarray:
        offset = x - self.centre_at(t)
        r = np.maximum(np.linalg.norm(offset, axis=1), 1.0)
        return -G * (self.enclosed_mass(r, t) / r**3)[:, None] * offset

    def field_acceleration(self, s: System) -> np.ndarray:
        return self.acceleration(s.x, s.t, s.G)


# --------------------------------------------------------------------------- black holes

def einstein_factor(r: np.ndarray, w: np.ndarray, gm: float) -> np.ndarray:
    """1 + 3 h^2 / (c^2 r^2) for positions ``r`` and velocities ``w`` relative to a hole of mass gm/G.

    Multiplying Newton's pull GM/r^2 by this gives the orbit equation of general
    relativity for a test body around a non-spinning black hole,
    u'' + u = GM/h^2 + 3 G M u^2 / c^2 (u = 1/r): exactly the Schwarzschild orbit shapes.
    """
    cross = np.cross(r, w)
    return 1.0 + 3.0 * np.einsum("ij,ij->i", cross, cross) / (C_LIGHT**2 * np.einsum("ij,ij->i", r, r))


@dataclass
class BlackHole:
    """Makes one body a non-spinning (Schwarzschild) black hole for everything orbiting it.

    Adds Einstein's correction to Newton's pull, a = -(3 G M h^2 / c^2 r^4) r_hat, where h
    is the orbiting body's angular momentum per unit mass. Orbits then follow exactly the
    shapes general relativity predicts: their closest points precess (by 3 pi r_s / (a (1-e^2))
    per orbit far out, much more close in), nothing orbits stably inside 3 r_s, and a body
    coming in with too little angular momentum spirals in. Timing very close to the hole
    is only approximate. The hole gets the equal and opposite push, so momentum is conserved.
    """

    name: str

    def schwarzschild_radius(self, s: System) -> float:
        return 2.0 * s.G * s.m[s.names.index(self.name)] / C_LIGHT**2

    def _relative(self, s: System):
        k = s.names.index(self.name)
        r, w = s.x - s.x[k], s.v - s.v[k]
        cross = np.cross(r, w)
        h2 = np.einsum("ij,ij->i", cross, cross)
        r2 = np.einsum("ij,ij->i", r, r)
        r2[k] = np.inf
        return k, r, h2, r2

    def field_acceleration(self, s: System) -> np.ndarray:
        k, r, h2, r2 = self._relative(s)
        a = -(3.0 * s.G * s.m[k] / C_LIGHT**2) * (h2 / r2**2.5)[:, None] * r
        a[k] = -(s.m[:, None] * a).sum(axis=0) / s.m[k]
        return a

    def field_energy(self, s: System) -> float:
        """The correction's potential energy, -G M m h^2 / (c^2 r^3) per body, so total energy stays conserved."""
        k, _, h2, r2 = self._relative(s)
        return float(-(s.G * s.m[k] / C_LIGHT**2) * np.sum(s.m * h2 / r2**1.5))

    def step_limit(self, s: System, extra: np.ndarray | None = None, fraction: float = 0.005) -> float:
        """A timestep [s] small enough for whatever is closest to the hole: ``fraction`` of the
        orbital timescale sqrt(r^3 / GM) there. ``extra`` adds more positions (e.g. debris)."""
        k = s.names.index(self.name)
        others = np.delete(s.x, k, axis=0)
        if extra is not None and len(extra):
            others = np.vstack([others, extra])
        if not len(others):
            return np.inf
        r = float(np.min(np.linalg.norm(others - s.x[k], axis=1)))
        return fraction * math.sqrt(r**3 / (s.G * s.m[k]))


def tidal_radius(star_radius: float, star_mass: float, hole_mass: float) -> float:
    """Distance [m] inside which a black hole's tides tear a star apart: R* (M_hole / M*)^(1/3)."""
    return star_radius * (hole_mass / star_mass) ** (1.0 / 3.0)


@dataclass
class Debris:
    """Massless fragments, e.g. a star torn apart by tides, moving in a black hole's gravity.

    They follow the same relativistic orbits as the bodies (Newton times einstein_factor).
    """

    x: np.ndarray                    # (P, 3) positions [m]
    v: np.ndarray                    # (P, 3) velocities [m/s]
    bound_fraction: float = 0.0      # share of the fragments on closed orbits when they were made
    swallowed: int = 0
    escaped: int = 0

    def accelerations(self, centre: np.ndarray, centre_velocity: np.ndarray, gm: float) -> np.ndarray:
        r = self.x - centre
        factor = einstein_factor(r, self.v - centre_velocity, gm)
        return -gm * (factor / np.einsum("ij,ij->i", r, r) ** 1.5)[:, None] * r

    def step(self, dt: float, centre: np.ndarray, centre_velocity: np.ndarray, gm: float) -> None:
        """Kick-drift-kick, like the bodies (the hole barely moves during one step)."""
        self.v += 0.5 * dt * self.accelerations(centre, centre_velocity, gm)
        self.x += dt * self.v
        self.v += 0.5 * dt * self.accelerations(centre, centre_velocity, gm)

    def clear(self, centre: np.ndarray, capture_radius: float, escape_radius: float) -> int:
        """Drop fragments that fell inside ``capture_radius`` (swallowed) or flew past
        ``escape_radius`` (gone for good). Returns how many were swallowed this time."""
        r = np.linalg.norm(self.x - centre, axis=1)
        inside, outside = r < capture_radius, r > escape_radius
        keep = ~(inside | outside)
        self.x, self.v = self.x[keep], self.v[keep]
        self.swallowed += int(inside.sum())
        self.escaped += int(outside.sum())
        return int(inside.sum())


def tidal_disruption(s: System, index: int, hole: str, fragments: int, rng: np.random.Generator) -> Debris:
    """Tear body ``index`` apart and replace it with ``fragments`` pieces of debris.

    The fragments fill the star's volume and all keep the star's velocity (the
    usual "frozen-in" approximation). The hole pulls harder on the near side than
    the far side, so they end up with a spread of orbital energies: roughly half
    on closed orbits that will fall back, half flung away for good.
    """
    k = s.names.index(hole)
    centre, velocity, radius = s.x[index].copy(), s.v[index].copy(), s.radius[index]
    offsets = rng.normal(size=(fragments, 3))
    offsets *= (rng.random(fragments) ** (1.0 / 3.0) / np.linalg.norm(offsets, axis=1))[:, None]  # inside a ball
    x = centre + radius * offsets
    v = np.repeat(velocity[np.newaxis, :], fragments, axis=0)
    gm, rel_x, rel_v = s.G * s.m[k], x - s.x[k], v - s.v[k]
    r = np.linalg.norm(rel_x, axis=1)
    cross = np.cross(rel_x, rel_v)
    h2 = np.einsum("ij,ij->i", cross, cross)
    # Orbital energy including the relativistic term (the same one BlackHole.field_energy uses).
    energy = 0.5 * np.einsum("ij,ij->i", rel_v, rel_v) - gm / r - gm * h2 / (C_LIGHT**2 * r**3)
    remove_body(s, index)
    return Debris(x, v, bound_fraction=float(np.mean(energy < 0)))
