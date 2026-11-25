import math

import numpy as np
import pytest

from gravity_sandbox import physics
from gravity_sandbox.presets import supernova_finale
from gravity_sandbox.units import AU, G, M_SUN, orbital_period

STEPS_PER_ORBIT = 500


def two_body(eccentricity: float = 0.0):
    """Sun + Earth-mass planet starting at perihelion of an orbit with a = 1 AU."""
    m_planet = 5.97e24
    r_peri = AU * (1 - eccentricity)
    v_peri = math.sqrt(G * (M_SUN + m_planet) * (1 + eccentricity) / r_peri)
    system = physics.System(
        names=["Sun", "Planet"],
        m=[M_SUN, m_planet],
        x=[[0, 0, 0], [r_peri, 0, 0]],
        v=[[0, 0, 0], [0, v_peri, 0]],
    )
    system.recenter()
    return system, orbital_period(M_SUN + m_planet, AU)


def energy_errors(system, integrator, dt, steps):
    e0 = system.energy()
    errors = np.empty(steps)
    for k in range(steps):
        integrator(system, dt)
        errors[k] = abs((system.energy() - e0) / e0)
    return errors


def test_forces_are_equal_and_opposite():
    rng = np.random.default_rng(1)
    m = rng.uniform(1e23, 1e30, 6)
    x = rng.normal(0.0, AU, (6, 3))
    forces = m[:, None] * physics.accelerations(x, m)
    assert np.linalg.norm(forces.sum(axis=0)) < 1e-12 * np.abs(forces).sum()


def test_matches_newtons_law_for_two_bodies():
    a = physics.accelerations(np.array([[0.0, 0.0, 0.0], [AU, 0.0, 0.0]]), np.array([M_SUN, 1.0]))
    assert a[1, 0] == pytest.approx(-G * M_SUN / AU**2, rel=1e-12)
    assert a[0, 0] == pytest.approx(G * 1.0 / AU**2, rel=1e-12)


def test_softening_keeps_overlapping_bodies_finite():
    a = physics.accelerations(np.zeros((2, 3)), np.array([1e30, 1e30]), softening=1e9)
    assert np.all(np.isfinite(a))


def test_circular_orbit_closes_after_one_period():
    system, period = two_body()
    start = system.x[1].copy()
    steps = 2000
    for _ in range(steps):
        physics.leapfrog(system, period / steps)
    assert np.linalg.norm(system.x[1] - start) < 1e-4 * AU


def test_leapfrog_beats_euler_and_symplectic_errors_stay_bounded():
    errors = {}
    for name, integrator in physics.INTEGRATORS.items():
        system, period = two_body(eccentricity=0.5)
        errors[name] = energy_errors(system, integrator, period / STEPS_PER_ORBIT, 20 * STEPS_PER_ORBIT)

    leapfrog = errors["Leapfrog"]
    semi_implicit = errors["Semi-implicit Euler (2D version)"]
    explicit = errors["Explicit Euler"]

    assert leapfrog.max() < 1e-3
    assert semi_implicit.max() > 10 * leapfrog.max()
    assert explicit.max() > 100 * leapfrog.max()

    def first_and_last_orbit(err):
        return err[:STEPS_PER_ORBIT].max(), err[-STEPS_PER_ORBIT:].max()

    for err in (leapfrog, semi_implicit):         # symplectic: the error oscillates but doesn't grow
        first, last = first_and_last_orbit(err)
        assert last < 1.5 * first
    first, last = first_and_last_orbit(explicit)  # not symplectic: the error keeps growing
    assert last > 2 * first


def test_momentum_is_conserved():
    system = supernova_finale().build_system()
    scale = np.sum(system.m * np.linalg.norm(system.v, axis=1))
    for _ in range(2000):
        physics.leapfrog(system, 4320.0)
    assert np.linalg.norm(system.momentum()) < 1e-12 * scale
