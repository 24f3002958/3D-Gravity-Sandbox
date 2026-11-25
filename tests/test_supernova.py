import math

import numpy as np
import pytest

from gravity_sandbox import physics, presets
from gravity_sandbox.units import AU, DAY, G, M_CHANDRASEKHAR, M_SUN, circular_velocity

ENERGY = 1.5e44


def test_inspiral_follows_the_gravitational_wave_chirp():
    """r = r0 (1 - t/T)^(1/4) while the orbit stays roughly circular, and momentum is untouched."""
    m, r0 = M_SUN, 0.5 * AU
    v = circular_velocity(2 * m, r0) / 2
    system = physics.System(["A", "B"], [m, m], [[-r0 / 2, 0, 0], [r0 / 2, 0, 0]], [[0, -v, 0], [0, v, 0]])
    inspiral = physics.Inspiral("A", "B", merge_time=400 * DAY, start_distance=r0, merge_distance=0.1 * AU)
    dt = 0.02 * DAY
    separation_at = {0.5: None, 0.9: None}
    while physics.apply_inspiral(system, inspiral, dt) is None:
        physics.leapfrog(system, dt)
        r = np.linalg.norm(system.x[1] - system.x[0])
        for fraction in separation_at:
            if separation_at[fraction] is None and system.t >= fraction * inspiral.merge_time:
                separation_at[fraction] = r
        assert system.t < 2 * inspiral.merge_time, "the pair never merged"
    assert separation_at[0.5] == pytest.approx(r0 * 0.5**0.25, rel=0.01)
    assert separation_at[0.9] == pytest.approx(r0 * 0.1**0.25, rel=0.03)
    assert 0.98 < system.t / inspiral.merge_time < 1.03
    assert np.linalg.norm(system.momentum()) < 1e-12 * m * v


def test_remove_body():
    system = physics.System(["A", "B", "C"], [1.0, 2.0, 3.0], np.eye(3), np.zeros((3, 3)), radius=[1.0, 2.0, 3.0])
    physics.remove_body(system, 1)
    assert system.names == ["A", "C"]
    assert list(system.m) == [1.0, 3.0] and list(system.radius) == [1.0, 3.0]
    assert system.x.shape == system.v.shape == system.a.shape == (2, 3)


def test_debris_carries_the_explosion_energy_with_type_ia_speeds():
    mass = 1.8 * M_SUN
    ejecta = physics.Ejecta.from_explosion(np.zeros(3), np.zeros(3), 0.0, mass, ENERGY, 200_000,
                                           np.random.default_rng(3), clumps=50)
    assert 0.5 * mass * np.mean(ejecta.speeds**2) == pytest.approx(ENERGY, rel=0.02)
    v_e = math.sqrt(ENERGY / (6 * mass))                       # exponential profile: mean speed is 3 v_e
    assert np.mean(ejecta.speeds) == pytest.approx(3 * v_e, rel=0.01)
    assert np.all(np.diff(ejecta.speeds) >= 0)
    assert np.allclose(np.linalg.norm(ejecta.directions, axis=1), 1.0)


def test_debris_pulls_like_the_star_until_the_blast_has_passed():
    """Newton's shell theorem: the full pull while the debris is all inside, none once it has gone by."""
    mass = 1.8 * M_SUN
    ejecta = physics.Ejecta.from_explosion(np.zeros(3), np.zeros(3), 0.0, mass, ENERGY, 5000,
                                           np.random.default_rng(1))
    planet = np.array([[2 * AU, 0.0, 0.0]])
    star_pull = G * mass / (2 * AU) ** 2
    early = ejecta.acceleration(planet, 60.0, G)               # one minute in: all the debris is inside 2 AU
    assert np.linalg.norm(early) == pytest.approx(star_pull, rel=1e-9) and early[0, 0] < 0
    late = ejecta.acceleration(planet, 60 * DAY, G)            # two months in: it has all gone past 2 AU
    assert np.linalg.norm(late) < 1e-3 * star_pull
    inside = ejecta.enclosed_mass(np.linspace(0.5, 10, 50) * AU, 1 * DAY)    # 865 to 17,300 km/s after a day
    assert np.all(np.diff(inside) >= 0) and 0 < inside[0] < inside[-1] <= mass


def test_supernova_finale_runs_on_schedule_and_sets_the_planets_free():
    preset = presets.supernova_finale()
    system = preset.build_system()
    inspiral = preset.build_inspiral(system)
    planets = [n for n in system.names if not n.startswith("White dwarf")]
    start = {n: np.linalg.norm(system.x[system.names.index(n)]) for n in planets}
    widest = {n: 0.0 for n in planets}
    pair = None
    while pair is None:
        assert not physics.step_with_collisions(system, preset.dt_max, physics.leapfrog, system.radius)
        pair = physics.apply_inspiral(system, inspiral, preset.dt_max)
        com = system.center_of_mass()
        for n in planets:
            change = abs(np.linalg.norm(system.x[system.names.index(n)] - com) / start[n] - 1)
            widest[n] = max(widest[n], change)
        assert system.t < 1300 * DAY, "the white dwarfs never merged"

    assert system.t / DAY == pytest.approx(1103.6, rel=0.01)       # about 37 s at the default 30 days/s
    assert max(widest.values()) < 0.05                             # every planet kept a steady orbit
    event = physics.merge(system, *pair)
    star = system.names.index(event.survivor)
    mass, velocity = system.m[star], system.v[star].copy()
    assert mass == pytest.approx(1.8 * M_SUN) and mass > M_CHANDRASEKHAR

    physics.remove_body(system, star)                              # Type Ia: the star becomes debris
    system.fields.append(physics.Ejecta.from_explosion(event.position, velocity, system.t, mass, ENERGY, 2000,
                                                       np.random.default_rng(5)))
    system.update_accelerations()
    speeds = np.linalg.norm(system.v, axis=1)
    distances = np.linalg.norm(system.x - event.position, axis=1)
    for _ in range(int(120 * DAY / preset.dt_max)):
        physics.leapfrog(system, preset.dt_max)
    # The blast passes in hours, long before the orbits can bend much, then they coast in straight lines.
    assert np.allclose(np.linalg.norm(system.v, axis=1), speeds, rtol=5e-3)
    assert np.all(np.linalg.norm(system.x - event.position, axis=1) > distances)      # all heading away
