import math

import numpy as np
import pytest

from gravity_sandbox import physics, presets
from gravity_sandbox.units import AU, DAY

OFFLINE_PRESETS = [key for key in presets.PRESETS if key != "solar-live"]    # solar-live needs the internet


@pytest.mark.parametrize("key", OFFLINE_PRESETS)
def test_preset_builds_a_recentred_system(key):
    preset = presets.PRESETS[key]()
    system = preset.build_system()
    assert system.n == len(preset.bodies) >= 3
    assert np.all(np.isfinite(system.a))
    assert np.linalg.norm(system.center_of_mass()) < 1e-6 * AU
    assert np.linalg.norm(system.momentum()) < 1e-12 * np.sum(system.m * np.linalg.norm(system.v, axis=1))
    assert preset.dt_max > 0 and preset.days_per_second > 0


def test_figure_eight_returns_to_its_start_after_one_period():
    system = presets.figure_eight().build_system()
    start = system.x.copy()
    steps = 4000
    dt = presets.figure_eight_period() / steps
    for _ in range(steps):
        physics.leapfrog(system, dt)
    assert np.abs(system.x - start).max() < 1e-4 * AU


def heliocentric_period(name: str, max_days: float = 420, dt: float = 0.25 * DAY) -> float:
    """Days for a planet to sweep 360 degrees around the Sun, starting from the JPL snapshot."""
    system = presets.solar_system().build_system()
    sun, body = system.names.index("Sun"), system.names.index(name)

    def angle() -> float:
        rel = system.x[body] - system.x[sun]
        return math.atan2(rel[1], rel[0])

    previous, swept = angle(), 0.0
    while system.t < max_days * DAY:
        physics.leapfrog(system, dt)
        current = angle()
        step = (current - previous + math.pi) % (2 * math.pi) - math.pi
        if swept + step >= 2 * math.pi:     # interpolate inside the step where the lap completes
            return (system.t - dt + dt * (2 * math.pi - swept) / step) / DAY
        swept += step
        previous = current
    raise AssertionError(f"{name} did not complete an orbit in {max_days} days")


@pytest.mark.parametrize("name, real_days", [("Mercury", 87.969), ("Venus", 224.701), ("Earth", 365.256)])
def test_planet_years_match_reality(name, real_days):
    assert heliocentric_period(name) == pytest.approx(real_days, rel=1e-3)


def test_scene_list():
    assert set(presets.PRESETS) == {"supernova", "solar", "solar-live", "figure8", "blackhole"}
