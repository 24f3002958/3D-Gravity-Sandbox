import math

import numpy as np
import pytest

from gravity_sandbox import physics, presets
from gravity_sandbox.units import R_EARTH, R_SUN

OFFLINE_PRESETS = [key for key in presets.PRESETS if key != "solar-live"]


def test_head_on_merger_conserves_mass_momentum_and_volume():
    system = physics.System(
        names=["Heavy", "Light"],
        m=[6e24, 3e24],
        x=[[-1e8, 0, 0], [1e8, 0, 0]],
        v=[[2e4, 0, 0], [-3e4, 500, 0]],
        radius=[R_EARTH, R_EARTH],
    )
    mass, momentum = system.m.sum(), system.momentum().copy()
    events = []
    for _ in range(500):
        events += physics.step_with_collisions(system, 60.0, physics.leapfrog, system.radius)
        if events:
            break
    assert len(events) == 1 and system.n == 1
    event = events[0]
    assert (event.survivor, event.absorbed) == ("Heavy", "Light")      # the heavier body survives
    assert system.names == ["Heavy"]
    assert system.m[0] == pytest.approx(mass)
    assert np.allclose(system.momentum(), momentum, rtol=0, atol=1e-12 * np.abs(momentum).max())
    assert system.radius[0] == pytest.approx(np.cbrt(2) * R_EARTH)    # volumes add up
    assert event.growth == pytest.approx(np.cbrt(2))


def test_contact_is_found_even_when_a_step_jumps_right_over_the_star():
    radii = np.array([R_SUN, R_EARTH])
    x0 = np.array([[0.0, 0.0, 0.0], [-1.6e9, 1e8, 0.0]])
    x1 = np.array([[0.0, 0.0, 0.0], [1.6e9, 1e8, 0.0]])
    reach = R_SUN + R_EARTH
    assert min(np.linalg.norm(x0[1]), np.linalg.norm(x1[1])) > reach      # both ends of the step are outside
    assert physics.might_touch(x0, x1, radii)
    contact = physics.first_contact(x0, x1, radii)
    assert contact is not None
    i, j, t = contact
    assert (i, j) == (0, 1)
    assert t == pytest.approx((1.6e9 - math.sqrt(reach**2 - 1e8**2)) / 3.2e9)    # moment of first touch


def test_near_misses_and_point_masses_do_not_collide():
    x0 = np.array([[0.0, 0.0, 0.0], [-1.6e9, 3 * R_SUN, 0.0]])
    x1 = np.array([[0.0, 0.0, 0.0], [1.6e9, 3 * R_SUN, 0.0]])
    assert physics.first_contact(x0, x1, np.array([R_SUN, R_EARTH])) is None
    through_x0 = np.array([[0.0, 0.0, 0.0], [-1.6e9, 0.0, 0.0]])
    through_x1 = np.array([[0.0, 0.0, 0.0], [1.6e9, 0.0, 0.0]])
    assert physics.first_contact(through_x0, through_x1, np.zeros(2)) is None    # radius 0 = point mass


def test_collisions_can_be_switched_off():
    def overlapping_pair():
        return physics.System(["A", "B"], [1e24, 1e24], [[0, 0, 0], [1e6, 0, 0]], [[0, 0, 0], [0, 0, 0]],
                              radius=[R_EARTH, R_EARTH])

    system = overlapping_pair()
    assert physics.step_with_collisions(system, 1.0, physics.leapfrog, None) == []
    assert system.n == 2
    system = overlapping_pair()
    assert len(physics.step_with_collisions(system, 1.0, physics.leapfrog, system.radius)) == 1
    assert system.n == 1


@pytest.mark.parametrize("key", OFFLINE_PRESETS)
def test_no_bodies_start_overlapping(key):
    """Otherwise switching collisions on would merge bodies the moment a preset starts."""
    bodies = presets.PRESETS[key]().bodies
    positions = np.array([b.position for b in bodies])
    i, j = np.triu_indices(len(bodies), k=1)
    distance = np.linalg.norm(positions[i] - positions[j], axis=1)
    for sizes in ([b.radius for b in bodies], [b.display_radius for b in bodies]):
        sizes = np.array(sizes)
        assert np.all(distance > sizes[i] + sizes[j])
