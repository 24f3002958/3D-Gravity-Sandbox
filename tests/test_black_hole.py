import math

import numpy as np
import pytest

from gravity_sandbox import physics, presets
from gravity_sandbox.units import AU, C_LIGHT, G, M_SUN, R_SUN, schwarzschild_radius

M_HOLE = 4.3e6 * M_SUN
GM = G * M_HOLE
R_S = schwarzschild_radius(M_HOLE)


def hole_and_body(position, velocity, body_radius=0.0):
    s = physics.System(["Hole", "Body"], [M_HOLE, 1.0], [[0.0, 0.0, 0.0], position], [[0.0, 0.0, 0.0], velocity],
                       radius=[R_S, body_radius])
    s.fields.append(physics.BlackHole("Hole"))
    s.update_accelerations()
    return s


def circular(r):
    """Circular orbit in the relativistic model: h^2 = GM r / (1 - 1.5 r_s / r)."""
    return [r, 0.0, 0.0], [0.0, math.sqrt(GM * r / (1 - 1.5 * R_S / r)) / r, 0.0]


def run(s, until, fraction=0.005):
    hole = s.fields[0]
    events = []
    while s.t < until and s.n > 1:
        dt = max(hole.step_limit(s, fraction=fraction), 0.05)
        events += physics.step_with_collisions(s, dt, physics.leapfrog, s.radius)
        yield s, events


def test_schwarzschild_radius_of_sagittarius_a():
    assert R_S / AU == pytest.approx(0.0849, rel=2e-3)


def test_precession_far_out_matches_einstein():
    """Periapsis advance per orbit -> 3 pi r_s / (a (1 - e^2)) when the orbit is far from the hole."""
    r_peri, r_apo = 300 * R_S, 600 * R_S
    s = hole_and_body(*presets.apoapsis_start(GM, r_peri, r_apo, 0.0, 0.0))
    directions, previous = [], None
    for state, _ in run(s, 4.5 * 2 * math.pi * math.sqrt((450 * R_S) ** 3 / GM)):
        r = np.linalg.norm(state.x[1])
        if previous is not None and previous[0] > previous[1] < r:          # just passed periapsis
            rel, w = state.x[1], state.v[1]
            ecc = np.cross(w, np.cross(rel, w)) / GM - rel / r                # Runge-Lenz direction
            directions.append(math.atan2(ecc[1], ecc[0]))
        previous = (previous[1], r) if previous is not None else (r, r)
    advances = [(b - a) % (2 * math.pi) for a, b in zip(directions, directions[1:])]
    expected = 3 * math.pi * R_S / (2 * r_peri * r_apo / (r_peri + r_apo))
    assert len(advances) >= 3
    assert np.mean(advances) == pytest.approx(expected, rel=0.03)


def test_nothing_orbits_stably_inside_three_horizon_radii():
    outside = hole_and_body(*circular(3.3 * R_S))
    outside.x[1] *= 1.001                                     # a small nudge
    radii = [np.linalg.norm(s.x[1]) / R_S for s, _ in run(outside, 2 * 3600.0)]
    assert outside.n == 2 and 3.2 < min(radii) and max(radii) < 3.45

    inside = hole_and_body(*circular(2.9 * R_S))
    inside.x[1] *= 0.999                                      # the same nudge, inside the last stable orbit
    for state, events in run(inside, 2 * 3600.0):
        pass
    assert events and events[0].absorbed == "Body"            # it spiralled in and crossed the horizon


@pytest.mark.parametrize("factor, captured", [(0.98, True), (1.02, False)])
def test_capture_needs_less_than_2_rs_c_of_angular_momentum(factor, captured):
    r0, h = 200 * R_S, factor * 2 * R_S * C_LIGHT
    speed = math.sqrt(2 * GM / r0 + 2 * GM * h * h / (C_LIGHT**2 * r0**3))    # zero total energy
    v_t = h / r0
    s = hole_and_body([r0, 0.0, 0.0], [-math.sqrt(speed**2 - v_t**2), v_t, 0.0])
    closest = np.inf
    for state, events in run(s, 2 * r0 / math.sqrt(2 * GM / r0) * 3):
        if state.n > 1:
            closest = min(closest, np.linalg.norm(state.x[1]))
    assert bool(events) == captured
    if not captured:
        assert 2 * R_S < closest < 2.6 * R_S                 # it whirled round just outside 2 r_s and left


def test_energy_is_conserved_including_the_relativistic_term():
    s = hole_and_body(*presets.apoapsis_start(GM, 20 * R_S, 60 * R_S, 0.0, 30.0))
    e0 = s.energy()
    worst = max(abs(state.energy() / e0 - 1) for state, _ in run(s, 3 * 2 * math.pi * math.sqrt((40 * R_S) ** 3 / GM)))
    assert worst < 1e-4


def test_tidal_disruption_leaves_about_half_the_star_bound():
    r_t = physics.tidal_radius(R_SUN, M_SUN, M_HOLE)
    assert r_t / R_S == pytest.approx(8.9, rel=0.01)
    v = math.sqrt(2 * GM / (r_t * (1 - R_S / r_t)))            # zero relativistic energy, moving sideways
    s = physics.System(["Hole", "Star"], [M_HOLE, M_SUN], [[0, 0, 0], [r_t, 0, 0]], [[0, 0, 0], [0, v, 0]],
                       radius=[R_S, R_SUN])
    s.fields.append(physics.BlackHole("Hole"))
    debris = physics.tidal_disruption(s, 1, "Hole", 4000, np.random.default_rng(0))
    assert s.names == ["Hole"] and len(debris.x) == 4000
    assert 0.35 < debris.bound_fraction < 0.65
    assert np.all(np.linalg.norm(debris.x - [r_t, 0, 0], axis=1) <= R_SUN * (1 + 1e-9))
    debris.step(10.0, s.x[0], s.v[0], GM)
    assert debris.clear(s.x[0], 2 * R_S, 4000 * R_S) == 0     # nothing falls in straight away


def test_lensing_makes_two_images_and_hides_the_shadow(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from gravity_sandbox.app import Camera, Starfield, View

    view = View(Camera(yaw=0.0, pitch=0.0, distance=50 * R_S / AU), (800, 600))
    hole_dir = view.to_camera(np.zeros((1, 3)))[0]
    distance = np.linalg.norm(hole_dir)
    axis = hole_dir / distance
    side = np.cross(axis, [0.0, 1.0, 0.0])
    side /= np.linalg.norm(side)
    theta_e = math.sqrt(2 * R_S / AU / distance)
    behind = np.cos(1e-4) * axis + np.sin(1e-4) * side          # almost exactly behind the hole
    far = np.cos(1.0) * axis + np.sin(1.0) * side               # a radian away
    directions, colours, big = Starfield.lensed(np.array([behind, far]), np.full((2, 3), 100.0),
                                                np.array([False, False]), view, np.zeros(3), R_S / AU)
    angles = np.arccos(np.clip(directions @ axis, -1, 1))
    # The star behind the hole gets a second image on the far side of the ring. The far star's second
    # image would sit at about tE^2 / 1 rad, inside the shadow, so it's hidden: three images in all.
    assert len(directions) == 3
    assert angles[0] == pytest.approx(theta_e, rel=1e-3) and angles[2] == pytest.approx(theta_e, rel=1e-3)
    assert angles[1] == pytest.approx(1.0 + theta_e**2, rel=1e-2)   # far away: barely moved
    assert np.all(angles > 2.6 * R_S / AU / distance)            # nothing shows inside the shadow
    assert colours[0, 0] > 100.0                                 # the star behind the hole is magnified


def test_black_hole_scene_builds():
    preset = presets.black_hole()
    system = preset.build_system()
    assert any(isinstance(f, physics.BlackHole) for f in system.fields)
    assert preset.trail_length > 600 and system.n == 6
