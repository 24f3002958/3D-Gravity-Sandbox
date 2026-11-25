"""Runs the viewer off-screen (SDL's dummy video driver) to catch drawing and input errors."""
import numpy as np
import pytest

pygame = pytest.importorskip("pygame")

from gravity_sandbox import physics  # noqa: E402
from gravity_sandbox.units import DAY  # noqa: E402


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from gravity_sandbox.app import App

    viewer = App("supernova", size=(640, 400))
    yield viewer
    pygame.quit()


def test_every_offline_preset_draws(app):
    for key in (pygame.K_2, pygame.K_4, pygame.K_6, pygame.K_7):   # 3 downloads from JPL, so it's left out
        app.handle_key(key)
        for _ in range(10):
            app.step_frame()
        assert np.all(np.isfinite(app.system.x))
        assert app.trails.count == 10


def test_keys_1_and_5_no_longer_have_scenes(app):
    name = app.preset.name
    app.handle_key(pygame.K_1)
    app.handle_key(pygame.K_5)
    assert app.preset.name == name


def test_controls_and_camera_do_not_crash(app):
    keys = [pygame.K_i, pygame.K_i, pygame.K_t, pygame.K_l, pygame.K_g, pygame.K_RIGHTBRACKET,
            pygame.K_LEFTBRACKET, pygame.K_UP, pygame.K_DOWN, pygame.K_c, pygame.K_h, pygame.K_SPACE, pygame.K_r]
    for key in keys:
        app.handle_key(key)
        app.step_frame()
    app.camera.rotate(40, -25)
    app.camera.zoom(3)
    app.step_frame()
    assert not app.paused                                   # R restarted the preset, which unpauses


def test_k_cycles_collision_modes(app):
    from gravity_sandbox.app import COLLISION_MODES

    seen = []
    for _ in range(3):
        app.handle_key(pygame.K_k)
        seen.append(COLLISION_MODES[app.collision_mode])
        radii = app.collision_radii()
        assert (radii is None) == (seen[-1] == "off")
    assert seen == ["drawn size", "off", "real size"]


def test_supernova_explodes_in_slow_motion_and_keeps_drawing(app):
    assert app.inspiral is not None and app.inspiral_countdown()
    app.days_per_second = 2000                 # races through the 3-year inspiral in about 90 frames
    for _ in range(150):
        app.step_frame()
        if app.supernovae:
            break
    assert app.supernovae, "the white dwarfs never exploded"
    nova = app.supernovae[0]
    assert nova.ejecta.age(app.system.t) == 0.0            # the frame stopped at the moment of the explosion
    assert app.speed_cap is not None and app.speed_cap < 0.1
    assert any(isinstance(source, physics.Ejecta) for source in app.system.fields)
    assert not any(name.startswith("White dwarf") for name in app.system.names)
    assert app.system.n == 5 and not app.stars and app.inspiral is None
    assert len(app.colours) == len(app.white_dwarf) == len(app.radii) == app.trails.points.shape[0] == 5
    assert "SUPERNOVA" in app.status and app.screen_flash > 0 and app.anchor is not None
    assert app.supernova_report().startswith("supernova")

    for _ in range(30):                        # slow-motion frames: debris, glow and blast front get drawn
        app.step_frame()
    assert app.speed_cap is not None and nova.ejecta.age(app.system.t) < DAY
    assert not app.blast_settled               # the planets still feel the debris
    app.handle_key(pygame.K_UP)                # Up takes over from the slow motion
    assert app.speed_cap is None
    app.days_per_second = 2000
    for _ in range(60):                        # the blast passes, the energy readout settles
        app.step_frame()
    assert app.blast_settled and app.auto_frame


def test_black_hole_scene_swallows_and_spaghettifies(app):
    app.handle_key(pygame.K_7)
    assert app.hole is not None and app.lens() is not None and app.black_hole_report()
    app.days_per_second = 10                   # the adaptive timestep still keeps every step accurate
    for _ in range(200):
        app.step_frame()
        if app.debris is not None:
            break
    assert "White dwarf" not in app.system.names               # crossed the event horizon
    assert "Sun-like star" not in app.system.names and app.debris is not None
    assert len(app.colours) == len(app.is_hole) == len(app.radii) == app.trails.points.shape[0] == app.system.n
    assert 0.3 < app.debris.bound_fraction < 0.7
    for _ in range(30):                        # the stream, the flare and the lensed sky all get drawn
        app.step_frame()
    assert len(app.black_hole_report()) == 2
    app.handle_key(pygame.K_k)
    app.handle_key(pygame.K_k)                 # collisions off: the horizon still swallows things
    assert app.collision_radii() is not None


def test_follow_cycles_through_the_bodies_and_back(app):
    n = app.system.n
    for _ in range(n + 1):
        app.handle_key(pygame.K_f)
        app.step_frame()
    assert app.follow == -1                                 # back on the centre of mass
    app.handle_key(pygame.K_f, pygame.KMOD_SHIFT)
    assert app.follow == n - 1
