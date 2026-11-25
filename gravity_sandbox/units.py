"""Physical constants and orbit helpers.

The simulation works in SI units (metres, kilograms, seconds) everywhere.
Only the viewer converts positions to astronomical units for drawing.
"""
from __future__ import annotations

import math

G = 6.67430e-11          # gravitational constant [m^3 kg^-1 s^-2] (CODATA 2018)
C_LIGHT = 299_792_458.0  # speed of light [m/s] (exact)
AU = 1.495978707e11      # astronomical unit [m] (exact, IAU 2012)
DAY = 86_400.0           # [s]
YEAR = 365.25 * DAY      # Julian year [s]

# G*M of the Sun is known far more precisely than G or M on their own,
# so the Sun's mass is derived from it.
GM_SUN = 1.32712440041279419e20   # [m^3 s^-2] (JPL DE440)
M_SUN = GM_SUN / G                # about 1.988e30 kg
R_SUN = 6.957e8                   # nominal solar radius [m] (IAU 2015)
R_EARTH = 6.371e6                 # mean Earth radius [m]
M_CHANDRASEKHAR = 1.44 * M_SUN    # the most a white dwarf can weigh before it collapses or explodes


def schwarzschild_radius(mass: float) -> float:
    """Event-horizon radius [m] of a non-spinning black hole of ``mass`` [kg]: r_s = 2GM/c^2."""
    return 2.0 * G * mass / C_LIGHT**2


def circular_velocity(central_mass: float, radius: float) -> float:
    """Speed [m/s] for a circular orbit of ``radius`` [m] around a much heavier ``central_mass`` [kg]."""
    return math.sqrt(G * central_mass / radius)


def orbital_period(total_mass: float, semi_major_axis: float) -> float:
    """Kepler's third law: period [s] of a two-body orbit."""
    return 2.0 * math.pi * math.sqrt(semi_major_axis**3 / (G * total_mass))
