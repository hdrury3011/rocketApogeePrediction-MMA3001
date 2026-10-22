"""Atmosphere models returning (density, speed of sound) for a given altitude.

Altitude is treated as geopotential altitude above mean sea level (m).
"""
import numpy as np

G0 = 9.80665        # standard gravity (m/s^2)
R_AIR = 287.053     # specific gas constant of dry air (J/(kg K))
GAMMA = 1.4         # ratio of specific heats
T0 = 288.15         # sea-level temperature (K)
P0 = 101325.0       # sea-level pressure (Pa)
LAPSE = 0.0065      # tropospheric lapse rate (K/m)
H_MIN = -500.0      # lowest altitude accepted (m)
H_MAX = 11000.0     # top of the troposphere (m); model not valid above this

_EXPONENT = G0 / (R_AIR * LAPSE)


def isa(h):
    """International Standard Atmosphere, troposphere only.

    Parameters
    ----------
    h : float
        Altitude above mean sea level (m), within [-500, 11000].

    Returns
    -------
    tuple of float
        (rho, a): air density (kg/m^3) and speed of sound (m/s).

    Raises
    ------
    ValueError
        If h is outside the range where the model is valid.
    """
    if not (H_MIN <= h <= H_MAX):
        raise ValueError(f"altitude {h:.1f} m outside ISA troposphere "
                         f"[{H_MIN:.0f}, {H_MAX:.0f}] m")
    T = T0 - LAPSE * h
    p = P0 * (T / T0) ** _EXPONENT
    rho = p / (R_AIR * T)
    a = np.sqrt(GAMMA * R_AIR * T)
    return rho, a


def constant_atmosphere(rho=1.225, a=340.29):
    """Return an atmosphere function with fixed density and speed of sound.

    Used to verify the numerical model against the constant-k analytical
    solution, and to quantify the error of assuming constant density.

    Parameters
    ----------
    rho : float
        Air density (kg/m^3), must be > 0.
    a : float
        Speed of sound (m/s), must be > 0.

    Returns
    -------
    callable
        Function h -> (rho, a) ignoring h.
    """
    if rho <= 0 or a <= 0:
        raise ValueError("rho and a must be positive")

    def atmosphere(h):
        return rho, a

    return atmosphere
