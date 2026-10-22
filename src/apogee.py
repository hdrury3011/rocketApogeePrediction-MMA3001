"""Analytical benchmark for vertical coast with quadratic drag.

Model: dv/dt = -g - k*v**2, with k = rho*C_D*S/(2*m) constant.
"""
import numpy as np


def _check_inputs(v0, k, g):
    """Raise ValueError for physically invalid inputs."""
    if v0 < 0:
        raise ValueError("v0 must be >= 0 (m/s)")
    if k < 0:
        raise ValueError("k must be >= 0 (1/m)")
    if g <= 0:
        raise ValueError("g must be > 0 (m/s^2)")


def analytic_apogee_height(v0, k, g=9.81):
    """Height gained above burnout for vertical coast with quadratic drag.

    Parameters
    ----------
    v0 : float
        Burnout velocity (m/s), must be >= 0.
    k : float
        Drag parameter rho*CD*S/(2m) (1/m), must be >= 0.
    g : float
        Gravitational acceleration (m/s^2), must be > 0.

    Returns
    -------
    float
        Height gained until v = 0 (m).

    Raises
    ------
    ValueError
        If v0 < 0, k < 0 or g <= 0.
    """
    _check_inputs(v0, k, g)
    if k == 0:  # drag-free limit, avoids dividing by zero
        return v0**2 / (2 * g)
    # log1p keeps precision when k*v0**2/g is tiny
    return np.log1p(k * v0**2 / g) / (2 * k)


def analytic_time_to_apogee(v0, k, g=9.81):
    """Time from burnout to apogee for vertical coast with quadratic drag.

    Parameters
    ----------
    v0 : float
        Burnout velocity (m/s), must be >= 0.
    k : float
        Drag parameter rho*CD*S/(2m) (1/m), must be >= 0.
    g : float
        Gravitational acceleration (m/s^2), must be > 0.

    Returns
    -------
    float
        Time from burnout until v = 0 (s).

    Raises
    ------
    ValueError
        If v0 < 0, k < 0 or g <= 0.
    """
    _check_inputs(v0, k, g)
    if k == 0:  # drag-free limit
        return v0 / g
    return np.arctan(np.sqrt(k / g) * v0) / np.sqrt(g * k)
