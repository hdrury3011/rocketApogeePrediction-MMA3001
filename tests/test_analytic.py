import numpy as np
import pytest
from scipy.integrate import solve_ivp

from apogee import analytic_apogee_height, analytic_time_to_apogee

G = 9.81


def test_height_vacuum_limit():
    v0 = 250.0
    assert np.isclose(analytic_apogee_height(v0, k=1e-12), v0**2 / (2 * G), rtol=1e-6)


def test_time_vacuum_limit():
    v0 = 250.0
    assert np.isclose(analytic_time_to_apogee(v0, k=1e-12), v0 / G, rtol=1e-6)


def test_zero_drag_is_vacuum():
    v0 = 250.0
    assert analytic_apogee_height(v0, 0.0) == v0**2 / (2 * G)
    assert analytic_time_to_apogee(v0, 0.0) == v0 / G


def test_drag_lowers_apogee_and_time():
    v0, k = 250.0, 1.8e-4
    assert analytic_apogee_height(v0, k) < v0**2 / (2 * G)
    assert analytic_time_to_apogee(v0, k) < v0 / G


def test_matches_numerical_solution():
    """Independent check of the formulas against a tight-tolerance solve_ivp run."""
    v0, k = 280.0, 1.0e-4

    def rhs(t, y):
        return [y[1], -G - k * y[1] ** 2]

    def apogee_event(t, y):
        return y[1]

    apogee_event.terminal = True
    apogee_event.direction = -1

    sol = solve_ivp(rhs, [0, 200], [0.0, v0], events=apogee_event, rtol=1e-12, atol=1e-12)
    assert np.isclose(sol.y[0, -1], analytic_apogee_height(v0, k), rtol=1e-8)
    assert np.isclose(sol.t[-1], analytic_time_to_apogee(v0, k), rtol=1e-8)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        analytic_apogee_height(v0=-10.0, k=1e-4)
    with pytest.raises(ValueError):
        analytic_apogee_height(v0=100.0, k=-1e-4)
    with pytest.raises(ValueError):
        analytic_time_to_apogee(v0=100.0, k=1e-4, g=0.0)
