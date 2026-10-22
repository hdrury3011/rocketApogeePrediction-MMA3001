import numpy as np
import pytest

from apogee import analytic_apogee_height, analytic_time_to_apogee
from integrators import euler_step, rk4_step, integrate_to_apogee

G = 9.81
K = 1.0e-4
V0 = 280.0


def coast_rhs(t, y):
    """dh/dt = v, dv/dt = -g - k v^2 (constant k)."""
    return np.array([y[1], -G - K * y[1] ** 2])


def apogee_error(step, dt):
    t_a, h_a = integrate_to_apogee(coast_rhs, [0.0, V0], dt, step)
    return abs(h_a - analytic_apogee_height(V0, K, G)), t_a


def observed_order(step, dts):
    errs = [apogee_error(step, dt)[0] for dt in dts]
    return [np.log(errs[i] / errs[i + 1]) / np.log(dts[i] / dts[i + 1])
            for i in range(len(errs) - 1)]


# --- Unit tests of a single step against hand-derived results -------------

def test_euler_step_linear_ode():
    lam, dt, y0 = -2.0, 0.1, np.array([3.0])
    y1 = euler_step(lambda t, y: lam * y, 0.0, y0, dt)
    assert np.allclose(y1, y0 * (1 + lam * dt))


def test_rk4_step_linear_ode():
    lam, dt, y0 = -2.0, 0.1, np.array([3.0])
    z = lam * dt
    expected = y0 * (1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24)
    y1 = rk4_step(lambda t, y: lam * y, 0.0, y0, dt)
    assert np.allclose(y1, expected, rtol=1e-12)


def test_rk4_step_uses_time_correctly():
    """RK4 integrates y' = t^3 exactly in one step; catches wrong t arguments."""
    dt = 0.5
    y1 = rk4_step(lambda t, y: np.array([t**3]), 0.0, np.array([0.0]), dt)
    assert np.allclose(y1, dt**4 / 4, rtol=1e-12)


def test_steps_accept_plain_lists():
    f = lambda t, y: [y[1], -9.81]                      # noqa: E731
    assert np.allclose(euler_step(f, 0.0, [0.0, 10.0], 0.1), [1.0, 9.019])
    assert rk4_step(f, 0.0, [0.0, 10.0], 0.1).shape == (2,)


# --- Whole-trajectory tests against the analytical benchmark --------------

def test_euler_close_to_analytic():
    err, t_a = apogee_error(euler_step, 0.01)
    assert err / analytic_apogee_height(V0, K, G) < 1e-3
    assert np.isclose(t_a, analytic_time_to_apogee(V0, K, G), rtol=1e-3)


def test_rk4_close_to_analytic():
    err, t_a = apogee_error(rk4_step, 0.1)
    assert err / analytic_apogee_height(V0, K, G) < 1e-8
    assert np.isclose(t_a, analytic_time_to_apogee(V0, K, G), rtol=1e-8)


def test_euler_is_first_order():
    orders = observed_order(euler_step, [0.1, 0.05, 0.025])
    assert all(0.9 < p < 1.1 for p in orders)


def test_rk4_is_fourth_order():
    orders = observed_order(rk4_step, [0.4, 0.2, 0.1])
    assert all(3.8 < p < 4.2 for p in orders)


# --- Robustness ------------------------------------------------------------

def test_invalid_dt_raises():
    with pytest.raises(ValueError):
        integrate_to_apogee(coast_rhs, [0.0, V0], 0.0, rk4_step)


def test_already_at_apogee_returns_immediately():
    t_a, h_a = integrate_to_apogee(coast_rhs, [100.0, 0.0], 0.1, rk4_step)
    assert t_a == 0.0 and h_a == 100.0


def test_t_max_raises():
    with pytest.raises(RuntimeError):
        integrate_to_apogee(coast_rhs, [0.0, V0], 0.1, rk4_step, t_max=1.0)
