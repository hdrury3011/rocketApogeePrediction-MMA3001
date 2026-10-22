"""Fixed-step ODE integrators for the vertical coast phase.

State vector convention: y = [h, v], with h the altitude (m) and v the
vertical velocity (m/s). The right-hand side f(t, y) returns dy/dt as a
NumPy array of the same shape as y.
"""
import numpy as np


def euler_step(f, t, y, dt):
    """Advance y by one explicit Euler step.

    Parameters
    ----------
    f : callable
        Right-hand side f(t, y) -> dy/dt (array, same shape as y).
    t : float
        Current time (s).
    y : array_like
        Current state.
    dt : float
        Step size (s).

    Returns
    -------
    numpy.ndarray
        State at t + dt.
    """
    y = np.asarray(y, dtype=float)
    y_n_plus_1 = y + dt * np.asarray(f(t, y), dtype=float)
    return y_n_plus_1


def rk4_step(f, t, y, dt):
    """Advance y by one classical fourth-order Runge-Kutta step.

    Same parameters and return value as euler_step.

    Notes
    -----
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt/2 * k1)
    k3 = f(t + dt/2, y + dt/2 * k2)
    k4 = f(t + dt,   y + dt   * k3)
    y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    """
    y = np.asarray(y, dtype=float)
    k1 = np.asarray(f(t, y), dtype=float)
    k2 = np.asarray(f(t + dt/2, y + dt/2 * k1), dtype=float)
    k3 = np.asarray(f(t + dt/2, y + dt/2 * k2), dtype=float)
    k4 = np.asarray(f(t + dt, y + dt * k3), dtype=float)
    y_n_plus_1 = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y_n_plus_1


def integrate_to_apogee(f, y0, dt, step, t0=0.0, t_max=1000.0, n_bisect=60):
    """Integrate until the vertical velocity reaches zero (apogee).

    Parameters
    ----------
    f : callable
        Right-hand side f(t, y) -> dy/dt.
    y0 : array_like
        Initial state [h0, v0].
    dt : float
        Fixed step size (s), must be > 0.
    step : callable
        Single-step function with signature step(f, t, y, dt),
        e.g. euler_step or rk4_step.
    t0 : float
        Start time (s).
    t_max : float
        Maximum integration time (s) before giving up.
    n_bisect : int
        Bisection iterations used to locate apogee inside the final step.

    Returns
    -------
    tuple of float
        (t_apogee, h_apogee): time (s) and altitude (m) at v = 0.

    Raises
    ------
    ValueError
        If dt <= 0.
    RuntimeError
        If apogee is not reached within t_max.

    Notes
    -----
    If v0 <= 0 the function returns immediately.

    Do not stop at the first step where v < 0 and interpolate linearly:
    that costs O(dt^2) accuracy and caps RK4 well below fourth order.
    Instead, once the next full step would make v <= 0, find the partial
    step tau in (0, dt] such that v = 0 after step(f, t, y, tau), using a
    root-finder (bisection), and return the state from that step.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    y = np.asarray(y0, dtype=float)
    if y[1] <= 0:
        return t0, y[0]

    t = t0
    while t - t0 < t_max:
        y_next = step(f, t, y, dt)
        if y_next[1] > 0:               # still rising: accept the full step
            t, y = t + dt, y_next
            continue
        # The full step overshoots apogee: v(0) > 0 and v(dt) <= 0, so the
        # partial step length tau with v(tau) = 0 is bracketed in (0, dt].
        lo, hi = 0.0, dt
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            if step(f, t, y, mid)[1] > 0:
                lo = mid
            else:
                hi = mid
        tau = 0.5 * (lo + hi)
        return t + tau, step(f, t, y, tau)[0]
    raise RuntimeError("apogee not reached before t_max")
