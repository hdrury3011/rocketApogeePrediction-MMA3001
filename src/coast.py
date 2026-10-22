"""Vertical coast model after motor burnout.

State y = [h, v]. Equations of motion (no thrust, no wind, zero angle of attack):

    dh/dt = v
    dv/dt = -g - D / m,      D = 0.5 * rho(h) * C_D(Mach, flap) * S * v * |v|

with Mach = |v| / a(h). Apogee is the first time v = 0.
"""
import numpy as np
from scipy.integrate import solve_ivp

from config import G, S_REF
from integrators import integrate_to_apogee, rk4_step


def _validate(mass, v0, h0, area, g):
    if mass <= 0:
        raise ValueError("mass must be > 0 (kg)")
    if v0 < 0:
        raise ValueError("v0 must be >= 0 (m/s)")
    if area <= 0:
        raise ValueError("area must be > 0 (m^2)")
    if g <= 0:
        raise ValueError("g must be > 0 (m/s^2)")


def make_rhs(mass, cd_model, atmosphere, flap=0.0, area=S_REF, g=G,
             counter=None):
    """Build the right-hand side f(t, y) of the coast equations.

    Parameters
    ----------
    mass : float
        Rocket mass after burnout (kg), > 0.
    cd_model : DragModel or callable
        If `flap` is a number, a DragModel (its fast `at_flap` slice is used)
        or any callable mach -> C_D. If `flap` is a function of time, a
        DragModel is required (called as cd_model(mach, flap(t))).
    atmosphere : callable
        Function h -> (rho, a).
    flap : float or callable
        Flap extension (%), constant or a function of time t (s).
    area : float
        Reference area (m^2).
    g : float
        Gravitational acceleration (m/s^2).
    counter : list of int, optional
        If given, counter[0] is incremented on every evaluation. Used to
        compare the cost of different integrators.

    Returns
    -------
    callable
        f(t, y) -> numpy.ndarray([dh/dt, dv/dt]).
    """
    if mass <= 0:
        raise ValueError("mass must be > 0 (kg)")
    if callable(flap):
        cd_of = lambda m, t: cd_model(m, flap(t))      # noqa: E731
    elif hasattr(cd_model, 'at_flap'):
        cd_fixed = cd_model.at_flap(flap)
        cd_of = lambda m, t: cd_fixed(m)               # noqa: E731
    else:
        cd_of = lambda m, t: cd_model(m)               # noqa: E731

    half_s_over_m = 0.5 * area / mass

    def rhs(t, y):
        if counter is not None:
            counter[0] += 1
        h, v = y
        rho, a = atmosphere(h)
        cd = cd_of(abs(v) / a, t)
        return np.array([v, -g - half_s_over_m * rho * cd * v * abs(v)])

    return rhs


def simulate_apogee(mass, cd_model, atmosphere, v0, h0, flap=0.0,
                    area=S_REF, g=G, dt=0.05, step=rk4_step, counter=None):
    """Fixed-step prediction of the coast apogee.

    Parameters
    ----------
    mass : float
        Mass after burnout (kg), > 0.
    cd_model, atmosphere, flap, area, g, counter
        See `make_rhs`.
    v0 : float
        Burnout velocity (m/s), >= 0.
    h0 : float
        Burnout altitude (m).
    dt : float
        Step size (s), > 0.
    step : callable
        Single-step integrator, e.g. euler_step or rk4_step.

    Returns
    -------
    tuple of float
        (t_apogee, h_apogee): time after burnout (s) and altitude (m).

    Raises
    ------
    ValueError
        For non-physical inputs.
    """
    _validate(mass, v0, h0, area, g)
    rhs = make_rhs(mass, cd_model, atmosphere, flap, area, g, counter)
    return integrate_to_apogee(rhs, [h0, v0], dt, step)


def reference_apogee(mass, cd_model, atmosphere, v0, h0, flap=0.0,
                     area=S_REF, g=G, rtol=1e-11, atol=1e-9):
    """High-accuracy apogee from SciPy's adaptive DOP853, used as a reference.

    Same inputs as `simulate_apogee` (without dt/step). Returns
    (t_apogee, h_apogee).
    """
    _validate(mass, v0, h0, area, g)
    rhs = make_rhs(mass, cd_model, atmosphere, flap, area, g)

    def apogee_event(t, y):
        return y[1]

    apogee_event.terminal = True
    apogee_event.direction = -1
    sol = solve_ivp(rhs, [0.0, 500.0], [h0, v0], method='DOP853',
                    events=apogee_event, rtol=rtol, atol=atol)
    if sol.t_events[0].size == 0:
        raise RuntimeError("apogee not reached within 500 s")
    return float(sol.t_events[0][0]), float(sol.y_events[0][0][0])


def trajectory(mass, cd_model, atmosphere, v0, h0, flap=0.0, area=S_REF,
               g=G, n=400):
    """Sampled coast trajectory up to apogee, for plotting.

    Returns
    -------
    t, h, v : numpy.ndarray
        Time (s), altitude (m) and velocity (m/s) at n points.
    """
    t_a, _ = reference_apogee(mass, cd_model, atmosphere, v0, h0, flap,
                              area, g)
    rhs = make_rhs(mass, cd_model, atmosphere, flap, area, g)
    t_eval = np.linspace(0.0, t_a, n)
    sol = solve_ivp(rhs, [0.0, t_a], [h0, v0], method='DOP853',
                    t_eval=t_eval, rtol=1e-10, atol=1e-9)
    return sol.t, sol.y[0], sol.y[1]
