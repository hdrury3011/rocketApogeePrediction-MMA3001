"""Root-finding utilities used to calibrate unknowns against flight data.

Two problems are solved:

* the rocket mass (not supplied) such that the predicted apogee without
  airbrakes equals the measured one, and
* the constant flap extension that produces a given apogee.

Both reduce to solving g(x) = 0 for a scalar x with a sign change inside a
bracket, so a hand-written bisection is compared with SciPy's Brent method.
"""
from scipy.optimize import brentq

from coast import simulate_apogee
from config import G, S_REF
from integrators import rk4_step


def bisect(func, lo, hi, xtol=1e-6, max_iter=200):
    """Find a root of a continuous function by bisection.

    Parameters
    ----------
    func : callable
        Scalar function with func(lo) and func(hi) of opposite sign.
    lo, hi : float
        Bracket, lo < hi.
    xtol : float
        Stop when the bracket is narrower than this.
    max_iter : int
        Iteration limit.

    Returns
    -------
    root : float
    n_eval : int
        Number of function evaluations used.

    Raises
    ------
    ValueError
        If the bracket is invalid or does not contain a sign change.
    RuntimeError
        If the tolerance is not reached within max_iter iterations.
    """
    if not lo < hi:
        raise ValueError("need lo < hi")
    f_lo, f_hi = func(lo), func(hi)
    n_eval = 2
    if f_lo == 0:
        return lo, n_eval
    if f_hi == 0:
        return hi, n_eval
    if f_lo * f_hi > 0:
        raise ValueError("func(lo) and func(hi) must have opposite signs")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = func(mid)
        n_eval += 1
        if f_mid == 0 or (hi - lo) < xtol:
            return mid, n_eval
        if f_lo * f_mid < 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    raise RuntimeError("bisection did not converge")


def calibrate_mass(target_apogee, cd_model, atmosphere, v0, h0, flap=0.0,
                   bracket=(5.0, 300.0), method='bisect', dt=0.05,
                   step=rk4_step, area=S_REF, g=G, xtol=1e-4):
    """Find the mass for which the simulated apogee equals `target_apogee`.

    Apogee increases with mass (a heavier rocket is slowed less by the same
    drag), so the residual is monotonic and has one root in a valid bracket.

    Parameters
    ----------
    target_apogee : float
        Measured apogee altitude (m).
    cd_model, atmosphere, v0, h0, flap, area, g, dt, step
        See `coast.simulate_apogee`.
    bracket : tuple of float
        Mass search interval (kg).
    method : {'bisect', 'brent'}
        Root finder.
    xtol : float
        Mass tolerance (kg).

    Returns
    -------
    mass : float
        Calibrated mass (kg).
    n_sim : int
        Number of coast simulations used.

    Raises
    ------
    ValueError
        If the target cannot be reached inside the bracket or the method
        is unknown.
    """
    def residual(m):
        return simulate_apogee(m, cd_model, atmosphere, v0, h0, flap, area,
                               g, dt, step)[1] - target_apogee

    if method == 'bisect':
        return bisect(residual, bracket[0], bracket[1], xtol=xtol)
    if method == 'brent':
        counter = [0]

        def counted(m):
            counter[0] += 1
            return residual(m)

        root = brentq(counted, bracket[0], bracket[1], xtol=xtol)
        return root, counter[0]
    raise ValueError("method must be 'bisect' or 'brent'")


def flap_for_apogee(target_apogee, mass, cd_model, atmosphere, v0, h0,
                    dt=0.05, step=rk4_step, area=S_REF, g=G, xtol=1e-3):
    """Constant flap extension (%) from burnout that gives `target_apogee`.

    Raises
    ------
    ValueError
        If the target lies outside the apogees reachable with 0-100 % flap.
    """
    def residual(flap):
        return simulate_apogee(mass, cd_model, atmosphere, v0, h0, flap,
                               area, g, dt, step)[1] - target_apogee

    return bisect(residual, 0.0, 100.0, xtol=xtol)
