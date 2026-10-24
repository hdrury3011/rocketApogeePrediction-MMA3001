"""Regression model of the drag coefficient C_D(Mach, flap).

The CFD table gives C_D on a coarse grid (9 Mach values by 3 flap settings).
Extra runs scattered over the same range are noisy, so interpolating through
them would reproduce the noise. A least-squares polynomial surface averages
it out instead and uses every sample:

    C_D(M, f) = sum_ij a_ij T_i(x) T_j(z),   x, z in [-1, 1] scaled from M, f

with T the Chebyshev polynomials (well conditioned, unlike raw powers). The
polynomial degrees are chosen by k-fold cross-validation, not by hand.
"""
import numpy as np
from numpy.polynomial import chebyshev as cheb


def _check_samples(mach, flap, cd):
    mach, flap, cd = (np.asarray(a, dtype=float) for a in (mach, flap, cd))
    if not (mach.ndim == flap.ndim == cd.ndim == 1
            and mach.size == flap.size == cd.size):
        raise ValueError("mach, flap and cd must be 1-D arrays of equal length")
    if mach.size == 0:
        raise ValueError("no samples")
    if not np.all(np.isfinite(mach) & np.isfinite(flap) & np.isfinite(cd)):
        raise ValueError("samples must be finite")
    if np.any(cd <= 0):
        raise ValueError("all drag coefficients must be positive")
    return mach, flap, cd


def _scale(values, lo, hi):
    """Map [lo, hi] onto [-1, 1]."""
    return 2.0 * (values - lo) / (hi - lo) - 1.0


class RegressionDragModel:
    """Least-squares polynomial surface C_D(Mach, flap).

    It has the same interface as `drag.DragModel` (`__call__` and `at_flap`),
    so the coast model and the calibration can use either. Build it with
    `RegressionDragModel.fit`.

    Mach numbers outside the sampled range are clamped to its ends, as in
    `DragModel`, because a polynomial extrapolates badly. A flap setting
    outside the sampled range raises ValueError.

    Parameters
    ----------
    coef : numpy.ndarray, shape (deg_mach + 1, deg_flap + 1)
        Chebyshev coefficients a_ij.
    mach_range, flap_range : tuple of float
        Sampled ranges used to scale the inputs.
    n_samples : int
        Number of samples the model was fitted to.
    """

    def __init__(self, coef, mach_range, flap_range, n_samples=0):
        self.coef = np.asarray(coef, dtype=float)
        if self.coef.ndim != 2:
            raise ValueError("coef must be 2-D")
        self.mach_range = (float(mach_range[0]), float(mach_range[1]))
        self.flap_range = (float(flap_range[0]), float(flap_range[1]))
        if not (self.mach_range[0] < self.mach_range[1]
                and self.flap_range[0] < self.flap_range[1]):
            raise ValueError("ranges must have lower < upper")
        self.deg_mach = self.coef.shape[0] - 1
        self.deg_flap = self.coef.shape[1] - 1
        self.n_samples = int(n_samples)

    @property
    def n_params(self):
        """Number of fitted coefficients."""
        return self.coef.size

    @classmethod
    def fit(cls, mach, flap, cd, deg_mach, deg_flap, check_positive=True):
        """Fit the surface to scattered samples by linear least squares.

        Parameters
        ----------
        mach, flap, cd : array_like
            Samples (Mach number, flap extension in %, drag coefficient).
        deg_mach, deg_flap : int
            Polynomial degrees, >= 0.
        check_positive : bool
            Reject a fit that predicts C_D <= 0 anywhere in the sampled box.

        Returns
        -------
        RegressionDragModel

        Raises
        ------
        ValueError
            For bad samples or degrees, fewer samples than coefficients,
            a degenerate Mach or flap range, or a non-positive fitted C_D.
        """
        mach, flap, cd = _check_samples(mach, flap, cd)
        if deg_mach < 0 or deg_flap < 0:
            raise ValueError("degrees must be >= 0")
        n_par = (deg_mach + 1) * (deg_flap + 1)
        if mach.size < n_par:
            raise ValueError(f"{mach.size} samples cannot determine "
                             f"{n_par} coefficients")
        m_rng = (mach.min(), mach.max())
        f_rng = (flap.min(), flap.max())
        if m_rng[0] == m_rng[1] or f_rng[0] == f_rng[1]:
            raise ValueError("samples must span a range of Mach and flap")
        design = cheb.chebvander2d(_scale(mach, *m_rng), _scale(flap, *f_rng),
                                   [deg_mach, deg_flap])
        coef, *_ = np.linalg.lstsq(design, cd, rcond=None)
        model = cls(coef.reshape(deg_mach + 1, deg_flap + 1), m_rng, f_rng,
                    mach.size)
        if check_positive:
            mm = np.linspace(*m_rng, 60)
            ff = np.linspace(*f_rng, 21)
            grid = model.predict(*np.meshgrid(mm, ff))
            if np.any(grid <= 0):
                raise ValueError("fitted C_D is not positive everywhere in "
                                 "the sampled range")
        return model

    def predict(self, mach, flap):
        """Vectorised C_D, with Mach clamped to the sampled range.

        Unlike `__call__` this does not range-check the flap, so it can be
        used on arrays for fitting and plotting.
        """
        m = np.clip(np.asarray(mach, dtype=float), *self.mach_range)
        m, f = np.broadcast_arrays(m, np.asarray(flap, dtype=float))
        return cheb.chebval2d(_scale(m, *self.mach_range),
                              _scale(f, *self.flap_range), self.coef)

    def _check_flap(self, flap):
        if flap < self.flap_range[0] or flap > self.flap_range[1]:
            raise ValueError(f"flap {flap} % outside sampled range "
                             f"[{self.flap_range[0]}, {self.flap_range[1]}]")

    def __call__(self, mach, flap=0.0):
        """Drag coefficient at the given Mach number and flap extension (%).

        Raises
        ------
        ValueError
            If flap lies outside the sampled range.
        """
        self._check_flap(flap)
        return float(self.predict(mach, flap))

    def at_flap(self, flap):
        """Return a faster function C_D(Mach) for one fixed flap setting.

        The flap dependence is collapsed into a single Chebyshev series in
        Mach once, instead of on every call of the right-hand side.

        Raises
        ------
        ValueError
            If flap lies outside the sampled range.
        """
        self._check_flap(flap)
        z = _scale(float(flap), *self.flap_range)
        series = cheb.chebval(z, self.coef.T)        # coefficients in Mach
        lo, hi = self.mach_range

        def cd_of_mach(m):
            return float(cheb.chebval(_scale(min(max(m, lo), hi), lo, hi),
                                      series))
        return cd_of_mach


def cross_validate(mach, flap, cd, deg_mach, deg_flap, k=5, repeats=3,
                   seed=0, test_mask=None):
    """Repeated k-fold cross-validation of one polynomial degree pair.

    Each fold is held out, the surface is fitted to everything else and the
    held-out samples are predicted. Errors are relative, prediction / sample - 1.

    Parameters
    ----------
    mach, flap, cd : array_like
        Samples.
    deg_mach, deg_flap : int
        Polynomial degrees.
    k : int
        Number of folds, >= 2.
    repeats : int
        Number of random re-shuffles.
    seed : int
        Seed of the shuffling, for repeatability.
    test_mask : array_like of bool, optional
        Only these samples are ever held out; the others are always used for
        fitting. Use it to hold out only the noisy runs and keep every gridded
        run in the training set.

    Returns
    -------
    dict
        rmse (relative RMS error), bias (mean relative error), se (standard
        error of the fold RMSEs) and n_params.

    Raises
    ------
    ValueError
        If k < 2 or there are fewer held-out samples than folds.
    """
    mach, flap, cd = _check_samples(mach, flap, cd)
    if k < 2:
        raise ValueError("k must be >= 2")
    idx = np.arange(cd.size) if test_mask is None else \
        np.flatnonzero(np.asarray(test_mask, dtype=bool))
    if idx.size < k:
        raise ValueError("fewer held-out samples than folds")
    all_err, fold_rmse = [], []
    for r in range(repeats):
        perm = np.random.default_rng(seed + r).permutation(idx)
        for test in np.array_split(perm, k):
            train = np.setdiff1d(np.arange(cd.size), test)
            model = RegressionDragModel.fit(mach[train], flap[train], cd[train],
                                            deg_mach, deg_flap,
                                            check_positive=False)
            err = model.predict(mach[test], flap[test]) / cd[test] - 1.0
            all_err.append(err)
            fold_rmse.append(np.sqrt(np.mean(err**2)))
    err = np.concatenate(all_err)
    return dict(rmse=float(np.sqrt(np.mean(err**2))), bias=float(err.mean()),
                se=float(np.std(fold_rmse, ddof=1) / np.sqrt(len(fold_rmse))),
                n_params=(deg_mach + 1) * (deg_flap + 1))


def select_degrees(mach, flap, cd, mach_degrees=range(2, 10),
                   flap_degrees=range(1, 4), **cv_kwargs):
    """Choose polynomial degrees by cross-validation (one-standard-error rule).

    Every (deg_mach, deg_flap) pair is cross-validated. The chosen model is
    the one with the fewest coefficients whose RMS error is within one
    standard error of the best, so extra terms are added only if they help.

    Returns
    -------
    best : tuple of int
        Selected (deg_mach, deg_flap).
    table : list of dict
        One entry per pair with deg_mach, deg_flap, n_params, rmse, bias, se.
    """
    table = []
    for dm in mach_degrees:
        for df in flap_degrees:
            try:
                res = cross_validate(mach, flap, cd, dm, df, **cv_kwargs)
            except ValueError:          # too few samples for this many terms
                continue
            table.append(dict(deg_mach=dm, deg_flap=df, **res))
    if not table:
        raise ValueError("no degree pair could be fitted to these samples")
    top = min(table, key=lambda t: t['rmse'])
    ok = [t for t in table if t['rmse'] <= top['rmse'] + top['se']]
    pick = min(ok, key=lambda t: (t['n_params'], t['rmse']))
    return (pick['deg_mach'], pick['deg_flap']), table


def source_offset(mach, flap, cd, is_offset, deg_mach, deg_flap):
    """Relative offset of one group of samples from the rest.

    Fits the polynomial surface plus a constant shift for the flagged
    samples, C_D = surface + delta * flag, and returns delta relative to the
    mean C_D of the flagged samples, with its standard error. It measures
    how far the noisy runs sit from the gridded runs once Mach and flap are
    accounted for.

    Returns
    -------
    delta, se : float
        Relative offset and its standard error.

    Raises
    ------
    ValueError
        If the flag is constant or there are too few samples.
    """
    mach, flap, cd = _check_samples(mach, flap, cd)
    flag = np.asarray(is_offset, dtype=bool)
    if flag.all() or not flag.any():
        raise ValueError("need both flagged and unflagged samples")
    m_rng, f_rng = (mach.min(), mach.max()), (flap.min(), flap.max())
    design = cheb.chebvander2d(_scale(mach, *m_rng), _scale(flap, *f_rng),
                               [deg_mach, deg_flap])
    design = np.column_stack([design, flag.astype(float)])
    if design.shape[0] <= design.shape[1]:
        raise ValueError("too few samples")
    coef, *_ = np.linalg.lstsq(design, cd, rcond=None)
    resid = cd - design @ coef
    dof = design.shape[0] - design.shape[1]
    cov = resid @ resid / dof * np.linalg.pinv(design.T @ design)
    scale = cd[flag].mean()
    return float(coef[-1] / scale), float(np.sqrt(cov[-1, -1]) / scale)
