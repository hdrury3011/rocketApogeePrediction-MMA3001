"""Drag coefficient C_D(Mach, flap) built from the CFD data set.

Only the zero-pitch rows are used, because the coast model is a vertical,
zero-angle-of-attack flight.
"""
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


def load_cd_table(path):
    """Read the zero-pitch drag-coefficient table from the CFD csv file.

    Parameters
    ----------
    path : str or path-like
        Location of the project csv file. Required columns are
        'Mach Number', 'Pitch (deg)', 'Roll (deg)', 'Flap (% extension)'
        and 'Total C_D'.

    Returns
    -------
    mach : numpy.ndarray, shape (n_mach,)
        Mach numbers, ascending, all > 0.
    flap : numpy.ndarray, shape (n_flap,)
        Flap extensions (%), ascending.
    cd : numpy.ndarray, shape (n_mach, n_flap)
        Total drag coefficient.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required columns are missing, the (Mach, flap) grid is incomplete
        or the two roll angles disagree by more than 1e-6 at zero pitch.
    """
    df = pd.read_csv(path, index_col=0)
    needed = ['Mach Number', 'Pitch (deg)', 'Roll (deg)',
              'Flap (% extension)', 'Total C_D']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"csv is missing columns: {missing}")

    # Mach 0 rows are placeholders (all forces are zero) and are dropped.
    d = df[(df['Pitch (deg)'] == 0) & (df['Mach Number'] > 0)]
    spread = d.groupby(['Mach Number', 'Flap (% extension)'])['Total C_D'].agg(
        lambda s: s.max() - s.min())
    if spread.max() > 1e-6:
        raise ValueError("C_D at zero pitch depends on roll angle; "
                         "the vertical-flight table is ambiguous")

    table = d.pivot_table(index='Mach Number', columns='Flap (% extension)',
                          values='Total C_D', aggfunc='first')
    if table.isna().any().any():
        raise ValueError("(Mach, flap) grid at zero pitch is incomplete")
    return (table.index.to_numpy(dtype=float),
            table.columns.to_numpy(dtype=float),
            table.to_numpy(dtype=float))


class DragModel:
    """Interpolated drag coefficient C_D(Mach, flap).

    Interpolation is done along Mach for each tabulated flap setting
    (linear or cubic spline) and then linearly in flap extension.
    Mach numbers outside the table are clamped to its end values, not
    extrapolated, because the CFD data say nothing about them and a spline
    would extrapolate unreliably.

    Parameters
    ----------
    mach : array_like, shape (n_mach,)
        Strictly increasing Mach numbers, at least 2 points.
    flap : array_like, shape (n_flap,)
        Strictly increasing flap extensions (%), at least 2 points.
    cd : array_like, shape (n_mach, n_flap)
        Drag coefficients, all > 0.
    method : {'linear', 'cubic'}
        Interpolation along Mach.

    Raises
    ------
    ValueError
        For inconsistent shapes, non-monotonic axes, non-positive C_D
        or an unknown method.
    """

    def __init__(self, mach, flap, cd, method='linear'):
        self.mach = np.asarray(mach, dtype=float)
        self.flap = np.asarray(flap, dtype=float)
        self.cd = np.asarray(cd, dtype=float)
        if self.cd.shape != (self.mach.size, self.flap.size):
            raise ValueError("cd must have shape (len(mach), len(flap))")
        if self.mach.size < 2 or self.flap.size < 2:
            raise ValueError("need at least two Mach and two flap points")
        if np.any(np.diff(self.mach) <= 0) or np.any(np.diff(self.flap) <= 0):
            raise ValueError("mach and flap must be strictly increasing")
        if np.any(self.cd <= 0):
            raise ValueError("all drag coefficients must be positive")
        if method not in ('linear', 'cubic'):
            raise ValueError("method must be 'linear' or 'cubic'")
        self.method = method
        self._splines = ([CubicSpline(self.mach, self.cd[:, j])
                          for j in range(self.flap.size)]
                         if method == 'cubic' else None)

    def _cd_columns(self, m):
        """C_D along Mach m (clamped) for every tabulated flap setting."""
        m = min(max(m, self.mach[0]), self.mach[-1])
        if self.method == 'cubic':
            return np.array([float(s(m)) for s in self._splines])
        return np.array([np.interp(m, self.mach, self.cd[:, j])
                         for j in range(self.flap.size)])

    def __call__(self, mach, flap=0.0):
        """Drag coefficient at the given Mach number and flap extension (%).

        Raises
        ------
        ValueError
            If flap lies outside the tabulated range.
        """
        if flap < self.flap[0] or flap > self.flap[-1]:
            raise ValueError(f"flap {flap} % outside table range "
                             f"[{self.flap[0]}, {self.flap[-1]}]")
        return float(np.interp(flap, self.flap, self._cd_columns(mach)))

    def at_flap(self, flap):
        """Return a faster function C_D(Mach) for one fixed flap setting.

        The flap blend is done once here instead of on every call, which
        matters because the ODE right-hand side is evaluated thousands of
        times.

        Parameters
        ----------
        flap : float
            Flap extension (%), inside the tabulated range.

        Returns
        -------
        callable
            Function mach -> C_D.
        """
        if flap < self.flap[0] or flap > self.flap[-1]:
            raise ValueError(f"flap {flap} % outside table range "
                             f"[{self.flap[0]}, {self.flap[-1]}]")
        # Blend the table columns once: C_D(Mach) at this flap setting.
        col = np.array([np.interp(flap, self.flap, row) for row in self.cd])
        lo, hi = self.mach[0], self.mach[-1]
        if self.method == 'cubic':
            spline = CubicSpline(self.mach, col)
            return lambda m: float(spline(min(max(m, lo), hi)))
        return lambda m: float(np.interp(m, self.mach, col))
