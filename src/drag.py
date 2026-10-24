"""Drag coefficient C_D(Mach, flap) built from the CFD data set.

The coast model is a vertical, zero-angle-of-attack flight, so the zero-pitch
rows are the default. Other pitch angles with a complete grid are available
for the angle-of-attack sensitivity study.
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


_NEEDED = ['Mach Number', 'Pitch (deg)', 'Roll (deg)',
           'Flap (% extension)', 'Total C_D']


GRID_SOURCE = 'HPR_grid'     # 'Source' label of the regular-grid CFD rows


def _read_cfd(path, grid_only=True):
    """Read the csv and check the columns every loader needs.

    A merged file carries a 'Source' column (see `load_cd_points`). The table
    loaders need the regular Mach-by-flap grid, so by default they keep only
    the rows labelled `GRID_SOURCE`; the scattered rows are for regression.
    """
    df = pd.read_csv(path, index_col=0)
    missing = [c for c in _NEEDED if c not in df.columns]
    if missing:
        raise ValueError(f"csv is missing columns: {missing}")
    if grid_only and 'Source' in df.columns:
        df = df[df['Source'] == GRID_SOURCE]
        if df.empty:
            raise ValueError(f"no rows with Source == '{GRID_SOURCE}'")
    return df


def _drag_column(df, component):
    """Drag coefficient column to tabulate.

    'total' is the csv column 'Total C_D'. In the CFD file this is the
    magnitude of the whole aerodynamic force over q*S, which equals the drag
    only at zero pitch; away from zero it also contains the normal force.

    'wind' is the drag along the flight direction, built from the
    force-coefficient vector: C_axial * cos(pitch) + C_normal * sin(pitch),
    where C_axial = -C_x and C_normal = hypot(C_y, C_z). It equals C_axial at
    zero pitch.
    """
    if component == 'total':
        return df['Total C_D']
    if component != 'wind':
        raise ValueError("component must be 'total' or 'wind'")
    if 'C_D vector' not in df.columns:
        raise ValueError("csv is missing columns: ['C_D vector']")
    vec = np.stack(df['C_D vector'].apply(ast.literal_eval))
    alpha = np.radians(df['Pitch (deg)'].to_numpy(dtype=float))
    c_normal = np.hypot(vec[:, 1], vec[:, 2])
    return pd.Series(-vec[:, 0] * np.cos(alpha) + c_normal * np.sin(alpha),
                     index=df.index)


def load_cd_table(path, pitch=0, roll=None, component='total'):
    """Read a drag-coefficient table C_D(Mach, flap) from the CFD csv file.

    The coast model assumes zero angle of attack, so the default is the
    zero-pitch slice. Other pitch angles are used only for the angle-of-attack
    sensitivity study; they are accepted only if the (Mach, flap) grid at that
    pitch is complete and spans the same Mach range as the zero-pitch table,
    because the model clamps outside its table and would otherwise silently
    reuse an end value.

    Parameters
    ----------
    path : str or path-like
        Location of the project csv file. Required columns are
        'Mach Number', 'Pitch (deg)', 'Roll (deg)', 'Flap (% extension)'
        and 'Total C_D'.
    pitch : float
        Pitch angle (deg), i.e. the angle of attack, to extract.
    roll : float or None
        Roll angle (deg) to extract. If None, zero pitch requires the roll
        angles to agree (the table is otherwise ambiguous) and any other
        pitch averages over the available roll angles.
    component : {'total', 'wind'}
        'total' uses the csv column 'Total C_D', which is the drag at zero
        pitch. 'wind' uses the drag along the flight direction, which is the
        right quantity at non-zero pitch (see `_drag_column`).

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
        If required columns are missing, the requested pitch or roll is not
        in the file, the (Mach, flap) grid is incomplete or shorter than the
        full Mach range, or the roll angles disagree by more than 1e-6 at
        zero pitch.
    """
    df = _read_cfd(path)
    df = df.assign(cd=_drag_column(df, component))

    # Mach 0 rows are placeholders (all forces are zero) and are dropped.
    df = df[df['Mach Number'] > 0]
    all_mach = np.sort(df['Mach Number'].unique())
    d = df[df['Pitch (deg)'] == pitch]
    if d.empty:
        raise ValueError(f"no rows at pitch {pitch} deg")
    if roll is not None:
        d = d[d['Roll (deg)'] == roll]
        if d.empty:
            raise ValueError(f"no rows at pitch {pitch} deg, roll {roll} deg")
    elif pitch == 0:
        spread = d.groupby(['Mach Number', 'Flap (% extension)'])['cd'].agg(
            lambda s: s.max() - s.min())
        if spread.max() > 1e-6:
            raise ValueError("C_D at zero pitch depends on roll angle; "
                             "the vertical-flight table is ambiguous")

    table = d.pivot_table(index='Mach Number', columns='Flap (% extension)',
                          values='cd', aggfunc='mean')
    if table.isna().any().any() or table.shape[0] != all_mach.size:
        raise ValueError(f"(Mach, flap) grid at pitch {pitch} deg is "
                         "incomplete or covers only part of the Mach range")
    return (table.index.to_numpy(dtype=float),
            table.columns.to_numpy(dtype=float),
            table.to_numpy(dtype=float))


def load_cd_points(paths, pitch=0, component='wind'):
    """Scattered drag-coefficient samples from one or more CFD csv files.

    Unlike `load_cd_table` this keeps every sample, whatever its Mach number
    and flap setting, so gridded and scattered (noisy) runs can be pooled.
    Samples at the same (source, Mach, flap), such as the two roll angles of
    a gridded run, are averaged so that they are not counted twice.

    Parameters
    ----------
    paths : str, path-like or list of those
        csv file(s). If a file has a 'Source' column its labels are kept;
        otherwise every row is labelled with the file name.
    pitch : float
        Pitch angle (deg) to extract.
    component : {'total', 'wind'}
        See `load_cd_table`.

    Returns
    -------
    pandas.DataFrame
        Columns 'mach', 'flap', 'cd', 'source', sorted by Mach number.

    Raises
    ------
    ValueError
        If no samples exist at that pitch.
    """
    if isinstance(paths, (str, bytes)) or hasattr(paths, '__fspath__'):
        paths = [paths]
    frames = []
    for path in paths:
        df = _read_cfd(path, grid_only=False)
        df = df.assign(cd=_drag_column(df, component))
        if 'Source' not in df.columns:
            df = df.assign(Source=Path(path).stem)
        df = df[(df['Mach Number'] > 0) & (df['Pitch (deg)'] == pitch)]
        frames.append(df[['Source', 'Mach Number', 'Flap (% extension)', 'cd']])
    pts = pd.concat(frames, ignore_index=True)
    if pts.empty:
        raise ValueError(f"no samples at pitch {pitch} deg")
    pts = (pts.groupby(['Source', 'Mach Number', 'Flap (% extension)'],
                       as_index=False)['cd'].mean()
           .rename(columns={'Source': 'source', 'Mach Number': 'mach',
                            'Flap (% extension)': 'flap'}))
    return pts.sort_values(['mach', 'flap', 'source']).reset_index(drop=True)


def complete_pitches(path):
    """Pitch angles (deg) in the csv that have a complete C_D table.

    Returns
    -------
    list of float
        Ascending pitch angles for which `load_cd_table` succeeds.
    """
    df = _read_cfd(path)
    out = []
    for p in sorted(df['Pitch (deg)'].unique()):
        try:
            load_cd_table(path, pitch=p)
        except ValueError:
            continue
        out.append(float(p))
    return out


def cd_roll_spread(path, pitch, component='total'):
    """Largest difference in C_D between the two roll angles at one pitch.

    Returns
    -------
    float
        max |C_D(roll_a) - C_D(roll_b)| over the whole (Mach, flap) grid.

    Raises
    ------
    ValueError
        If the file does not have exactly two roll angles at that pitch,
        or the grid is incomplete.
    """
    df = _read_cfd(path)
    rolls = np.sort(df[df['Pitch (deg)'] == pitch]['Roll (deg)'].unique())
    if rolls.size != 2:
        raise ValueError(f"need exactly two roll angles at pitch {pitch} deg")
    a = load_cd_table(path, pitch, roll=rolls[0], component=component)[2]
    b = load_cd_table(path, pitch, roll=rolls[1], component=component)[2]
    return float(np.max(np.abs(a - b)))


def cd_vs_pitch(path, mach, flap, component='total'):
    """C_D against pitch angle at one tabulated Mach number and flap setting.

    Roll angles are averaged. Unlike `load_cd_table`, this also returns the
    pitch angles that exist only at low Mach number.

    Returns
    -------
    pitch, cd : numpy.ndarray
        Pitch angles (deg, ascending) and the matching mean C_D.

    Raises
    ------
    ValueError
        If no rows match the Mach number and flap setting.
    """
    df = _read_cfd(path)
    df = df.assign(cd=_drag_column(df, component))
    sel = (np.isclose(df['Mach Number'], mach)
           & np.isclose(df['Flap (% extension)'], flap))
    if not sel.any():
        raise ValueError(f"no rows at Mach {mach}, flap {flap} %")
    s = df[sel].groupby('Pitch (deg)')['cd'].mean()
    return s.index.to_numpy(dtype=float), s.to_numpy(dtype=float)


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
