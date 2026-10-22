"""Centre of pressure from the CFD force and moment data.

A transverse force acting on the body axis at x_cp, seen from a moment
reference point at x_ref, produces

    My = -d * Fz,   Mz = d * Fy,   d = x_cp - x_ref.

Solving in the least-squares sense for d (valid at any roll angle):

    d = (-My*Fz + Mz*Fy) / (Fy**2 + Fz**2)
"""
import ast

import numpy as np
import pandas as pd

FN_MIN = 1.0           # N; ignore rows with less transverse force than this
PITCH_LIN_MIN = 2      # deg; lowest pitch used for any CoP value
PITCH_LIN_MAX = 10     # deg; top of the linear range used for the fit


def load_cfd_dataset(path, ref_idx=0):
    """Read the CFD csv and return flat arrays for the CoP calculation.

    Parameters
    ----------
    path : str or path-like
        Project csv file.
    ref_idx : int
        Index of the moment reference station (0 -> first x location).

    Returns
    -------
    dict
        Keys 'mach', 'pitch', 'roll', 'flap', 'Fy', 'Fz', 'My', 'Mz'
        (arrays of shape (N,)) and 'x_ref' (float). Rows with zero drag
        (the Mach 0 placeholders) are removed.

    Raises
    ------
    ValueError
        If ref_idx is outside the available moment stations.
    """
    df = pd.read_csv(path, index_col=0)
    df = df[df['Total Drag (N)'] > 1e-6].reset_index(drop=True)

    drag_vec = np.stack(df['drag vector (N)'].apply(ast.literal_eval))
    moments = np.stack(df['Moments (Nm)'].apply(ast.literal_eval))
    x = np.array(ast.literal_eval(df['Moment x_locations'].iloc[0]))
    if not 0 <= ref_idx < x.size:
        raise ValueError(f"ref_idx must be in [0, {x.size - 1}]")

    return {
        'mach': df['Mach Number'].to_numpy(),
        'pitch': df['Pitch (deg)'].to_numpy(),
        'roll': df['Roll (deg)'].to_numpy(),
        'flap': df['Flap (% extension)'].to_numpy(),
        'Fy': drag_vec[:, 1], 'Fz': drag_vec[:, 2],
        'My': moments[:, ref_idx, 1], 'Mz': moments[:, ref_idx, 2],
        'x_ref': float(x[ref_idx]),
    }


def center_of_pressure(Fy, Fz, My, Mz, x_ref, fn_min=FN_MIN):
    """Centre-of-pressure x location for each case.

    Parameters
    ----------
    Fy, Fz : array_like
        Transverse force components (N).
    My, Mz : array_like
        Moment components about the reference point (N m).
    x_ref : float
        x location of the moment reference point (m).
    fn_min : float
        Cases with transverse force below this (N) return NaN, since the
        ratio of two near-zero numbers is meaningless.

    Returns
    -------
    numpy.ndarray
        x_cp (m), NaN where the transverse force is too small.
    """
    Fy, Fz, My, Mz = (np.asarray(a, dtype=float) for a in (Fy, Fz, My, Mz))
    fn2 = Fy**2 + Fz**2
    x_cp = np.full(fn2.shape, np.nan)
    ok = np.sqrt(fn2) > fn_min
    x_cp[ok] = x_ref + (-My[ok] * Fz[ok] + Mz[ok] * Fy[ok]) / fn2[ok]
    return x_cp


def cp_by_condition(data, pitch_min=PITCH_LIN_MIN, pitch_max=PITCH_LIN_MAX):
    """Centre of pressure per (Mach, flap, roll) from a regression.

    Within the linear range of pitch angles, the best single d for all
    cases of one condition is sum(num) / sum(den), where
    num = -My*Fz + Mz*Fy and den = Fy^2 + Fz^2. This weights the
    cases with more normal force more heavily, which suppresses noise.

    Parameters
    ----------
    data : dict
        Output of `load_cfd_dataset`.
    pitch_min, pitch_max : float
        Pitch range (deg) used in the fit.

    Returns
    -------
    pandas.DataFrame
        Columns mach, flap, roll, x_cp (m) and s_cp (m aft of x = 0).
    """
    sel = (data['pitch'] >= pitch_min) & (data['pitch'] <= pitch_max)
    df = pd.DataFrame({
        'mach': data['mach'][sel], 'flap': data['flap'][sel],
        'roll': data['roll'][sel],
        'num': (-data['My'] * data['Fz'] + data['Mz'] * data['Fy'])[sel],
        'den': (data['Fy']**2 + data['Fz']**2)[sel],
    })
    g = df.groupby(['mach', 'flap', 'roll'])[['num', 'den']].sum()
    out = (data['x_ref'] + g['num'] / g['den']).rename('x_cp').reset_index()
    out['s_cp'] = -out['x_cp']
    return out
