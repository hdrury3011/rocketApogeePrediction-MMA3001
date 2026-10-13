import ast

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
from scipy.integrate import simpson, trapezoid
from scipy.interpolate import CubicSpline
import time

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
DATA_PATH      = 'Project Data Set.csv'
REF_IDX        = 0        # index of the moment reference station (0 -> x = 0)
FN_MIN         = 1.0      # N; ignore rows with less normal force than this
PITCH_LIN_MIN  = 2        # deg; lowest pitch used for any CoP value (and for the linear-range fit)
PITCH_LIN_MAX  = 10       # deg; top of the linear range used for the per-condition CoP fit

# ----------------------------------------------------------------------
# Load + clean
# ----------------------------------------------------------------------
dataSet = pd.read_csv(DATA_PATH, index_col=0)

print("N before cleaning:", len(dataSet))
dataSetOriginalLength = len(dataSet)
mask = dataSet['Total Drag (N)'] > 1e-6
dataSet = dataSet[mask].reset_index(drop=True)
dataSetCleanedLength = len(dataSet)
print("N after cleaning:", len(dataSet))
print(f"Removed {dataSetOriginalLength - dataSetCleanedLength} Rows of Mach# = 0 Data\n")

# ----------------------------------------------------------------------
# Scalar arrays (N,)
# ----------------------------------------------------------------------
mach           = dataSet['Mach Number'].to_numpy()
pitch          = dataSet['Pitch (deg)'].to_numpy()
roll           = dataSet['Roll (deg)'].to_numpy()
flap           = dataSet['Flap (% extension)'].to_numpy()
totalDrag      = dataSet['Total Drag (N)'].to_numpy()
totalDragCoeff = dataSet['Total C_D'].to_numpy()

# ----------------------------------------------------------------------
# 3-component vector columns (N, 3)
# ----------------------------------------------------------------------
dragVector        = np.stack(dataSet['drag vector (N)'].apply(ast.literal_eval))
dragCoeffVectors  = np.stack(dataSet['C_D vector'].apply(ast.literal_eval))
wallMomentsMaxAbs = np.stack(dataSet['Wall Moments Max Abs (Nm)'].apply(ast.literal_eval))

# Lengthwise stations (same for every row)
x = np.array(ast.literal_eval(dataSet['Moment x_locations'].iloc[0]))
s = -x   # distance aft of x = 0

# ----------------------------------------------------------------------
# Station-resolved arrays (N, 66)
# ----------------------------------------------------------------------
momentVectors      = np.stack(dataSet['Moments (Nm)'].apply(ast.literal_eval))
momentCoeffVectors = np.stack(dataSet['C_M vectors'].apply(ast.literal_eval))
momentLocations    = np.stack(dataSet['Moment locations'].apply(ast.literal_eval))

Mx_all = momentVectors[:, :, 0]
My_all = momentVectors[:, :, 1]
Mz_all = momentVectors[:, :, 2]

# ----------------------------------------------------------------------
# Collapse to flat (N,) arrays
# ----------------------------------------------------------------------
# Forces (body axes): one value per case
Fx = dragVector[:, 0]
Fy = dragVector[:, 1]
Fz = dragVector[:, 2]

# Moments at a single reference station: one value per case
x_ref = x[REF_IDX]
Mx = Mx_all[:, REF_IDX]
My = My_all[:, REF_IDX]
Mz = Mz_all[:, REF_IDX]

# Print the shapes and data types of the arrays
print("mach               :", mach.shape, mach.dtype)
print("dragVector         :", dragVector.shape, dragVector.dtype)
print("x                  :", x.shape, x.dtype)
print("momentVectors      :", momentVectors.shape, momentVectors.dtype)
print("momentCoeffVectors :", momentCoeffVectors.shape, momentCoeffVectors.dtype)
print("momentLocations    :", momentLocations.shape, momentLocations.dtype)
print("Fx, Fy, Fz         :", Fx.shape, Fy.shape, Fz.shape)
print("Mx, My, Mz         :", Mx.shape, My.shape, Mz.shape)
print(f"Moment reference station: x = {x_ref} m\n")

# ----------------------------------------------------------------------
# Center of pressure
# ----------------------------------------------------------------------
# A force acting on the axis at x_cp, seen from a reference point at x_ref, gives
#   My = -d*Fz,  Mz = d*Fy,   d = x_cp - x_ref
# Least-squares solution for d (valid for any roll angle):
#   d = (-My*Fz + Mz*Fy) / (Fy^2 + Fz^2)

Fn2 = Fy**2 + Fz**2      # normal force squared
Fn  = np.sqrt(Fn2)       # total normal (transverse) force magnitude

# At pitch ~ 0 the transverse force is not an angle-of-attack normal force: it is noise
# or flap asymmetry (up to several N at higher Mach), so the CoP there is meaningless.
# Require a real pitch angle as well as a non-trivial normal force.
valid = (pitch >= PITCH_LIN_MIN) & (Fn > FN_MIN)

x_cp = np.full_like(Fn, np.nan)
x_cp[valid] = x_ref + (-My[valid] * Fz[valid] + Mz[valid] * Fy[valid]) / Fn2[valid]

s_cp = -x_cp             # distance aft of x = 0 (the nose, if that is the origin)

print(f"Valid CoP rows: {valid.sum()} of {len(valid)}")
print("First 10 valid x_cp [m]:", np.round(x_cp[valid][:10], 3))
print("First 10 valid s_cp [m]:", np.round(s_cp[valid][:10], 3), "\n")

# Simple roll-0-only check: x_cp = x_ref - My/Fz
roll0 = valid & (roll == 0)
x_cp_simple = x_ref - My[roll0] / Fz[roll0]
print("Max |full - simple| at roll 0 [m]:", np.max(np.abs(x_cp[roll0] - x_cp_simple)), "\n")

# ----------------------------------------------------------------------
# Sanity check: single-station shortcut vs all 66 stations
# ----------------------------------------------------------------------
Fn2_safe = np.where(Fn2 > 1e-9, Fn2, np.nan)
d_all    = (-My_all * Fz[:, None] + Mz_all * Fy[:, None]) / Fn2_safe[:, None]
x_cp_all = x[None, :] + d_all                                  # (N, 66)
spread   = np.nanmax(np.nanstd(x_cp_all[valid], axis=1))
print(f"Max spread of x_cp across the 66 stations: {spread:.2e} m (should be ~0)\n")

# ----------------------------------------------------------------------
# Per-condition CoP (regression through the origin over the linear pitch range)
# ----------------------------------------------------------------------
lin = (pitch >= PITCH_LIN_MIN) & (pitch <= PITCH_LIN_MAX)

df = pd.DataFrame({
    'mach': mach, 'flap': flap, 'roll': roll, 'lin': lin,
    'num': -My * Fz + Mz * Fy,
    'den': Fn2,
})
df = df[df['lin']]

g = df.groupby(['mach', 'flap', 'roll'])[['num', 'den']].sum()
cp_table = (x_ref + g['num'] / g['den']).rename('x_cp [m]').reset_index()
cp_table['s_cp [m aft of x=0]'] = -cp_table['x_cp [m]']

pd.set_option('display.width', 200)
print("Center of pressure by condition (pitch %d-%d deg):" % (PITCH_LIN_MIN, PITCH_LIN_MAX))
print(cp_table.round(3).to_string(index=False))

# Compact view: Mach down the side, flap/roll across the top
cp_pivot = cp_table.pivot(index='mach', columns=['flap', 'roll'], values='s_cp [m aft of x=0]')
print("\nDistance aft of x = 0 [m]:")
print(cp_pivot.round(3))

# ----------------------------------------------------------------------
# Plot: CoP vs Mach
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
for (fl, rl), col in cp_pivot.items():
    ax.plot(col.index, col.values, marker='o', label=f'Flap {fl}%, roll {rl}°')
ax.set_xlabel('Mach number')
ax.set_ylabel('CoP, distance aft of x = 0 [m]')
ax.set_title('Center of pressure vs Mach')
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig('cop_vs_mach.png', dpi=150)
plt.show()