"""Known inputs for the rocket and the flight data supplied with the project.

All quantities are SI unless the name says otherwise.
"""
import numpy as np

FT = 0.3048                        # metres per foot
IN = 0.0254                        # metres per inch

# --- Supplied flight information -------------------------------------------
V_BURNOUT = 280.0                  # vertical velocity at motor burnout (m/s)
H_BURNOUT = 1800.0 * FT            # altitude at motor burnout, 1800 ft (m)
APOGEE_NO_BRAKES = 3450.0          # apogee without airbrakes (m)
APOGEE_WITH_BRAKES = 11000.0 * FT  # apogee with airbrakes, 11000 ft (m)

# --- Geometry --------------------------------------------------------------
D_REF = 6.0 * IN                   # reference diameter, 6 in (m)
S_REF = np.pi * (D_REF / 2) ** 2   # reference area (m^2)

# --- Verification benchmark ------------------------------------------------
K_BENCHMARK = 8.8e-5               # k = rho*C_D*S/(2m) of the analytical benchmark (1/m)

# --- Physical constants ----------------------------------------------------
G = 9.81                           # gravitational acceleration (m/s^2)
