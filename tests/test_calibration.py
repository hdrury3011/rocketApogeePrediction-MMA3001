import numpy as np
import pytest

from atmosphere import isa
from calibration import bisect, calibrate_mass, flap_for_apogee
from coast import simulate_apogee
from drag import DragModel

MACH = np.linspace(0.1, 0.9, 9)
FLAP = np.array([0.0, 50.0, 100.0])
CD = np.column_stack([np.full(9, 0.40), np.full(9, 0.46), np.full(9, 0.55)])
MODEL = DragModel(MACH, FLAP, CD)
V0, H0 = 280.0, 548.64


def test_bisect_finds_known_root():
    root, n = bisect(lambda x: x**3 - 2.0, 0.0, 2.0, xtol=1e-10)
    assert np.isclose(root, 2.0 ** (1 / 3), atol=1e-9)
    assert n < 60


def test_bisect_rejects_bad_brackets():
    with pytest.raises(ValueError):
        bisect(lambda x: x**2 + 1.0, -1.0, 1.0)      # no sign change
    with pytest.raises(ValueError):
        bisect(lambda x: x, 1.0, 0.0)                # reversed bracket


def test_bisect_root_at_endpoint():
    assert bisect(lambda x: x, 0.0, 1.0)[0] == 0.0


@pytest.mark.parametrize("method", ["bisect", "brent"])
def test_calibrate_mass_recovers_known_mass(method):
    """Round trip: simulate with a known mass, then recover it from the apogee."""
    true_mass = 37.5
    target = simulate_apogee(true_mass, MODEL, isa, V0, H0)[1]
    mass, n = calibrate_mass(target, MODEL, isa, V0, H0, method=method, xtol=1e-6)
    assert np.isclose(mass, true_mass, atol=1e-4)
    assert n > 0


def test_calibrate_mass_unreachable_target_raises():
    with pytest.raises(ValueError):
        calibrate_mass(1e6, MODEL, isa, V0, H0)       # above vacuum apogee


def test_calibrate_mass_unknown_method():
    with pytest.raises(ValueError):
        calibrate_mass(3000.0, MODEL, isa, V0, H0, method="newton")


def test_flap_for_apogee_round_trip():
    target = simulate_apogee(35.0, MODEL, isa, V0, H0, flap=40.0)[1]
    flap, _ = flap_for_apogee(target, 35.0, MODEL, isa, V0, H0, xtol=1e-5)
    assert np.isclose(flap, 40.0, atol=1e-3)


def test_flap_for_apogee_unreachable_raises():
    with pytest.raises(ValueError):
        flap_for_apogee(1e5, 35.0, MODEL, isa, V0, H0)
