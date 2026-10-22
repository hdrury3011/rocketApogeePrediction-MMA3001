import numpy as np
import pytest

from atmosphere import isa, constant_atmosphere


def test_sea_level_values():
    rho, a = isa(0.0)
    assert np.isclose(rho, 1.225, rtol=2e-3)
    assert np.isclose(a, 340.29, rtol=1e-3)


def test_tropopause_values():
    """Published ISA values at 11 km: rho = 0.3639 kg/m^3, a = 295.07 m/s."""
    rho, a = isa(11000.0)
    assert np.isclose(rho, 0.3639, rtol=2e-3)
    assert np.isclose(a, 295.07, rtol=1e-3)


def test_density_decreases_with_altitude():
    hs = np.linspace(0, 11000, 50)
    rhos = [isa(h)[0] for h in hs]
    assert np.all(np.diff(rhos) < 0)


def test_out_of_range_raises():
    with pytest.raises(ValueError):
        isa(12000.0)
    with pytest.raises(ValueError):
        isa(-1000.0)


def test_constant_atmosphere():
    atm = constant_atmosphere(1.1, 340.0)
    assert atm(0.0) == atm(5000.0) == (1.1, 340.0)
    with pytest.raises(ValueError):
        constant_atmosphere(rho=-1.0)
