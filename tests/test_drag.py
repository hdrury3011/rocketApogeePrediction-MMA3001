import numpy as np
import pytest

from drag import DragModel, load_cd_table

MACH = np.array([0.1, 0.3, 0.5, 0.7])
FLAP = np.array([0.0, 50.0, 100.0])
CD = np.array([[0.40, 0.45, 0.50],
               [0.38, 0.44, 0.51],
               [0.37, 0.46, 0.55],
               [0.39, 0.48, 0.58]])


@pytest.mark.parametrize("method", ["linear", "cubic"])
def test_interpolation_passes_through_table_nodes(method):
    model = DragModel(MACH, FLAP, CD, method)
    for i, m in enumerate(MACH):
        for j, f in enumerate(FLAP):
            assert np.isclose(model(m, f), CD[i, j], rtol=1e-12)


def test_linear_midpoint():
    model = DragModel(MACH, FLAP, CD, "linear")
    assert np.isclose(model(0.2, 25.0), 0.25 * (CD[0, 0] + CD[1, 0] + CD[0, 1] + CD[1, 1]))


def test_mach_outside_table_is_clamped():
    model = DragModel(MACH, FLAP, CD, "cubic")
    assert model(0.0, 0.0) == model(0.1, 0.0)
    assert model(2.0, 100.0) == model(0.7, 100.0)


@pytest.mark.parametrize("method", ["linear", "cubic"])
def test_at_flap_matches_full_model(method):
    model = DragModel(MACH, FLAP, CD, method)
    for flap in (0.0, 20.0, 50.0, 80.0, 100.0):
        fast = model.at_flap(flap)
        for m in (0.05, 0.1, 0.25, 0.42, 0.7, 0.9):
            assert np.isclose(fast(m), model(m, flap), rtol=1e-12)


def test_invalid_inputs_raise():
    model = DragModel(MACH, FLAP, CD)
    with pytest.raises(ValueError):
        model(0.3, 120.0)
    with pytest.raises(ValueError):
        model.at_flap(-5.0)
    with pytest.raises(ValueError):
        DragModel(MACH, FLAP, CD[:, :2])
    with pytest.raises(ValueError):
        DragModel(MACH[::-1], FLAP, CD)
    with pytest.raises(ValueError):
        DragModel(MACH, FLAP, -CD)
    with pytest.raises(ValueError):
        DragModel(MACH, FLAP, CD, "quintic")


def test_load_real_table(csv_path):
    mach, flap, cd = load_cd_table(csv_path)
    assert mach[0] == pytest.approx(0.1) and mach[-1] == pytest.approx(0.9)
    assert list(flap) == [0.0, 50.0, 100.0]
    assert cd.shape == (9, 3)
    assert np.all(cd[:, 2] > cd[:, 1]) and np.all(cd[:, 1] > cd[:, 0])  # more flap, more drag


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_cd_table(tmp_path / "nope.csv")
