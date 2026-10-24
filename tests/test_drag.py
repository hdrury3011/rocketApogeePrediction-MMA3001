import numpy as np
import pytest

import pandas as pd

from drag import (DragModel, cd_roll_spread, cd_vs_pitch, complete_pitches,
                  load_cd_table)

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


# ---------------------------------------------------------------------------
# Pitch (angle of attack) options, tested on a small synthetic csv so that
# they run without the real CFD data.
# ---------------------------------------------------------------------------
def _write_csv(path):
    """Two Mach values and two flap settings; pitch 0 and 1 are complete,
    pitch 3 exists only at the lower Mach number."""
    rows = []
    for pitch, machs in ((0, (0.1, 0.5)), (1, (0.1, 0.5)), (3, (0.1,))):
        for mach in machs:
            for flap in (0, 100):
                for roll in (0, 45):
                    cx = -(0.40 + 0.1 * mach + 0.1 * flap / 100)
                    cn = 0.2 * pitch * (1.0 if roll == 0 else 0.9)
                    rows.append({'Mach Number': mach, 'Pitch (deg)': pitch,
                                 'Roll (deg)': roll, 'Flap (% extension)': flap,
                                 'Total C_D': float(np.hypot(cx, cn)),
                                 'C_D vector': f'[{cx}, 0.0, {-cn}]'})
    pd.DataFrame(rows).to_csv(path)
    return path


def test_default_is_zero_pitch_table(tmp_path):
    path = _write_csv(tmp_path / "d.csv")
    a = load_cd_table(path)
    b = load_cd_table(path, pitch=0)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_wind_component_matches_formula(tmp_path):
    path = _write_csv(tmp_path / "d.csv")
    _, _, cd = load_cd_table(path, pitch=1, roll=0, component='wind')
    alpha = np.radians(1.0)
    cx, cn = 0.40 + 0.1 * 0.1, 0.2
    assert cd[0, 0] == pytest.approx(cx * np.cos(alpha) + cn * np.sin(alpha))


def test_total_component_overstates_drag_at_pitch(tmp_path):
    """'Total C_D' contains the normal force, so it exceeds the drag along
    the flight direction once the pitch is non-zero."""
    path = _write_csv(tmp_path / "d.csv")
    total = load_cd_table(path, pitch=1, roll=0)[2]
    wind = load_cd_table(path, pitch=1, roll=0, component='wind')[2]
    assert np.all(total > wind)


def test_roll_none_averages_at_nonzero_pitch(tmp_path):
    path = _write_csv(tmp_path / "d.csv")
    r0 = load_cd_table(path, pitch=1, roll=0, component='wind')[2]
    r45 = load_cd_table(path, pitch=1, roll=45, component='wind')[2]
    mean = load_cd_table(path, pitch=1, component='wind')[2]
    assert np.allclose(mean, 0.5 * (r0 + r45))
    assert cd_roll_spread(path, 1, 'wind') == pytest.approx(np.max(np.abs(r0 - r45)))


def test_partial_mach_pitch_is_rejected(tmp_path):
    """Pitch 3 has only Mach 0.1; using it would silently clamp at 0.1."""
    path = _write_csv(tmp_path / "d.csv")
    with pytest.raises(ValueError, match="incomplete"):
        load_cd_table(path, pitch=3)
    assert complete_pitches(path) == [0.0, 1.0]


def test_pitch_and_roll_validation(tmp_path):
    path = _write_csv(tmp_path / "d.csv")
    with pytest.raises(ValueError):
        load_cd_table(path, pitch=7)
    with pytest.raises(ValueError):
        load_cd_table(path, pitch=1, roll=90)
    with pytest.raises(ValueError):
        load_cd_table(path, component='lift')


def test_cd_vs_pitch_lists_every_pitch(tmp_path):
    path = _write_csv(tmp_path / "d.csv")
    pitch, cd = cd_vs_pitch(path, 0.1, 0, component='wind')
    assert list(pitch) == [0.0, 1.0, 3.0]
    assert np.all(np.diff(cd) > 0)          # more angle, more drag
    with pytest.raises(ValueError):
        cd_vs_pitch(path, 0.7, 0)


def test_real_data_pitch_tables(csv_path):
    assert complete_pitches(csv_path) == [0.0, 1.0, 2.0]
    base = load_cd_table(csv_path)[2]
    wind2 = load_cd_table(csv_path, pitch=2, component='wind')[2]
    total2 = load_cd_table(csv_path, pitch=2)[2]
    assert np.all(wind2[:, 0] > base[:, 0])                 # drag grows with pitch
    assert np.max(wind2[:, 0] / base[:, 0]) < 1.15          # by a few percent
    assert np.max(total2[:, 0] / base[:, 0]) > 1.4          # the trap: force magnitude
