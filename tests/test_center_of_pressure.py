import numpy as np

from center_of_pressure import center_of_pressure, load_cfd_dataset, cp_by_condition


def synthetic(x_cp_true, x_ref, roll_deg, n=5, seed=0):
    rng = np.random.default_rng(seed)
    fn = rng.uniform(50, 500, n)
    roll = np.radians(roll_deg)
    Fy, Fz = fn * np.sin(roll), fn * np.cos(roll)
    d = x_cp_true - x_ref
    return Fy, Fz, -d * Fz, d * Fy


def test_recovers_known_cp_at_any_roll():
    for roll in (0.0, 45.0, 120.0):
        Fy, Fz, My, Mz = synthetic(-2.5, 0.0, roll)
        assert np.allclose(center_of_pressure(Fy, Fz, My, Mz, 0.0), -2.5)


def test_result_is_independent_of_reference_point():
    x_cp_true = -2.4
    for x_ref in (0.0, -1.0, -3.25):
        Fy, Fz, My, Mz = synthetic(x_cp_true, x_ref, 30.0)
        assert np.allclose(center_of_pressure(Fy, Fz, My, Mz, x_ref), x_cp_true)


def test_small_force_gives_nan():
    out = center_of_pressure([0.1], [0.1], [5.0], [5.0], 0.0, fn_min=1.0)
    assert np.isnan(out[0])


def test_real_data_is_self_consistent(csv_path):
    """The CoP must not depend on which of the 66 stations is used as reference."""
    a = load_cfd_dataset(csv_path, ref_idx=0)
    b = load_cfd_dataset(csv_path, ref_idx=40)
    sel = a['pitch'] >= 2
    cp_a = center_of_pressure(a['Fy'], a['Fz'], a['My'], a['Mz'], a['x_ref'])[sel]
    cp_b = center_of_pressure(b['Fy'], b['Fz'], b['My'], b['Mz'], b['x_ref'])[sel]
    assert np.nanmax(np.abs(cp_a - cp_b)) < 1e-3


def test_real_data_cp_is_aft_in_plausible_range(csv_path):
    table = cp_by_condition(load_cfd_dataset(csv_path))
    assert table['s_cp'].between(2.0, 3.0).all()
