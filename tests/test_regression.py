import numpy as np
import pandas as pd
import pytest

from drag import GRID_SOURCE, load_cd_points, load_cd_table
from regression import (RegressionDragModel, cross_validate, select_degrees,
                        source_offset)


def _surface(m, f):
    """A known smooth C_D surface, polynomial of degree (2, 1)."""
    return 0.40 + 0.10 * m + 0.20 * m**2 + 0.001 * f + 0.0002 * f * m


N_CORNER = 4                       # the first samples pin the corners of the range
HOLDOUT = np.arange(204) >= N_CORNER   # cross-validation holds out the rest only


@pytest.fixture
def samples():
    """204 samples; the first four are the corners of the Mach-flap range so a
    held-out sample is never an extrapolation beyond the fitted range."""
    rng = np.random.default_rng(1)
    m = np.r_[[0.1, 0.1, 0.9, 0.9], rng.uniform(0.1, 0.9, 200)]
    f = np.r_[[0.0, 100.0, 0.0, 100.0], rng.uniform(0.0, 100.0, 200)]
    return m, f, _surface(m, f)


def test_fit_recovers_known_polynomial_exactly(samples):
    m, f, cd = samples
    model = RegressionDragModel.fit(m, f, cd, 2, 1)
    for mm, ff in ((0.1, 0.0), (0.45, 37.0), (0.9, 100.0)):
        assert model(mm, ff) == pytest.approx(_surface(mm, ff), rel=1e-10)
    assert model.n_params == 6 and model.n_samples == 204


def test_at_flap_matches_call(samples):
    m, f, cd = samples
    model = RegressionDragModel.fit(m, f, cd, 4, 2)
    for flap in (0.0, 12.5, 100.0):
        fast = model.at_flap(flap)
        for mach in (0.05, 0.1, 0.33, 0.9, 1.4):
            assert fast(mach) == pytest.approx(model(mach, flap), rel=1e-12)


def test_mach_clamped_and_flap_checked(samples):
    m, f, cd = samples
    model = RegressionDragModel.fit(m, f, cd, 2, 1)
    assert model(5.0, 50.0) == model(model.mach_range[1], 50.0)
    assert model(0.0, 50.0) == model(model.mach_range[0], 50.0)
    with pytest.raises(ValueError):
        model(0.5, 150.0)
    with pytest.raises(ValueError):
        model.at_flap(-1.0)


def test_predict_broadcasts_scalar_flap(samples):
    m, f, cd = samples
    model = RegressionDragModel.fit(m, f, cd, 2, 1)
    out = model.predict(np.linspace(0.1, 0.9, 7), 40.0)
    assert out.shape == (7,)


def test_fit_rejects_bad_input(samples):
    m, f, cd = samples
    with pytest.raises(ValueError):
        RegressionDragModel.fit(m[:5], f[:5], cd[:5], 3, 3)      # too few samples
    with pytest.raises(ValueError):
        RegressionDragModel.fit(m, f[:-1], cd, 2, 1)             # length mismatch
    with pytest.raises(ValueError):
        RegressionDragModel.fit(m, f, -cd, 2, 1)                 # non-positive C_D
    with pytest.raises(ValueError):
        RegressionDragModel.fit(m, f, cd, -1, 1)                 # negative degree
    with pytest.raises(ValueError):
        RegressionDragModel.fit(np.full(10, 0.5), np.linspace(0, 1, 10),
                                np.ones(10), 0, 1)               # no Mach range
    bad = cd.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError):
        RegressionDragModel.fit(m, f, bad, 2, 1)


def test_fit_rejects_surface_that_goes_negative():
    m = np.linspace(0.1, 0.9, 40)
    f = np.zeros(40)
    cd = np.clip(0.05 - 0.4 * (m - 0.5) ** 2 + 0.1, 0.02, None)
    cd = np.where(m < 0.3, 0.001, cd)           # steep drop towards zero
    with pytest.raises(ValueError):
        RegressionDragModel.fit(m, f + np.linspace(0, 100, 40), cd, 9, 1)


def test_noise_free_cross_validation_error_is_tiny(samples):
    m, f, cd = samples
    res = cross_validate(m, f, cd, 2, 1, test_mask=HOLDOUT)
    assert res['rmse'] < 1e-10 and abs(res['bias']) < 1e-10


def test_cross_validation_detects_underfit(samples):
    m, f, cd = samples
    under = cross_validate(m, f, cd, 1, 1, test_mask=HOLDOUT)['rmse']
    right = cross_validate(m, f, cd, 2, 1, test_mask=HOLDOUT)['rmse']
    assert under > 1e6 * right


def test_cross_validation_is_repeatable_and_validates(samples):
    m, f, cd = samples
    a = cross_validate(m, f, cd + 0.001 * np.sin(50 * m), 3, 1, seed=3, test_mask=HOLDOUT)
    b = cross_validate(m, f, cd + 0.001 * np.sin(50 * m), 3, 1, seed=3,
                       test_mask=HOLDOUT)
    assert a == b
    with pytest.raises(ValueError):
        cross_validate(m, f, cd, 2, 1, k=1)
    with pytest.raises(ValueError):
        cross_validate(m, f, cd, 2, 1, k=5, test_mask=np.arange(204) < 3)


def test_test_mask_only_holds_out_flagged_samples(samples):
    """With the mask, un-flagged samples are always in the training set."""
    m, f, cd = samples
    noisy = np.arange(204) >= 104
    cd_bad = cd.copy()
    cd_bad[N_CORNER:104] *= 1.5            # the un-held-out half is biased
    res = cross_validate(m, f, cd_bad, 2, 1, test_mask=noisy)
    assert res['rmse'] > 0.01      # the bias shows up on the held-out half


def test_select_degrees_prefers_simplest_adequate_model(samples):
    m, f, cd = samples
    rng = np.random.default_rng(5)
    noisy = cd * (1 + 0.005 * rng.standard_normal(cd.size))
    (dm, df), table = select_degrees(m, f, noisy, test_mask=HOLDOUT)
    assert (dm, df) == (2, 1)
    assert all(t['n_params'] >= 6 or t['rmse'] > 0 for t in table)
    with pytest.raises(ValueError):
        select_degrees(m[:3], f[:3], cd[:3])


def test_source_offset_recovers_known_shift(samples):
    m, f, cd = samples
    flag = np.arange(204) % 2 == 0
    shifted = np.where(flag, cd * 1.01, cd)
    delta, se = source_offset(m, f, shifted, flag, 2, 1)
    assert delta == pytest.approx(0.01, abs=2e-4) and se >= 0
    with pytest.raises(ValueError):
        source_offset(m, f, cd, np.zeros(204, bool), 2, 1)


# ---------------------------------------------------------------------------
# Loading scattered samples (synthetic csv, no real data needed)
# ---------------------------------------------------------------------------
def _csv(path, rows, source=None):
    recs = []
    for mach, pitch, roll, flap, cd in rows:
        rec = {'Mach Number': mach, 'Pitch (deg)': pitch, 'Roll (deg)': roll,
               'Flap (% extension)': flap, 'Total C_D': cd,
               'C_D vector': f'[{-cd}, 0.0, 0.0]'}
        if source:
            rec['Source'] = source
        recs.append(rec)
    pd.DataFrame(recs).to_csv(path)
    return path


def test_load_points_averages_rolls_and_labels_by_file(tmp_path):
    a = _csv(tmp_path / 'grid.csv', [(0.1, 0, 0, 0, 0.4), (0.1, 0, 45, 0, 0.4),
                                     (0.0, 0, 0, 0, 0.0), (0.1, 2, 0, 0, 0.5)])
    b = _csv(tmp_path / 'scatter.csv', [(0.13, 0, 0, 10, 0.41)])
    pts = load_cd_points([a, b])
    assert len(pts) == 2                       # roll pair merged, Mach 0 dropped
    assert set(pts['source']) == {'grid', 'scatter'}
    assert pts['mach'].is_monotonic_increasing
    assert len(load_cd_points(a, pitch=2)) == 1
    with pytest.raises(ValueError):
        load_cd_points(a, pitch=9)


def test_load_points_keeps_source_column_and_tables_use_grid_only(tmp_path):
    rows = [(m, 0, 0, f, 0.4 + 0.05 * m + 0.0005 * f)
            for m in (0.1, 0.5) for f in (0, 100)]
    grid = pd.DataFrame([{'Mach Number': m, 'Pitch (deg)': p, 'Roll (deg)': r,
                          'Flap (% extension)': f, 'Total C_D': cd,
                          'C_D vector': f'[{-cd}, 0.0, 0.0]',
                          'Source': GRID_SOURCE} for m, p, r, f, cd in rows])
    scatter = grid.iloc[:1].assign(**{'Mach Number': 0.3, 'Source': 'HPR_noisy',
                                      'Flap (% extension)': 37})
    path = tmp_path / 'merged.csv'
    pd.concat([grid, scatter], ignore_index=True).to_csv(path)
    pts = load_cd_points(path)
    assert set(pts['source']) == {GRID_SOURCE, 'HPR_noisy'} and len(pts) == 5
    mach, flap, cd = load_cd_table(path)       # grid rows only: still complete
    assert list(mach) == [0.1, 0.5] and cd.shape == (2, 2)


# ---------------------------------------------------------------------------
# Real data (skipped without the csv files)
# ---------------------------------------------------------------------------
def test_regression_on_real_samples(combined_path):
    pts = load_cd_points(combined_path)
    assert (pts['source'] == GRID_SOURCE).sum() == 27
    assert (pts['source'] != GRID_SOURCE).sum() == 320
    noisy = (pts['source'] != GRID_SOURCE).to_numpy()
    m, f, y = (pts[c].to_numpy() for c in ('mach', 'flap', 'cd'))
    model = RegressionDragModel.fit(m, f, y, 5, 2)
    grid = ~noisy
    err = model.predict(m[grid], f[grid]) / y[grid] - 1
    assert np.sqrt(np.mean(err**2)) < 0.01     # within 1 % of the gridded runs
    cv = cross_validate(m, f, y, 5, 2, test_mask=noisy)
    assert cv['rmse'] < 0.0065                 # better than linear table, 0.68 %
    assert model(0.83, 0.0) < model(0.83, 100.0)
