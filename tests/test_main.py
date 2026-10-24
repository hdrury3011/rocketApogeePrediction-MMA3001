import main


def test_missing_data_file_returns_error(tmp_path, capsys):
    code = main.main(['--data', str(tmp_path / 'missing.csv'), '--outdir', str(tmp_path)])
    assert code == 1
    assert 'data file not found' in capsys.readouterr().err


def test_full_run_without_plots(csv_path, tmp_path):
    """Smoke test: the whole study runs and reports the expected headline results."""
    code = main.main(['--data', str(csv_path), '--outdir', str(tmp_path), '--no-plots'])
    assert code == 0
    text = (tmp_path / 'results.md').read_text(encoding='utf-8')
    assert 'Mass calibration' in text and 'Airbrake validation' in text
    assert 'Angle-of-attack sensitivity' in text


def test_full_run_with_regression_section(combined_path, tmp_path):
    """A merged csv adds the regression section, with the table results unchanged."""
    code = main.main(['--data', str(combined_path), '--outdir', str(tmp_path), '--no-plots'])
    assert code == 0
    text = (tmp_path / 'results.md').read_text(encoding='utf-8')
    assert 'Regression drag model from all CFD samples' in text
    assert '35.9923' in text                    # same calibrated mass as the grid file


def test_missing_extra_file_returns_error(csv_path, tmp_path, capsys):
    code = main.main(['--data', str(csv_path), '--extra-data', str(tmp_path / 'nope.csv'),
                      '--outdir', str(tmp_path)])
    assert code == 1
    assert 'extra data file not found' in capsys.readouterr().err
