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
