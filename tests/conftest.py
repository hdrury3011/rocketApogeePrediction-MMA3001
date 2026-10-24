"""Shared pytest fixtures."""
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def csv_path():
    """Path to the CFD data set, or skip the test if it is not available.

    The csv is not stored in the repository, so tests that need it skip
    cleanly on a fresh clone. Set APOGEE_DATA to point at a copy.
    """
    candidates = [os.environ.get("APOGEE_DATA"),
                  ROOT / "data" / "project_data_set.csv",
                  ROOT / "data" / "Project Data Set.csv"]
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    pytest.skip("CFD data set not found (set APOGEE_DATA or put it in data/)")


@pytest.fixture(scope="session")
def combined_path():
    """Path to the merged CFD file (with a 'Source' column), or skip the test.

    Set APOGEE_COMBINED to point at a copy, or put combined_cfd_data.csv in data/.
    """
    candidates = [os.environ.get("APOGEE_COMBINED"),
                  ROOT / "data" / "combined_cfd_data.csv"]
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    pytest.skip("merged CFD file not found (set APOGEE_COMBINED or put "
                "combined_cfd_data.csv in data/)")
