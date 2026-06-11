"""Resolve data and model paths from the repository root (not the process CWD)."""

from pathlib import Path


def project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parent.parent


def ml_data_dir() -> Path:
    """Return the directory containing the ML datasets."""
    return project_root() / "ml" / "data"


def ml_model_dir() -> Path:
    """Return the directory used to store trained model artifacts."""
    return project_root() / "ml" / "model"


def prepared_data_path() -> Path:
    """Return the canonical prepared dataset path."""
    return ml_data_dir() / "prepared_data.csv"


def carbon_country_csv() -> Path:
    """Return the country emission factor lookup table path."""
    return ml_data_dir() / "carbone_kwh_country.csv"


def ensure_model_dir() -> Path:
    """Create the model directory if needed and return it."""
    mdir = ml_model_dir()
    mdir.mkdir(parents=True, exist_ok=True)
    return mdir
