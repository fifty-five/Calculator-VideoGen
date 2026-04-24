"""Resolve data and model paths from the repository root (not the process CWD)."""

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def ml_data_dir() -> Path:
    return project_root() / "ml" / "data"


def ml_model_dir() -> Path:
    return project_root() / "ml" / "model"


def prepared_data_path() -> Path:
    return ml_data_dir() / "prepared_data.csv"


def carbon_country_csv() -> Path:
    return ml_data_dir() / "carbone_kwh_country.csv"


def ensure_model_dir() -> Path:
    mdir = ml_model_dir()
    mdir.mkdir(parents=True, exist_ok=True)
    return mdir
