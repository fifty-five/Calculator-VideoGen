"""Sanity checks for repository-root path resolution."""

from ml.paths import ml_data_dir, ml_model_dir, prepared_data_path, project_root


def test_project_root_points_at_repo_with_run_py() -> None:
    root = project_root()
    assert (root / "run.py").is_file()


def test_data_and_model_paths_under_ml() -> None:
    assert prepared_data_path() == ml_data_dir() / "prepared_data.csv"
    assert ml_model_dir() == project_root() / "ml" / "model"
