import os
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def run_cli():
    """Invoke `python run.py ...` from the project root and return (returncode, stdout, stderr)."""
    def _run(*args, config=None):
        argv = [sys.executable, "run.py", *[str(a) for a in args]]
        if config is not None:
            argv += ["--config", str(config)]
        result = subprocess.run(
            argv,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        return result.returncode, result.stdout, result.stderr
    return _run


@pytest.fixture
def empty_yaml(tmp_path):
    """Path to a yaml file that holds no keys (parsed as None)."""
    p = tmp_path / "empty.yaml"
    p.write_text("")
    return p
