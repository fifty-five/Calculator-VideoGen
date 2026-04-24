from typing import Any

import pandas as pd
import pytest

from ml.compute_wh import emission_factor, prepare_frames


@pytest.mark.parametrize(
    ("frames", "arch", "expected"),
    [
        (49, "hybrid", 1),
        (50, "hybrid", 2),
        (100, "hybrid", 3),
        (100, "unet", 100),
        (100, "dit", 100),
        (0, "hybrid", 0),
    ],
)
def test_prepare_frames(frames: int, arch: str, expected: int) -> None:
    assert prepare_frames(frames, arch) == expected


def test_emission_factor_uses_patched_table(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_read_csv(_path: Any, **_: Any) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "country": ["Testland"],
                "Emission factor": [100.0],
            }
        )

    monkeypatch.setattr("ml.compute_wh.pd.read_csv", fake_read_csv)
    wh, run_t = 1000.0, 100.0
    carbon_embodied, carbon_electricity, water_used = emission_factor(
        "Testland", wh, run_t
    )
    assert carbon_embodied == pytest.approx(0.20139540255138402, rel=1e-9, abs=1e-12)
    assert carbon_electricity == pytest.approx(156.0)
    assert water_used == pytest.approx(0.546, rel=1e-9)


def test_emission_factor_unknown_country_fallback_factor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_read_csv(_path: Any, **_: Any) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "country": ["Somewhere"],
                "Emission factor": [50.0],
            }
        )

    monkeypatch.setattr("ml.compute_wh.pd.read_csv", fake_read_csv)
    wh, run_t = 1000.0, 100.0
    carbon_embodied, carbon_electricity, water_used = emission_factor(
        "NotInTable", wh, run_t
    )
    assert carbon_embodied == pytest.approx(0.20139540255138402, rel=1e-9, abs=1e-12)
    assert carbon_electricity == pytest.approx(343.2)
    assert water_used == pytest.approx(0.546, rel=1e-9)
