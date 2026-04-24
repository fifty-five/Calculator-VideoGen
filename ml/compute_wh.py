import math
from pathlib import Path

import joblib
import pandas as pd

from ml.ml_runtime import Videorun_timePredictor
from ml.ml_wh import VideoEnergyPredictor
from ml.paths import (
    carbon_country_csv,
    ensure_model_dir,
    ml_model_dir,
    prepared_data_path,
)

ENERGY_METADATA = "energy_best_models_metadata.joblib"
RUNTIME_METADATA = "runtime_best_models_metadata.joblib"


def _load_or_migrate_metadata(model_dir: Path, canonical: Path, legacy_glob: str):
    if canonical.exists():
        return joblib.load(canonical)
    legacy_files = sorted(model_dir.glob(legacy_glob))
    for leg in legacy_files:
        data = joblib.load(leg)
        joblib.dump(data, canonical)
        return data
    return None


def emission_factor(country: str, wh: float, run_time: float) -> tuple:
    emission_factor_csv = pd.read_csv(carbon_country_csv(), header=0)
    pue = 1.56
    water_usage = 0.35  # L/ kWh

    wh_w_pue = wh * pue
    try:
        country_factor = emission_factor_csv.loc[
            emission_factor_csv["country"] == country
        ]["Emission factor"].values[0]
    except (IndexError, KeyError):
        country_factor = 220.0  # Glbal avg server EF for electricity
    gpu_embodied_co2 = 143.0  # avg kgCO2e to create a GPU
    gpu_lifetime_years = 3.0
    gpu_utilization = 0.75
    carbon_electricity = country_factor * (wh_w_pue / 1000)  # gCO2
    water_used = wh_w_pue / 1000 * water_usage  # l/kWh

    seconds_in_3_years = 60 * 60 * 24 * 365.25 * gpu_lifetime_years
    carbon_embodied = (
        (run_time / seconds_in_3_years) / gpu_utilization * gpu_embodied_co2
    ) * 1000  # gCO2e
    return carbon_embodied, carbon_electricity, water_used


def prepare_frames(frames: float, arch: str):
    if arch == "hybrid":
        return math.ceil(frames / 49)
    return frames


def run_ml(
    steps: float,
    res: float,
    frames: float,
    fps: int,
    duration: int,
    params: float,
    arch: str,
    input_type: str = "text",
    country: str = "France",
):
    """
    Predict energy and run_time with uncertainties

    Args:
        steps: denoising steps
        res: resolution (pixels)
        frames: number of frames
        params: model parameters (billions)
        arch: architecture (dit, hybrid, unet)
        input_type: text or image
        country: for carbon emission factor

    Returns:
        dict with predictions and uncertainties
    """

    if any(v <= 0 for v in (steps, res, frames, params, fps, duration)):
        return {
            "error": f"Invalid input: steps={steps}, res={res}, frames={frames}, params={params}. All must be > 0"
        }

    ensure_model_dir()
    model_dir = ml_model_dir()
    data_csv = str(prepared_data_path())
    energy_path = model_dir / ENERGY_METADATA
    runtime_path = model_dir / RUNTIME_METADATA

    energy_predictor = VideoEnergyPredictor(data_file=data_csv, model_dir=model_dir)
    run_time_predictor = Videorun_timePredictor(data_file=data_csv, model_dir=model_dir)

    e_meta = _load_or_migrate_metadata(
        model_dir, energy_path, "best_models_wh_*.joblib"
    )
    if e_meta is not None:
        energy_predictor.best_models = e_meta
    else:
        energy_predictor.train_all_architectures()
        joblib.dump(energy_predictor.best_models, energy_path)

    r_meta = _load_or_migrate_metadata(
        model_dir, runtime_path, "best_models_run_time_*.joblib"
    )
    if r_meta is not None:
        run_time_predictor.best_models = r_meta
    else:
        run_time_predictor.train_all_architectures()
        joblib.dump(run_time_predictor.best_models, runtime_path)

    frames = prepare_frames(frames, arch)
    pred_wh = energy_predictor.predict(
        arch, steps, res, frames, fps, duration, params, input_type
    )
    pred_run_time = run_time_predictor.predict(
        arch, steps, res, frames, fps, duration, params, input_type
    )

    if "error" in pred_wh:
        return {"error": f"Energy prediction failed: {pred_wh['error']}"}
    if "error" in pred_run_time:
        return {"error": f"run_time prediction failed: {pred_run_time['error']}"}

    total_carbon_embodied, total_carbon_electricity, total_water_used = emission_factor(
        country, pred_wh["energy_wh"], pred_run_time["run_time_s"]
    )
    best_case_carbon_embodied, best_case_carbon_electricity, best_case_water_used = (
        emission_factor(
            country,
            max(0, pred_wh["energy_wh"] - pred_wh["margin_95_wh"]),
            max(0, pred_run_time["run_time_s"] - pred_run_time["margin_95_s"]),
        )
    )
    (
        worst_case_carbon_carbon_embodied,
        worst_case_carbon_electricity,
        worst_case_water_used,
    ) = emission_factor(
        country,
        pred_wh["energy_wh"] + pred_wh["margin_95_wh"],
        pred_run_time["run_time_s"] + pred_run_time["margin_95_s"],
    )

    min_wh = 2.0  # Minimum energy value from dataset
    min_run_time = 4.0  # Minimum run_time value from dataset

    return {
        "energy": {
            "value_wh": max(min_wh, round(pred_wh["energy_wh"], 2)),
            "uncertainty_wh": pred_wh["uncertainty_wh"],
            "margin_95_wh": pred_wh["margin_95_wh"],
            "best_case_wh": round(
                max(min_wh, pred_wh["energy_wh"] - pred_wh["margin_95_wh"]), 2
            ),
            "worst_case_wh": round(
                max(min_wh, pred_wh["energy_wh"] + pred_wh["margin_95_wh"]), 2
            ),
            "model": pred_wh["model"],
            "r2": pred_wh["r2_score"],
        },
        "run_time": {
            "value_s": max(min_run_time, round(pred_run_time["run_time_s"], 2)),
            "value_min": max(
                min_run_time / 60, round(pred_run_time["run_time_min"], 2)
            ),
            "uncertainty_s": pred_run_time["uncertainty_s"],
            "margin_95_s": pred_run_time["margin_95_s"],
            "best_case_s": round(
                max(
                    min_run_time,
                    pred_run_time["run_time_s"] - pred_run_time["margin_95_s"],
                ),
                2,
            ),
            "worst_case_s": round(
                max(
                    min_run_time,
                    pred_run_time["run_time_s"] + pred_run_time["margin_95_s"],
                ),
                2,
            ),
            "model": pred_run_time["model"],
            "r2": pred_run_time["r2_score"],
        },
        "carbon": {
            "value_gco2e": round(
                max(0.01, total_carbon_embodied + total_carbon_electricity), 2
            ),
            "best_case_gco2e": round(
                max(0.01, best_case_carbon_embodied + best_case_carbon_electricity), 2
            ),
            "worst_case_gco2e": round(
                max(
                    0.01,
                    worst_case_carbon_carbon_embodied + worst_case_carbon_electricity,
                ),
                2,
            ),
            "g_co2_embodied": round(max(0.01, total_carbon_embodied), 2),
            "g_co2_electricity": round(max(0.01, total_carbon_electricity), 2),
        },
        "water_used": {
            "value_water_used": round(max(0.01, total_water_used), 2),
            "best_case_water_used": round(max(0.01, best_case_water_used), 2),
            "worst_case_water_used": round(max(0.01, worst_case_water_used), 2),
        },
    }
