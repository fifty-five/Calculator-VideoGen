from ml.ml_wh import VideoEnergyPredictor
from ml.ml_runtime import Videorun_timePredictor
import pandas as pd
import joblib
import os


def emission_factor(country: str, wh: float, run_time: float) -> tuple:
    emission_factor_csv = pd.read_csv("./ml/data/carbone_kwh_country.csv", header=0)
    PUE = 1.56
    WATER_USAGE = 0.35  # L/ kWh

    wh_w_pue = wh * PUE
    try:
        country_factor = emission_factor_csv.loc[emission_factor_csv["country"] == country]["Emission factor"].values[0]
    except (IndexError, KeyError):
        country_factor = 220.0  # Glbal avg server EF for electricity
    GPU_EMBODIED_CO2 = 143.0  # avg kgCO2e to create a GPU
    GPU_LIFETIME_YEARS = 3.0
    GPU_UTILIZATION = 0.75
    carbon_electricity = country_factor * (wh_w_pue / 1000)  # gCO2
    water_used = wh_w_pue / 1000 * WATER_USAGE  # l/kWh

    seconds_in_3_years = 60 * 60 * 24 * 365.25 * GPU_LIFETIME_YEARS
    carbon_embodied = ((run_time / seconds_in_3_years) / GPU_UTILIZATION * GPU_EMBODIED_CO2) * 1000  # gCO2e
    return carbon_embodied, carbon_electricity, water_used


def run_ml(steps: float, res: float, frames: float, fps: int, duration: int, params: float, arch: str,
           input_type: str = "text", country: str = "France"):
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

    # Safety check: avoid zero or invalid values
    if steps <= 0 or res <= 0 or frames <= 0 or params <= 0 or fps <= 0 or duration <= 0:
        return {"error": f"Invalid input: steps={steps}, res={res}, frames={frames}, params={params}. All must be > 0"}

    # Check if models already exist
    wh_best_models_path = f"./ml/model/best_models_wh_{arch}.joblib"
    run_time_best_models_path = f"./ml/model/best_models_run_time_{arch}.joblib"

    # Initialize predictors
    energy_predictor = VideoEnergyPredictor(data_file="./ml/data/prepared_data.csv")
    run_time_predictor = Videorun_timePredictor(data_file="./ml/data/prepared_data.csv")

    # Load best_models if they exist, otherwise train
    if os.path.exists(wh_best_models_path):
        energy_predictor.best_models = joblib.load(wh_best_models_path)
    else:
        energy_predictor.train_all_architectures()
        joblib.dump(energy_predictor.best_models, wh_best_models_path)

    if os.path.exists(run_time_best_models_path):
        run_time_predictor.best_models = joblib.load(run_time_best_models_path)
    else:
        run_time_predictor.train_all_architectures()
        joblib.dump(run_time_predictor.best_models, run_time_best_models_path)

    # Make predictions with uncertainties
    pred_wh = energy_predictor.predict(arch, steps, res, frames, fps, duration, params, input_type)
    pred_run_time = run_time_predictor.predict(arch, steps, res, frames, fps, duration, params, input_type)

    if "error" in pred_wh:
        return {"error": f"Energy prediction failed: {pred_wh['error']}"}
    if "error" in pred_run_time:
        return {"error": f"run_time prediction failed: {pred_run_time['error']}"}

    # Calculate carbon emissions (with protection against negative values)
    total_carbon_embodied, total_carbon_electricity, total_water_used = emission_factor(country, pred_wh["energy_wh"], pred_run_time["run_time_s"])
    best_case_carbon_embodied, best_case_carbon_electricity, best_case_water_used = emission_factor(country, max(0, pred_wh["energy_wh"] - pred_wh["margin_95_wh"]),
                                                                                                    max(0, pred_run_time["run_time_s"] - pred_run_time["margin_95_s"]))
    worst_case_carbon_carbon_embodied, worst_case_carbon_electricity, worst_case_water_used = emission_factor(country, pred_wh["energy_wh"] + pred_wh["margin_95_wh"],
                                                                                                              pred_run_time["run_time_s"] + pred_run_time["margin_95_s"])

    MIN_WH = 2.0  # Minimum energy value from dataset
    MIN_run_time = 4.0  # Minimum run_time value from dataset

    return {
        "energy": {
            "value_wh": max(MIN_WH, round(pred_wh["energy_wh"], 2)),
            "uncertainty_wh": pred_wh["uncertainty_wh"],
            "margin_95_wh": pred_wh["margin_95_wh"],
            "best_case_wh": round(max(MIN_WH, pred_wh["energy_wh"] - pred_wh["margin_95_wh"]), 2),
            "worst_case_wh": round(max(MIN_WH, pred_wh["energy_wh"] + pred_wh["margin_95_wh"]), 2),
            "model": pred_wh["model"],
            "r2": pred_wh["r2_score"]
        },
        "run_time": {
            "value_s": max(MIN_run_time, round(pred_run_time["run_time_s"], 2)),
            "value_min": max(MIN_run_time / 60, round(pred_run_time["run_time_min"], 2)),
            "uncertainty_s": pred_run_time["uncertainty_s"],
            "margin_95_s": pred_run_time["margin_95_s"],
            "best_case_s": round(max(MIN_run_time, pred_run_time["run_time_s"] - pred_run_time["margin_95_s"]), 2),
            "worst_case_s": round(max(MIN_run_time, pred_run_time["run_time_s"] + pred_run_time["margin_95_s"]), 2),
            "model": pred_run_time["model"],
            "r2": pred_run_time["r2_score"]
        },
        "carbon": {
            "value_gco2e": round(max(0.01, total_carbon_embodied + total_carbon_electricity), 2),
            "best_case_gco2e": round(max(0.01, best_case_carbon_embodied + best_case_carbon_electricity), 2),
            "worst_case_gco2e": round(max(0.01, worst_case_carbon_carbon_embodied + worst_case_carbon_electricity), 2),
            "g_co2_embodied": round(max(0.01, total_carbon_embodied), 2),
            "g_co2_electricity": round(max(0.01, total_carbon_electricity), 2)
        },
        "water_used": {
            "value_water_used": round(max(0.01, total_water_used), 2),
            "best_case_water_used": round(max(0.01, best_case_water_used), 2),
            "worst_case_water_used": round(max(0.01, worst_case_water_used), 2),
        }
    }
