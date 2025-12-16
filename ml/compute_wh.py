from ml.ml_wh import VideoEnergyPredictor
from ml.ml_duration import VideoDurationPredictor
import pandas as pd
import joblib
import os


def emission_factor(country: str, wh: float, run_time: float) -> tuple:
    emission_factor_csv = pd.read_csv("./ml/data/carbone_kwh_country.csv", header=0)
    try:
        country_factor = emission_factor_csv.loc[emission_factor_csv["country"] == country]["Emission factor"].values[0]
    except (IndexError, KeyError):
        country_factor = 233.0  # gCO2/kWh
    carbone_kwh_g = country_factor * (wh / 1000)
    GPU_EMBODIED_CO2 = 143.0  # kgCO2e pour fabrication GPU
    GPU_LIFETIME_YEARS = 3.0
    GPU_UTILIZATION = 0.75

    seconds_in_3_years = 60 * 60 * 24 * 365.25 * GPU_LIFETIME_YEARS
    carbone_embodied = (run_time / seconds_in_3_years) / GPU_UTILIZATION * GPU_EMBODIED_CO2  # kgCO2e
    total_carbone = carbone_embodied * 1000 + carbone_kwh_g  # gCO2e
    return total_carbone


def run_ml(steps: float, res: float, frames: float, params: float, arch: str,
           input_type: str = "text", country: str = "France"):
    """
    Predict energy and duration with uncertainties

    Args:
        steps: denoising steps
        res: resolution (pixels)
        frames: number of frames
        params: model parameters (billions)
        arch: architecture (dit, cog, unet)
        input_type: text or image
        country: for carbon emission factor

    Returns:
        dict with predictions and uncertainties
    """

    # Safety check: avoid zero or invalid values
    if steps <= 0 or res <= 0 or frames <= 0 or params <= 0:
        return {"error": f"Invalid input: steps={steps}, res={res}, frames={frames}, params={params}. All must be > 0"}

    # Check if models already exist
    wh_best_models_path = f"./ml/model/best_models_wh.joblib"
    duration_best_models_path = f"./ml/model/best_models_duration.joblib"

    # Initialize predictors
    energy_predictor = VideoEnergyPredictor(data_file="./ml/data/prepared_data.csv")
    duration_predictor = VideoDurationPredictor(data_file="./ml/data/prepared_data.csv")

    # Load best_models if they exist, otherwise train
    if os.path.exists(wh_best_models_path):
        energy_predictor.best_models = joblib.load(wh_best_models_path)
    else:
        energy_predictor.train_all_architectures()
        joblib.dump(energy_predictor.best_models, wh_best_models_path)

    if os.path.exists(duration_best_models_path):
        duration_predictor.best_models = joblib.load(duration_best_models_path)
    else:
        duration_predictor.train_all_architectures()
        joblib.dump(duration_predictor.best_models, duration_best_models_path)

    # Make predictions with uncertainties
    pred_wh = energy_predictor.predict(arch, steps, res, frames, params, input_type)
    pred_duration = duration_predictor.predict(arch, steps, res, frames, params, input_type)

    if "error" in pred_wh:
        return {"error": f"Energy prediction failed: {pred_wh['error']}"}
    if "error" in pred_duration:
        return {"error": f"Duration prediction failed: {pred_duration['error']}"}

    # Calculate carbon emissions (with protection against negative values)
    total_carbon = float(emission_factor(country, pred_wh["energy_wh"], pred_duration["duration_s"]))
    best_case_carbon = float(emission_factor(country, max(0, pred_wh["energy_wh"] - pred_wh["margin_95_wh"]),
                                              max(0, pred_duration["duration_s"] - pred_duration["margin_95_s"])))
    worst_case_carbon = float(emission_factor(country, pred_wh["energy_wh"] + pred_wh["margin_95_wh"],
                                               pred_duration["duration_s"] + pred_duration["margin_95_s"]))

    MIN_WH = 2.0  # Minimum energy value from dataset
    MIN_DURATION = 4.0  # Minimum duration value from dataset

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
        "duration": {
            "value_s": max(MIN_DURATION, round(pred_duration["duration_s"], 2)),
            "value_min": max(MIN_DURATION / 60, round(pred_duration["duration_min"], 2)),
            "uncertainty_s": pred_duration["uncertainty_s"],
            "margin_95_s": pred_duration["margin_95_s"],
            "best_case_s": round(max(MIN_DURATION, pred_duration["duration_s"] - pred_duration["margin_95_s"]), 2),
            "worst_case_s": round(max(MIN_DURATION, pred_duration["duration_s"] + pred_duration["margin_95_s"]), 2),
            "model": pred_duration["model"],
            "r2": pred_duration["r2_score"]
        },
        "carbon": {
            "value_gco2e": round(max(0.01, total_carbon), 2),
            "best_case_gco2e": round(max(0.01, best_case_carbon), 2),
            "worst_case_gco2e": round(max(0.01, worst_case_carbon), 2)
        }
    }
