from ml.ml import CarbonImpactModel
import pandas as pd
import numpy as np
import joblib
import os


def emission_factor(country: str, wh: float, run_time: float) -> tuple:
    emission_factor_csv = pd.read_csv("./ml/data/carbone_kwh_country.csv", header=0)
    try:
        country_factor = emission_factor_csv.loc[emission_factor_csv["country"] == country]["Emission factor"]
    except (IndexError, KeyError):
        print(f"⚠️ Country '{country}' not found, using France default (233 gCO2/kWh)")
        country_factor = 233.0  # gCO2/kWh
    carbone_wh = country_factor.values * (wh / 1000)
    print("carbone wh = ", carbone_wh)
    GPU_EMBODIED_CO2 = 143.0  # kgCO2e pour fabrication GPU
    GPU_LIFETIME_YEARS = 3.0
    GPU_UTILIZATION = 0.75

    seconds_in_3_years = 60 * 60 * 24 * 365.25 * GPU_LIFETIME_YEARS
    carbone_embodied = (run_time / seconds_in_3_years) * GPU_UTILIZATION * GPU_EMBODIED_CO2  # kgCO2e

    print("carbone embodied = ", carbone_embodied * 1000)
    total_carbone = carbone_embodied * 1000 + carbone_wh  # gCO2e
    return total_carbone


def run_ml(to_pred: np.ndarray, model_type: str, country: str, new_random_forest=False):
    for param in to_pred:
        if param == 0:
            return 0

    if new_random_forest or not os.path.exists(f"./ml/model/carbon_impact_model_{model_type}.joblib"):
        # 1️⃣ Initialize modèle de base
        model_wh = CarbonImpactModel(input_file="./ml/data/prepared_data_wh.csv", target_col="Wh", model_out=f"./ml/model/carbon_impact_model_{model_type}.joblib")
        model_wh.prepare_data_wh(model_type=model_type)
        model_wh.train("wh_" + model_type)
    else:
        model_wh = joblib.load(f"./ml/model/carbon_impact_model_{model_type}.joblib")

    if not os.path.exists(f"./ml/model/run_time_model_{model_type}.joblib"):
        model_rt = CarbonImpactModel(input_file="./ml/data/prepared_data_duration.csv", target_col="duration", model_out=f"./ml/model/run_time_model_{model_type}.joblib")
        model_rt.prepare_data_duration(model_type=model_type)
        model_rt.train("duration_" + model_type)
    else:
        model_rt = joblib.load(f"./ml/model/run_time_model_{model_type}.joblib")

    to_pred_wh_normalized = model_wh.scaler.transform(to_pred.reshape(1, -1))
    pred_wh = model_wh.model.predict(to_pred_wh_normalized)[0]

    to_pred_rt_normalized = model_rt.scaler.transform(to_pred.reshape(1, -1))
    pred_rt = model_rt.model.predict(to_pred_rt_normalized)[0]

    total_carbon = float(emission_factor(country, pred_wh, pred_rt))
    return (pred_wh, pred_rt, total_carbon)
