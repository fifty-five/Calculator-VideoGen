"""
carbon_impact_model.py

Predicts carbon impact (kgCO2e) for LLM runs using an ensemble of 12 regression models.

Includes:
- Automatic preprocessing (numeric + categorical)
- Feature engineering
- Ensemble learning via StackingRegressor
- Comparison of all regressors vs ensemble
- Future prediction and model persistence
"""


import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler


# Regressors
from sklearn.ensemble import (
    RandomForestRegressor,
)

warnings.filterwarnings("ignore")


class CarbonImpactModel:

    def __init__(self,
                 input_file="data.csv",
                 target_col="carbon_kgco2e",
                 ts_col="timestamp",
                 random_state=42,
                 model_out="./ml/model/carbon_impact_model.joblib"):

        self.input_file = input_file
        self.target_col = target_col
        self.ts_col = ts_col
        self.random_state = random_state
        self.model_out = model_out

        self.model = None
        self.scaler = StandardScaler()  # ✅ Un seul scaler pour X
        self.feature_cols = []
        self.numeric_cols = []
        self.categorical_cols = []

    def _load_data(self, path=None):
        path = path or self.input_file
        if not os.path.exists(path):
            raise FileNotFoundError(f"File '{path}' not found.")
        df = pd.read_csv(path)
        return df

    def add_energy_metrics(self, df):
        dur_s = df["duration_generate"].astype(float)
        eg = df["energy_generate_gpu"].astype(float)

        # si on interprète comme kWh, puissance moyenne GPU en W :
        power_if_kwh = (eg * 1000.0) / (dur_s / 3600.0)  # W
        plausible_kwh = (power_if_kwh.between(10, 5000)).mean()  # fraction plausible

        if plausible_kwh > 0.5:
            factor_to_Wh = 1000.0
        else:
            factor_to_Wh = 1.0

        energy_gpu_Wh = df["energy_generate_gpu"].astype(float) * factor_to_Wh
        energy_cpu_Wh = df["energy_generate_cpu"].astype(float) * factor_to_Wh
        energy_ram_Wh = df["energy_generate_ram"].astype(float) * factor_to_Wh
        # energy_total_Wh = energy_gpu_Wh, energy_cpu_Wh, energy_ram_Wh.sum(axis=1)
        energy_total_Wh = energy_ram_Wh + energy_gpu_Wh + energy_cpu_Wh
        return round(energy_total_Wh)

    def prepare_data_wh(self, model_type: str):
        csv_hf = pd.read_csv("./ml/data/exp_wan_all.csv", header=0)

        csv_55 = pd.read_csv("./ml/data/data_55_analysis.csv", header=1)
        csv_55 = csv_55.loc[csv_55["Model"] == model_type]

        res_55 = csv_55["Output resolution (Total pixels)"].astype(float).values.ravel()
        frames_55 = csv_55["Frames"].astype(float).values.ravel()
        steps_55 = csv_55["Steps"].astype(float).values.ravel()
        Wh_55 = csv_55["Total Wh"].astype(float).values.ravel()

        res_hf = (csv_hf["width"].astype(float) * csv_hf["height"].astype(float))
        frames_hf = csv_hf["num_frames"]
        steps_hf = csv_hf["steps"]
        Wh_hf = self.add_energy_metrics(csv_hf)

        steps = np.concatenate([steps_hf, steps_55])
        res = np.concatenate((res_hf, res_55))
        frames = np.concatenate((frames_hf, frames_55))
        Wh = np.concatenate((Wh_hf, Wh_55))

        df = pd.DataFrame({
            'steps': steps,
            'res': res,
            'frames': frames,
            'Wh': Wh
        })

        n_test = int(len(df) * 0.15)
        idxs = np.random.choice(len(df), size=n_test, replace=False)

        df_test = df.iloc[idxs].copy()
        df_train = df.drop(idxs).copy()

        feature_cols = ['steps', 'res', 'frames']

        # Entraîner le scaler sur le train set
        self.scaler.fit(df_train[feature_cols])

        # Appliquer la normalisation
        df_train[feature_cols] = self.scaler.transform(df_train[feature_cols])
        df_test[feature_cols] = self.scaler.transform(df_test[feature_cols])

        df_train.to_csv("./ml/data/prepared_data_wh.csv", index=False)
        df_test.to_csv("./ml/data/future_wh.csv", index=False)

    def prepare_data_duration(self, model_type: str):
        csv_hf = pd.read_csv("./ml/data/exp_wan_all.csv", header=0)

        csv_55 = pd.read_csv("./ml/data/data_55_analysis.csv", header=1)
        csv_55 = csv_55.loc[csv_55["Model"] == model_type]

        res_55 = csv_55["Output resolution (Total pixels)"].astype(float).values.ravel()
        frames_55 = csv_55["Frames"].astype(float).values.ravel()
        steps_55 = csv_55["Steps"].astype(float).values.ravel()
        duration_55 = csv_55["Duration"].astype(float).values.ravel()

        res_hf = (csv_hf["width"].astype(float) * csv_hf["height"].astype(float))
        frames_hf = csv_hf["num_frames"]
        steps_hf = csv_hf["steps"]
        duration_hf = csv_hf["duration_generate"].astype(float).values.ravel()

        steps = np.concatenate([steps_hf, steps_55])
        res = np.concatenate((res_hf, res_55))
        frames = np.concatenate((frames_hf, frames_55))
        duration = np.concatenate((duration_hf, duration_55))

        df = pd.DataFrame({
            'steps': steps,
            'res': res,
            'frames': frames,
            'duration': duration
        })

        df = df.dropna().reset_index().drop(["index"], axis=1)
        # n_test = int(len(df) * 0.15)
        # idxs = np.random.choice(len(df), size=n_test, replace=False)

        feature_cols = ['steps', 'res', 'frames']

        # Entraîner le scaler sur le train set
        self.scaler.fit(df[feature_cols])

        # Appliquer la normalisation
        df[feature_cols] = self.scaler.transform(df[feature_cols])

        df.to_csv("./ml/data/prepared_data_duration.csv", index=False)

    def _build_model(self):
        model = RandomForestRegressor(n_estimators=250, random_state=self.random_state, n_jobs=-1)
        return model

    # ===============================
    # Training
    # ===============================
    def train(self, to_pred):
        df = self._load_data()
        df = df[df[self.target_col].notna()].copy()

        exclude = {self.target_col, self.ts_col}
        self.feature_cols = [c for c in df.columns if c not in exclude]
        self.categorical_cols = [c for c in self.feature_cols if df[c].dtype in ("object", "bool")]
        self.numeric_cols = [c for c in self.feature_cols if c not in self.categorical_cols]

        print(f"Numeric features: {len(self.numeric_cols)} | Categorical features: {len(self.categorical_cols)}")

        X = df[self.feature_cols]
        y = df[self.target_col].astype(float)

        self.model = self._build_model()

        print("\nTraining RandomForest...")
        self.model.fit(X, y)

        # print("input file = ", self.input_file)
        # print("to_pred = ", to_pred)
        # y = pd.read_csv(self.input_file)[self.target_col]
        # y_pred = self.predict(self.input_file)
        # mae = mean_absolute_error(y, y_pred)
        # rmse = mean_squared_error(y, y_pred, squared=False)
        # r2 = r2_score(y, y_pred)

        # print(f"\nEvaluation:\nMAE={mae:.4f} | RMSE={rmse:.4f} | R²={r2:.4f}")
        # to_pred_normalized = self.scaler.transform(to_pred.reshape(1, -1))
        # print(to_pred_normalized)
        # pred = self.model.predict(to_pred_normalized)
        # print(pred)
        self.save_model()
        # self._plot_predictions(y_test, y_pred)
        # return mae, rmse, r2

    # ===============================
    # Predict New Data
    # ===============================
    def predict(self, future_file):
        if not os.path.exists(future_file):
            raise FileNotFoundError(f"Future file '{future_file}' not found.")
        if self.model is None:
            self.load_model()

        df_future = self._load_data(future_file)
        X_future = df_future[self.feature_cols]
        print(self.feature_cols)
        preds = self.model.predict(X_future)
        df_future["predicted_" + self.target_col] = preds

        out_file = "future_with_predictions.csv"
        df_future.to_csv(out_file, index=False)
        print(f"Predictions saved to {out_file}")
        return preds

    # ===============================
    # Save / Load
    # ===============================
    def save_model(self):
        joblib.dump(self, self.model_out)
        print(f"✅ Model saved as '{self.model_out}'")

    # ===============================
    # Helper: Plot predictions
    # ===============================
    def _plot_predictions(self, y_true, y_pred):
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
        plt.xlabel("Actual Carbon Impact")
        plt.ylabel("Predicted Carbon Impact")
        plt.title("Predicted vs Actual Carbon Impact")
        plt.tight_layout()
        plt.show()
