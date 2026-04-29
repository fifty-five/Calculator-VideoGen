"""
Video Generation Energy (Wh) Predictor
Tests multiple regression models for each architecture (dit, hybrid, unet)
Models: LinearRegression, Ridge, SVR, ExtraTrees, RandomForest, GradientBoosting
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.paths import ml_model_dir, prepared_data_path
from ml.tabular_base import BaseTabularPredictor


class VideoEnergyPredictor(BaseTabularPredictor):
    def __init__(
        self,
        data_file: str | None = None,
        model_dir: Path | None = None,
    ) -> None:
        super().__init__(
            data_file or str(prepared_data_path()),
            model_dir or ml_model_dir(),
        )

    @property
    def target_column(self) -> str:
        return "Wh"

    def _shape_architecture_data(
        self, arch_name: str, df_arch: pd.DataFrame
    ) -> pd.DataFrame:
        if arch_name == "hybrid":
            df_out = df_arch.copy()
            df_out["frames"] = np.ceil(df_out["frames"] / 49)
            return df_out
        return df_arch

    def _choose_best(
        self, _arch_name: str, n_samples: int, arch_results: list
    ) -> dict | None:
        if not arch_results:
            return None
        if n_samples < 70:
            return next(r for r in arch_results if r["model"] == "Ridge")
        return max(arch_results, key=lambda x: x["r2"])

    def _model_and_scaler_prefixes(self) -> tuple[str, str]:
        return "best_model_wh", "scaler_wh"

    def predict(
        self, arch, steps, res, frames, fps, duration, params, input_type="text"
    ):
        """Make prediction with uncertainty"""
        res_factor_hybrid = 0.000045
        res_factor_unet = 0.000012
        refs_res = {"hybrid": 345600, "unet": 589824, "dit": res}
        try:
            model = joblib.load(self.model_dir / f"best_model_wh_{arch}.joblib")
            scaler = joblib.load(self.model_dir / f"scaler_wh_{arch}.joblib")
            best = self.best_models[arch]
            input_image = 1 if input_type.lower() == "image" else 0
            input_text = 1 if input_type.lower() == "text" else 0
            res_arch = refs_res[arch]
            x_row = np.array(
                [
                    [
                        steps,
                        res_arch,
                        frames,
                        params,
                        duration,
                        fps,
                        input_image,
                        input_text,
                    ]
                ]
            )
            x_scaled = scaler.transform(x_row)
            pred = model.predict(x_scaled)[0]
            pred = max(0, pred)
            if arch == "hybrid":
                res_delta = (res - res_arch) * res_factor_hybrid
                pred += res_delta
            elif arch == "unet":
                res_delta = (res - res_arch) * res_factor_unet * (params / 1.5)
                pred += res_delta
            return {
                "energy_wh": round(pred, 2),
                "uncertainty_wh": round(best["mae"], 2),
                "margin_95_wh": round(1.96 * best["rmse"], 2),
                "r2_score": round(best["r2"], 3),
                "model": best["model"],
            }
        except Exception as e:
            return {"error": str(e)}
