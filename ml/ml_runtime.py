"""
Video Generation run_time Predictor
Tests multiple regression models for each architecture (dit, hybrid, unet)
Predicts: run_time (seconds)
Models: LinearRegression, Ridge, SVR, ExtraTrees, RandomForest, GradientBoosting
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.paths import ml_model_dir, prepared_data_path
from ml.tabular_base import BaseTabularPredictor


def _select_best_run_time(n_samples: int, arch_results: list) -> dict | None:
    if not arch_results:
        return None
    if n_samples >= 100:
        tree_models = [
            r
            for r in arch_results
            if r["model"] in ("ExtraTrees", "RandomForest", "GradientBoosting")
        ]
        if tree_models:
            return max(tree_models, key=lambda x: x["r2"])
        return max(arch_results, key=lambda x: x["r2"])
    stable_models = [r for r in arch_results if (r["r2"] - r["cv_r2"]) <= 0.08]
    if stable_models:
        return min(stable_models, key=lambda x: x["mae"])
    return min(arch_results, key=lambda x: (x["r2"] - x["cv_r2"], x["mae"]))


class Videorun_timePredictor(BaseTabularPredictor):
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
        return "run_time"

    def _shape_architecture_data(
        self, arch_name: str, df_arch: pd.DataFrame
    ) -> pd.DataFrame:
        df_work = df_arch.copy()
        if arch_name == "hybrid":
            df_work.loc[:, "frames"] = np.ceil(df_work["frames"] / 49)
        return df_work.dropna().reset_index(drop=True)

    def _choose_best(
        self, _arch_name: str, n_samples: int, arch_results: list
    ) -> dict | None:
        if not arch_results:
            return None
        return _select_best_run_time(n_samples, arch_results)

    def _model_and_scaler_prefixes(self) -> tuple[str, str]:
        return "best_model_run_time", "scaler_run_time"

    def predict(
        self, arch, steps, res, frames, fps, duration, params, input_type="text"
    ):
        """Make prediction with uncertainty"""
        try:
            model = joblib.load(self.model_dir / f"best_model_run_time_{arch}.joblib")
            scaler = joblib.load(self.model_dir / f"scaler_run_time_{arch}.joblib")
            best = self.best_models[arch]
            input_image = 1 if input_type.lower() == "image" else 0
            input_text = 1 if input_type.lower() == "text" else 0
            feature_names = [
                "steps",
                "res",
                "frames",
                "params",
                "duration",
                "fps",
                "input_image",
                "input_text",
            ]
            x_in = pd.DataFrame(
                [
                    [
                        steps,
                        res,
                        frames,
                        params,
                        duration,
                        fps,
                        input_image,
                        input_text,
                    ]
                ],
                columns=feature_names,
            )
            x_scaled = scaler.transform(x_in)
            pred = model.predict(x_scaled)[0]
            pred = max(0, pred)
            return {
                "run_time_s": round(pred, 2),
                "run_time_min": round(pred / 60, 2),
                "uncertainty_s": round(best["mae"], 2),
                "margin_95_s": round(1.96 * best["rmse"], 2),
                "r2_score": round(best["r2"], 3),
                "model": best["model"],
            }
        except Exception as e:
            return {"error": str(e)}
