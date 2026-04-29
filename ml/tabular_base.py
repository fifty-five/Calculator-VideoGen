"""Shared tabular training scaffold for energy and run-time predictors."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

LOG = logging.getLogger(__name__)


def _regression_model_catalog() -> dict:
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "SVR_rbf": SVR(kernel="rbf", C=100, epsilon=0.1),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        ),
    }


def _fit_and_score(
    model_name: str, model, scaler: StandardScaler, x_train, y_train, y_test, x_test
) -> dict | None:
    try:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_pred = np.clip(y_pred, 0, None)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring="r2")
        return {
            "model": model_name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "cv_r2": cv_scores.mean(),
            "cv_r2_std": cv_scores.std(),
            "n_test": len(x_test),
            "model_obj": model,
            "scaler": scaler,
        }
    except Exception as exc:
        LOG.debug("Skipping %s: %s", model_name, exc)
        return None


class BaseTabularPredictor(ABC):
    def __init__(self, data_file: str, model_dir: Path) -> None:
        self.data_file = data_file
        self.model_dir = model_dir
        self.base_features = ["steps", "res", "frames", "params", "duration", "fps"]
        self.feature_cols: list[str] | None = None
        self.results: dict = {}
        self.best_models: dict = {}

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_prep = df.copy()
        if "Input type" in df_prep.columns:
            input_dummies = pd.get_dummies(df_prep["Input type"], prefix="input")
            df_prep = pd.concat([df_prep, input_dummies], axis=1)
            input_cols = sorted(input_dummies.columns)
            if self.feature_cols is None:
                self.feature_cols = self.base_features + list(input_cols)
            for col in input_cols:
                if col not in df_prep.columns:
                    df_prep[col] = 0
        return df_prep

    def get_models(self) -> dict:
        return _regression_model_catalog()

    @property
    @abstractmethod
    def target_column(self) -> str: ...

    @abstractmethod
    def _shape_architecture_data(
        self, arch_name: str, df_arch: pd.DataFrame
    ) -> pd.DataFrame: ...

    @abstractmethod
    def _choose_best(
        self, arch_name: str, n_samples: int, arch_results: list
    ) -> dict | None: ...

    @abstractmethod
    def _model_and_scaler_prefixes(self) -> tuple[str, str]: ...

    def train_architecture(self, arch_name: str) -> None:
        df = pd.read_csv(self.data_file)
        df = self.prepare_features(df)
        assert self.feature_cols is not None
        df_arch = df[df["architecture"] == arch_name].copy()
        n_samples = len(df_arch)
        if n_samples < 5:
            self.results[arch_name] = {}
            return
        shaped = self._shape_architecture_data(arch_name, df_arch)
        x_df = shaped[self.feature_cols]
        y = shaped[self.target_column]
        x_train, x_test, y_train, y_test = train_test_split(
            x_df, y, test_size=0.25, random_state=42
        )
        scaler = StandardScaler()
        x_train_s = scaler.fit_transform(x_train)
        x_test_s = scaler.transform(x_test)
        arch_results: list = []
        for model_name, model in self.get_models().items():
            result = _fit_and_score(
                model_name, model, scaler, x_train_s, y_train, y_test, x_test_s
            )
            if result is not None:
                arch_results.append(result)
        self.results[arch_name] = arch_results
        best = self._choose_best(arch_name, n_samples, arch_results)
        if best is not None:
            self.best_models[arch_name] = best

    def save_models(self) -> None:
        best_pfx, scaler_pfx = self._model_and_scaler_prefixes()
        for arch, best in self.best_models.items():
            mpath = self.model_dir / f"{best_pfx}_{arch}.joblib"
            spath = self.model_dir / f"{scaler_pfx}_{arch}.joblib"
            joblib.dump(best["model_obj"], mpath)
            joblib.dump(best["scaler"], spath)

    def train_all_architectures(self) -> None:
        df = pd.read_csv(self.data_file)
        df = self.prepare_features(df)
        for arch in sorted(df["architecture"].unique()):
            self.train_architecture(arch)
        self.save_models()
