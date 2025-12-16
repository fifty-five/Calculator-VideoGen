"""
Video Generation Energy (Wh) Predictor
Tests multiple regression models for each architecture (dit, cog, unet)
Models: LinearRegression, Ridge, SVR, ExtraTrees, RandomForest, GradientBoosting
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class VideoEnergyPredictor:
    def __init__(self, data_file="prepared_data.csv"):
        self.data_file = data_file
        self.base_features = ['steps', 'res', 'frames', 'params']
        self.feature_cols = None
        self.results = {}
        self.best_models = {}

    def prepare_features(self, df):
        df_prep = df.copy()
        if 'Input type' in df_prep.columns:
            input_dummies = pd.get_dummies(df_prep['Input type'], prefix='input')
            df_prep = pd.concat([df_prep, input_dummies], axis=1)
            input_cols = sorted(list(input_dummies.columns))
            if self.feature_cols is None:
                self.feature_cols = self.base_features + input_cols
            for col in input_cols:
                if col not in df_prep.columns:
                    df_prep[col] = 0
        return df_prep

    def get_models(self):
        """Return dict of regression models to test"""
        return {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'SVR_rbf': SVR(kernel='rbf', C=100, epsilon=0.1),
            'ExtraTrees': ExtraTreesRegressor(
                n_estimators=100, max_depth=10, min_samples_leaf=2,
                random_state=42, n_jobs=-1
            ),
            'RandomForest': RandomForestRegressor(
                n_estimators=100, max_depth=10, min_samples_leaf=2,
                random_state=42, n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                subsample=0.8, random_state=42
            ),
        }

    def train_architecture(self, arch_name):
        """Train all models on one architecture"""
        df = pd.read_csv(self.data_file)
        df = self.prepare_features(df)

        df_arch = df[df['architecture'] == arch_name]
        n_samples = len(df_arch)

        if n_samples < 5:
            self.results[arch_name] = {}
            return

        X = df_arch[self.feature_cols]
        y = df_arch['Wh']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        models = self.get_models()
        arch_results = []

        for model_name, model in models.items():
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred = np.clip(y_pred, 0, None)

                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)

                cv_scores = cross_val_score(model, X_train_scaled, y_train,
                                            cv=5, scoring='r2')

                arch_results.append({
                    'model': model_name,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'cv_r2': cv_scores.mean(),
                    'cv_r2_std': cv_scores.std(),
                    'n_test': len(X_test),
                    'model_obj': model,
                    'scaler': scaler
                })
            except Exception as e:
                pass

        # Store results
        self.results[arch_name] = arch_results

        # Find best model
        if arch_results:
            best = sorted(arch_results, key=lambda x: x['r2'], reverse=True)[0]
            self.best_models[arch_name] = best

    def save_models(self):
        """Save best models"""
        for arch in self.best_models:
            best = self.best_models[arch]
            model_path = f"./ml/model/best_model_wh_{arch}.joblib"
            scaler_path = f"./ml/model/scaler_wh_{arch}.joblib"
            joblib.dump(best['model_obj'], model_path)
            joblib.dump(best['scaler'], scaler_path)

    def predict(self, arch, steps, res, frames, params, input_type="text"):
        """Make prediction with uncertainty"""
        try:
            model = joblib.load(f"./ml/model/best_model_wh_{arch}.joblib")
            scaler = joblib.load(f"./ml/model/scaler_wh_{arch}.joblib")
            best = self.best_models[arch]

            # Prepare features
            input_image = 1 if input_type.lower() == "image" else 0
            input_text = 1 if input_type.lower() == "text" else 0

            X = np.array([[steps, res, frames, params, input_image, input_text]])
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)[0]
            pred = max(0, pred)  # No negative energy

            return {
                "energy_wh": round(pred, 2),
                "uncertainty_wh": round(best['mae'], 2),
                "margin_95_wh": round(1.96 * best['rmse'], 2),
                "r2_score": round(best['r2'], 3),
                "model": best['model']
            }
        except Exception as e:
            return {"error": str(e)}

    def train_all_architectures(self):
        """Train all models for all architectures"""
        df = pd.read_csv(self.data_file)
        df = self.prepare_features(df)

        for arch in sorted(df['architecture'].unique()):
            self.train_architecture(arch)

        self.save_models()
