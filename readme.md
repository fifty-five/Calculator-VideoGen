# Video Generation Carbon Emissions Calculator

A production-ready web calculator to estimate **carbon emissions (gCO2e)** for video generation models, including both operational energy consumption and embodied carbon from hardware manufacturing.

## Overview

This calculator predicts the total carbon footprint of video generation models using regression models trained on real benchmark data. It combines:
- **Energy consumption** (Wh) prediction based on model parameters
- **Runtime duration** (seconds) prediction
- **Operational emissions** (from electricity usage with country-specific carbon intensity)
- **Embodied emissions** (from GPU manufacturing, amortized over 3-year device lifetime)
- **Uncertainty quantification** (95% confidence intervals)

## Features

- **Dual prediction models**: Separate scikit-learn regressors for energy (Wh) and duration (seconds)
- **Model comparison**: Tests 6 regression algorithms (LinearRegression, Ridge, SVR, ExtraTrees, RandomForest, GradientBoosting) and selects best per architecture
- **Adaptive selection**: Prefers tree-based models for large datasets (better extrapolation), stability-based selection for small datasets
- **Complete carbon footprint**: Operational + embodied emissions with uncertainty bounds
- **Country-specific emission factors**: Uses national electricity carbon intensity (233 gCO2/kWh for France default)
- **Support for 20+ models** including Sora, Mochi, CogVideoX, WAN2.1, Stable Video Diffusion, etc.
- **Separate models** for DiT, hybrid, and U-Net architectures
- **Fixed parameters**: Denoising steps (50) and FPS (24)
- **YAML-based configuration** with optional input_type selection

## Supported Models

### DiT Architecture
- Sora, Veo, Latte
- WAN2.1, WAN2.2
- Mochi, Runway Gen-4
- ContentV, MAGI-1

### U-Net Architecture
- AnimateDiff
- Stable Video Diffusion
- Runway (Gen-1, Gen-2)
- Pika, ModelScopeT2V
- Lumiere, MagicVideoazy

### Hybrids
- CogVideoX (5B, 2B, 1.5)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure you have the required data files
# - ml/data/exp_wan_all.csv
# - ml/data/data_55_analysis.csv
```

## Usage

### 1. Configure Input

Edit `input.yaml` with your generation parameters:

```yaml
# ============================================================
# VIDEO GENERATION CARBON CALCULATOR - Configuration
# ============================================================

model: CogVideoX-5B        # Model name (see supported models below)
duration: 2                # Duration in seconds
resolution_height: 480     # Height in pixels
resolution_witdh: 720      # Width in pixels
input_type: text           # text or image (optional, defaults to text)
country: France            # Country for emission factor
```

**Fixed Parameters (coded in `run.py`):**
- `denoising_steps`: 50 (fixed for optimal performance)
- `fps`: 24 (fixed, total_frames = duration × 24)

**Note:** `total_frames` is automatically calculated as `duration × FPS` where FPS=24

### 2. Run the Calculator

```bash
python run.py
```

### 3. View Results

**Console Output Example:**
```
📥 INPUTS:
  Model: CogVideoX-5B (hybrid, 5.0B params)
  Steps: 60
  Resolution: 576x1024
  Frames: 48

📊 RESULTS:
  Energy: 31.14 ± 1.93 Wh
    Model: SVR_rbf (R²=0.982)
    95% interval: 25.90 - 36.38 Wh

  run_time: 855.90 ± 62.87 s (14.26 min)
    Model: ExtraTrees (R²=0.943)
    95% interval: 672.97 - 1038.83 s

  Carbon emissions: 21.25 gCO2e
    Embodied: 1.72 gCO2e
    Electricity: 19.53 gCO2e
    95% interval: 17.60 - 24.91 gCO2e

  Water used: 0.02 L
    95% interval: 0.01 - 0.02 L
```

**JSON Output** (returned as Python dict):
```python
{
  'inputs': {
    'model': 'CogVideoX-5B',
    'steps': 50,
    'resolution': '480x720',
    'frames': 48
  },
  'predictions': {
    'energy': {
      'value_wh': 5.63,
      'uncertainty_wh': 1.57,
      'margin_95_wh': 7.63,
      'best_case_wh': 2.00,
      'worst_case_wh': 9.96,
      'model': 'SVR_rbf',
      'r2': 0.988
    },
    'duration': {
      'value_s': 147.32,
      'value_min': 2.46,
      'uncertainty_s': 69.90,
      'margin_95_s': 192.45,
      'best_case_s': 4.00,
      'worst_case_s': 340.77,
      'model': 'ExtraTrees',
      'r2': 0.937
    },
    'carbon': {
      'value_gco2e': 0.48,
      'best_case_gco2e': 0.04,
      'worst_case_gco2e': 1.01
    }
    'water_used': {
      'value_water_used': 0.02,
      'best_case_water_used': 0.01,
      'worst_case_water_used': 0.03
    }
  }
}
```

## How It Works

### 1. Data Preparation

**Energy Data** (`prepared_data.csv`):
- Loads from `ml/data/prepared_data.csv` (pre-processed)
- Features: `steps`, `res` (total pixels), `frames`, `params` (model size in billions)
- One-hot encodes `Input type` (text/image)
- Target: `Wh` (Watt-hours)

**Duration Data** (`prepared_data.csv`):
- Same dataset, different target
- Features: `steps`, `res`, `frames`, `params`, `input_type`
- Target: `duration` (seconds)

### 2. Model Training & Selection

**Architecture:** Separate models for `dit`, `hybrid`, and `unet`

**Algorithm Comparison:**
Tests 6 regressors per architecture:
1. LinearRegression
2. Ridge
3. SVR
4. ExtraTrees
5. RandomForest
6. GradientBoosting

**Model Selection Logic:**
- **Large datasets (n ≥ 100)**: Prefer tree-based models (ExtraTrees, RandomForest, GradientBoosting)
  - Reason: Better extrapolation beyond training range
  - LinearRegression/Ridge avoid due to poor extrapolation
- **Small datasets (n < 100)**: Prioritize stability
  - Filter models with overfitting gap ≤ 0.08 (R² - CV_R² ≤ 0.08)
  - Among stable models, select lowest MAE
  - Fallback: lowest gap, then lowest MAE

**Metrics Tracked:**
- R² score (test set)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- CV R² (5-fold cross-validation)

### 3. Prediction Pipeline

```
Input (steps, res, frames, params, input_type)
         ↓
Input Validation (all > 0)
         ↓
Load Cached Models or Train If Missing
         ↓
StandardScaler Normalization
         ↓
Regression Model Prediction
         ↓
Energy (Wh) + Duration (s) predictions with uncertainty
         ↓
Apply Safety Floors: max(MIN_WH=2.0, pred_wh), max(MIN_DURATION=4.0, pred_s)
         ↓
Calculate 3 Carbon Scenarios (current, best case, worst case)
         ↓
Return JSON with predictions, uncertainties, R² scores
```

**Validation Rules:**
- If any input ≤ 0 → Return error
- Energy predictions: `max(2.0 Wh, prediction)`
- Duration predictions: `max(4.0 s, prediction)`
- Carbon emissions: `max(0.01 gCO2e, prediction)`

**Uncertainty Quantification:**
- **Margin calculation:** 1.96 × RMSE (95% confidence interval)
- **Best case:** prediction - margin (lower bound)
- **Worst case:** prediction + margin (upper bound)
- **Floor protection:** All intervals clamped to minimum values

### 4. Carbon Emissions Calculation

**Formula:**
```
# 1. Power usage effectiveness
PUE = 1.56
wh_w_pue = wh * PUE  # wh

# 2. Operational emissions (from electricity consumption)
country_factor = emission_factor_csv[country]  # gCO2/kWh
operational_co2 = country_factor × (Wh / 1000)  # gCO2

# 3. Embodied emissions (from GPU manufacturing)
GPU_EMBODIED_CO2 = 143.0  # kgCO2e (NVIDIA A100)
GPU_LIFETIME_YEARS = 3.0
GPU_UTILIZATION = 0.75

seconds_in_3_years = 60 × 60 × 24 × 365.25 × 3
embodied_co2 = (runtime_s / seconds_in_3_years) / 0.75 × 143.0 × 1000  # gCO2

# 4. Water used
water_usage =  0.35L/kWh
water_used = wh_w_pue / 1000 * water_usage

# 5. Total emissions
total_emissions = operational_co2 + embodied_co2  # gCO2e
```

**Emission Factors by Country:**
- France: 33 gCO2/kWh
- USA: 402 gCO2/kWh
- China: 541 gCO2/kWh
- Germany: 333 gCO2/kWh
- Default (unknown country): 220 gCO2/kWh (Glbal avg server EF for electricity)

## Model Performance

Typical performance metrics vary by model type and target:



## Project Structure

```
final_prog/
├── run.py                          # Main entry point (fixed FPS=24, steps=50)
├── input.yaml                      # User configuration (model, duration, resolution, input_type)
├── utils.py                        # YAML loading and validation utilities
├── readme.md                       # This file
│
├── ml/
│   ├── compute_wh.py              # Main prediction orchestration + carbon calculation
│   ├── ml_wh.py                   # VideoEnergyPredictor class (6 algorithm comparison)
│   ├── ml_duration.py             # VideoDurationPredictor class (6 algorithm comparison)
│   │
│   ├── data/
│   │   ├── prepared_data.csv           # Combined training data (steps, res, frames, params, duration, Wh, Input type)
│   │   └── carbone_kwh_country.csv     # Country emission factors (gCO2/kWh)
│   │
│   └── model/
│       ├── best_models_wh.joblib               # Cached energy models dict for all architectures
│       ├── best_models_duration.joblib         # Cached duration models dict for all architectures
│       ├── best_model_wh_dit.joblib            # DiT energy model
│       ├── best_model_wh_hybrid.joblib            # hybrid energy model
│       ├── best_model_wh_unet.joblib           # U-Net energy model
│       ├── best_model_duration_dit.joblib      # DiT duration model
│       ├── best_model_duration_hybrid.joblib      # hybrid duration model
│       ├── best_model_duration_unet.joblib     # U-Net duration model
│       ├── scaler_wh_dit.joblib                # Energy scaler (DiT)
│       └── ... (scaler files for each arch)
```

**Cache Auto-Generation:**
- First run: `best_models_wh.joblib` and `best_models_duration.joblib` are created
- These files store the best model + metrics for each architecture
- Subsequent runs load from cache (~3.3s vs ~35s)

## Summary

| Component | Description |
|-----------|-------------|
| **Prediction** | Tests 6 algorithms, selects best per architecture |
| **Caching** | Models saved on first run (~35s), loaded on subsequent runs (~3.3s) |
| **Energy** | Predicts Wh with 95% confidence interval |
| **Duration** | Predicts seconds with 95% confidence interval |
| **Carbon** | Operational (electricity) + Embodied (GPU manufacturing) |
| **Safety** | Minimum floors: 2.0 Wh, 4.0 s, 0.01 gCO2e | (based on the data minimum)
| **Scalability** | Supports 20+ video generation models across 3 architectures |


