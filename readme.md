# Video Generation Carbon Emissions Calculator

A machine learning-based tool to estimate **carbon emissions (gCO2e)** for text-to-video generation models, including both operational energy consumption and embodied carbon from hardware manufacturing.

## Overview

This calculator predicts the total carbon footprint of video generation models using Random Forest regressors trained on real benchmark data from DiT (Diffusion Transformer) and U-Net architectures. It combines:
- **Energy consumption** (Wh) prediction
- **Runtime duration** (seconds) prediction
- **Operational emissions** (from electricity usage)
- **Embodied emissions** (from GPU manufacturing, amortized over device lifetime)

## Features

- **Dual prediction models**: Separate Random Forests for energy consumption (Wh) and runtime (seconds)
- **Complete carbon footprint**: Operational + embodied emissions
- **Country-specific emission factors**: Uses national electricity carbon intensity
- **Support for 20+ models** including Sora, Mochi, CogVideoX, WAN2.1, Stable Video Diffusion, etc.
- **Separate models** for DiT and U-Net architectures
- **Input validation** with zero-handling and model-specific constraints
- **YAML-based** configuration

## Supported Models

### DiT Architecture
- Sora, Veo, Latte
- WAN2.1, WAN2.2
- CogVideoX (5B, 2B, 1.5)
- Mochi, Runway Gen-4
- ContentV, MAGI-1

### U-Net Architecture
- AnimateDiff
- Stable Video Diffusion
- Runway (Gen-1, Gen-2)
- Pika, ModelScopeT2V
- Lumiere, MagicVideoazy

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
model: WAN2.1              # Model name
duration: 81               # Duration in seconds
fps: 8                     # Frames per second
resolution_height: 508     # Height in pixels
resolution_witdh: 1088     # Width in pixels (note: typo in key name)
denoising_steps: 20        # Number of denoising steps
country: France            # Country for emission factor (e.g., France, USA, China)
```

**Note:** `frame` is automatically calculated as `duration × fps`

### 2. Run the Calculator

```bash
python run.py
```

### 3. View Results

**Console Output:**
```
📥 INPUTS:
  Model: WAN2.1 (dit)
  Steps: 20
  Resolution: 508x1088
  Frames: 648  # (81s × 8fps)

📊 RESULTS:
  Energy: 45.32 Wh
  RunTime: 185.67 s
  Total emissions: 10.56 gCO2e
```

**YAML Output** (saved automatically):
```yaml
inputs:
  model: WAN2.1
  steps: 20
  resolution: 508x1088
  frames: 648
results:
  energy_wh: 45.32
  runtime: 185.67
  Total emissions: 10.56  # in gCO2e
```

## How It Works

### 1. Data Preparation (`ml.py`)

**Energy Model (`prepare_data_wh`)**:
- Loads experimental data from WAN benchmarks (`exp_wan_all.csv`) and Open-Sora data (`data_55_analysis.csv`)
- Filters by model type (DiT or U-Net)
- **For DiT models**: Combines WAN and Open-Sora datasets
- **For U-Net models**: Uses only Open-Sora data
- Splits into train/test (85%/15%) with fixed random seed for reproducibility
- **Normalizes features** (steps, resolution, frames) using StandardScaler
- Saves to `prepared_data_wh.csv` and `future_wh.csv`

**Runtime Model (`prepare_data_duration`)**:
- Same data sources and filtering logic as energy model
- Extracts `duration_generate` from WAN and `Duration` from Open-Sora
- **For DiT**: Combines both datasets
- **For U-Net**: Uses only Open-Sora data
- Removes NaN values with `dropna()`
- Normalizes features and saves to `prepared_data_duration.csv`

### 2. Model Training

- **Algorithm:** Random Forest Regressor (250 trees, `random_state=42`, `n_jobs=-1`)
- **Two separate models:**
  1. **Energy model** (`carbon_impact_model_{dit|unet}.joblib`): Predicts Wh consumption
  2. **Runtime model** (`run_time_model_{dit|unet}.joblib`): Predicts execution time in seconds
- **Features:** 
  - `steps`: Number of denoising steps
  - `res`: Total pixels (height × width)
  - `frames`: Number of frames generated
- **Targets:** 
  - Energy consumption in Watt-hours (Wh)
  - Runtime in seconds (s)

### 3. Prediction Pipeline

```python
Input → Validation → Normalization → Random Forest → Energy (Wh) + Runtime (s)
                                                              ↓
                                    Emission Factor Calculation
                                                              ↓
                              Operational CO2 + Embodied CO2 → Total gCO2e
```

**Validation Rules:**
- If `frames ≤ 0` → Return 0 gCO2e
- If `resolution ≤ 0` → Return 0 gCO2e
- If `steps ≤ 0` → Return 0 gCO2e

### 4. Carbon Emissions Calculation (`emission_factor` function)

**Formula:**
```python
# 1. Operational emissions (from electricity consumption)
country_factor = emission_factor_csv[country]  # gCO2/kWh
carbone_operational = country_factor × (Wh / 1000)  # gCO2

# 2. Embodied emissions (from GPU manufacturing)
GPU_EMBODIED_CO2 = 143 kgCO2e  # NVIDIA A100 fabrication
GPU_LIFETIME = 3 years
GPU_UTILIZATION = 0.75

seconds_in_3_years = 60 × 60 × 24 × 365.25 × 3
carbone_embodied = (runtime_s / seconds_in_3_years) / 0.75 × 143 × 1000  # gCO2

# 3. Total emissions
total_emissions = carbone_operational + carbone_embodied  # gCO2e
```

**Emission Factors:**
- Loaded from `carbone_kwh_country.csv`
- Country-specific electricity carbon intensity (gCO2/kWh)
- Examples: France (233), USA (389), China (555)
- Default fallback: France (233 gCO2/kWh)

### 5. Model Selection

The calculator automatically loads the appropriate models:
- **DiT models** → `carbon_impact_model_dit.joblib` + `run_time_model_dit.joblib`
- **U-Net models** → `carbon_impact_model_unet.joblib` + `run_time_model_unet.joblib`

## Model Performance

Typical performance metrics vary by model type and target:

**Energy Model (Wh prediction):**
- Random Forest with 250 trees
- Trained on normalized features
- Performance depends on training data quality

**Runtime Model (Duration prediction):**
- Random Forest with 250 trees
- Handles variable-length datasets
- NaN values removed during preprocessing

## Architecture

```
final_prog/
├── run.py                  # Main entry point
├── input.yaml             # User configuration
├── utils.py               # YAML loading utilities
├── ml/
│   ├── ml.py              # CarbonImpactModel class
│   ├── compute_wh.py      # Prediction and emission calculation
│   ├── data/
│   │   ├── exp_wan_all.csv              # WAN benchmark data
│   │   ├── data_55_analysis.csv         # Open-Sora benchmark data
│   │   ├── carbone_kwh_country.csv      # Country emission factors
│   │   ├── prepared_data_wh.csv         # Normalized energy training data
│   │   ├── prepared_data_duration.csv   # Normalized runtime training data
│   │   ├── future_wh.csv                # Energy test set
│   │   └── future_duration.csv          # Runtime test set (optional)
│   └── model/
│       ├── carbon_impact_model_dit.joblib   # DiT energy model + scaler
│       ├── run_time_model_dit.joblib        # DiT runtime model + scaler
│       ├── carbon_impact_model_unet.joblib  # U-Net energy model + scaler
│       └── run_time_model_unet.joblib       # U-Net runtime model + scaler
```

## Key Components

### `CarbonImpactModel` Class

**Methods:**
- `prepare_data_wh(model_type)`: Load, combine, normalize energy datasets
- `prepare_data_duration(model_type)`: Load, combine, normalize runtime datasets
- `train(to_pred)`: Train Random Forest model on prepared data
- `predict(future_file)`: Predict for new data from CSV file
- `save_model()`: Persist model + scaler to disk using joblib
- `add_energy_metrics(df)`: Convert WAN benchmark energy units to Wh

### `run_ml()` Function

Main prediction interface:
```python
def run_ml(to_pred: np.ndarray, model_type: str, country: str, new_random_forest=False):
    """
    Args:
        to_pred: [steps, resolution, frames]
        model_type: "dit" or "unet"
        country: Country name for emission factor (e.g., "France", "USA")
        new_random_forest: Force retraining
    
    Returns:
        tuple: (energy_wh: float, runtime_s: float, total_emissions_gCO2e: float)
    """
```

### `emission_factor()` Function

Calculates total carbon emissions:
```python
def emission_factor(country: str, wh: float, rt: float) -> float:
    """
    Args:
        country: Country for emission factor lookup
        wh: Energy consumption in Watt-hours
        rt: Runtime in seconds
    
    Returns:
        float: Total emissions in gCO2e (operational + embodied)
    """
```

## Data Normalization

**Critical:** The model uses `StandardScaler` to normalize inputs during training. The same scaler is saved with the model and automatically applied during prediction.

**Training:**
```python
scaler.fit(X_train)  # Learn mean and std
X_normalized = scaler.transform(X_train)
model.fit(X_normalized, y)
```

**Prediction:**
```python
X_new_normalized = scaler.transform(X_new)  # Apply same transformation
y_pred = model.predict(X_new_normalized)
```

## Retraining the Model

To retrain with new data:

```python
from ml.compute_wh import run_ml
import numpy as np

# Force retraining
result = run_ml(
    to_pred=np.array([50, 1920*1080, 24]),
    model_type="dit",
    country="France",
    new_random_forest=True  # Triggers retraining
)
# Returns: (energy_wh, runtime_s, total_emissions_gCO2e)
```

## Limitations

1. **Extrapolation:** The model may be less accurate for parameters far outside the training data range
2. **Architecture-specific:** Performance varies between DiT and U-Net models
3. **Hardware-agnostic:** Does not account for specific GPU/hardware configurations
