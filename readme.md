# Video Generation Environmental Impact Calculator

A comprehensive tool to predict the **environmental footprint** of video generation models, including energy consumption, runtime, carbon emissions (operational + embodied), and water usage.

## Overview

This calculator predicts the complete environmental impact of video generation using machine learning models trained on real benchmark data from 15+ state-of-the-art video generation models. The system provides:

- **Energy consumption** (Wh) with 95% confidence intervals
- **Runtime duration** (seconds) with uncertainty quantification
- **Carbon emissions** (gCO2e): operational + embodied emissions
- **Water usage** (L): data center cooling water consumption
- **Architecture-specific models**: Separate predictors for DiT, U-Net, and Hybrid architectures
- **Automated model selection**: Compares 6 regression algorithms and selects the best performer

## Features

✅ **Dual Prediction System**
- Separate ML models for energy (Wh) and runtime (seconds)
- 6 algorithm comparison per architecture: LinearRegression, Ridge, SVR, ExtraTrees, RandomForest, GradientBoosting

✅ **Complete Environmental Footprint**
- **Operational emissions**: Electricity consumption with country-specific carbon intensity
- **Embodied emissions**: GPU manufacturing carbon, amortized over 3-year lifetime
- **Water usage**: Data center cooling water (0.35 L/kWh)
- **PUE factor**: 1.56 for realistic data center efficiency

✅ **Intelligent Model Selection**
- Large datasets (n≥100): Prioritizes tree-based models for better extrapolation
- Small datasets (n<100): Selects most stable models (overfitting gap ≤ 0.08)
- Automatic caching: First run trains models (~35s), subsequent runs load cache (~3s)

✅ **Architecture Support**
- **DiT** (Diffusion Transformer): Sora, Mochi, WAN2.1, Veo, Latte
- **U-Net**: AnimateDiff, Stable Video Diffusion, Pika, Lumiere
- **Hybrid**: CogVideoX (5B, 2B)

✅ **Production-Ready**
- Input validation with safety floors
- YAML configuration
- Uncertainty quantification (95% CI)
- Batch processing support (`all_models.py`)

## Supported Models

### DiT Architecture (Diffusion Transformer)
- **Sora** (10B params)
- **Veo** (10B params)
- **Latte-XL** (0.67B params)
- **WAN2.1-T2V-1.3B** (1.3B params)
- **WAN2.1-T2V-14B** (14B params)
- **Mochi 1** (10B params)
- **ContentV** (8B params)

### U-Net Architecture
- **AnimateDiff** (0.9B params)
- **Stable Video Diffusion** (1.5B params)
- **Pika 1.0** (1.5B params)
- **ModelScopeT2V** (1.7B params)
- **Lumiere** (5B params)
- **MagicVideo-V2** (1.5B params)
- **Runway Gen-2** (1.5B params)

### Hybrid Architecture (Transformer + 3D VAE)
- **CogVideoX-5B** (5B params)
- **CogVideoX-2B** (2B params)

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
# Clone or navigate to project directory
cd final_prog

# Install dependencies
pip install pandas numpy scikit-learn pyyaml joblib polars

# Verify data files exist
# - ml/data/prepared_data.csv
# - ml/data/carbone_kwh_country.csv
```

## Quick Start

### 1. Configure Input

Edit [`input.yaml`](input.yaml) with your video generation parameters:

```yaml
model: CogVideoX-5B              # Model name (see supported models)
duration: 2                      # Video duration in seconds
resolution_height: 480           # Height in pixels
resolution_witdh: 720           # Width in pixels (note: typo preserved for compatibility)
fps: 24                         # Frames per second
denoising_steps: 50             # Number of denoising steps
input_type: text                # "text" or "image" (default: text)
country: France                 # Country for carbon intensity factor
```

**Parameter Guidelines:**
- `duration`: 1-10 seconds (typical range)
- `resolution_height` × `resolution_witdh`: 480×720 to 1080×1920
- `fps`: 8, 16, 24, 30 (common values)
- `denoising_steps`: 20-100 (50 is optimal for most models)
- `input_type`: "text" (text-to-video) or "image" (image-to-video)
- `total_frames`: Auto-calculated as `duration × fps`

### 2. Run Single Prediction

```bash
python run.py
```

### 3. Output

**Console Output:**
```
📥 INPUTS:
  Model: CogVideoX-5B (hybrid, 5.0B params)
  Steps: 50
  Resolution: 480x720
  Frames: 48

📊 RESULTS:
  Energy: 5.63 ± 1.57 Wh
    Model: SVR_rbf (R²=0.988)
    95% interval: 2.00 - 13.26 Wh

  run_time: 147.32 ± 69.90 s (2.46 min)
    Model: ExtraTrees (R²=0.937)
    95% interval: 4.00 - 339.77 s

  Carbon emissions: 0.48 gCO2e
    Embodied: 0.03 gCO2e
    Electricity: 0.45 gCO2e
    95% interval: 0.04 - 1.01 gCO2e

  Water used: 0.003 L
    95% interval: 0.002 - 0.007 L
```

**Return Value (Python dict):**
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
      'value_wh': 5.63,              # Predicted energy consumption
      'uncertainty_wh': 1.57,         # Mean absolute error
      'margin_95_wh': 3.08,          # 95% CI margin (1.96×RMSE)
      'best_case_wh': 2.56,          # Lower bound (max(MIN, pred-margin))
      'worst_case_wh': 8.71,         # Upper bound
      'model': 'SVR_rbf',            # Selected algorithm
      'r2': 0.988                    # R² score
    },
    'run_time': {
      'value_s': 147.32,             # Predicted runtime (seconds)
      'value_min': 2.46,             # Runtime in minutes
      'uncertainty_s': 69.90,        # Mean absolute error
      'margin_95_s': 137.01,         # 95% CI margin
      'best_case_s': 10.31,          # Lower bound
      'worst_case_s': 284.33,        # Upper bound
      'model': 'ExtraTrees',
      'r2': 0.937
    },
    'carbon': {
      'value_gco2e': 0.48,           # Total carbon (operational + embodied)
      'g_co2_embodied': 0.03,        # GPU manufacturing carbon
      'g_co2_electricity': 0.45,     # Electricity carbon
      'best_case_gco2e': 0.18,       # Best case scenario
      'worst_case_gco2e': 0.78       # Worst case scenario
    },
    'water_used': {
      'value_water_used': 0.003,     # Water consumption (L)
      'best_case_water_used': 0.001,
      'worst_case_water_used': 0.005
    }
  }
}
```

### 4. Batch Processing (All Models)

To compare all supported models at once:

```bash
python all_models.py
```

This will:
- Run predictions for all 17 models
- Save results to `result_all_models.csv`
- Use fixed parameters: 720p, 8s duration, 24fps, 50 steps, image input

## Advanced Usage

### Adding New Models

Edit [`run.py`](run.py), add to `model_configs` dict:

```python
"YourModel": {"arch": "dit", "params": 15.0}  # arch: dit, unet, or hybrid
```

### Changing Default Safety Floors

Edit [`ml/compute_wh.py`](ml/compute_wh.py#L103):

```python
MIN_WH = 2.0         # Minimum energy (Wh)
MIN_run_time = 4.0   # Minimum runtime (seconds)
```

### Re-training Models

Delete cached models to force retraining:

```bash
rm ml/model/best_models_*.joblib
python run.py  # Will retrain (~35 seconds)
```

## Technical Details

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Input                           │
│  (model, duration, resolution, fps, steps, input_type)      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ↓
        ┌────────────────────────────┐
        │   Model Configuration      │
        │  (arch, params from dict)  │
        └────────┬───────────────────┘
                 │
                 ↓
        ┌────────────────────────────┐
        │   Check Model Cache        │
        │  (best_models_*.joblib)    │
        └────┬───────────────┬───────┘
             │               │
       Cache │               │ No cache
       exists│               │
             ↓               ↓
        ┌────────┐    ┌──────────────┐
        │  Load  │    │ Train Models │
        │ Models │    │ (~35 seconds)│
        └────┬───┘    └──────┬───────┘
             │               │
             └───────┬───────┘
                     ↓
        ┌────────────────────────────┐
        │  Feature Preparation       │
        │ (steps, res, frames, etc.) │
        └────────┬───────────────────┘
                 │
                 ↓
        ┌────────────────────────────┐
        │  StandardScaler Transform  │
        └────────┬───────────────────┘
                 │
         ┌───────┴───────┐
         │               │
         ↓               ↓
    ┌─────────┐    ┌──────────┐
    │ Energy  │    │ Runtime  │
    │Predictor│    │Predictor │
    └────┬────┘    └────┬─────┘
         │              │
         └──────┬───────┘
                ↓
    ┌───────────────────────┐
    │ Emission Factor Calc  │
    │  • PUE = 1.56         │
    │  • Country factor     │
    │  • Embodied carbon    │
    │  • Water usage        │
    └───────┬───────────────┘
            │
            ↓
    ┌───────────────────────┐
    │  Final Results Dict   │
    │  • Energy (Wh)        │
    │  • Runtime (s)        │
    │  • Carbon (gCO2e)     │
    │  • Water (L)          │
    │  • 95% CI intervals   │
    └───────────────────────┘
```

### Data Pipeline

**1. Data Preparation** ([`ml/data/prepared_data.csv`](ml/data/prepared_data.csv))

The training dataset contains benchmark measurements from 15+ video generation models:

| Column | Description | Example |
|--------|-------------|---------|
| `architecture` | Model type | dit, unet, hybrid |
| `steps` | Denoising steps | 50 |
| `res` | Total pixels (height × width) | 345,600 (480×720) |
| `frames` | Number of frames | 48 |
| `fps` | Frames per second | 24 |
| `duration` | Video duration (seconds) | 2 |
| `params` | Model parameters (billions) | 5.0 |
| `Input type` | Input modality | text, image |
| `Wh` | Measured energy (target) | 5.63 |
| `run_time` | Measured runtime (target) | 147.32 |

**Special Processing for Hybrid Architecture:**
- CogVideoX models use 49-frame chunks for processing
- Frame count is normalized: `frames_normalized = ceil(frames / 49)`

**2. Model Training** ([`ml/ml_wh.py`](ml/ml_wh.py), [`ml/ml_runtime.py`](ml/ml_runtime.py))

**Architecture-Specific Training:**
- Data is split by `architecture` column (dit, unet, hybrid)
- Each architecture trains 6 regression algorithms independently
- Feature engineering: One-hot encodes `Input type` → `input_image`, `input_text`

**Algorithm Comparison:**
```python
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'SVR_rbf': SVR(kernel='rbf', C=100, epsilon=0.1),
    'ExtraTrees': ExtraTreesRegressor(n_estimators=100, max_depth=10),
    'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100)
}
```

**Train/Test Split:**
- 75% training, 25% testing
- StandardScaler normalization on training set
- 5-fold cross-validation for stability assessment

**Model Selection Logic:**

| Dataset Size | Selection Strategy |
|-------------|-------------------|
| **n ≥ 100** | Prefer tree-based models (ExtraTrees, RandomForest, GradientBoosting)<br>Reason: Better extrapolation beyond training range |
| **n < 100** | Priority: Stability (overfitting gap ≤ 0.08)<br>Secondary: Lowest MAE<br>Fallback: Lowest gap → lowest MAE |

*Overfitting gap = R² (test) - R² (cross-validation)*

**3. Prediction** ([`ml/compute_wh.py`](ml/compute_wh.py))

**Input Validation:**
```python
if steps <= 0 or res <= 0 or frames <= 0 or params <= 0:
    return {"error": "Invalid input"}
```

**Prediction Pipeline:**
1. Prepare input features: `[steps, res, frames, fps, duration, params, input_image, input_text]`
2. Convert to DataFrame with column names (avoids StandardScaler warnings)
3. Load cached scaler and transform features
4. Predict with best model for architecture
5. Apply safety floors: `max(MIN_WH=2.0, prediction)`, `max(MIN_run_time=4.0, prediction)`
6. Calculate uncertainty: `margin_95 = 1.96 × RMSE`

**4. Carbon Emissions** ([`ml/compute_wh.py:emission_factor`](ml/compute_wh.py#L8))

```python
# Constants
PUE = 1.56                        # Power Usage Effectiveness
WATER_USAGE = 0.35                # L/kWh
GPU_EMBODIED_CO2 = 143.0          # kgCO2e (NVIDIA A100)
GPU_LIFETIME_YEARS = 3.0
GPU_UTILIZATION = 0.75

# Calculations
wh_with_pue = wh * PUE
operational_co2 = country_factor * (wh_with_pue / 1000)  # gCO2

seconds_in_lifetime = 60 * 60 * 24 * 365.25 * 3
embodied_co2 = (runtime_s / seconds_in_lifetime) / 0.75 * 143.0 * 1000  # gCO2

water_used = wh_with_pue / 1000 * 0.35  # L

total_co2 = operational_co2 + embodied_co2  # gCO2e
```

**Country Emission Factors** ([`ml/data/carbone_kwh_country.csv`](ml/data/carbone_kwh_country.csv)):

| Country | gCO2/kWh | Source |
|---------|----------|--------|
| France | 33 | Low-carbon mix (nuclear) |
| United States | 402 | Mixed grid |
| China | 541 | Coal-heavy |
| Germany | 333 | Renewables + coal |
| **Default** | 220 | Global server average |

### Uncertainty Quantification

**95% Confidence Intervals:**
```python
margin_95 = 1.96 * RMSE  # Assumes normal distribution

best_case = max(MIN_VALUE, prediction - margin_95)
worst_case = prediction + margin_95
```

**Propagation Through Carbon Calculation:**
- 3 scenarios calculated: nominal, best case, worst case
- Each scenario uses corresponding energy and runtime values
- Final uncertainties account for both prediction and carbon factor uncertainties

## Project Structure

```
final_prog/
├── run.py                             # Main entry point
├── all_models.py                      # Batch processing script (all models)
├── input.yaml                         # User configuration
├── utils.py                           # YAML/CSV utilities, validation
├── result_all_models.csv             # Batch results output
├── readme.md                          # This file
│
└── ml/                                # Machine learning modules
    ├── compute_wh.py                  # Prediction orchestration + carbon calc
    ├── ml_wh.py                       # Energy predictor (6 algorithms)
    ├── ml_runtime.py                  # Runtime predictor (6 algorithms)
    │
    ├── data/
    │   ├── prepared_data.csv          # Training data (steps, res, frames, Wh, run_time)
    │   └── carbone_kwh_country.csv    # Country emission factors (gCO2/kWh)
    │
    └── model/                         # Model cache (auto-generated)
        ├── best_models_wh_dit.joblib
        ├── best_models_wh_unet.joblib
        ├── best_models_wh_hybrid.joblib
        ├── best_models_run_time_dit.joblib
        ├── best_models_run_time_unet.joblib
        ├── best_models_run_time_hybrid.joblib
        ├── best_model_wh_dit.joblib           # Selected energy model (DiT)
        ├── best_model_wh_unet.joblib          # Selected energy model (U-Net)
        ├── best_model_wh_hybrid.joblib        # Selected energy model (Hybrid)
        ├── scaler_wh_dit.joblib               # StandardScaler (DiT energy)
        ├── scaler_wh_unet.joblib
        ├── scaler_wh_hybrid.joblib
        ├── best_model_run_time_dit.joblib     # Selected runtime model (DiT)
        ├── best_model_run_time_unet.joblib
        ├── best_model_run_time_hybrid.joblib
        ├── scaler_run_time_dit.joblib         # StandardScaler (DiT runtime)
        ├── scaler_run_time_unet.joblib
        └── scaler_run_time_hybrid.joblib
```

### Key Files

| File | Purpose |
|------|---------|
| [`run.py`](run.py) | Main script: loads config, runs prediction, displays results |
| [`all_models.py`](all_models.py) | Batch processing: runs all models, exports CSV |
| [`input.yaml`](input.yaml) | User configuration: model, resolution, duration, etc. |
| [`utils.py`](utils.py) | Helper functions: YAML loading, validation, CSV export |
| [`ml/compute_wh.py`](ml/compute_wh.py) | Orchestrates energy + runtime prediction, calculates carbon |
| [`ml/ml_wh.py`](ml/ml_wh.py) | VideoEnergyPredictor class (6-algorithm comparison) |
| [`ml/ml_runtime.py`](ml/ml_runtime.py) | Videorun_timePredictor class (6-algorithm comparison) |
| `ml/data/prepared_data.csv` | Training dataset (benchmark measurements) |
| `ml/data/carbone_kwh_country.csv` | Country-specific carbon intensity factors |

### Model Caching

**First Run (~35 seconds):**
- Trains 6 algorithms per architecture (dit, unet, hybrid)
- Tests each on energy and runtime prediction
- Selects best model per architecture
- Saves to `ml/model/best_models_*.joblib`

**Subsequent Runs (~3 seconds):**
- Loads cached models from disk
- Skips training entirely

**Force Retrain:**
```bash
rm ml/model/best_models_*.joblib
python run.py
```

## Performance Benchmarks

### Typical Model Performance

| Architecture | Energy R² | Runtime R² | Best Algorithm (Energy) | Best Algorithm (Runtime) |
|-------------|-----------|------------|------------------------|-------------------------|
| **DiT** | 0.95-0.99 | 0.91-0.95 | SVR, ExtraTrees | ExtraTrees, GradientBoosting |
| **U-Net** | 0.93-0.97 | 0.89-0.93 | RandomForest, SVR | RandomForest |
| **Hybrid** | 0.97-0.99 | 0.93-0.96 | SVR, ExtraTrees | ExtraTrees |

### Prediction Speed

| Operation | Time |
|-----------|------|
| First run (with training) | ~35 seconds |
| Subsequent runs (cached) | ~3 seconds |
| Single prediction | <0.1 seconds |
| Batch (17 models) | ~5 seconds |

## Limitations & Assumptions

### Data Limitations
- **Training data size**: Limited benchmark measurements (~20-150 samples per architecture)
- **Model coverage**: 15+ models, may not generalize to completely new architectures
- **Parameter ranges**: Predictions most accurate within training distribution
  - Steps: 20-100
  - Resolution: 240p-1080p
  - Duration: 1-10 seconds
  - Params: 0.6-24B

### Assumptions
- **GPU type**: Assumes NVIDIA A100 equivalent (143 kgCO2e embodied)
- **Data center PUE**: Fixed at 1.56 (industry average)
- **GPU lifetime**: 3 years amortization
- **GPU utilization**: 75% capacity factor
- **Water usage**: 0.35 L/kWh (cooling)
- **Electricity factors**: National averages, doesn't account for time-of-day variance

### Known Issues
- **Hybrid architecture**: Frame normalization (÷49) specific to CogVideoX, may not apply to future models
- **Small sample sizes**: Some architectures have n<100, leading to higher uncertainty
- **Extrapolation**: Predictions outside training range may be less accurate

## Troubleshooting

### Common Errors

**1. Model not found:**
```
ValueError: Error, model can't be handled or is badly written
```
→ Check model name in `input.yaml` matches exactly (case-sensitive)

**2. Missing data files:**
```
FileNotFoundError: ml/data/prepared_data.csv
```
→ Ensure data files exist in `ml/data/` directory

**3. Invalid parameters:**
```
{"error": "Invalid input: steps=-1, ..."}
```
→ All numeric parameters must be > 0

**4. Pandas SettingWithCopyWarning:**
- **Fixed**: Code now uses `.copy()` and `.loc[]` for DataFrame operations

**5. StandardScaler feature name warnings:**
- **Fixed**: Predictions now use pandas DataFrame with proper column names

### Debug Mode

Enable verbose output by checking model selection:

```python
# In ml/ml_wh.py or ml/ml_runtime.py
print(f"Architecture: {arch_name}")
print(f"Best model: {best_name}")
print(f"R²: {best_metrics['r2']:.3f}")
```

## Summary

| Component | Description |
|-----------|-------------|
| **Prediction** | Tests 6 algorithms, selects best per architecture |
| **Caching** | Models saved on first run (~35s), loaded on subsequent runs (~3s) |
| **Energy** | Predicts Wh with 95% confidence interval |
| **Runtime** | Predicts seconds with 95% confidence interval |
| **Carbon** | Operational (electricity) + Embodied (GPU manufacturing) |
| **Water** | Data center cooling water usage (0.35 L/kWh) |
| **Safety** | Minimum floors: 2.0 Wh, 4.0 s (based on data minimum) |
| **Scalability** | Supports 17+ video generation models across 3 architectures |

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{video_generation_carbon_calculator,
  title = {Video Generation Environmental Impact Calculator},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/sustainability_project}
}
```

## License

[Specify your license]

## Contact

[Your contact information]

---

**Last Updated:** January 2026  
**Version:** 1.0


