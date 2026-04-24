# Video Generation Environmental Impact Calculator

A comprehensive tool to predict the **environmental footprint** of video generation models, including energy consumption, runtime, carbon emissions (operational + embodied), and water usage.

## Overview

This calculator predicts the complete environmental impact of video generation using machine learning models trained on real benchmark data (see [`ml/data/prepared_data.csv`](ml/data/prepared_data.csv)). The CLI supports **15** named models in [`run.py`](run.py) `MODEL_CONFIGS`. The system provides:

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

✅ **Intelligent Model Selection** (independent for energy vs runtime)
- **Runtime** (`Videorun_timePredictor`): For n≥100, prefers tree-based models (ExtraTrees, RandomForest, GradientBoosting). For n<100, prefers stable models (test vs CV R² gap ≤ 0.08), then lowest MAE.
- **Energy** (`VideoEnergyPredictor`): For n<70, selects **Ridge**; otherwise picks the best test R² among candidates.
- **Caching**: First run trains both stacks (~35s), subsequent runs load joblib from `ml/model/` (~3s when cache is warm)

✅ **Architecture Support**
- **DiT** (Diffusion Transformer): e.g. Sora, Mochi, WAN2.1, VEO, Latte-XL
- **U-Net**: AnimateDiff, Stable Video Diffusion, Pika, Lumiere
- **Hybrid**: CogVideoX (5B, 2B)

✅ **Production-Ready**
- Input validation with safety floors
- CLI (argparse) with optional YAML fallback
- Single-line JSON output on stdout (pipeline-friendly)
- Docker + docker-compose setup
- Uncertainty quantification (95% CI)
- You can script batch runs (loop over `MODEL_CONFIGS` in [`run.py`](run.py))

## Supported Models

### DiT Architecture (Diffusion Transformer)
- **Sora** (10B params)
- **VEO** (10B params) — use this exact name with `--model` (case-sensitive; matches [`run.py`](run.py) `MODEL_CONFIGS`)
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

### Hybrid Architecture (Transformer + 3D VAE)
- **CogVideoX-5B** (5B params)
- **CogVideoX-2B** (2B params)

## Installation

### Prerequisites
- Python 3.10+ and pip (the codebase uses 3.10+ typing syntax, e.g. `dict | None`), **or** Docker

### Local setup

```bash
# Clone or navigate to project directory
cd Calculator-VideoGen

# Install runtime dependencies
pip install -r requirements.txt

# (Optional) dev dependencies for linting + tests
pip install -r requirements-dev.txt
```

### Docker setup

```bash
docker build -t calculator-videogen .
# or
docker compose build
```

A `docker build` copies the build context (see [`Dockerfile`](Dockerfile) `COPY . .`), so any existing `ml/model/` cache and `ml/data/` on the machine you build on are included. A fresh clone without a local model cache will still **train on first run** in the container (~35s) before cached behavior applies.

## Quick Start

The tool is CLI-first and writes a single line of minified JSON to stdout. `input.yaml` is still supported as an **optional fallback** — CLI flags always win, and if no YAML is found the CLI alone must supply every parameter.

### Run locally

```bash
python run.py \
  --model CogVideoX-5B \
  --duration 5 \
  --fps 24 \
  --resolution-height 1280 \
  --resolution-width 720 \
  --denoising-steps 40 \
  --input-type image \
  --country China
```

### Run in Docker

```bash
docker run --rm calculator-videogen \
  --model CogVideoX-5B --duration 5 --fps 24 \
  --resolution-height 1280 --resolution-width 720 \
  --denoising-steps 40 --input-type image --country China
```

Or with compose:

```bash
docker compose run --rm videogen \
  --model CogVideoX-5B --duration 5 --fps 24 \
  --resolution-height 1280 --resolution-width 720 \
  --denoising-steps 40 --input-type image --country China
```

### CLI flags

| Flag | Type | Description |
|------|------|-------------|
| `--model` | str | Model name (see supported models) |
| `--duration` | int | Video duration in seconds |
| `--resolution-height` | int | Height in pixels |
| `--resolution-width` | int | Width in pixels |
| `--fps` | int | Frames per second |
| `--denoising-steps` | int | Number of denoising steps |
| `--input-type` | str | `text` or `image` |
| `--country` | str | Country for carbon intensity factor |
| `--config` | path | Optional YAML file. Defaults to `input.yaml` in cwd if present. |

**Parameter guidelines:**
- `--duration`: 1-10 seconds (typical range)
- `--resolution-height` × `--resolution-width`: 480×720 to 1080×1920
- `--fps`: 8, 16, 24, 30 (common values)
- `--denoising-steps`: 20-100 (50 is optimal for most models)
- `--input-type`: `text` (text-to-video) or `image` (image-to-video)
- `total_frames` is auto-calculated as `duration × fps`

### YAML fallback

`input.yaml` (or any file passed via `--config`) can supply defaults for missing CLI flags. The legacy `resolution_witdh` key is still accepted for backward compatibility.

```yaml
model: CogVideoX-5B
duration: 2
resolution_height: 480
resolution_width: 720
fps: 24
denoising_steps: 50
input_type: text
country: France
```

### Output

stdout contains a single line of minified JSON (exit 0). Errors go to stderr with a non-zero exit code. Piping into `jq` is the easy way to pretty-print:

```bash
python run.py --model CogVideoX-5B ... | jq .
```

```json
{
  "inputs": {
    "model": "CogVideoX-5B",
    "steps": 50,
    "resolution": "480x720",
    "frames": 48
  },
  "predictions": {
    "energy": {
      "value_wh": 5.63,
      "uncertainty_wh": 1.57,
      "margin_95_wh": 3.08,
      "best_case_wh": 2.56,
      "worst_case_wh": 8.71,
      "model": "SVR_rbf",
      "r2": 0.988
    },
    "run_time": {
      "value_s": 147.32,
      "value_min": 2.46,
      "uncertainty_s": 69.90,
      "margin_95_s": 137.01,
      "best_case_s": 10.31,
      "worst_case_s": 284.33,
      "model": "ExtraTrees",
      "r2": 0.937
    },
    "carbon": {
      "value_gco2e": 0.48,
      "g_co2_embodied": 0.03,
      "g_co2_electricity": 0.45,
      "best_case_gco2e": 0.18,
      "worst_case_gco2e": 0.78
    },
    "water_used": {
      "value_water_used": 0.003,
      "best_case_water_used": 0.001,
      "worst_case_water_used": 0.005
    }
  }
}
```

#### Batch processing (all models)

There is no canonical `all_models.py` in this repository (it may be gitignored in some checkouts or maintained locally). To compare all supported models, write a small loop that invokes `run.py` once per model name in [`run.py`](run.py) `MODEL_CONFIGS` (15 models) and collect stdout JSON or CSV as needed.

## Advanced Usage

### Adding New Models

Edit [`run.py`](run.py), add to the `MODEL_CONFIGS` dict:

```python
"YourModel": {"arch": "dit", "params": 15.0}  # arch: dit, unet, or hybrid
```

### Changing Default Safety Floors

Edit the safety floors in [`ml/compute_wh.py`](ml/compute_wh.py) (search for `min_wh` / `min_run_time`):

```python
min_wh = 2.0         # Minimum energy (Wh)
min_run_time = 4.0   # Minimum runtime (seconds)
```

### Re-training Models

Delete cached models to force retraining. The next CLI invocation will retrain (~35 seconds):

```bash
rm -f ml/model/energy_best_models_metadata.joblib \
      ml/model/runtime_best_models_metadata.joblib \
      ml/model/best_models_wh_*.joblib \
      ml/model/best_models_run_time_*.joblib
python run.py --model CogVideoX-5B --duration 5 --fps 24 \
  --resolution-height 1280 --resolution-width 720 \
  --denoising-steps 40 --input-type image --country China
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
        │(energy_* / runtime_* metadata)│
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

The training dataset contains benchmark rows from multiple source runs and architectures (the CLI exposes **15** named model presets; see `MODEL_CONFIGS` in [`run.py`](run.py)):

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

**Model selection (runtime, `Videorun_timePredictor`)**

| Training samples (per architecture) | Selection strategy |
|-------------|-------------------|
| **n ≥ 100** | Prefer tree-based models (ExtraTrees, RandomForest, GradientBoosting) for extrapolation; else best test R² |
| **n < 100** | Prefer stability (R² test − R² CV ≤ 0.08), then lowest MAE; else minimize gap then MAE |

*Overfitting gap = R² (test) − mean CV R²* (see [`ml/ml_runtime.py`](ml/ml_runtime.py)).

**Model selection (energy, `VideoEnergyPredictor`)**

| Training samples (per architecture) | Selection strategy |
|-------------|-------------------|
| **n < 70** | **Ridge** (forced) |
| **n ≥ 70** | Best test R² among the six candidates |

(Implemented in [`ml/ml_wh.py`](ml/ml_wh.py) via the shared [`ml/tabular_base.py`](ml/tabular_base.py) base class.)

**3. Prediction** ([`ml/compute_wh.py`](ml/compute_wh.py))

**Input Validation** (all must be positive):
```python
if any(v <= 0 for v in (steps, res, frames, params, fps, duration)):
    return {"error": "Invalid input: ..."}
```

**Prediction Pipeline:**
1. Prepare input features: `[steps, res, frames, fps, duration, params, input_image, input_text]`
2. Convert to DataFrame with column names (avoids StandardScaler warnings)
3. Load cached scaler and transform features
4. Predict with best model for architecture
5. Apply safety floors: `max(min_wh=2.0, prediction)` for Wh, `max(min_run_time=4.0, ...)` for seconds (see [`ml/compute_wh.py`](ml/compute_wh.py))
6. Calculate uncertainty: `margin_95 = 1.96 × RMSE`

**4. Carbon Emissions** — `emission_factor` in [`ml/compute_wh.py`](ml/compute_wh.py)

```python
# Constants
PUE = 1.56                        # Power Usage Effectiveness
WATER_USAGE = 0.35                # L/kWh
GPU_EMBODIED_CO2 = 143.0          # kgCO2e (generic GPU figure in code; see Limitations)
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
Calculator-VideoGen/
├── run.py                             # CLI entry point
├── utils.py                           # YAML loading helpers
├── input.yaml                         # Optional fallback config
├── requirements.txt                   # Runtime dependencies
├── requirements-dev.txt               # Format, lint, type-check + test dependencies
├── Dockerfile                         # Runtime container image
├── docker-compose.yml                 # Compose service for convenient CLI runs
├── .pylintrc                          # Lint config
├── pyproject.toml                     # isort (black profile) + mypy defaults
├── readme.md                          # This file
│
├── tests/                             # Pytest: CLI and unit tests
│   ├── conftest.py                    # Shared fixtures
│   ├── test_cli.py                    # Black-box CLI tests
│   ├── test_compute_wh.py             # Unit tests (emission + frame rules)
│   └── test_paths.py                  # Repo path resolution sanity checks
│
└── ml/                                # Machine learning modules
    ├── compute_wh.py                  # Prediction orchestration + carbon calc
    ├── paths.py                       # Resolve ml/data and ml/model from repo root
    ├── tabular_base.py                # Shared sklearn training loop for both predictors
    ├── ml_wh.py                       # Energy predictor (6 algorithms)
    ├── ml_runtime.py                  # Runtime predictor (6 algorithms)
    │
    ├── data/
    │   ├── prepared_data.csv          # Training data (steps, res, frames, Wh, run_time)
    │   └── carbone_kwh_country.csv    # Country emission factors (gCO2/kWh)
    │
    └── model/                         # Model cache (auto-generated)
        ├── energy_best_models_metadata.joblib  # In-memory best-model metadata (all arch; canonical)
        ├── runtime_best_models_metadata.joblib # Same for runtime
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
| [`input.yaml`](input.yaml) | Optional fallback config (CLI flags override) |
| [`utils.py`](utils.py) | Helper functions: YAML loading, validation, CSV export |
| [`Dockerfile`](Dockerfile) | Runtime container image (python:3.11-slim) |
| [`docker-compose.yml`](docker-compose.yml) | Compose service wrapping the CLI |
| [`requirements.txt`](requirements.txt) | Runtime dependencies (version-pinned for sklearn/joblib cache compatibility) |
| [`requirements-dev.txt`](requirements-dev.txt) | Format, lint, type-check, and test dependencies |
| [`pyproject.toml`](pyproject.toml) | `isort` (black profile) and `mypy` defaults |
| [`tests/test_cli.py`](tests/test_cli.py) | Black-box CLI tests |
| [`tests/test_compute_wh.py`](tests/test_compute_wh.py) | Unit tests for `emission_factor` and `prepare_frames` |
| [`ml/compute_wh.py`](ml/compute_wh.py) | Orchestrates energy + runtime prediction, calculates carbon |
| [`ml/paths.py`](ml/paths.py) | `project_root` / `ml` data and model `Path` helpers (CWD-independent) |
| [`ml/tabular_base.py`](ml/tabular_base.py) | Shared base class for training loops |
| [`ml/ml_wh.py`](ml/ml_wh.py) | `VideoEnergyPredictor` (6-algorithm comparison) |
| [`ml/ml_runtime.py`](ml/ml_runtime.py) | `Videorun_timePredictor` (6-algorithm comparison) |
| `ml/data/prepared_data.csv` | Training dataset (benchmark measurements) |
| `ml/data/carbone_kwh_country.csv` | Country-specific carbon intensity factors |

### Model Caching

**First Run (~35 seconds):**
- Trains 6 algorithms per architecture (dit, unet, hybrid)
- Tests each on energy and runtime prediction
- Selects best model per architecture
- Writes per-arch `best_model_*.joblib` / `scaler_*.joblib` plus canonical `energy_best_models_metadata.joblib` and `runtime_best_models_metadata.joblib` (legacy `best_models_wh_*.joblib` / `best_models_run_time_*.joblib` are migrated on first load)

**Subsequent Runs (~3 seconds):**
- Loads cached models from disk
- Skips training entirely

**Force Retrain:** Remove metadata bundles and, if present, any legacy per-arch metadata copies; per-arch model/scaler joblibs are re-created on train.

```bash
rm -f ml/model/energy_best_models_metadata.joblib \
      ml/model/runtime_best_models_metadata.joblib \
      ml/model/best_models_wh_*.joblib \
      ml/model/best_models_run_time_*.joblib
python run.py --model CogVideoX-5B --duration 5 --fps 24 \
  --resolution-height 1280 --resolution-width 720 \
  --denoising-steps 40 --input-type image --country China
```

## Development

### Install dev dependencies

```bash
pip install -r requirements-dev.txt
```

`requirements-dev.txt` pulls in the runtime stack from `requirements.txt` and adds **pytest** (tests), **pylint** (lint), **black** and **isort** (formatting), **mypy** (static typing), **types-PyYAML** and **pandas-stubs** (stubs for cleaner mypy on YAML/CSV helpers and tests), and [`pyproject.toml`](pyproject.toml) configures **isort** with the **black** profile (so import order matches the formatter) and **mypy** defaults (`follow_imports = silent`, `ignore_missing_imports` for third-party wheels without full stubs).

### Format (Black + isort)

Check formatting and import order without writing files:

```bash
black --check run.py utils.py tests/ ml/
isort --check run.py utils.py tests/ ml/
```

Apply fixes (writes files in place):

```bash
black run.py utils.py tests/ ml/
isort run.py utils.py tests/ ml/
```

### Lint

```bash
pylint run.py utils.py ml/*.py tests/*.py
```

### Type check (mypy)

```bash
mypy run.py utils.py tests/
```

`ml/` is omitted from the default invocation so mypy does not treat the same file as both `ml.*` and a top-level module. Third-party libraries (e.g. sklearn) rely on `ignore_missing_imports` in [`pyproject.toml`](pyproject.toml) until you add explicit stubs or tighten the config.

### Run tests

```bash
pytest -q
```

**CLI tests** in [`tests/test_cli.py`](tests/test_cli.py) drive `run.py` as a subprocess. Covered scenarios: all flags happy path, YAML fallback, legacy `resolution_witdh` typo acceptance, missing-param errors, unknown model errors, and stdout-cleanliness (no stray prints or emoji). If the local model cache under `ml/model/` is absent, the first test run will train models (~35s); subsequent runs reuse the cache.

**Unit tests:** [`tests/test_compute_wh.py`](tests/test_compute_wh.py) for `emission_factor` and `prepare_frames` (no full ML run); [`tests/test_paths.py`](tests/test_paths.py) checks [`ml/paths.py`](ml/paths.py) resolution against the repo layout.

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
| Batch (15 models, warm cache) | A few seconds (inference only; no retrain) |

## Limitations & Assumptions

### Data Limitations
- **Training data size**: Limited benchmark measurements (~20-150 samples per architecture)
- **Model coverage**: 15 names in `MODEL_CONFIGS`; behavior may not generalize to architectures or sizes outside the training data
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
error: unknown model 'Nonsense'
```
→ Check `--model` (or the `model` key in your YAML) matches a supported name exactly (case-sensitive).

**2. Missing data files:**

Errors when reading `prepared_data.csv` or `carbone_kwh_country.csv` mean the `ml/data/` files are missing or unreadable. Paths are resolved from the **repository root** (see [`ml/paths.py`](ml/paths.py)), not the shell’s current working directory, but the `ml/data/` tree must be present in the project.

**3. Invalid parameters:**
```
{"error": "Invalid input: steps=-1, ..."}
```
→ All numeric parameters must be > 0

**4. Pandas `SettingWithCopyWarning`:**
- **Fixed**: Code now uses `.copy()` and `.loc[]` for DataFrame operations

**5. StandardScaler feature name warnings:**
- **Fixed**: Predictions now use pandas DataFrame with proper column names

### Debug / training diagnostics

Fitting each candidate regressor is wrapped in `try`/`except` in [`ml/tabular_base.py`](ml/tabular_base.py). Failed fits are logged at **DEBUG** (model name and exception). The stock [`run.py`](run.py) does not configure `logging`, so you will not see these lines until you set the log level, for example in a **local** scratch script or REPL *before* importing `ml` modules:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
# then import and invoke training or run.py programmatically
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
| **Scalability** | 15 named models in `MODEL_CONFIGS` across 3 architectures (DiT, U-Net, hybrid) |

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

**Last Updated:** April 2026


