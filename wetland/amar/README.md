# Wetland Water Quality Prediction

This repository contains two Streamlit applications for forecasting wetland treatment outcomes from inlet and weather data.

- `bod_predictor.py`: Seasonal Outlet BOD forecasting using Gradient Boosting Regression.
- `water_treatment_predictor.py`: Multi-parameter outlet prediction using per-target Random Forest models.

## Abstract-Style Project Overview

Constructed wetlands are cost-effective and energy-efficient wastewater treatment systems, but their behavior changes significantly across seasons. In monsoon-influenced conditions, variation in rainfall, humidity, and hydraulic loading introduces nonlinear dynamics in pollutant removal, especially in BOD.

The BOD module in this project uses a data-driven workflow to improve forecast reliability:

- Gradient Boosting Regression as the core model.
- Seasonal feature engineering from week index using cyclical encoding.
- Data preprocessing with missing-value cleanup, IQR-based outlier clipping, and feature scaling.
- K-fold cross-validation for robustness and generalization assessment.

This design supports reliable near real-time prediction and improves readiness for proactive wetland operation decisions.

## Key Features

- Interactive Streamlit dashboards.
- Upload-your-own CSV training support.
- Model diagnostics: R2, RMSE, feature importance, and seasonal R2 view.
- Prediction-oriented UI for fast what-if analysis.

## Applications

### 1) BOD Predictor

File: `bod_predictor.py`

- Model: `GradientBoostingRegressor`
- Inputs: `Weeks`, `Inlet_BOD`, `Weather_tavg`, `Weather_prcp`, `Weather_wspd`, `Weather_rhum`
- Engineered features: `Week_Sin`, `Week_Cos`
- Preprocessing: type cleanup, NA removal, IQR clipping, scaling (`StandardScaler`)
- Validation: K-fold cross-validation
- Outputs:
	- Predicted `Outlet_BOD`
	- Treatment efficiency
	- Status badge (Safe / Warning / Critical)
	- Feature-importance plot
	- Seasonal performance diagnostics

### 2) Wastewater Quality Predictor

File: `water_treatment_predictor.py`

- Model: Separate `RandomForestRegressor` per outlet parameter
- Scope: Predicts multiple outlet water-quality targets from inlet + weather features
- Views: Data Analysis, Model Training, Prediction, and Model Performance

## Repository Contents

- `bod_predictor.py`
- `water_treatment_predictor.py`
- `requirements.txt`
- `BOD.csv`, `BOD_50weeks.csv`, `water_data.csv`
- Additional working data files in repository root as available

## Environment Setup

1. Create and activate a virtual environment.

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

## Run

From `wetland/amar`:

```bash
streamlit run bod_predictor.py
```

or

```bash
streamlit run water_treatment_predictor.py
```

## Methodology Notes (BOD Module)

- Target variable: `Outlet_BOD`
- Core predictors: inlet BOD + meteorology + seasonal progression
- Seasonal encoding:
	- `Week_Sin = sin(2 * pi * week / 52)`
	- `Week_Cos = cos(2 * pi * week / 52)`
- Robustness strategy: K-fold cross-validation with fold-wise R2 aggregation
- Interpretation strategy: model-based feature importance and seasonal segment diagnostics

## Requirements

Dependencies are pinned in `requirements.txt` and include:

- `streamlit`
- `pandas`
- `numpy`
- `scikit-learn`
- `plotly`
- `openpyxl`
- Additional analysis stack: `xgboost`, `joblib`, `matplotlib`, `seaborn`

## Troubleshooting

- If Streamlit fails to launch, confirm the virtual environment is active.
- If CSV load fails, verify the expected column structure and delimiter.
- If plots do not render, ensure `plotly` is installed in the active environment.

## License

No license file is currently included. Add one before public release.
