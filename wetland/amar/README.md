# Water Treatment Prediction System 💧

A Streamlit application that predicts outlet water quality parameters based on inlet conditions and weather data using machine learning.

## Overview

This application analyzes water treatment data and uses Random Forest regression models to predict outlet water quality parameters. The system takes into account:
- **Inlet Parameters**: BOD, COD, TDS, EC, NH4, NO3, DO, pH
- **Weather Data**: Temperature (avg, min, max), precipitation, wind speed, pressure
- **Outlet Predictions**: All corresponding outlet parameters

## Features

1. **Data Analysis**: 
   - Dataset overview and statistics
   - Correlation analysis with heatmaps
   - Time series visualization of inlet vs outlet parameters

2. **Model Training**: 
   - Random Forest models for each outlet parameter
   - Performance metrics (R², RMSE, MSE)
   - Feature importance analysis

3. **Prediction Interface**: 
   - User-friendly input forms
   - Real-time predictions
   - Treatment efficiency calculations

4. **Model Performance**: 
   - Actual vs predicted visualizations
   - Residual analysis
   - Model validation metrics

## Data Files Required

Make sure the following CSV files are in your Downloads folder:
- `inlet values.csv`
- `outlet values.csv` 
- `to train data (1).csv`
- `weather values.csv`

## Installation

1. Install required packages:
```bash
pip install streamlit pandas numpy scikit-learn plotly
```

## Running the Application

1. Navigate to the application directory:
```bash
cd /Users/pavan
```

2. Run the Streamlit app:
```bash
streamlit run water_treatment_predictor.py
```

3. Open your web browser and go to the URL displayed in the terminal (usually `http://localhost:8501`)

## How to Use

### Step 1: Data Analysis
- Start by exploring the "Data Analysis" page to understand your dataset
- Review correlations and time series trends

### Step 2: Model Training
- Go to "Model Training" page and wait for models to train
- Review performance metrics and feature importance

### Step 3: Make Predictions
- Navigate to "Prediction" page
- Input your inlet parameters and weather data
- Click "Predict Outlet Values" to get results
- Review treatment efficiency calculations

### Step 4: Evaluate Performance
- Check "Model Performance" page for detailed analysis
- Review actual vs predicted plots and residual analysis

## Model Information

- **Algorithm**: Random Forest Regressor
- **Features**: 14 input features (8 inlet + 6 weather parameters)
- **Targets**: 8 outlet parameters
- **Training**: Individual models for each outlet parameter
- **Preprocessing**: StandardScaler normalization

## Input Parameters

### Inlet Parameters
- BOD (mg/l): Biochemical Oxygen Demand
- COD (mg/l): Chemical Oxygen Demand  
- TDS (mg/l): Total Dissolved Solids
- EC (mS/cm): Electrical Conductivity
- NH4 (mg/l): Ammonia
- NO3 (mg/l): Nitrate
- DO: Dissolved Oxygen
- pH: Acidity/Alkalinity level

### Weather Parameters
- Average Temperature (°C)
- Minimum Temperature (°C)
- Maximum Temperature (°C)
- Precipitation (mm)
- Wind Speed
- Pressure

## Output

The application provides:
- Predicted outlet values for all parameters
- Treatment efficiency percentages
- Interactive visualizations
- Model performance metrics

## Technical Details

- **Framework**: Streamlit
- **ML Library**: Scikit-learn
- **Visualization**: Plotly
- **Data Processing**: Pandas, Numpy
- **Model Type**: Ensemble (Random Forest)
- **Validation**: Train-test split with performance metrics

## Support

For any issues or questions, please ensure:
1. All CSV files are in the correct location
2. All dependencies are installed
3. Python version is compatible (3.7+)

---
Built with ❤️ using Streamlit and Machine Learning

# 💧 Wastewater Quality Prediction System

A comprehensive machine learning web application for predicting outlet water quality parameters based on inlet conditions and weather data.

## 📋 Features

### Core Functionality
- **Real-time Prediction**: Predict outlet water quality parameters using inlet conditions and weather data
- **Interactive UI**: Clean, modern Streamlit interface with sidebar input controls
- **Multi-parameter Prediction**: Simultaneously predicts 8 outlet parameters:
  - BOD (mg/l)
  - COD (mg/l) 
  - TDS (mg/l)
  - EC (mS/cm)
  - NH4 (mg/l)
  - NO3 (mg/l)
  - DO
  - pH

### Advanced Features
- **Feature Importance Analysis**: Visual representation of which parameters most influence predictions
- **Treatment Efficiency Calculation**: Shows removal/improvement percentages for each parameter
- **Batch Prediction**: Upload Excel files for processing multiple samples at once
- **Model Performance Metrics**: Displays correlation matrices and model statistics
- **Template Download**: Provides properly formatted templates for batch uploads

## 🚀 Quick Start

### Installation

1. **Clone or download the project files**
2. **Create a virtual environment:**
   ```bash
   python3 -m venv wastewater_env
   source wastewater_env/bin/activate
   ```

3. **Install required packages:**
   ```bash
   pip install pandas openpyxl streamlit scikit-learn xgboost joblib matplotlib seaborn plotly
   ```

### Running the Application

1. **Train the model (first time only):**
   ```bash
   python train_model.py
   ```

2. **Launch the Streamlit app:**
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Open your browser** and navigate to the displayed URL (typically `http://localhost:8501`)

## 📊 Usage Instructions

### Single Prediction
1. **Enter Parameters**: Use the sidebar to input water quality and weather parameters
2. **Click Predict**: Press the "🔮 Predict Outlet Quality" button
3. **View Results**: Check the results table and visualizations

### Batch Prediction
1. **Go to Batch Prediction tab**
2. **Download template** to see the required format
3. **Upload your Excel file** with multiple samples
4. **Process and download results** as CSV

## 🧠 Model Details

- **Algorithm**: Random Forest Regressor with MultiOutput wrapper
- **Features**: 9 input parameters (8 water quality + 1 weather)
- **Targets**: 8 outlet water quality parameters
- **Performance**: Uses standardized features and cross-validation

## 📁 File Structure

```
wastewater_prediction/
├── streamlit_app.py           # Main Streamlit application
├── train_model.py            # Model training and evaluation
├── data_preprocessing.py     # Data loading and preprocessing
├── wastewater_model.pkl     # Trained model (generated)
└── README.md                # This documentation
```

---

**Built with ❤️ using Streamlit, scikit-learn, and Python**
