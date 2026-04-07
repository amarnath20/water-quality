import streamlit as st
try:
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except Exception:
    pd = None
    np = None
    SKLEARN_AVAILABLE = False

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
import os
import webbrowser
import threading
import time


@st.cache_data
def load_and_prepare_data(uploaded_file=None):
    """Load and prepare the training data.

    If `uploaded_file` is provided (Streamlit uploaded file), use it. Otherwise try a set of
    common local paths. Returns a cleaned DataFrame or None on failure.
    """
    # If pandas/scikit-learn aren't available, return a small demo DataFrame
    if not SKLEARN_AVAILABLE or pd is None:
        demo = {
            'Weeks': list(range(1, 13)),
            'Inlet_BOD': [200, 195, 190, 185, 180, 175, 170, 165, 160, 155, 150, 145],
            'Inlet_COD': [450]*12,
            'Inlet_TDS': [1400]*12,
            'Inlet_EC': [2.1]*12,
            'Inlet_NH4': [9]*12,
            'Inlet_NO3': [20]*12,
            'Inlet_DO': [0.5]*12,
            'Inlet_pH': [8.8]*12,
            'Outlet_BOD': [50,48,47,45,44,43,42,41,40,39,38,37],
            'Outlet_COD': [120]*12,
            'Outlet_TDS': [1000]*12,
            'Outlet_EC': [1.5]*12,
            'Outlet_NH4': [3]*12,
            'Outlet_NO3': [12]*12,
            'Outlet_DO': [6]*12,
            'Outlet_pH': [7.2]*12,
            'Weather_tavg': [27]*12,
            'Weather_tmin': [23]*12,
            'Weather_tmax': [33]*12,
            'Weather_prcp': [5]*12,
            'Weather_wspd': [10]*12,
            'Weather_wpgt': [1010]*12,
        }
        try:
            import pandas as _pd
            return _pd.DataFrame(demo)
        except Exception:
            return demo

    # When pandas is available, attempt to load the CSV(s)
    try:
        if uploaded_file is not None:
            train_data = pd.read_csv(uploaded_file, skiprows=2)
        else:
            tried = []
            paths_to_try = [
                "./to_train_data.csv",
                "./data/to_train_data.csv",
                r"C:\Users\user\Downloads\to train data (1).csv",
                '/Users/pavan/Downloads/to train data (1).csv'
            ]

            train_data = None
            for p in paths_to_try:
                try:
                    train_data = pd.read_csv(p, skiprows=2)
                    break
                except Exception:
                    tried.append(p)

            if train_data is None:
                raise FileNotFoundError(f"Could not find training CSV. Tried: {tried}")

        # Clean column names and handle the structure
        columns = [
            'Weeks',
            'Inlet_BOD', 'Inlet_COD', 'Inlet_TDS', 'Inlet_EC', 'Inlet_NH4', 'Inlet_NO3', 'Inlet_DO', 'Inlet_pH',
            'sep1',
            'Outlet_BOD', 'Outlet_COD', 'Outlet_TDS', 'Outlet_EC', 'Outlet_NH4', 'Outlet_NO3', 'Outlet_DO', 'Outlet_pH',
            'sep2',
            'Weather_tavg', 'Weather_tmin', 'Weather_tmax', 'Weather_prcp', 'Weather_wspd', 'Weather_wpgt'
        ]

        train_data.columns = columns

        # Drop separator columns
        train_data = train_data.drop(['sep1', 'sep2'], axis=1)

        # Convert to numeric, handling any non-numeric values
        for col in train_data.columns:
            if col != 'Weeks':
                train_data[col] = pd.to_numeric(train_data[col], errors='coerce')

        # Remove any rows with NaN values
        train_data = train_data.dropna()

        return train_data

    except Exception as e:
        # When running in Streamlit show the error; otherwise raise so callers can handle it
        try:
            st.error(f"Error loading data: {str(e)}")
            return None
        except Exception:
            # If Streamlit UI isn't available, re-raise
            raise

@st.cache_data
def train_models(data):
    """Train machine learning models for each outlet parameter"""
    
    # Features: Inlet parameters + Weather parameters
    inlet_features = ['Inlet_BOD', 'Inlet_COD', 'Inlet_TDS', 'Inlet_EC', 'Inlet_NH4', 'Inlet_NO3', 'Inlet_DO', 'Inlet_pH']
    weather_features = ['Weather_tavg', 'Weather_tmin', 'Weather_tmax', 'Weather_prcp', 'Weather_wspd', 'Weather_wpgt']
    feature_columns = inlet_features + weather_features
    
    # Target variables: Outlet parameters
    target_columns = ['Outlet_BOD', 'Outlet_COD', 'Outlet_TDS', 'Outlet_EC', 'Outlet_NH4', 'Outlet_NO3', 'Outlet_DO', 'Outlet_pH']
    
    X = data[feature_columns]
    
    models = {}
    scalers = {}
    performance_metrics = {}
    
    # Train a separate model for each outlet parameter
    for target in target_columns:
        y = data[target]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        models[target] = model
        scalers[target] = scaler
        performance_metrics[target] = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    return models, scalers, performance_metrics, feature_columns

def predict_outlet_values(models, scalers, feature_columns, input_data):
    """Make predictions for all outlet parameters"""
    predictions = {}
    
    # Create input dataframe
    input_df = pd.DataFrame([input_data], columns=feature_columns)
    
    for target, model in models.items():
        scaler = scalers[target]
        
        # Scale input data
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        predictions[target] = max(0, prediction)  # Ensure non-negative values
    
    return predictions

# File uploader in sidebar (so users can provide their own CSV)
st.sidebar.subheader("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload training CSV (skip first 2 rows)", type=["csv"]) 

# Load data (uploaded file takes precedence)
data = load_and_prepare_data(uploaded_file)

# Optional direct redirect: enter a full URL and the app will navigate there immediately.
st.sidebar.markdown("---")
redirect_input = st.sidebar.text_input("Go to external URL (include http:// or https://)", value="")
if redirect_input:
    # Minimal validation
    if redirect_input.startswith("http://") or redirect_input.startswith("https://"):
        # Inject JS to redirect the current page
        js = f"<script>window.location.href = '{redirect_input}';</script>"
        st.components.v1.html(js)
    else:
        st.sidebar.error("Please include http:// or https:// in the URL")

# Add page selection radio button in sidebar
page = st.sidebar.radio(
    "Select Page",
    ["Data Analysis", "Model Training", "Prediction", "Model Performance"]
)

if data is not None:
    
    if page == "Data Analysis":
        st.header("📊 Data Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Dataset Overview")
            st.write(f"**Number of samples:** {len(data)}")
            st.write(f"**Number of features:** {data.shape[1] - 1}")  # Excluding 'Weeks'
            st.dataframe(data.head(10))

        with col2:
            st.subheader("Data Statistics")
            st.dataframe(data.describe())

        # Correlation heatmap
        st.subheader("Feature Correlations")
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        corr_matrix = data[numeric_cols].corr()

        fig = px.imshow(corr_matrix, 
                       title="Correlation Matrix",
                       aspect="auto",
                       color_continuous_scale="RdBu_r")
        st.plotly_chart(fig, width='stretch')

        # Time series plots
        st.subheader("Time Series Analysis")

        # Inlet vs Outlet comparison
        fig = make_subplots(rows=2, cols=4, 
                          subplot_titles=('BOD', 'COD', 'TDS', 'EC', 'NH4', 'NO3', 'DO', 'pH'))

        parameters = ['BOD', 'COD', 'TDS', 'EC', 'NH4', 'NO3', 'DO', 'pH']

        for i, param in enumerate(parameters):
            row = (i // 4) + 1
            col = (i % 4) + 1

            fig.add_trace(
                go.Scatter(x=data['Weeks'], y=data[f'Inlet_{param}'], 
                          name=f'Inlet {param}', line=dict(color='red')),
                row=row, col=col
            )
            fig.add_trace(
                go.Scatter(x=data['Weeks'], y=data[f'Outlet_{param}'], 
                          name=f'Outlet {param}', line=dict(color='blue')),
                row=row, col=col
            )

        fig.update_layout(height=600, title_text="Inlet vs Outlet Parameters Over Time")
        st.plotly_chart(fig, width='stretch')

    elif page == "Model Training":
        st.header("🤖 Model Training")

        with st.spinner("Training models..."):
            models, scalers, performance_metrics, feature_columns = train_models(data)

        st.success("Models trained successfully!")

        # Display performance metrics
        st.subheader("Model Performance")

        metrics_df = pd.DataFrame({
            'Parameter': list(performance_metrics.keys()),
            'R² Score': [metrics['r2'] for metrics in performance_metrics.values()],
            'RMSE': [metrics['rmse'] for metrics in performance_metrics.values()],
            'MSE': [metrics['mse'] for metrics in performance_metrics.values()]
        })

        st.dataframe(metrics_df)

        # Feature importance
        st.subheader("Feature Importance")

        # Get feature importance from one of the models (BOD as example)
        sample_model = models['Outlet_BOD']
        importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': sample_model.feature_importances_
        }).sort_values('Importance', ascending=False)

        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                     title='Feature Importance (Random Forest - BOD Model)')
        st.plotly_chart(fig, width='stretch')

        # Store models in session state
        st.session_state['models'] = models
        st.session_state['scalers'] = scalers
        st.session_state['feature_columns'] = feature_columns
        st.session_state['performance_metrics'] = performance_metrics

    elif page == "Prediction":
        st.header("🔮 Make Predictions")

        if 'models' not in st.session_state:
            st.warning("Please train the models first by visiting the 'Model Training' page.")
        else:
            st.subheader("Input Parameters")

            col1, col2, col3 = st.columns(3)

            # Inlet parameters
            with col1:
                st.write("**Inlet Parameters**")
                inlet_bod = st.number_input("BOD (mg/l)", value=200.0, min_value=0.0)
                inlet_cod = st.number_input("COD (mg/l)", value=450.0, min_value=0.0)
                inlet_tds = st.number_input("TDS (mg/l)", value=1400.0, min_value=0.0)
                inlet_ec = st.number_input("EC (mS/cm)", value=2.1, min_value=0.0)

            with col2:
                st.write("**More Inlet Parameters**")
                inlet_nh4 = st.number_input("NH4 (mg/l)", value=9.0, min_value=0.0)
                inlet_no3 = st.number_input("NO3 (mg/l)", value=20.0, min_value=0.0)
                inlet_do = st.number_input("DO", value=0.5, min_value=0.0)
                inlet_ph = st.number_input("pH", value=8.8, min_value=0.0, max_value=14.0)

            # Weather parameters
            with col3:
                st.write("**Weather Parameters**")
                weather_tavg = st.number_input("Average Temperature (°C)", value=27.0)
                weather_tmin = st.number_input("Min Temperature (°C)", value=23.0)
                weather_tmax = st.number_input("Max Temperature (°C)", value=33.0)
                weather_prcp = st.number_input("Precipitation (mm)", value=5.0, min_value=0.0)
                weather_wspd = st.number_input("Wind Speed", value=10.0, min_value=0.0)
                weather_wpgt = st.number_input("Pressure", value=1010.0, min_value=0.0)

            if st.button("Predict Outlet Values", type="primary"):
                # Prepare input data
                input_data = [
                    inlet_bod, inlet_cod, inlet_tds, inlet_ec, inlet_nh4, inlet_no3, inlet_do, inlet_ph,
                    weather_tavg, weather_tmin, weather_tmax, weather_prcp, weather_wspd, weather_wpgt
                ]

                # Make predictions
                predictions = predict_outlet_values(
                    st.session_state['models'], 
                    st.session_state['scalers'], 
                    st.session_state['feature_columns'], 
                    input_data
                )

                st.success("Predictions completed!")

                # Display predictions
                st.subheader("Predicted Outlet Values")

                pred_df = pd.DataFrame({
                    'Parameter': ['BOD (mg/l)', 'COD (mg/l)', 'TDS (mg/l)', 'EC (mS/cm)', 
                                'NH4 (mg/l)', 'NO3 (mg/l)', 'DO', 'pH'],
                    'Predicted Value': [
                        predictions['Outlet_BOD'], predictions['Outlet_COD'], 
                        predictions['Outlet_TDS'], predictions['Outlet_EC'],
                        predictions['Outlet_NH4'], predictions['Outlet_NO3'], 
                        predictions['Outlet_DO'], predictions['Outlet_pH']
                    ]
                })

                st.dataframe(pred_df, width='stretch')

                # Visualization
                fig = px.bar(pred_df, x='Parameter', y='Predicted Value',
                           title='Predicted Outlet Water Quality Parameters')
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, width='stretch')

                # Treatment efficiency
                st.subheader("Treatment Efficiency")
                efficiency_data = {
                    'Parameter': ['BOD', 'COD', 'TDS', 'NH4', 'NO3'],
                    'Inlet': [inlet_bod, inlet_cod, inlet_tds, inlet_nh4, inlet_no3],
                    'Outlet': [predictions['Outlet_BOD'], predictions['Outlet_COD'], 
                             predictions['Outlet_TDS'], predictions['Outlet_NH4'], predictions['Outlet_NO3']],
                }

                efficiency_df = pd.DataFrame(efficiency_data)
                efficiency_df['Removal Efficiency (%)'] = (
                    (efficiency_df['Inlet'] - efficiency_df['Outlet']) / efficiency_df['Inlet'] * 100
                )

                st.dataframe(efficiency_df, width='stretch')

    elif page == "Model Performance":
        st.header("📈 Model Performance Analysis")

        if 'performance_metrics' not in st.session_state:
            st.warning("Please train the models first by visiting the 'Model Training' page.")
        else:
            performance_metrics = st.session_state['performance_metrics']

            # Performance summary
            st.subheader("Model Performance Summary")

            metrics_df = pd.DataFrame({
                'Parameter': list(performance_metrics.keys()),
                'R² Score': [f"{metrics['r2']:.3f}" for metrics in performance_metrics.values()],
                'RMSE': [f"{metrics['rmse']:.3f}" for metrics in performance_metrics.values()]
            })

            st.dataframe(metrics_df, width='stretch')

            # Actual vs Predicted plots
            st.subheader("Actual vs Predicted Values")

            # Create subplot for all parameters
            fig = make_subplots(rows=2, cols=4, 
                              subplot_titles=[param.replace('Outlet_', '') for param in performance_metrics.keys()])

            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

            for i, (param, metrics) in enumerate(performance_metrics.items()):
                row = (i // 4) + 1
                col = (i % 4) + 1

                # Scatter plot of actual vs predicted
                fig.add_trace(
                    go.Scatter(x=metrics['y_test'], y=metrics['y_pred'], 
                             mode='markers', name=param.replace('Outlet_', ''),
                             marker=dict(color=colors[i])),
                    row=row, col=col
                )

                # Add diagonal line (perfect prediction line)
                min_val = min(min(metrics['y_test']), min(metrics['y_pred']))
                max_val = max(max(metrics['y_test']), max(metrics['y_pred']))
                fig.add_trace(
                    go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                             mode='lines', line=dict(dash='dash', color='black'),
                             showlegend=False),
                    row=row, col=col
                )

            fig.update_layout(height=600, title_text="Actual vs Predicted Values for All Parameters")
            fig.update_xaxes(title_text="Actual")
            fig.update_yaxes(title_text="Predicted")
            st.plotly_chart(fig, width='stretch')

            # Residuals plot
            st.subheader("Residual Analysis")

            selected_param = st.selectbox("Select parameter for residual analysis:", 
                                        list(performance_metrics.keys()))

            metrics = performance_metrics[selected_param]
            residuals = metrics['y_test'] - metrics['y_pred']

            col1, col2 = st.columns(2)

            with col1:
                # Residuals vs Predicted
                fig = px.scatter(x=metrics['y_pred'], y=residuals,
                               title=f'Residuals vs Predicted ({selected_param})',
                               labels={'x': 'Predicted Values', 'y': 'Residuals'})
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig, width='stretch')

            with col2:
                # Residuals histogram
                fig = px.histogram(x=residuals, title=f'Residuals Distribution ({selected_param})',
                                 labels={'x': 'Residuals'})
                st.plotly_chart(fig, width='stretch')

else:
    st.error("Could not load the training data. Please check if the files exist in the Downloads folder.")

# Footer
st.markdown("---")
st.markdown("**Water Treatment Prediction System** - Built with Streamlit and Machine Learning")

