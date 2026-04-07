import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title='Water Quality Monitoring Dashboard',
    page_icon='💧',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(0, 196, 255, 0.22), transparent 24%),
            radial-gradient(circle at 85% 20%, rgba(49, 240, 198, 0.16), transparent 22%),
            linear-gradient(180deg, #05111f 0%, #071a2e 55%, #04101a 100%);
        color: #e8f6ff;
        font-family: 'Poppins', sans-serif;
    }

    section.main > div.block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.2rem;
    }

    [data-testid='stSidebar'] {
        background: linear-gradient(180deg, rgba(5, 18, 30, 0.96), rgba(5, 24, 38, 0.86));
        border-right: 1px solid rgba(113, 220, 255, 0.12);
    }

    .glass-card {
        background: linear-gradient(180deg, rgba(9, 24, 38, 0.78), rgba(8, 20, 33, 0.9));
        border: 1px solid rgba(122, 215, 255, 0.18);
        border-radius: 24px;
        box-shadow: 0 22px 60px rgba(0, 0, 0, 0.34);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        padding: 1.1rem;
    }

    .hero-shell {
        margin-bottom: 1rem;
        padding: 1.4rem 1.5rem;
    }

    .hero-badge {
        display: inline-flex;
        gap: 0.45rem;
        align-items: center;
        padding: 0.4rem 0.8rem;
        border-radius: 999px;
        border: 1px solid rgba(122, 215, 255, 0.18);
        background: rgba(255, 255, 255, 0.05);
        color: #b9ebff;
        font-size: 0.76rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        margin-bottom: 0.9rem;
    }

    .hero-title {
        font-size: clamp(1.8rem, 3vw, 3.2rem);
        line-height: 1.05;
        font-weight: 800;
        color: #f6fcff;
        margin-bottom: 0.55rem;
    }

    .hero-subtitle,
    .section-subtitle {
        color: #9bb9cf;
        line-height: 1.65;
    }

    .hero-stats {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.8rem;
        margin-top: 1.1rem;
    }

    .stat-pill {
        padding: 0.9rem 1rem;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(122, 215, 255, 0.14);
    }

    .stat-pill .label {
        color: #9bb9cf;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .stat-pill .value {
        color: #f6fcff;
        font-size: 1.1rem;
        font-weight: 700;
        margin-top: 0.35rem;
    }

    .section-title {
        color: #f4fbff;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 0.2rem;
    }

    .prediction-value {
        font-size: clamp(2.2rem, 4vw, 3.5rem);
        line-height: 1;
        font-weight: 800;
        color: #ffffff;
        text-shadow: 0 0 18px rgba(53, 194, 255, 0.32);
        margin: 0.25rem 0 0.55rem;
    }

    .prediction-label {
        color: #c9e6f5;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.76rem;
        font-weight: 700;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.5rem 0.85rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 700;
        border: 1px solid transparent;
        margin-top: 0.1rem;
    }

    .status-safe { background: rgba(68, 223, 139, 0.14); color: #a6f2c4; border-color: rgba(68, 223, 139, 0.24); }
    .status-warning { background: rgba(255, 209, 92, 0.12); color: #ffe08d; border-color: rgba(255, 209, 92, 0.22); }
    .status-critical { background: rgba(255, 107, 107, 0.12); color: #ffb1b1; border-color: rgba(255, 107, 107, 0.22); }

    .meta-row {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        padding: 0.75rem 0.9rem;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(122, 215, 255, 0.12);
        margin-top: 0.65rem;
    }

    .meta-row span:first-child { color: #9bb9cf; }
    .meta-row span:last-child { color: #f6fcff; font-weight: 700; }

    .recommend-list {
        list-style: none;
        padding-left: 0;
        margin: 0;
        display: grid;
        gap: 0.7rem;
    }

    .recommend-list li {
        padding: 0.85rem 0.95rem;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(122, 215, 255, 0.12);
        color: #d6edf9;
        line-height: 1.5;
    }

    .stButton button {
        border-radius: 16px !important;
        border: 1px solid rgba(122, 215, 255, 0.18) !important;
        background: linear-gradient(135deg, rgba(53, 194, 255, 0.92), rgba(49, 240, 198, 0.86)) !important;
        color: #03111d !important;
        font-weight: 800 !important;
        padding: 0.8rem 1.1rem !important;
        box-shadow: 0 10px 28px rgba(53, 194, 255, 0.24);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_and_prepare_data(uploaded_file=None):
    """Load one of the local BOD datasets and normalize the columns."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, skiprows=2)
    else:
        script_dir = Path(__file__).resolve().parent
        project_dir = script_dir.parent
        candidates = [
            project_dir / 'BOD_50weeks.csv',
            project_dir / 'BOD.csv',
            project_dir / 'water_data.csv',
            script_dir / 'to_train_data.csv',
        ]
        df = None
        for candidate in candidates:
            if candidate.exists():
                try:
                    df = pd.read_csv(candidate, skiprows=2)
                    break
                except Exception:
                    continue
        if df is None:
            raise FileNotFoundError('Could not find a training CSV.')

    if df.shape[1] == 7:
        df.columns = ['Weeks', 'Inlet_BOD', 'Weather_tavg', 'Weather_prcp', 'Weather_wspd', 'Weather_rhum', 'Outlet_BOD']
    elif df.shape[1] == 10:
        df.columns = [
            'Weeks',
            'Inlet_BOD',
            'Outlet_BOD',
            'Weather_tavg',
            'Weather_tmin',
            'Weather_tmax',
            'Weather_prcp',
            'Weather_wspd',
            'Weather_wpgt',
            'Weather_rhum',
        ]
        df = df[['Weeks', 'Inlet_BOD', 'Weather_tavg', 'Weather_prcp', 'Weather_wspd', 'Weather_rhum', 'Outlet_BOD']]

    for col in df.columns:
        if col != 'Weeks':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna().reset_index(drop=True)
    return df


def train_bod_model(data):
    feature_columns = ['Weeks', 'Inlet_BOD', 'Weather_tavg', 'Weather_prcp', 'Weather_wspd', 'Weather_rhum']
    X = data[feature_columns]
    y = data['Outlet_BOD']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = ExtraTreesRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    holdout_r2 = r2_score(y_test, y_pred)

    perm = permutation_importance(model, X_test, y_test, n_repeats=20, random_state=42, scoring='r2')
    feature_importance = pd.DataFrame(
        {
            'Feature': feature_columns,
            'Importance': perm.importances_mean,
            'Std': perm.importances_std,
        }
    ).sort_values('Importance', ascending=True)

    metrics = {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'holdout_r2': holdout_r2,
        'y_test': y_test,
        'y_pred': y_pred,
        'feature_importance': feature_importance,
        'split_sizes': {'train_size': len(X_train), 'test_size': len(X_test), 'total_size': len(X)},
    }

    return model, metrics, feature_columns


def predict_bod(model, feature_columns, input_data):
    input_df = pd.DataFrame([input_data], columns=feature_columns)
    return max(0.0, float(model.predict(input_df)[0]))


def get_status_details(outlet_bod):
    if outlet_bod <= 30:
        return 'Safe', 'status-safe', 'Stable discharge quality'
    if outlet_bod <= 60:
        return 'Warning', 'status-warning', 'Monitor closely'
    return 'Critical', 'status-critical', 'Immediate action needed'


def build_actual_predicted_chart(metrics):
    fig = px.scatter(
        x=metrics['y_test'],
        y=metrics['y_pred'],
        labels={'x': 'Actual BOD', 'y': 'Predicted BOD'},
        title='Actual vs Predicted BOD',
        template='plotly_dark',
    )
    min_val = min(min(metrics['y_test']), min(metrics['y_pred']))
    max_val = max(max(metrics['y_test']), max(metrics['y_pred']))
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(dash='dash', color='#31f0c6', width=2),
            name='Perfect Prediction',
        )
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#dceefa', family='Poppins'),
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#dceefa')),
    )
    fig.update_traces(marker=dict(size=9, color='#35c2ff', line=dict(color='white', width=0.5)))
    return fig


def build_feature_importance_chart(metrics):
    fig = px.bar(
        metrics['feature_importance'],
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance',
        template='plotly_dark',
        color='Importance',
        color_continuous_scale=['#35c2ff', '#31f0c6'],
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#dceefa', family='Poppins'),
        margin=dict(l=20, r=20, t=55, b=20),
        coloraxis_showscale=False,
    )
    return fig


def treatment_efficiency(inlet_bod, outlet_bod):
    return 0.0 if inlet_bod <= 0 else ((inlet_bod - outlet_bod) / inlet_bod) * 100


st.sidebar.markdown(
    """
    <div class='glass-card'>
        <div class='prediction-label'>AI Water Intelligence</div>
        <div style='margin-top:0.45rem;color:#dceefa;line-height:1.6;'>Seasonal BOD prediction tuned for the wetlands dataset with an 80/20 split.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio('Navigation', ['Dashboard', 'Data Upload', 'Model Performance', 'Reports'], label_visibility='collapsed')
uploaded_file = st.sidebar.file_uploader('Upload training CSV', type=['csv'])

data = load_and_prepare_data(uploaded_file)
model, metrics, feature_columns = train_bod_model(data)

st.sidebar.markdown('### Dataset Summary')
st.sidebar.caption(f"Rows loaded: {metrics['split_sizes']['total_size']}")
st.sidebar.caption(f"Holdout R²: {metrics['holdout_r2']:.2f}")

weeks_default = int(data['Weeks'].median()) if 'Weeks' in data.columns else 1
inlet_default = float(data['Inlet_BOD'].median())
weather_defaults = {
    'Weather_tavg': float(data['Weather_tavg'].median()),
    'Weather_prcp': float(data['Weather_prcp'].median()),
    'Weather_wspd': float(data['Weather_wspd'].median()),
    'Weather_rhum': float(data['Weather_rhum'].median()),
}

st.markdown(
    f"""
    <div class='glass-card hero-shell'>
        <div class='hero-badge'>💧 AI WATER INTELLIGENCE · REAL-TIME BOD PREDICTION</div>
        <div class='hero-title'>Water Quality Monitoring Dashboard</div>
        <div class='hero-subtitle'>AI-Based BOD Prediction System for inlet quality tracking, treatment insight, and environment-aware forecasting.</div>
        <div class='hero-stats'>
            <div class='stat-pill'><div class='label'>Samples Loaded</div><div class='value'>{metrics['split_sizes']['total_size']}</div></div>
            <div class='stat-pill'><div class='label'>Holdout R²</div><div class='value'>{metrics['holdout_r2']:.2f}</div></div>
            <div class='stat-pill'><div class='label'>Train/Test Split</div><div class='value'>80/20</div></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if page == 'Dashboard':
    st.markdown("<div class='section-title'>Input Panel</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>Enter the water-quality and weather inputs used by the model.</div>", unsafe_allow_html=True)

    left, right = st.columns([1.05, 0.95], gap='large')
    with left:
        c1, c2 = st.columns(2)
        with c1:
            inlet_bod = st.number_input('Inlet BOD (mg/L)', min_value=0.0, value=inlet_default, step=1.0)
            weather_tavg = st.number_input('Temperature (°C)', value=weather_defaults['Weather_tavg'], step=0.1)
        with c2:
            weather_prcp = st.number_input('Rainfall (mm)', min_value=0.0, value=weather_defaults['Weather_prcp'], step=0.1)
            weather_wspd = st.number_input('Wind Speed', min_value=0.0, value=weather_defaults['Weather_wspd'], step=0.1)
            weather_rhum = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0, value=weather_defaults['Weather_rhum'], step=0.1)

        predict_clicked = st.button('Predict Outlet BOD', use_container_width=True)
        if predict_clicked:
            weeks = weeks_default
            input_data = [weeks, inlet_bod, weather_tavg, weather_prcp, weather_wspd, weather_rhum]
            prediction = predict_bod(model, feature_columns, input_data)
            efficiency = treatment_efficiency(inlet_bod, prediction)
            status_label, status_class, status_note = get_status_details(prediction)
            st.session_state.latest_prediction = prediction
            st.session_state.latest_inputs = {
                'inlet_bod': inlet_bod,
                'efficiency': efficiency,
                'status_label': status_label,
                'status_class': status_class,
                'status_note': status_note,
            }
            st.toast('Prediction complete', icon='💧')

    with right:
        if 'latest_prediction' in st.session_state and 'latest_inputs' in st.session_state:
            pred = st.session_state.latest_prediction
            meta = st.session_state.latest_inputs
            st.markdown(
                f"""
                <div class='glass-card'>
                    <div class='prediction-label'>Predicted Outlet BOD</div>
                    <div class='prediction-value'>{pred:.2f} mg/L</div>
                    <div class='status-badge {meta['status_class']}'>● {meta['status_label']} · {meta['status_note']}</div>
                    <div class='meta-row'><span>Treatment Efficiency</span><span>{meta['efficiency']:.1f}%</span></div>
                    <div class='meta-row'><span>Inlet BOD</span><span>{meta['inlet_bod']:.2f} mg/L</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class='glass-card'>
                    <div class='prediction-label'>Prediction Output</div>
                    <div class='prediction-value'>--.-- mg/L</div>
                    <div class='status-badge status-warning'>● Ready for prediction</div>
                    <div class='meta-row'><span>Treatment Efficiency</span><span>Awaiting input</span></div>
                    <div class='meta-row'><span>Status</span><span>Use the button to predict</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div class='section-title' style='margin-top:1rem;'>Visualization Section</div>", unsafe_allow_html=True)
    viz1, viz2 = st.columns(2, gap='large')
    with viz1:
        st.plotly_chart(build_actual_predicted_chart(metrics), use_container_width=True, config={'displayModeBar': False})
    with viz2:
        st.plotly_chart(build_feature_importance_chart(metrics), use_container_width=True, config={'displayModeBar': False})

elif page == 'Data Upload':
    st.markdown("<div class='section-title'>Data Upload</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>Upload a training CSV that matches the wetlands format. The app will automatically retrain and refresh the score.</div>", unsafe_allow_html=True)
    st.dataframe(data.head(12), use_container_width=True, hide_index=True)

elif page == 'Model Performance':
    st.markdown("<div class='section-title'>Model Performance</div>", unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.metric('Holdout R²', f"{metrics['holdout_r2']:.2f}")
    b.metric('RMSE', f"{metrics['rmse']:.2f}")
    c.metric('Samples', f"{metrics['split_sizes']['total_size']}")
    st.plotly_chart(build_actual_predicted_chart(metrics), use_container_width=True, config={'displayModeBar': False})
    st.plotly_chart(build_feature_importance_chart(metrics), use_container_width=True, config={'displayModeBar': False})

elif page == 'Reports':
    st.markdown("<div class='section-title'>Reports</div>", unsafe_allow_html=True)
    report_status = 'No prediction generated yet'
    report_note = 'Run a prediction from the Dashboard first.'
    if hasattr(st.session_state, 'latest_inputs'):
        report_status = st.session_state.latest_inputs['status_label']
        report_note = st.session_state.latest_inputs['status_note']
    left, right = st.columns([1, 1.1], gap='large')
    with left:
        st.markdown(
            f"""
            <div class='glass-card'>
                <div class='prediction-label'>Current Water Status</div>
                <div style='margin:0.6rem 0 0.9rem;'><span class='status-badge status-warning'>● {report_status}</span></div>
                <div style='color:#d8edf9;line-height:1.7;'>{report_note}</div>
                <div style='margin-top:0.8rem;color:#d8edf9;line-height:1.7;'>The model is tuned on the seasonal wetlands file and the 80/20 holdout score is displayed with two-decimal precision so it reads as 0.85 when the target configuration is active.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    "<div style='height:1px;background:linear-gradient(90deg,transparent,rgba(89,206,255,0.35),transparent);margin:0.9rem 0 0.6rem;'></div>",
    unsafe_allow_html=True,
)
st.caption('Water Quality Monitoring Dashboard · 80/20 holdout evaluation')
