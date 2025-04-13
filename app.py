import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import requests
from PIL import Image
from tensorflow.keras.preprocessing import image
import plotly.graph_objects as go
import joblib
from datetime import datetime

# Page configuration
st.set_page_config(page_title="AirSafe Innovators", layout="wide")

# Logo
st.image("logo.png", width=200)

# Custom CSS
st.markdown("""
    <style>
    /* Styles for tabs */
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        justify-content: center;
        background-color: #1f77b4;
        padding: 10px 0;
        border-radius: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 10px 20px;
        margin: 0 5px;
        border-radius: 5px;
        transition: background-color 0.3s;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #ff7f0e;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #ff7f0e;
    }
    /* General styling */
    .stApp {
        background-color: #f0f2f6;
        font-family: 'Arial', sans-serif;
    }
    h1, h2, h3 {
        color: #1f77b4;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #ff7f0e;
    }
    .stTextInput>div>input {
        border-radius: 5px;
        border: 1px solid #1f77b4;
    }
    .stFileUploader>div {
        border-radius: 5px;
        border: 1px solid #1f77b4;
    }
    /* Styles for AQI alerts */
    .aqi-alert-good {
        background-color: #00e400;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        margin: 10px 0;
    }
    .aqi-alert-moderate {
        background-color: #ffff00;
        color: black;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        margin: 10px 0;
    }
    .aqi-alert-unhealthy-sensitive {
        background-color: #ff7e00;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        margin: 10px 0;
    }
    .aqi-alert-unhealthy {
        background-color: #ff0000;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        margin: 10px 0;
    }
    .aqi-alert-very-unhealthy {
        background-color: #8f3f97;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        margin: 10px 0;
    }
    .aqi-alert-hazardous {
        background-color: #7e0023;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize models and scalers
custom_objects = {'mse': tf.keras.losses.MeanSquaredError()}
cnn_lstm_model = None
sky_cnn_model = None
scaler_x = None
scaler_y = None
scalers_available = False

# Load CNN-LSTM model
cnn_lstm_path = 'cnn_lstm_model.h5'
if os.path.exists(cnn_lstm_path):
    try:
        cnn_lstm_model = tf.keras.models.load_model(cnn_lstm_path, custom_objects=custom_objects)
        if os.path.exists('scaler_x.pkl') and os.path.exists('scaler_y.pkl'):
            scaler_x = joblib.load('scaler_x.pkl')
            scaler_y = joblib.load('scaler_y.pkl')
            scalers_available = True
        else:
            st.warning("Scalers not found. CNN-LSTM predictions disabled.")
    except Exception as e:
        st.error(f"Error loading CNN-LSTM model: {str(e)}")
else:
    st.error(f"CNN-LSTM model not found at '{cnn_lstm_path}'.")
    st.warning("City predictions disabled.")

# Load sky image model
sky_model_path = 'air_quality_model.keras'
if os.path.exists(sky_model_path):
    try:
        sky_cnn_model = tf.keras.models.load_model(sky_model_path, custom_objects=custom_objects)
    except Exception as e:
        st.error(f"Error loading sky image model: {str(e)}")
        st.warning("Image analysis disabled.")
else:
    st.error(f"Sky image model not found at '{sky_model_path}'.")
    st.warning("Image analysis disabled.")

# Utility functions
def fetch_air_quality(city):
    """Fetch air quality data for a given city using OpenWeather API."""
    api_key = "59a803008ea27684f963eb7da02fafef"
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={api_key}"
    try:
        geo_response = requests.get(geo_url).json()
        if not geo_response:
            return None, "City not found."
        lat, lon = geo_response[0]['lat'], geo_response[0]['lon']
        air_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
        air_response = requests.get(air_url).json()
        pollutants = air_response['list'][0]['components']
        aqi = air_response['list'][0]['main']['aqi']
        return {
            'pm2_5': pollutants['pm2_5'],
            'pm10': pollutants['pm10'],
            'no2': pollutants['no2'],
            'so2': pollutants['so2'],
            'co': pollutants['co'],
            'o3': pollutants['o3'],
            'aqi': aqi
        }, None
    except Exception as e:
        return None, f"Error fetching data: {str(e)}"

def preprocess_for_cnn_lstm(data, is_api_data=True):
    """Preprocess data for CNN-LSTM model."""
    if not scalers_available:
        return None
    df = pd.DataFrame([data]) if is_api_data else data.copy()
    if is_api_data:
        now = datetime.now()
        df['year'] = now.year
        df['month'] = now.month
        df['day'] = now.day
        df['hour'] = now.hour

    try:
        expected_features = scaler_x.feature_names_in_.tolist()
    except AttributeError:
        return None

    if not is_api_data:
        if len(df) < 30:
            return None

    feature_mapping = {
        'pm2_5': 'PM2.5', 'pm10': 'PM10', 'no2': 'NO2', 'co': 'CO',
        'o3': 'O3', 'so2': 'SO2', 'year': 'year', 'month': 'month',
        'day': 'day', 'hour': 'hour', 'temperature': 'temperature',
        'humidity': 'humidity', 'wind_speed': 'wind_speed', 'pressure': 'pressure'
    }

    X = pd.DataFrame(0.0, index=df.index, columns=expected_features)
    for input_col, scaler_col in feature_mapping.items():
        if input_col in df.columns and scaler_col in expected_features:
            X[scaler_col] = df[input_col]

    X.fillna(0.0, inplace=True)
    st.write(f"Preprocessed X shape: {X.shape}")
    try:
        X_scaled = scaler_x.transform(X)
        if is_api_data:
            X_seq = np.repeat(X_scaled[np.newaxis, :, :], 30, axis=1)
        else:
            Xs = [X_scaled[i:i+30] for i in range(len(X_scaled) - 30)]
            X_seq = np.array(Xs)
            if X_seq.shape[0] == 0:
                return None
        return X_seq
    except Exception as e:
        st.write(f"Error in preprocessing: {str(e)}")
        return None

def convert_aqi_to_api_scale(aqi_0_500):
    """Convert AQI from 0-500 scale to 1-5 scale."""
    if aqi_0_500 <= 50: return 1
    elif aqi_0_500 <= 100: return 2
    elif aqi_0_500 <= 150: return 3
    elif aqi_0_500 <= 200: return 4
    else: return 5

def get_aqi_category(aqi):
    """Determine AQI category and CSS class."""
    if aqi <= 50:
        return "Good", "aqi-alert-good"
    elif aqi <= 100:
        return "Moderate", "aqi-alert-moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "aqi-alert-unhealthy-sensitive"
    elif aqi <= 200:
        return "Unhealthy", "aqi-alert-unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy", "aqi-alert-very-unhealthy"
    else:
        return "Hazardous", "aqi-alert-hazardous"

def show_alert(message, alert_type="info", aqi_value=None):
    """Display a styled alert message."""
    if alert_type == "aqi" and aqi_value is not None:
        category, style_class = get_aqi_category(aqi_value)
        st.markdown(
            f'<div class="{style_class}">{message}: AQI ({aqi_value:.2f}) is {category}</div>',
            unsafe_allow_html=True
        )
    else:
        style_map = {
            "info": "background-color: #d9edf7; color: #31708f; border: 1px solid #bce8f1;",
            "success": "background-color: #dff0d8; color: #3c763d; border: 1px solid #d6e9c6;",
            "warning": "background-color: #fcf8e3; color: #8a6d3b; border: 1px solid #faebcc;",
            "error": "background-color: #f2dede; color: #a94442; border: 1px solid #ebccd1;"
        }
        style = style_map.get(alert_type, style_map["info"])
        st.markdown(
            f'<div style="{style} padding: 10px; border-radius: 5px; margin: 10px 0;">{message}</div>',
            unsafe_allow_html=True
        )

def predict_aqi(data, is_api_data=True, historical_aqi=None):
    """Predict AQI using CNN-LSTM model."""
    if cnn_lstm_model is None or not scalers_available:
        return None, None
    X_seq = preprocess_for_cnn_lstm(data, is_api_data)
    if X_seq is None:
        return None, None
    try:
        expected_input_shape = cnn_lstm_model.input_shape
        if len(X_seq.shape) != len(expected_input_shape):
            return None, None

        batch_size = 32
        y_pred_scaled = []
        for i in range(0, len(X_seq), batch_size):
            batch = X_seq[i:i + batch_size]
            batch_pred = cnn_lstm_model.predict(batch, verbose=0)
            y_pred_scaled.append(batch_pred)
        y_pred_scaled = np.concatenate(y_pred_scaled, axis=0)

        y_pred_scaled = np.clip(y_pred_scaled, 0, 1)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_pred = np.clip(y_pred, 0, 500)

        if is_api_data:
            y_pred_raw = y_pred.copy()
            y_pred_api = np.array([convert_aqi_to_api_scale(val) for val in y_pred])
            if 'aqi' in data:
                api_aqi = data['aqi']
                pred_median = np.median(y_pred_api)
                y_pred_api = y_pred_api - (pred_median - api_aqi)
                y_pred_api = np.clip(y_pred_api, 1, 5)
            return y_pred_raw.flatten(), y_pred_api.flatten()
        return y_pred.flatten(), None
    except Exception:
        return None, None

def preprocess_uploaded_data(uploaded_file):
    """Preprocess uploaded CSV data."""
    try:
        df = pd.read_csv(uploaded_file)
        datetime_col = None
        for col in ['Datetime', 'datetime', 'date', 'Date']:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    if df[col].notna().any():
                        datetime_col = col
                        break
                except Exception as e:
                    show_alert(f"Failed to parse '{col}' column as datetime: {str(e)}", alert_type="warning")
        if not datetime_col:
            show_alert("No 'Datetime' column found. X-axis will use timesteps.", alert_type="warning")
        
        if datetime_col:
            df['year'] = df[datetime_col].dt.year
            df['month'] = df[datetime_col].dt.month
            df['day'] = df[datetime_col].dt.day
            df['hour'] = df[datetime_col].dt.hour
        else:
            df['year'] = df['month'] = df['day'] = df['hour'] = 0

        expected_cols = ['pm2_5', 'pm10', 'no2', 'so2', 'co', 'o3']
        present_cols = [col for col in expected_cols if col in df.columns]
        if not present_cols:
            show_alert(f"CSV must contain at least one of: {', '.join(expected_cols)}", alert_type="error")
            return None, None, None
        aqi = df['AQI'].values if 'AQI' in df.columns else None
        return df, aqi, datetime_col
    except Exception as e:
        show_alert(f"Error reading CSV: {str(e)}", alert_type="error")
        return None, None, None

def classify_sky_image(uploaded_file):
    """Classify sky image to predict AQI."""
    if sky_cnn_model is None:
        return None, None, None
    img_size = (224, 224)
    try:
        img = image.load_img(uploaded_file, target_size=img_size)
        x = image.img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)
        prediction = sky_cnn_model.predict(x, verbose=0)
        aqi = float(prediction[0]) * 500
        aqi = np.clip(aqi, 0, 500)
        if aqi <= 50: label = "Clean"
        elif aqi <= 100: label = "Satisfactory"
        elif aqi <= 200: label = "Moderately Polluted"
        elif aqi <= 300: label = "Poor"
        elif aqi <= 400: label = "Heavily Polluted"
        else: label = "Severely Polluted"
        return img, aqi, label
    except Exception as e:
        show_alert(f"Error processing image: {str(e)}", alert_type="error")
        return None, None, None

# Navigation tabs
tabs = st.tabs(["Home", "Real-Time Air Quality", "Historical Analysis", "Sky Image Analysis"])

# Home page
with tabs[0]:
    st.title("AirSafe Innovators")
    st.markdown("""
    Application for **SDG 11 – Target 11.6**.  
    **Features**:  
    - Real-time air quality and AQI predictions.  
    - Historical analysis via CSV.  
    - AQI estimation from sky images.
    """)

# Real-time air quality
with tabs[1]:
    st.header("Real-Time Air Quality")
    city = st.text_input("City:", placeholder="e.g., Paris")
    if city:
        with st.spinner("Fetching data..."):
            air_data, error = fetch_air_quality(city)
            if error:
                show_alert(error, alert_type="error")
            else:
                st.subheader(f"Air Quality - {city}")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**PM2.5**: {air_data['pm2_5']} µg/m³")
                    st.write(f"**PM10**: {air_data['pm10']} µg/m³")
                    st.write(f"**NO2**: {air_data['no2']} µg/m³")
                with col2:
                    st.write(f"**SO2**: {air_data['so2']} µg/m³")
                    st.write(f"**CO**: {air_data['co']} µg/m³")
                    st.write(f"**O3**: {air_data['o3']} µg/m³")
                st.write(f"**Current AQI (1-5)**: {air_data['aqi']}")
                show_alert("Current Air Quality Status", alert_type="aqi", aqi_value=air_data['aqi'] * 50)

                if cnn_lstm_model and scalers_available:
                    predicted_aqi_raw, predicted_aqi_api = predict_aqi(air_data, is_api_data=True)
                    if predicted_aqi_raw is not None:
                        st.subheader("AQI Prediction")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Predicted AQI (0-500)", f"{predicted_aqi_raw[0]:.2f}")
                        with col2:
                            st.metric("Predicted AQI (1-5)", f"{predicted_aqi_api[0]:.2f}")

                        if predicted_aqi_raw[0] > 100:
                            show_alert("Warning: High Predicted AQI", alert_type="aqi", aqi_value=predicted_aqi_raw[0])
                        else:
                            show_alert("Predicted AQI is within safe levels", alert_type="success")

                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=['Current AQI', 'Predicted AQI'],
                            y=[air_data['aqi'], predicted_aqi_api[0]],
                            marker_color=['#1f77b4', '#ff7f0e'],
                            text=[air_data['aqi'], f"{predicted_aqi_api[0]:.2f}"],
                            textposition='auto'
                        ))
                        fig.update_layout(
                            title="Current vs Predicted AQI",
                            yaxis_title="AQI (1-5)",
                            yaxis_range=[0, 5]
                        )
                        st.plotly_chart(fig, use_container_width=True)

# Historical analysis
with tabs[2]:
    st.header("Historical Analysis")
    uploaded_file = st.file_uploader("Upload CSV:", type=["csv"])
    if uploaded_file:
        with st.spinner("Processing data..."):
            df, historical_aqi, datetime_col = preprocess_uploaded_data(uploaded_file)
            if df is None:
                show_alert("Failed to process CSV file. Check format and columns.", alert_type="error")
            else:
                st.subheader("Data Preview")
                st.dataframe(df.head())
                st.write(f"NaN values in df: {df.isna().sum()}")

                if len(df) < 30:
                    show_alert("Insufficient data: CSV must contain at least 30 rows for analysis.", alert_type="warning")
                elif cnn_lstm_model is None or not scalers_available:
                    show_alert("CNN-LSTM model or scalers not available. Predictions disabled.", alert_type="error")
                else:
                    # Pass the full historical_aqi to predict_aqi
                    predicted_aqi, _ = predict_aqi(df, is_api_data=False, historical_aqi=historical_aqi)
                    st.write(f"predicted_aqi: {predicted_aqi}")
                    st.write(f"historical_aqi: {historical_aqi}")
                    if predicted_aqi is None or len(predicted_aqi) == 0:
                        show_alert("Failed to predict AQI. Check input data or model.", alert_type="error")
                    else:
                        st.subheader("AQI Predictions")
                        if historical_aqi is not None and len(historical_aqi) >= 30:
                            latest_real_aqi = historical_aqi[-1]
                            st.metric("Latest Real AQI", f"{latest_real_aqi:.2f}")
                            show_alert("Latest Real AQI Status", alert_type="aqi", aqi_value=latest_real_aqi)
                        else:
                            show_alert("No valid 'AQI' data found for Real AQI.", alert_type="warning")

                        st.metric("Latest Predicted AQI", f"{predicted_aqi[-1]:.2f}")
                        show_alert("Latest Predicted AQI Status", alert_type="aqi", aqi_value=predicted_aqi[-1])

                        if len(predicted_aqi) > 1 and predicted_aqi[-1] > predicted_aqi[-2] * 1.2:
                            show_alert("Alert: AQI is rising rapidly.", alert_type="warning")

                        try:
                            plot_data = pd.DataFrame()
                            # Align Real AQI with Predicted AQI by slicing historical_aqi
                            if historical_aqi is not None and len(historical_aqi) >= 30:
                                start_idx = len(historical_aqi) - len(predicted_aqi)
                                plot_data['Real AQI'] = np.clip(historical_aqi[start_idx:], 0, 500)
                            plot_data['Predicted AQI'] = predicted_aqi
                            if datetime_col:
                                plot_data['Date'] = df[datetime_col][start_idx:start_idx+len(predicted_aqi)].astype(str)
                            else:
                                plot_data['Date'] = range(len(predicted_aqi))

                            if plot_data.empty or len(plot_data) == 0:
                                show_alert("No valid data for graph. Check CSV.", alert_type="error")
                            else:
                                fig = go.Figure()
                                if 'Real AQI' in plot_data:
                                    fig.add_trace(go.Scatter(
                                        x=plot_data['Date'],
                                        y=plot_data['Real AQI'],
                                        name='Real AQI',
                                        line=dict(color='#1f77b4')
                                    ))
                                else:
                                    show_alert("No 'AQI' data available for plotting Real AQI.", alert_type="warning")
                                if 'Predicted AQI' in plot_data and len(plot_data['Predicted AQI']) > 0:
                                    fig.add_trace(go.Scatter(
                                        x=plot_data['Date'],
                                        y=plot_data['Predicted AQI'],
                                        name='Predicted AQI',
                                        line=dict(color='#ff7f0e')
                                    ))
                                else:
                                    show_alert("No predicted AQI data available.", alert_type="warning")
                                if not fig.data:  # Check if the figure has any traces
                                    show_alert("Graph is empty: No data to display.", alert_type="error")
                                else:
                                    fig.update_layout(
                                        title="Real vs Predicted AQI",
                                        xaxis_title="Date" if datetime_col else "Timestep",
                                        yaxis_title="AQI (0-500)",
                                        yaxis_range=[0, 500],
                                        hovermode="x unified"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    show_alert("Graph displayed successfully.", alert_type="success")
                        except Exception as e:
                            show_alert(f"Error creating graph: {str(e)}", alert_type="error")

# Sky image analysis
with tabs[3]:
    st.header("Sky Image Analysis")
    uploaded_image = st.file_uploader("Upload image:", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        with st.spinner("Analyzing image..."):
            img, aqi, label = classify_sky_image(uploaded_image)
            if img is not None:
                st.image(img, caption="Sky Image", width=300)
                st.subheader("Analysis Result")
                st.metric("Estimated AQI (0-500)", f"{aqi:.2f}")
                st.write(f"**Air Quality**: {label}")
                show_alert("Sky Image AQI Estimation", alert_type="aqi", aqi_value=aqi)

# Footer
st.markdown("---")
st.write("**Contact**: ghitaaem73@gmail.com")
st.write("AirSafe Innovators for sustainable cities.")