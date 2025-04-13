# AirSafe Innovators

AirSafe Innovators is a Streamlit web application designed to support **Sustainable Development Goal (SDG) 11 – Target 11.6** by providing tools to monitor and predict air quality. 
The application offers real-time air quality data, historical analysis, and sky image-based AQI estimation, empowering users to make informed decisions for sustainable cities.

## Features

- Real-Time Air Quality:
  - Fetch current air quality data (PM2.5, PM10, NO2, SO2, CO, O3, AQI) for any city using the OpenWeather API.
  - Predict future AQI using a CNN-LSTM model.
  - Display alerts for high AQI levels.
 
  - ###  Result:

![Air Quality Result](https://github.com/ghitaaitm/AirSafe_Innovators/blob/main/screen.PNG?raw=true)

- Historical Analysis
  - Upload CSV files to analyze historical air quality data.
  - Predict AQI trends using the CNN-LSTM model.
  - Visualize real vs. predicted AQI with interactive Plotly graphs.
  - Alerts for data issues, predictions, and significant AQI trends.

- Sky Image Analysis:
  - Upload sky images (JPG, PNG, JPEG) to estimate AQI using a CNN model.
  - Display estimated AQI and air quality category with alerts.

- User-Friendly Interface:
  - Intuitive tabs: Home, Real-Time Air Quality, Historical Analysis, Sky Image Analysis.
  - Custom CSS for a modern, responsive design.
  - Alerts for errors, successes, and AQI statuses.

## Prerequisites

- Python 3.8+
- Required Python packages (listed in `requirements.txt`):
  - `streamlit`
  - `pandas`
  - `numpy`
  - `tensorflow`
  - `requests`
  - `pillow`
  - `plotly`
  - `joblib`
- Model and asset files (must be in the project directory):
  - `cnn_lstm_model.h5` (CNN-LSTM model for AQI predictions)
  - `scaler_x.pkl` (Input scaler for CNN-LSTM)
  - `scaler_y.pkl` (Output scaler for CNN-LSTM)
  - `air_quality_model.keras` (CNN model for sky image analysis)
  - `model_CNN_LSTM.py` 
  - `logo.png` (Application logo)


    ### Breakdown of Sections Added:

1. **Real-Time Air Quality**:
   - This section explains that users can fetch current air quality data from OpenWeather and also predicts future AQI using a CNN-LSTM model.
   
2. **Historical Analysis**:
   - Users can upload CSV files containing historical air quality data.
   - The application provides data analysis and visualization (using Plotly graphs) to show real vs. predicted AQI trends.
  
   - ### Historical Analysis Result:

![Air Quality Result](https://github.com/ghitaaitm/AirSafe_Innovators/blob/main/HiSTORICAL_Analysis.PNG?raw=true)

3. **Usage Section**:
   - This section helps users understand how to use the features by navigating to specific tabs in the app and interacting with them (e.g., entering a city name for real-time air quality or uploading a CSV file for historical analysis).


**Note**: Model files are not included in this repository due to their size. Download them from [this Google Drive link](https://drive.google.com/drive/folders/1m6gdHy4a9Q3DfhjQuUSya-1NPWLqEQuh?usp=sharing) and place them in the project directory.
 Installation
Clone the Repository:
   ```bash
   git clone https://github.com/your-username/AirSafe_Innovators.git
   cd AirSafe_Innovators
  ```

## Image Analysis Result

![Air Quality Result](https://raw.githubusercontent.com/ghitaaitm/AirSafe_Innovators/main/Image_Analyses_Result.PNG)


<span style="color: black; font-size: 20px;">.</span>

