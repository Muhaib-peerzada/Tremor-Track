import streamlit as st
import joblib
import pandas as pd
import numpy as np
import gdown
import os

# Google Drive file IDs
file_ids = {
    "earthquake_magnitude_predictor.pkl": "1MQxltXuyc_pcwAKs5P10wKPRjIymTLW2",
    "usgs_main.csv": "1V8QUguj83h2RxiZaZfNg2A8meIseVfuO"
}


def download_file(filename, file_id):
    if not os.path.exists(filename):  # Check if file already exists
        url = f"https://drive.google.com/uc?id={file_id}"
        st.info(f"📥 Downloading {filename} from Google Drive...")
        try:
            gdown.download(url, filename, quiet=False)
            st.success(f"✅ {filename} downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Error downloading {filename}: {e}")

download_file("earthquake_magnitude_predictor.pkl", file_ids["earthquake_magnitude_predictor.pkl"])
download_file("usgs_main.csv", file_ids["usgs_main.csv"])

try:
    model = joblib.load("earthquake_magnitude_predictor.pkl")
    num_imputer = joblib.load("numerical_imputer.pkl")
    cat_imputer = joblib.load("categorical_imputer.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    st.success("✅ All model files loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model files: {e}")

mag_types = [col.split("_")[1] for col in feature_columns if "magType" in col]

# Streamlit UI
st.title("       🌍 TremorTrack")
st.markdown("<h5 style='text-align: center; color: grey;'>Earthquake Magnitude Predictor</h5>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: grey;'>by Peerzada Mubashir</h5>", unsafe_allow_html=True)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Latitude", format="%.4f")
        lon = st.number_input("Longitude", format="%.4f")
    with col2:
        depth = st.number_input("Depth (km)", min_value=0.0)
        rms = st.number_input("RMS Value", min_value=0.0)

    mag_type = st.selectbox("Magnitude Type", mag_types)

    if st.form_submit_button("Predict Magnitude"):
        try:
            
            input_data = pd.DataFrame([[lat, lon, depth, rms, mag_type]], 
                                      columns=["latitude", "longitude", "depth", "rms", "magType"])
            
            # Preprocess
            num_features = ["latitude", "longitude", "depth", "rms"]
            cat_features = ["magType"]
            
            input_data[num_features] = num_imputer.transform(input_data[num_features])
            input_data[cat_features] = cat_imputer.transform(input_data[cat_features])
            
            # One-hot encode
            input_encoded = pd.get_dummies(input_data, columns=["magType"])
            
            # Ensure all feature columns are present
            for col in feature_columns:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0  # Add missing columns with 0
            
            # Reorder columns to match training data
            input_encoded = input_encoded[feature_columns]
            
            # Predict
            prediction = max(0, model.predict(input_encoded)[0])
            st.success(f"Predicted Magnitude: **{prediction:.2f}**")
        
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

# Additional note about the model
st.markdown(""" 
    **Note:**  
    This model uses a Random Forest Regressor to predict earthquake magnitudes based on features
     like latitude, longitude, depth, rms, and magType. After preprocessing the data 
    (handling missing values, encoding categorical features, and splitting into training and testing sets),
     hyperparameter tuning is done using RandomizedSearchCV.

    The model’s performance is evaluated with:

    - **MAE**: **0.3166**
    - **MSE**: **0.1861**
    - **R²**: **0.87**

    The model was able to predict earthquake magnitudes with reasonable accuracy, 
     as shown by the actual vs predicted comparison. Feature importance is also analyzed to 
    understand which attributes contribute most to the predictions.

    **Data Source**: The data used to train this model is sourced from the United States Geological Survey (USGS).
""")
