import streamlit as st
import pandas as pd
import joblib

# Load model and encoder
model = joblib.load("Fuel_Efiiciency_model.pkl")
encoder = joblib.load("label_encoder.pkl")

st.title("🚗 Fuel Efficiency Prediction")

# User Inputs
mpg = st.number_input("MPG", 0.0, 100.0)
cylinders = st.number_input("Cylinders", 0, 16)
displacement = st.number_input("Displacement", 0.0, 1000.0)
horsepower = st.number_input("Horsepower", 0.0, 1000.0)
weight = st.number_input("Weight", 0.0, 10000.0)
acceleration = st.number_input("Acceleration", 0.0, 50.0)
model_year = st.number_input("Model Year", 1900, 2100)
origin = st.number_input("Origin (1=USA, 2=Europe, 3=Asia)", 1, 3)

# FIXED variable name
car_name = st.selectbox("Car Name", encoder["car name"].classes_)

# Prediction Button
if st.button("Predict"):

    # Create DataFrame
    df = pd.DataFrame({
        "mpg": [mpg],
        "cylinders": [cylinders],
        "displacement": [displacement],
        "horsepower": [horsepower],
        "weight": [weight],
        "acceleration": [acceleration],
        "model year": [model_year],
        "origin": [origin],
        "car name": [car_name]
    })

    # Encode categorical columns
    for col in encoder:
        df[col] = encoder[col].transform(df[col])

    # Match training feature order
    df = df[model.feature_names_in_]

    # Prediction
    prediction = model.predict(df)

    # Output
    st.success(f"🚀 Predicted Fuel Efficiency: {prediction[0]:.2f} MPG")
