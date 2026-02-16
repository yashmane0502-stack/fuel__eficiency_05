import streamlit as st
import pandas as pd
import joblib

# Load model and encoder
model = joblib.load("Fuel_Efficiency_model.pkl")
encoder = joblib.load("fuel_label_encoder.pkl")

st.title("Fuel Efficiency Prediction")

cylinders = st.number_input("cylinders", 0, 12)
displacement = st.number_input("displacement", 0.0, 1000.0)
horsepower = st.number_input("horsepower", 0.0, 1000.0)
weight = st.number_input("weight", 0.0, 10000.0)
acceleration = st.number_input("acceleration", 0.0, 50.0)
model_year = st.number_input("model_year", 1970, 2025)
origin = st.selectbox("origin", encoder["origin"].classes_)
car_name = st.selectbox("car name", encoder["car name"].classes_)

df = pd.DataFrame({
    "cylinders": [cylinders],
    "displacement": [displacement],
    "horsepower": [horsepower],
    "weight": [weight],
    "acceleration": [acceleration],
    "model year": [model_year],
    "origin": [origin],
    "car name": [car_name]
})

if st.button("Predict"):
    
    # Encode categorical columns
    for col in encoder:
        df[col] = encoder[col].transform(df[col])

    # Match training column order
    df = df[model.feature_names_in_]

    prediction = model.predict(df)

    st.success(f"Fuel Efficiency (MPG): {prediction[0]:,.2f}")
