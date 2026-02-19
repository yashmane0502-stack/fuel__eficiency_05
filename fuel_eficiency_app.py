import streamlit as st
import pandas as pd
import pickle

# Load model safely
with open("Fuel_Efficiency_model.pkl", "rb") as f:
    model = pickle.load(f, encoding="latin1")

with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f, encoding="latin1")

st.title("🚗 Fuel Efficiency Prediction App")

# Inputs
mpg = st.number_input("MPG", 0.0, 100.0)
cylinders = st.number_input("Cylinders", 0, 16)
displacement = st.number_input("Displacement", 0.0, 1000.0)
horsepower = st.number_input("Horsepower", 0.0, 1000.0)
weight = st.number_input("Weight", 0.0, 10000.0)
acceleration = st.number_input("Acceleration", 0.0, 50.0)
model_year = st.number_input("Model Year", 1900, 2100)
origin = st.selectbox("Origin", [1, 2, 3])

# If encoder is dictionary
if isinstance(encoder, dict):
    car_name = st.selectbox("Car Name", encoder["car name"].classes_)
else:
    car_name = st.selectbox("Car Name", encoder.classes_)

# Predict button
if st.button("Predict Fuel Efficiency"):

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

    # Apply encoding safely
    if isinstance(encoder, dict):
        for col in encoder:
            df[col] = encoder[col].transform(df[col])
    else:
        df["car name"] = encoder.transform(df["car name"])

    # Match training column order
    df = df[model.feature_names_in_]

    prediction = model.predict(df)

    st.success(f"🚀 Predicted Fuel Efficiency: {prediction[0]:.2f} MPG")
