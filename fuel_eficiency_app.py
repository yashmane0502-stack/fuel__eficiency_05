import streamlit as st
import pandas as pd
import joblib

model = joblib.load("Fuel_Efiiciency_model.pkl")
encoder = joblib.load("label_encoder.pkl")

st.title("Fuel Efficiency Prediction")

mpg = st.number_input("mpg", 0.0, 100.0)
cylinders = st.number_input("cylinders", 0, 16)
displacement = st.number_input("displacement", 0.0, 1000.0)
horsepower = st.number_input("horsepower", 0.0, 1000.0)
weight = st.number_input("weight", 0.0, 10000.0)
acceleration = st.number_input("acceleration", 0.0, 50.0)
model_year = st.number_input("model year", 1900, 2100)
origin = st.number_input("origin", 1, 3)

# Handle encoder type safely
if isinstance(encoder, dict):
    car_name = st.selectbox("car name", encoder["car name"].classes_)
else:
    car_name = st.text_input("car name")

if st.button("Predict"):

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

    # Encoding
    if isinstance(encoder, dict):
        for col in encoder:
            df[col] = encoder[col].transform(df[col])
    else:
        df["car name"] = encoder.transform(df["car name"])

    # Match training column order
    df = df[model.feature_names_in_]

    prediction = model.predict(df)

    st.success(f"Fuel Eficiency: {prediction[0]:,.2f}")
