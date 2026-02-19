import streamlit as st
import pandas as pd
import joblib

model = joblib.load("Fuel_prediction_model(1).pkl")
encoder = joblib.load("label_encoder(5).pkl")

st.title("Fuel Efficiency Prediction")

mpg = st.number_input("mpg", 0.0, 100.0)
cylinders = st.number_input("cylinders", 0, 16)
displacement = st.number_input("displacement", 0.0, 1000.0)
horsepower = st.number_input("horsepower", 0.0, 1000.0)
weight = st.number_input("weight", 0.0, 10000.0)
acceleration = st.number_input("acceleration", 0.0, 50.0)
model_year = st.number_input("model year", 1900, 2100)
origin = st.number_input("origin", 1, 3)
car_name = st.selectbox("car name", encoder["car name"].classes_)

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

if st.button("Predict"):
    for col in encoder:
        df[col] = encoder[col].transform(df[col])

    df = df[model.feature_names_in_]

    prediction = model.predict(df)

    st.success(f"Fuel Efficiency: {prediction[0]:,.2f}")

