import streamlit as st
import pandas as pd
import joblib

# Load model and encoder
model = joblib.load("Fuel_Efiiciency_model.pkl")
encoder = joblib.load("label_encoder.pkl")

st.title("Fuel Efficiency Prediction App")

# Inputs (REMOVED mpg because that should be predicted)
cylinders = st.number_input("Enter Cylinders value", min_value=1)
displacement = st.number_input("Enter Displacement value", min_value=0.0)
horsepower = st.number_input("Enter Horsepower value", min_value=0.0)
weight = st.number_input("Enter Weight value", min_value=0.0)
acceleration = st.number_input("Enter Acceleration value", min_value=0.0)
model_year = st.number_input("Enter Model Year value", min_value=1970, max_value=2025)
origin = st.number_input("Enter Origin value (1=USA, 2=Europe, 3=Japan)", min_value=1, max_value=3)
car_name = st.text_input("Enter Car Name")

if st.button("Predict"):

    if car_name.strip() == "":
        st.warning("Please enter a car name.")
    else:

        # Create dataframe
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

        # Encode car name safely
        try:
            df["car name"] = encoder.transform(df["car name"])
        except:
            st.error("Car name not recognized by model encoder.")
            st.stop()

        # Reorder columns to match training
        df = df[model.feature_names_in_]

        # Prediction
        prediction = model.predict(df)

        st.success(f"Predicted Fuel Efficiency (MPG): {prediction[0]:.2f}")
