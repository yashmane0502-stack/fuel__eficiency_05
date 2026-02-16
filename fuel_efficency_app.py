import streamlit as st
import pandas as pd
import joblib

# Load model and encoder
model = joblib.load("Fuel_Efiiciency_model.pkl")
encoder = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="Fuel Efficiency Predictor", layout="centered")

st.title("🚗 Fuel Efficiency Prediction App")

st.write("Enter vehicle details below to predict fuel efficiency.")

# User Inputs
mpg = st.number_input("MPG", min_value=0.0)
cylinders = st.number_input("cylinders", min_value=1)
displacement = st.number_input("displacement", min_value=0.0)
horsepower = st.number_input("horsepower", min_value=0.0)
weight = st.number_input("weight", min_value=0.0)
acceleration = st.number_input("acceleration", min_value=0.0)
model_year = st.number_input("model year", min_value=1900)
origin = st.number_input("origin", min_value=1)
car_name = st.text_input("car name")

if st.button("Predict Fuel Efficiency"):

    if car_name.strip() == "":
        st.warning("Please enter car name")
    else:
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

        try:
            # Encoding
            if isinstance(encoder, dict):
                for col in encoder:
                    df[col] = encoder[col].transform(df[col])
            else:
                df["car name"] = encoder.transform(df["car name"])

            # Match training feature order
            df = df[model.feature_names_in_]

            # Prediction
            prediction = model.predict(df)

            st.success(f" Predicted Fuel Efficiency: {prediction[0]:.2f}")

        except Exception as e:
            st.error(f"Error: {e}")
