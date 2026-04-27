import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Title and description
st.title("🌱 Crop Recommendation System")
st.write("This app predicts the best crop based on soil nutrients and environmental conditions.")

# Load dataset (optional for display)
df = pd.read_csv("Crop_recommendation.csv")

# Show dataset
st.subheader("📊 Dataset")
st.dataframe(df.head())

# Load trained model
model = pickle.load(open("crop_model.pkl", "rb"))

# Input section
st.subheader("🔍 Enter Input Values")

N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, step=1)
P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, step=1)
K = st.number_input("Potassium (K)", min_value=0, max_value=200, step=1)
temperature = st.number_input("Temperature (°C)")
humidity = st.number_input("Humidity (%)")
ph = st.number_input("pH Value")
rainfall = st.number_input("Rainfall (mm)")

# Prediction button
if st.button("🌾 Predict Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)

    st.success(f"✅ Recommended Crop: **{prediction[0]}**")

# Visualization
st.subheader("📈 Visualization")
st.line_chart(df.select_dtypes(include=['int64','float64']))
