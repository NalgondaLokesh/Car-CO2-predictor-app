import streamlit as st
import joblib
import pandas as pd

# App settings (must be first Streamlit command)
st.set_page_config(page_title="Car CO₂ Emission Predictor", layout="centered")

# Add a clean, modern look with a soft gradient
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(120deg, #f8ffae 0%, #43c6ac 100%);
    }
    .main > div:first-child {
        background: rgba(255,255,255,0.92);
        border-radius: 16px;
        padding: 2rem 2rem 1rem 2rem;
        margin-top: 2rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.07);
    }
    .stButton>button {
        background: linear-gradient(90deg, #43c6ac 0%, #191654 100%);
        color: #fff;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        font-size: 1.1em;
        padding: 0.5em 2em;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: 0.2s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #191654 0%, #43c6ac 100%);
        color: #fff;
        transform: scale(1.04);
    }
    .stNumberInput>div>input {
        border-radius: 6px;
        border: 1px solid #43c6ac;
        padding: 0.4em 0.8em;
        font-size: 1.05em;
    }
    label, .stTextInput label, .stNumberInput label, .stSelectbox label {
        color: black !important;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load trained model
model = joblib.load("Co2_model.pkl")

st.title("🚗 Car CO₂ Emission Predictor")
st.markdown("<h3 style='color:#191654;'>Estimate <b>CO₂ emissions (g/km)</b> using engine and fuel specifications.</h3>", unsafe_allow_html=True)

# Input fields with black labels
engine_size = st.number_input("Engine Size (L)", min_value=0.0, step=0.1)
cylinders = st.number_input("Number of Cylinders", min_value=1, step=1)
fuel_comb = st.number_input("Fuel Consumption - Combined (L/100 km)", min_value=0.0, step=0.1)
fuel_city = st.number_input("Fuel Consumption - City (L/100 km)", min_value=0.0, step=0.1)
fuel_hwy = st.number_input("Fuel Consumption - Highway (L/100 km)", min_value=0.0, step=0.1)

# Predict button
if st.button("Predict CO₂ Emission"):
    input_df = pd.DataFrame([{
        "Engine Size(L)": engine_size,
        "Cylinders": cylinders,
        "Fuel Consumption Comb (L/100 km)": fuel_comb,
        "Fuel Consumption City (L/100 km)": fuel_city,
        "Fuel Consumption Hwy (L/100 km)": fuel_hwy
    }])

    prediction = model.predict(input_df)
    st.markdown(f"<span style='color:black;font-size:1.1em;'>🚗 Estimated CO₂ Emissions: <b>{prediction[0]:.2f} g/km</b></span>", unsafe_allow_html=True)

# Footer with project and social links
st.markdown(
    """
    <hr style='margin-top:2em;margin-bottom:0.5em;border:1px solid #bbb;'>
    <div style='text-align:center; font-size:1.05em; color:#222;'>
        A Project by <b>Nalgonda Lokesh</b><br>
        <a href='https://github.com/nalgondalokesh' target='_blank' style='margin:0 8px;'>GitHub</a> |
        <a href='https://www.linkedin.com/in/nalgondalokesh/' target='_blank' style='margin:0 8px;'>LinkedIn</a> |
        <a href='https://www.instagram.com/nalgondalokesh.ai/' target='_blank' style='margin:0 8px;'>Instagram</a>
        
    </div>
    """,
    unsafe_allow_html=True
)
