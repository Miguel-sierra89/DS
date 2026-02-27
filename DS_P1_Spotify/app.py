# ==============================
# pip install streamlit shap joblib scikit-learn matplotlib
# streamlit run app.py
# ==============================

# ==============================
# IMPORTS
# ==============================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")  # Necesario para Streamlit
import matplotlib.pyplot as plt

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Spotify Popularity Predictor")
st.title("🎵 Spotify Popularity Predictor")

# ==============================
# LOAD MODEL
# ==============================
model = joblib.load("models/random_forest.pkl")

# ==============================
# SIDEBAR INPUTS
# ==============================
st.sidebar.header("Audio Features")

acousticness = st.sidebar.slider("Acousticness", 0.0, 1.0, 0.3)
danceability = st.sidebar.slider("Danceability", 0.0, 1.0, 0.5)
duration_ms = st.sidebar.slider("Duration (ms)", 30000, 300000, 180000)
energy = st.sidebar.slider("Energy", 0.0, 1.0, 0.5)
instrumentalness = st.sidebar.slider("Instrumentalness", 0.0, 1.0, 0.0)
liveness = st.sidebar.slider("Liveness", 0.0, 1.0, 0.2)
loudness = st.sidebar.slider("Loudness (dB)", -60.0, 0.0, -10.0)
speechiness = st.sidebar.slider("Speechiness", 0.0, 1.0, 0.05)
tempo = st.sidebar.slider("Tempo", 50.0, 200.0, 120.0)
valence = st.sidebar.slider("Valence", 0.0, 1.0, 0.5)

key = st.sidebar.selectbox("Key (0-11)", list(range(12)))
mode = st.sidebar.selectbox("Mode (0=Minor, 1=Major)", [0, 1])
time_signature = st.sidebar.selectbox("Time Signature", [1, 2, 3, 4, 5])

# ==============================
# BUILD INPUT DICTIONARY
# ==============================
input_dict = {
    "acousticness": acousticness,
    "danceability": danceability,
    "duration_ms": duration_ms,
    "energy": energy,
    "instrumentalness": instrumentalness,
    "liveness": liveness,
    "loudness": loudness,
    "speechiness": speechiness,
    "tempo": tempo,
    "valence": valence,
}

# KEY DUMMIES (key_1 to key_11)
for i in range(1, 12):
    input_dict[f"key_{i}"] = 1 if key == i else 0

# MODE DUMMY (mode_1 only)
input_dict["mode_1"] = 1 if mode == 1 else 0

# TIME SIGNATURE DUMMIES
for ts in [1, 3, 4, 5]:
    input_dict[f"time_signature_{ts}"] = 1 if time_signature == ts else 0

# Convert to DataFrame
input_data = pd.DataFrame([input_dict])

# Ensure exact column order
input_data = input_data[model.feature_names_in_]

# ==============================
# PREDICTION + SHAP
# ==============================

if st.button("Predict Popularity"):

    # --------------------------
    # Prediction
    # --------------------------
    probability = model.predict_proba(input_data)[0][1]
    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")
    st.metric("Popularity Probability", f"{probability:.2%}")

    if prediction == 1:
        st.success("🔥 This track has high potential to be popular!")
    else:
        st.warning("⚠️ This track is unlikely to become popular.")

    # --------------------------
    # --------------------------
# SHAP Explanation
# --------------------------
st.subheader("🔍 Model Explanation")

explainer = shap.TreeExplainer(model)

# 🔥 Forma moderna (SHAP 0.50)
shap_values = explainer(input_data)

# Extraer SOLO una explicación y SOLO clase positiva
if len(shap_values.values.shape) == 3:
    # (1, n_features, 2)
    shap_values_to_plot = shap_values.values[0, :, 1]
    expected_value = explainer.expected_value[1]
else:
    # Regresión o forma diferente
    shap_values_to_plot = shap_values.values[0]
    expected_value = explainer.expected_value

fig, ax = plt.subplots()

shap.plots._waterfall.waterfall_legacy(
    expected_value,
    shap_values_to_plot,
    feature_names=input_data.columns,
    max_display=10,
    show=False
)

st.pyplot(fig)