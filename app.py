# app.py - (Main App Code)

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static
from scipy.special import expit
import os  # For potential real data handling

# Cache data simulation (mimics NASA MODIS/VIIRS)
@st.cache_data
def load_data():
    np.random.seed(42)  # For reproducibility
    lats = np.linspace(-10, 10, 100)
    lons = np.linspace(-180, -160, 100)
    LATS, LONS = np.meshgrid(lats, lons)
    chl_a = 0.5 + 2 * np.exp(0.1 * LATS) + np.random.normal(0, 0.2, (100, 100))
    sst = 20 + 5 * np.sin(np.deg2rad(lats))[:, None]
    ssha = 0.1 * np.cos(np.deg2rad(lons))[None, :]
    backscatter = 0.1 + 0.05 * chl_a
    return chl_a, sst, ssha, backscatter, lats, lons

# Model functions (HSI: Logistic Regression with iterative updates)
def fit_logistic(X, y, lr=0.01, epochs=50):
    weights = np.zeros(X.shape[1])
    for _ in range(epochs):
        z = np.dot(X, weights)
        grad = np.dot(X.T, (expit(z) - y)) / len(y)
        weights -= lr * grad
    return weights

def update_logistic(weights, X_new, y_new, lr=0.01, epochs=10):
    for _ in range(epochs):
        z = np.dot(X_new, weights)
        grad = np.dot(X_new.T, (expit(z) - y_new)) / len(y_new)
        weights -= lr * grad
    return weights

def ecological_link(Chl_a_flat, SSHA_flat):
    return 1 + 0.3 * np.maximum(Chl_a_flat - 1, 0) + 0.2 * np.abs(SSHA_flat)

# UI
st.title("SharkSentinel AI 🦈 - Iterative ML Edition")
st.markdown("""
Predict shark hotspots using NASA satellite data! Now with iterative machine learning updates from tag data.
This app simulates ocean data for demo; see README for real NASA integration.
""")

# Initialize session state for weights
if 'weights_hsi' not in st.session_state:
    st.session_state.weights_hsi = None

if st.button("Load Ocean Data"):
    with st.spinner("Fetching satellite data..."):
        chl_a, sst, ssha, backscatter, lats, lons = load_data()
        X = np.column_stack([sst.flatten(), chl_a.flatten(), ssha.flatten(), backscatter.flatten()])
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        
        # Initial HSI training if not done
        if st.session_state.weights_hsi is None:
            true_weights = np.array([0, 0.1, 0.3, 0.2, 0.15])  # Simulated initial weights
            z = np.dot(X_bias, true_weights)
            presence = (expit(z) > 0.5).astype(float)
            st.session_state.weights_hsi = fit_logistic(X_bias, presence)
        
        hsi_proba = expit(np.dot(X_bias, st.session_state.weights_hsi)).reshape(100, 100)
        
        # FIM
        E = ecological_link(chl_a.flatten(), ssha.flatten()).reshape(100, 100)
        temp_factor = np.clip(1 - np.abs(sst - 22) / 10, 0, 1)
        fim_intensity = hsi_proba * E * temp_factor
        
        # Map
        m = folium.Map(location=[0, -170], zoom_start=5)
        folium.HeatMap(list(zip(lats, lons, fim_intensity.flatten())), radius=15).add_to(m)
        folium_static(m)
        
        # Plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.imshow(fim_intensity, cmap='Reds', extent=[lons.min(), lons.max(), lats.min(), lats.max()])
        ax1.set_title("Foraging Hotspots")
        df_links = pd.DataFrame({'Chl_a': chl_a.flatten(), 'FIM': fim_intensity.flatten()})
        corr = df_links.corr().iloc[0,1]
        ax2.scatter(chl_a.flatten(), fim_intensity.flatten(), alpha=0.5)
        ax2.set_xlabel("Chlorophyll-a (mg/m³)"); ax2.set_ylabel("Foraging Intensity")
        ax2.set_title(f"Chl_a vs. FIM (r={corr:.2f})")
        plt.tight_layout()
        st.pyplot(fig)

# Tag uploader for iterative updates
uploaded_tag = st.file_uploader("Upload Tag Data for Iterative Update (CSV: lat,lon,prey,presence)")
if uploaded_tag:
    tag_data = pd.read_csv(uploaded_tag)
    # Simulate feature extraction from tags (use grid features for simplicity; real: interpolate lat/lon)
    chl_a, sst, ssha, backscatter, _, _ = load_data()
    X_new = np.column_stack([sst.flatten(), chl_a.flatten(), ssha.flatten(), backscatter.flatten()])
    X_new_bias = np.hstack([np.ones((X_new.shape[0], 1)), X_new])
    y_new = tag_data['presence'].values[:X_new.shape[0]]  # Truncate to match; real: handle properly
    # Iterative update
    if st.session_state.weights_hsi is not None:
        st.session_state.weights_hsi = update_logistic(st.session_state.weights_hsi, X_new_bias, y_new)
        st.success("Model updated iteratively with tag data! Re-run 'Load Ocean Data' to see changes.")
    else:
        st.warning("Load data first to initialize model.")

st.markdown("Built for NASA Space Apps 2025. See GitHub for source!")

# Optional: Real NASA fetch (uncomment and add dependencies: requests, xarray, io)
# import requests
# import xarray as xr
# import io
# @st.cache_data
# def load_real_nasa_data():
#     NASA_API_KEY = os.getenv("NASA_API_KEY")  # Set in environment
#     url = "https://oceandata.sci.gsfc.nasa.gov/opendap/MODISA/L3SMI/2025/001/AQUA_MODIS.20250101.L3m.DAY.CHL.chlor_a.4km.nc"
#     response = requests.get(url, auth=('username', NASA_API_KEY))
#     if response.status_code == 200:
#         ds = xr.open_dataset(io.BytesIO(response.content))
#         chl_a = ds['chlor_a'].values  # Subset as needed
#         lats = ds['lat'].values
#         lons = ds['lon'].values
#         # Add sim SST/SSHA or fetch similarly
#         return chl_a, ..., lats, lons
#     return None  # Fallback to sim
