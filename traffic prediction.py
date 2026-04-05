import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Smart Traffic Navigator",
    page_icon="🚦",
    layout="wide"
)

# -------------------------------
# 🎨 ROAD BACKGROUND
# -------------------------------
st.markdown("""
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1493238792000-8113da705763");
    background-size: cover;
    background-position: center;
}
.block-container {
    background: rgba(0,0,0,0.65);
    padding: 2rem;
    border-radius: 10px;
}
h1,h2,h3,h4,h5,h6,p,label {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.markdown("<h1 style='text-align:center;'>🚦 Smart Traffic Navigator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>ML + DL Traffic Prediction</p>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------------
# DATA
# -------------------------------
@st.cache_resource
def load_data():
    return pd.DataFrame({
        "distance": [2,5,10,15,20,25,30],
        "vehicles": [20,50,80,100,120,150,180],
        "hour": [6,8,9,17,18,20,22],
        "traffic": [0,1,2,2,2,1,0]
    })

data = load_data()
X = data[["distance","vehicles","hour"]].values
y = data["traffic"].values

# -------------------------------
# ML MODEL
# -------------------------------
@st.cache_resource
def train_ml():
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

ml_model = train_ml()

# -------------------------------
# DL MODEL
# -------------------------------
@st.cache_resource
def train_dl():
    model = Sequential([
        Dense(8, activation='relu', input_shape=(3,)),
        Dense(8, activation='relu'),
        Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(X, y, epochs=100, verbose=0)
    return model

dl_model = train_dl()

# -------------------------------
# FUNCTIONS
# -------------------------------
def predict_ml(distance, vehicles, hour):
    pred = ml_model.predict([[distance, vehicles, hour]])[0]
    return ["Low","Medium","High"][pred]

def predict_dl(distance, vehicles, hour):
    pred = dl_model.predict(np.array([[distance, vehicles, hour]]), verbose=0)
    return ["Low","Medium","High"][np.argmax(pred)]

def simulate_distance():
    return random.randint(2, 30)

def simulate_vehicles(hour):
    if 8 <= hour <= 10 or 17 <= hour <= 19:
        return random.randint(100, 200)
    return random.randint(20, 80)

def estimate_time(distance, traffic):
    speed = 40
    if traffic == "High":
        speed = 20
    elif traffic == "Medium":
        speed = 30
    return round(distance / speed * 60, 2)

# -------------------------------
# INPUT FORM
# -------------------------------
with st.form("route_form"):
    col1, col2 = st.columns(2)

    with col1:
        start = st.text_input("📍 Start Location")

    with col2:
        dest = st.text_input("📍 Destination")

    hour = st.slider("⏰ Time", 0, 23, 8)

    submit = st.form_submit_button("🚀 Predict")

# -------------------------------
# OUTPUT
# -------------------------------
if submit:
    if start == "" or dest == "":
        st.warning("⚠️ Enter both locations")
    else:
        dist = simulate_distance()
        veh = simulate_vehicles(hour)

        ml_pred = predict_ml(dist, veh, hour)
        dl_pred = predict_dl(dist, veh, hour)

        time = estimate_time(dist, ml_pred)

        st.markdown("## 📊 Results")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🛣 Distance", f"{dist} km")
        c2.metric("🚗 Vehicles", veh)
        c3.metric("ML Traffic", ml_pred)
        c4.metric("DL Traffic", dl_pred)

        st.markdown(f"### ⏱ Travel Time: {time} minutes")

        # -------------------------------
        # 🗺 ROUTE MAP (VISIBLE PATH)
        # -------------------------------
        st.markdown("### 🗺 Route Map")

        # Example coordinates (Hyderabad)
        start_coord = np.array([17.4401, 78.3489])
        end_coord = np.array([17.4948, 78.3996])

        # Generate route points
        points = 50
        lat = np.linspace(start_coord[0], end_coord[0], points)
        lon = np.linspace(start_coord[1], end_coord[1], points)

        # Add curve to simulate road
        curve = np.sin(np.linspace(0, 3, points)) * 0.01
        lat = lat + curve

        route_df = pd.DataFrame({
            "lat": lat,
            "lon": lon
        })

        st.map(route_df, zoom=11)

        # -------------------------------
        # SUMMARY
        # -------------------------------
        st.markdown("### 📍 Route Summary")
        st.write(f"{start} → {dest}")
        st.write(f"Estimated time: {time} minutes")