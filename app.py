import os
import subprocess
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import timedelta
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Smart AQI Dashboard", layout="wide")

# --------------------------------------------------
# TRAIN MODEL IF NOT EXISTS
# --------------------------------------------------
if not os.path.exists("model.pkl") or not os.path.exists("encoder.pkl"):
    subprocess.run(["python", "train_model.py"])

# --------------------------------------------------
# LOAD MODEL (CACHED)
# --------------------------------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    return model, encoder

model, le = load_model()

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv("air_quality.csv")
df["Date"] = pd.to_datetime(df["Date"])

# --------------------------------------------------
# SELECT CITY
# --------------------------------------------------
st.sidebar.title("🌆 Select City")
city = st.sidebar.selectbox("City", sorted(df["City"].unique()))

city_df = df[df["City"] == city].sort_values("Date")
latest = city_df.iloc[-1]

pm25 = latest["PM2.5"]
pm10 = latest["PM10"] if "PM10" in city_df.columns else 0

# --------------------------------------------------
# AQI CATEGORY
# --------------------------------------------------
def aqi_category(value):
    if value <= 30:
        return "Good", "#00E676"
    elif value <= 60:
        return "Moderate", "#FFD54F"
    elif value <= 90:
        return "Poor", "#FF7043"
    else:
        return "Unhealthy", "#FF1744"

status, color = aqi_category(pm25)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown(f"""
<h1 style='font-size:42px; margin-bottom:0;'>
🌍 Smart City Air Quality Intelligence System
</h1>
<p style='color:gray; margin-top:5px;'>
Real-time AQI Monitoring & 7-Day AI Forecasting Dashboard
</p>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------
st.markdown(f"""
<div style="
background: linear-gradient(135deg, {color}90 0%, #1c1f26 40%, #0e1117 100%);
padding:50px;
border-radius:30px;
box-shadow:0 0 60px {color}55;
">

<h4>🔴 LIVE AQI • {city}</h4>

<h1 style="
font-size:90px;
font-weight:900;
text-shadow:0px 0px 30px {color};
">
{int(pm25)}
</h1>

<h3>Air Quality is {status}</h3>

<p>
PM2.5: {round(pm25,2)} µg/m³ |
PM10: {round(pm10,2)}
</p>

</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# INFO CARDS
# --------------------------------------------------
col3, col4 = st.columns(2)

with col3:
    st.markdown("""
    <div style="background:#1c1f26;padding:25px;border-radius:15px;">
    <h4>🌤 Weather Snapshot</h4>
    <p>Temperature: 27°C</p>
    <p>Humidity: 58%</p>
    <p>Wind Speed: 4.7 km/h</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style="background:#1c1f26;padding:25px;border-radius:15px;">
    <h4>📊 AI Insight</h4>
    <p>This forecast uses time-series ML with lag & rolling features.</p>
    <p>Click below to generate 7-day prediction.</p>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# FORECAST
# --------------------------------------------------
st.markdown("## 📈 7-Day AI Forecast")

if st.button("Generate Forecast 🚀"):

    last_row = city_df.iloc[-1]
    lag1 = last_row["PM2.5"]
    roll3 = city_df["PM2.5"].tail(3).mean()
    last_date = last_row["Date"]

    predictions = []
    dates = []

    for i in range(1,8):

        next_date = last_date + timedelta(days=i)

        input_data = np.array([[
            le.transform([city])[0],
            next_date.year,
            next_date.month,
            next_date.day,
            lag1,
            roll3
        ]])

        pred = model.predict(input_data)[0]

        predictions.append(pred)
        dates.append(next_date)

        lag1 = pred
        roll3 = np.mean([roll3, pred])

    forecast_df = pd.DataFrame({
        "Date": dates,
        "Predicted PM2.5": predictions
    })

    fig = px.line(
        forecast_df,
        x="Date",
        y="Predicted PM2.5",
        markers=True,
        template="plotly_dark"
    )

    st.plotly_chart(fig, width="stretch")