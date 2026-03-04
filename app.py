import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import timedelta
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Smart AQI Dashboard", layout="wide")

# --------------------------------------------------
# TRAIN MODEL IF FILE NOT PRESENT
# --------------------------------------------------
def train_model():

    df = pd.read_csv("air_quality.csv")

    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(['City','Date'], inplace=True)

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day

    df = df.dropna(subset=['PM2.5'])

    df.fillna(df.median(numeric_only=True), inplace=True)

    df['PM2.5_lag1'] = df.groupby('City')['PM2.5'].shift(1)
    df['PM2.5_roll3'] = df.groupby('City')['PM2.5'].rolling(3).mean().reset_index(0,drop=True)

    df = df.dropna()

    le = LabelEncoder()
    df['City_encoded'] = le.fit_transform(df['City'])

    features = ['City_encoded','Year','Month','Day','PM2.5_lag1','PM2.5_roll3']
    X = df[features]
    y = df['PM2.5']

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X,y)

    pickle.dump(model, open("model.pkl","wb"))
    pickle.dump(le, open("encoder.pkl","wb"))

    return model, le


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
if not os.path.exists("model.pkl"):

    model, le = train_model()

else:

    model = pickle.load(open("model.pkl","rb"))
    le = pickle.load(open("encoder.pkl","rb"))

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
st.title("🌍 Smart City Air Quality Intelligence System")
st.caption("Real-time AQI Monitoring & 7-Day AI Forecasting Dashboard")

# --------------------------------------------------
# AQI DISPLAY
# --------------------------------------------------
st.markdown(f"""
### 🔴 LIVE AQI • {city}

# {int(pm25)}

Air Quality: **{status}**

PM2.5: {round(pm25,2)} µg/m³  
PM10: {round(pm10,2)}
""")

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
        markers=True
    )

    st.plotly_chart(fig)