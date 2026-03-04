# ==============================
# Smart AQI Forecast Training
# ==============================

import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("air_quality.csv")

# Convert Date
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values(['City', 'Date'], inplace=True)

# Extract time features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# Remove rows without PM2.5
df = df.dropna(subset=['PM2.5'])

# Fill other missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Create lag features
df['PM2.5_lag1'] = df.groupby('City')['PM2.5'].shift(1)
df['PM2.5_roll3'] = df.groupby('City')['PM2.5'].rolling(3).mean().reset_index(0,drop=True)

df = df.dropna()

# Encode city
le = LabelEncoder()
df['City_encoded'] = le.fit_transform(df['City'])

# Features
features = ['City_encoded', 'Year', 'Month', 'Day', 'PM2.5_lag1', 'PM2.5_roll3']
X = df[features]
y = df['PM2.5']

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(le, open("encoder.pkl", "wb"))

print("Model trained & saved successfully 💙")