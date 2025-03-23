import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
import plotly.express as px
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add Logo
st.image("logo.png", width=150)


# Generate synthetic time-series energy data
np.random.seed(42)
days = 365  # One year of data
hours = days * 24

time = np.arange(hours)
temperature = 20 + 10 * np.sin(2 * np.pi * time / 8760) + np.random.normal(0, 2, hours)  # Seasonal temp variation
demand = 500 + 50 * np.sin(2 * np.pi * time / 24) + np.random.normal(0, 10, hours)  # Daily energy demand pattern

# Simulate solar power production
def simulate_solar_power(time, max_capacity=1000):
    # Solar power is highest at midday (12 PM) and zero at night
    solar_power = max_capacity * np.sin(2 * np.pi * (time % 24) / 24)
    solar_power = np.where(solar_power < 0, 0, solar_power)  # No solar power at night
    return solar_power

# Simulate wind power production
def simulate_wind_power(time, max_capacity=800):
    # Wind power varies randomly but follows a daily pattern
    wind_power = max_capacity * np.random.weibull(2, size=len(time))
    return wind_power

# Simulate fossil fuel power production
def simulate_fossil_fuel_power(time, base_capacity=500):
    # Fossil fuel power is relatively constant with minor fluctuations
    fossil_fuel_power = base_capacity + 50 * np.random.randn(len(time))
    return fossil_fuel_power

# Simulate power production for each source
solar_power = simulate_solar_power(time)
wind_power = simulate_wind_power(time)
fossil_fuel_power = simulate_fossil_fuel_power(time)

# Total power production
total_power_production = solar_power + wind_power + fossil_fuel_power

# Simulate energy supply and consumption
supply = total_power_production * 0.9  # Assume supply is 90% of total production
consumption = demand * (1 + 0.05 * np.random.randn(hours))  # Simulated energy consumption

# Simulate transmission loss
transmission_loss = total_power_production * 0.05  # Assume 5% transmission loss

# Create a DataFrame
data = pd.DataFrame({
    'time': time,
    'temperature': temperature,
    'demand': demand,
    'supply': supply,
    'consumption': consumption,
    'solar_power': solar_power,
    'wind_power': wind_power,
    'fossil_fuel_power': fossil_fuel_power,
    'total_power_production': total_power_production,
    'transmission_loss': transmission_loss
})

# Split dataset into training and testing
X = data[['time', 'temperature', 'demand']]
y = data['consumption']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Prepare data for LSTM
X_lstm = np.array(X).reshape(X.shape[0], X.shape[1], 1)
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y, test_size=0.2, random_state=42)

# Build LSTM model
lstm_model = Sequential([
    LSTM(50, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=32, verbose=1)

def predict_energy(model, time, temperature, demand):
    input_data = np.array([[time, temperature, demand]])
    if model == "Linear Regression":
        return lr_model.predict(input_data)[0]
    elif model == "Random Forest":
        return rf_model.predict(input_data)[0]
    elif model == "LSTM":
        input_data = input_data.reshape(1, 3, 1)
        return lstm_model.predict(input_data)[0][0]
    return None

# Streamlit App
st.title("Energy Consumption Prediction & Visualization")

# Model Selection
model_choice = st.selectbox("Select Model", ["Linear Regression", "Random Forest", "LSTM"])
time_input = st.number_input("Time (hour)", min_value=0, max_value=hours, value=0)
temperature_input = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0)
demand_input = st.number_input("Energy Demand", min_value=0.0, max_value=1000.0, value=500.0)

if st.button("Predict Consumption"):
    prediction = predict_energy(model_choice, time_input, temperature_input, demand_input)
    st.write(f"Predicted Energy Consumption: {prediction:.2f}")

# Button to display power production and transmission loss
if st.button("Show Power Production and Transmission Loss"):
    # Get the row corresponding to the selected time
    selected_data = data[data['time'] == time_input]
    
    if not selected_data.empty:
        solar = selected_data['solar_power'].values[0]
        wind = selected_data['wind_power'].values[0]
        fossil_fuel = selected_data['fossil_fuel_power'].values[0]
        total_production = selected_data['total_power_production'].values[0]
        transmission_loss = selected_data['transmission_loss'].values[0]
        
        st.write(f"Solar Power at Time {time_input}: {solar:.2f} kWh")
        st.write(f"Wind Power at Time {time_input}: {wind:.2f} kWh")
        st.write(f"Fossil Fuel Power at Time {time_input}: {fossil_fuel:.2f} kWh")
        st.write(f"Total Power Production at Time {time_input}: {total_production:.2f} kWh")
        st.write(f"Transmission Loss at Time {time_input}: {transmission_loss:.2f} kWh")
    else:
        st.write("No data available for the selected time.")



# Data Visualization Dashboard
st.subheader("Energy Demand & Supply Trends")
fig1 = px.line(data.iloc[:100], x='time', y=['demand', 'supply'], 
              labels={'value': "Energy (kWh)", 'time': "Time (hours)"},
              title="Energy Demand and Supply Over Time")
st.plotly_chart(fig1)

st.subheader("Energy Consumption Trends")
fig2 = px.line(data.iloc[:100], x='time', y=['consumption'], 
              labels={'value': "Energy (kWh)", 'time': "Time (hours)"},
              title="Energy Consumption Over Time")
st.plotly_chart(fig2)

st.subheader("Power Production Trends")
fig3 = px.line(data.iloc[:100], x='time', y=['solar_power', 'wind_power', 'fossil_fuel_power', 'total_power_production'], 
              labels={'value': "Energy (kWh)", 'time': "Time (hours)"},
              title="Power Production Over Time")
st.plotly_chart(fig3)