
# app.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ------------------ Page Config ------------------
st.set_page_config(page_title="Stock Price Forecasting with LSTM", layout="wide")

# ------------------ Sidebar Hyperparameter Inputs ------------------
st.sidebar.title("LSTM Hyperparameters")

window_size = st.sidebar.slider("Window Size", min_value=10, max_value=100, value=30, step=10)
lstm_units = st.sidebar.selectbox("LSTM Units", options=[32, 50, 64, 128], index=1)
batch_size = st.sidebar.selectbox("Batch Size", options=[16, 32, 64], index=0)
dropout = st.sidebar.slider("Dropout Rate", min_value=0.0, max_value=0.5, step=0.05, value=0.0)
optimizer = st.sidebar.selectbox("Optimizer", options=["adam", "rmsprop"], index=0)

# ------------------ Main Title ------------------
st.title("LSTM-Based Stock Price Predictor")

# ------------------ Load and preprocess data ------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/M3-AAPL.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    data = df[['Close']]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    return df, data, scaled, scaler

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

df, raw_data, scaled_data, scaler = load_data()

# ------------------ Run Model Button ------------------
if st.button("Run Model"):

    # Create sequences
    X, y = create_sequences(scaled_data, window_size)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build model
    model = Sequential()
    model.add(LSTM(lstm_units, return_sequences=True, input_shape=(window_size, 1)))
    model.add(Dropout(dropout))
    model.add(LSTM(lstm_units))
    model.add(Dropout(dropout))
    model.add(Dense(1))

    opt = Adam() if optimizer == "adam" else RMSprop()
    model.compile(optimizer=opt, loss='mean_squared_error')

    # Train
    history = model.fit(X_train, y_train, epochs=30, batch_size=batch_size, validation_split=0.2, verbose=0)

    # Predict and inverse transform
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Evaluate
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mae = mean_absolute_error(y_actual, y_pred)
    mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
    train_loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]

    # Plot 1: Training vs Validation Loss
    st.subheader("Training vs Validation Loss")
    fig1, ax1 = plt.subplots(figsize=(4, 2))
    ax1.plot(history.history['loss'], label="Training Loss")
    ax1.plot(history.history['val_loss'], label="Validation Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    st.pyplot(fig1)

    # Plot 2: Actual vs Predicted Prices
    st.subheader("Actual vs Predicted Stock Prices")
    fig2, ax2 = plt.subplots(figsize=(4, 2))
    ax2.plot(y_actual, label="Actual Price")
    ax2.plot(y_pred, label="Predicted Price")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Stock Price")
    ax2.legend()
    st.pyplot(fig2)

    # Display metrics
    st.markdown('<style>div[data-testid="metric-container"] {font-size: 60% !important; margin: 0 10px 0 0;}</style>', unsafe_allow_html=True)
    st.subheader("Evaluation Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:.4f}")
    col2.metric("MAPE", f"{mape:.2f}%")
    col3.metric("RMSE", f"{rmse:.4f}")

    col4, col5 = st.columns(2)
    col4.metric("Train Loss", f"{train_loss:.6f}")
    col5.metric("Validation Loss", f"{val_loss:.6f}")
