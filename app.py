import streamlit as st
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from math import sqrt
os.environ['STREAMLIT_HOME'] = '/tmp'
os.environ['MPLCONFIGDIR'] = '/tmp'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

# Function to create sequences
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

# Load and preprocess data
df = pd.read_csv("data/M3-AAPL.csv")
df = df[['Close']]
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# Streamlit UI for hyperparameters
st.sidebar.title("LSTM Hyperparameters")
epochs = st.sidebar.slider("Epochs", 1, 100, 20)
batch_size = st.sidebar.slider("Batch Size", 1, 128, 32)
window_size = st.sidebar.slider("Window Size", 5, 100, 60)
units = st.sidebar.slider("LSTM Units", 10, 200, 50)
dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.2)

# Prepare data sequences
X, y = create_sequences(df_scaled, window_size)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split train/test
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# App title
st.title("LSTM Based Stock Price Predictor")

# Load pre-trained model if exists
model_path = "/tmp/model.h5"
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
    if os.path.exists(model_path):
        try:
            st.session_state.model = load_model(model_path)
            st.session_state.model_loaded = True
            st.success("Loaded pre-trained model from disk.")
        except Exception as e:
            st.error(f"Error loading model: {e}")

# Train on button click
if st.button("Run"):
    model = Sequential()
    model.add(LSTM(units, return_sequences=False, input_shape=(window_size, 1)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=0
    )

    os.makedirs("/tmp", exist_ok=True)
    model.save(model_path)

    st.session_state.model = model
    st.session_state.model_loaded = True
    st.success("Model trained and saved.")

    # Plot training vs validation loss
    st.subheader("Training vs Validation Loss")
    fig2, ax2 = plt.subplots()
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.legend()
    st.pyplot(fig2)

# Predictions and evaluation
if st.session_state.model_loaded:
    model = st.session_state.model
    predictions = model.predict(X_test)
    predictions_inv = scaler.inverse_transform(predictions)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Metrics
    mse = mean_squared_error(y_test_inv, predictions_inv)
    rmse = sqrt(mse)

    st.subheader("Performance Metrics")
    st.write(f"**MSE:** {mse:.4f}")
    st.write(f"**RMSE:** {rmse:.4f}")

    # Plot actual vs predicted
    st.subheader("Predicted vs Actual")
    fig, ax = plt.subplots()
    ax.plot(y_test_inv, label="Actual")
    ax.plot(predictions_inv, label="Predicted")
    ax.legend()
    st.pyplot(fig)
