# LSTM-Based Stock Price Prediction

This project leverages a Long Short-Term Memory (LSTM) neural network to predict future stock prices based on historical time series data. It includes hyperparameter tuning, experiment tracking with Weights & Biases (W&B), and a deployable Streamlit application.

---

## 🚀 Project Features

- LSTM-based time series forecasting  
- Hyperparameter tuning using W&B Sweeps  
- Experiment tracking with W&B (offline and online modes)  
- Streamlit app for interactive prediction and visualization  
- Pretrained model for immediate demo  
- Easy deployment on Streamlit Community Cloud

---

## 📁 File Structure

![File Structure](https://raw.githubusercontent.com/ecubeproject/LSTM_BASED-STOCK_PRICE_PREDICTION/main/file_structure.png)

---

## 📊 Sample Evaluation Metrics

| Metric       | Value    |
|--------------|----------|
| RMSE         | 3.90     |
| MAE          | 2.95     |
| MAPE (%)     | 1.63     |
| Train Loss   | 0.00065  |
| Val Loss     | 0.00091  |

---

## 🧪 Dependencies

Install using:

```bash
pip install -r requirements.txt
```

---

## 💻 Running the App Locally

```bash
streamlit run streamlit_lstm_app_modified.py
```

You can modify hyperparameters like `window_size`, `batch_size`, `lstm_units`, `optimizer`, and `dropout` directly from the UI.

---

## 🧠 Author

**aimldstejas**

Connect via [GitHub](https://github.com/ecubeproject) for more ML projects.
