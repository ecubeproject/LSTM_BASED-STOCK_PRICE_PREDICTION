# LSTM-Based Stock Price Predictor

Welcome to the **LSTM Stock Price Predictor** hosted on [Hugging Face Spaces](https://huggingface.co/spaces/minusquare/lstm-stock-price-predictor).  
This interactive Streamlit app predicts future stock prices using a Long Short-Term Memory (LSTM) neural network trained on historical time series data.

---

## Key Features

- LSTM-based time series forecasting  
- Interactive Streamlit UI with slider-controlled hyperparameters  
- Reproducible training with adjustable epochs, window size, batch size, dropout, etc.  
- Live visualization of training & validation loss  
- Real-time prediction vs actual price plotting  
- Automatically loads pretrained model for demo on app launch

---

## App Directory Structure

![File Structure](https://raw.githubusercontent.com/ecubeproject/LSTM_BASED-STOCK_PRICE_PREDICTION/main/file_structure.png)

---

---

## 🔬 Sample Evaluation Metrics

| Metric       | Value    |
|--------------|----------|
| RMSE         | ~3.90    |
| MAE          | ~2.95    |
| Train Loss   | ~0.00065 |
| Val Loss     | ~0.00091 |

---

##  Try it Out

Launch the app:  
👉 [Open in Hugging Face Spaces](https://huggingface.co/spaces/minusquare/lstm-stock-price-predictor)

You can change hyperparameters on the left sidebar and click **Run** to retrain the model.

---

##  Author

Built by **Tejas Desai**  
Connect via [GitHub](https://github.com/ecubeproject) for more AI/ML projects.

---

## Tech Stack

- Python, TensorFlow/Keras
- Streamlit for interactive UI
- Docker (for custom environment on HF)
- Hugging Face Spaces for deployment

---

## 💬 Note

This app runs fully in the browser on CPU. For best performance and reproducibility, use the recommended defaults or pretrained model (loaded on app launch).

