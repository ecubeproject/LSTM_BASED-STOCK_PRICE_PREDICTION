Here is your complete `README.md` file with the **file structure image embedded correctly** while keeping everything else unchanged:

---

````markdown
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

![File Structure](https://raw.githubusercontent.com/ecubeproject/LSTM_BASED-STOCK_PRICE_PREDICTION/main/structure.png)

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
````

---

## 💻 Running the App Locally

```bash
streamlit run streamlit_lstm_app_modified.py
```

You can modify hyperparameters like `window_size`, `batch_size`, `lstm_units`, `optimizer`, and `dropout` directly from the UI.

---

## 🌐 Deploy on Streamlit Community Cloud

1. Push code to a public GitHub repo.
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Click “New App”.
4. Connect your repo and choose `streamlit_lstm_app_modified.py` as the app entry point.
5. Add `best_model.h5`, `M3-AAPL.csv`, and `requirements.txt` to repo.
6. Deploy!

---

## 📌 Notes

* This project uses **offline mode** for W\&B experiment tracking. Enable online logging by removing `wandb.init(mode='offline')`.
* Model training results and sweeps can be found in the `wandb/` directory.
* The pretrained model is already saved as `best_model.h5` and loaded on app start.

---

## 🧠 Author

**aimldstejas**

Connect via [GitHub](https://github.com/ecubeproject) for more ML projects.

```

---

Let me know if you'd like this saved as a `.md` file or want a PDF export too.
```
