```markdown
# LSTM-Based Stock Price Forecasting App

This Streamlit app predicts future stock prices using a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) layers. It is trained on historical stock data for Apple Inc. (AAPL), and allows users to experiment with different model hyperparameters and visualize the results.

---

## Live Demo

[Click to launch app on Streamlit Cloud](https://share.streamlit.io/your-username/lstm-stock-app/main/streamlit_lstm_app_modified.py)

---

## Model Overview

- **Model Type**: LSTM-based RNN
- **Target**: Forecasting future stock prices using historical `Close` prices
- **Best Hyperparameters** (preloaded on launch):
  - `window_size = 30`
  - `lstm_units = 50`
  - `batch_size = 16`
  - `dropout = 0.0`
  - `optimizer = adam`

---

## Features

- Hyperparameter tuning using interactive widgets
- Real-time evaluation metrics:
  - MAE, MAPE, RMSE, Train Loss, Validation Loss
-  Visualizations:
  - Training vs Validation Loss
  - Actual vs Predicted Stock Prices

---

## File Structure

```

lstm-stock-app/
├── best\_model.h5                   # Pretrained best model
├── M3-AAPL.csv                     # Historical stock price data (2019–2024)
├── streamlit\_lstm\_app\_modified.py # Main Streamlit app
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation

````

---

## Requirements

Install dependencies locally using:

```bash
pip install -r requirements.txt
````

---

## Running Locally

```bash
streamlit run streamlit_lstm_app_modified.py
```

---

## Deployment

This app is deployed on [Streamlit Community Cloud](https://streamlit.io/cloud).

To deploy your own:

1. Fork or clone this repo
2. Push your version to GitHub
3. Deploy via Streamlit Cloud → **New App**

---

## Contact

Created by **\Tejas Desai**
[Email](mailto:aimldstejas@gmail.com)
[LinkedIn](https://www.linkedin.com/in/tejasddesaiindia/)

---

## License

This project is licensed under the MIT License.

```
