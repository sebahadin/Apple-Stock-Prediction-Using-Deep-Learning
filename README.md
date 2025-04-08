# 📈 Apple Stock Price Prediction using Deap Learning (LSTM)

This project demonstrates how to use a **Long Short-Term Memory (LSTM)** neural network to predict Apple (AAPL) stock prices. It includes a complete **Streamlit app** to visualize the model's performance and interactively adjust model parameters.

---

### 🚀 Features
- Predicts Apple stock closing prices using historical data.
- LSTM model built with PyTorch.
- Interactive web app built with Streamlit.
- Visual comparison of actual vs. predicted stock prices.
- Adjustable sequence length and training epochs.
- GPU support (if available).

---

### 📦 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/apple-stock-lstm.git
cd apple-stock-lstm
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the app:**
```bash
streamlit run app.py
```

---

### 📋 Requirements

- `torch`
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `streamlit`

Install with:
```bash
pip install torch pandas numpy matplotlib scikit-learn streamlit
```

---

### 🧠 How It Works

1. Downloads historical AAPL stock data from Plotly's GitHub.
2. Preprocesses the data (normalization, sequence generation).
3. Trains an LSTM model on the time series.
4. Evaluates the model and visualizes predictions.
5. Inverses normalization for real stock price plots.

---

### ⚙️ Parameters

You can customize the following via the sidebar in the app:
- `Sequence Length`: Number of past days to use for prediction.
- `Training Epochs`: Number of epochs to train the LSTM model.

---

### 📊 Example Outputs

- RMSE on test set after training
- Plot comparing predicted vs. actual prices (normalized and real scale)

---

### 📁 File Structure
```
.
├── app.py             # Streamlit web app
├── README.md          # Project description
└── requirements.txt   # Python dependencies
```

---

### 📌 To-Do
- [ ] Add model saving/loading support
- [ ] Forecast future prices
- [ ] Extend to other stocks or multi-feature inputs

---

### 🧑‍💻 Author

**Sebahadin Denur**  
NYU Abu Dhabi  
Feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/sebahadin-denur/) or contribute to the project!
