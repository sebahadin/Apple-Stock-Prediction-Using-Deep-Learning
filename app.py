import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# For matplotlib plotting inside Streamlit
import warnings
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)

# GPU Info
st.title("📈 Apple Stock Price Prediction (LSTM)")
st.write("Predicting AAPL stock prices using a trained LSTM model")

if torch.cuda.is_available():
    st.success(f"GPU Available: {torch.cuda.get_device_name(0)}")
else:
    st.warning("Running on CPU. GPU not available.")

# Load data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv"
    df = pd.read_csv(url)
    df = df[['Date', 'AAPL.Close']].dropna()
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()
st.subheader("Apple Stock Data Sample")
st.dataframe(df.tail())

# Plot original data
st.subheader("Apple Closing Price Over Time")
plt.figure(figsize=(10,4))
plt.plot(df['Date'], df['AAPL.Close'], color='blue')
plt.xlabel("Date")
plt.ylabel("AAPL.Close")
plt.title("Apple Stock Closing Price")
st.pyplot()

# Normalize
data = df[['AAPL.Close']].values
scaler = MinMaxScaler(feature_range=(-1, 1))
data_scaled = scaler.fit_transform(data)

# Sequence creation
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

SEQ_LENGTH = st.sidebar.slider("Sequence Length", min_value=10, max_value=100, value=50)
X, y = create_sequences(data_scaled, SEQ_LENGTH)

# Train-test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_t = torch.tensor(X_train).to(device)
y_train_t = torch.tensor(y_train).to(device)
X_test_t = torch.tensor(X_test).to(device)
y_test_t = torch.tensor(y_test).to(device)

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

model = LSTMModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training
EPOCHS = st.sidebar.slider("Training Epochs", 50, 300, 200)
train_button = st.button("🚀 Train the Model")

if train_button:
    progress = st.progress(0)
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            st.write(f"Epoch {epoch}/{EPOCHS}, Loss: {loss.item():.5f}")
        progress.progress((epoch+1) / EPOCHS)
    
    st.success("Training complete!")

    # Evaluation
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).cpu().numpy()
        y_true = y_test_t.cpu().numpy()
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        st.subheader(f"📊 Test RMSE: {rmse:.5f}")

        # Plot predictions (normalized)
        st.subheader("Normalized Predictions vs Actual")
        plt.figure(figsize=(10, 4))
        plt.plot(y_true, label='Actual', color='blue')
        plt.plot(preds, label='Predicted', color='red')
        plt.legend()
        st.pyplot()

        # Inverse transform
        preds_orig = scaler.inverse_transform(preds)
        y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Plot predictions (original scale)
        st.subheader("Predicted vs Actual Stock Price (Original Scale)")
        plt.figure(figsize=(10, 4))
        plt.plot(y_test_orig, label='Actual Price', color='blue')
        plt.plot(preds_orig, label='Predicted Price', color='red')
        plt.legend()
        plt.xlabel("Time Step")
        plt.ylabel("Price ($)")
        st.pyplot()
