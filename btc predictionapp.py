streamlit==1.35.0
pandas==2.2.2
numpy==1.26.4
yfinance==0.2.37
ta==0.11.0
scikit-learn==1.4.2
tensorflow==2.15.0
matplotlib==3.8.4
requests==2.31.0
import streamlit as st import pandas as pd import numpy as np import requests import yfinance as yf from ta.momentum import RSIIndicator from ta.trend import MACD, EMAIndicator from sklearn.preprocessing import MinMaxScaler from sklearn.ensemble import RandomForestClassifier from tensorflow.keras.models import Sequential from tensorflow.keras.layers import LSTM, Dense, Dropout import matplotlib.pyplot as plt import time

st.set_page_config(page_title="BTC AI Predictor Pro", layout="wide") st.title("BTC Live Predictor with Auto-Refresh (10 min)")

@st.cache_data(ttl=600) def get_binance_data(): try: url = "https://api.binance.com/api/v3/klines" params = {"symbol": "BTCUSDT", "interval": "1m", "limit": 1440}  # 1 day = 1440 mins response = requests.get(url, params=params, timeout=10) data = response.json() df = pd.DataFrame(data, columns=[ 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'trades', 'tbbav', 'tbqav', 'ignore']) df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms') df.set_index('timestamp', inplace=True) df = df[['open', 'high', 'low', 'close', 'volume']].astype(float) return df except: return None

@st.cache_data(ttl=600) def get_yfinance_data(): df = yf.download('BTC-USD', period='1d', interval='1m') if not df.empty: return df return None

@st.cache_data(ttl=600) def load_data(): df = get_binance_data() if df is None or df.empty: st.warning("Binance data failed. Trying YFinance...") df = get_yfinance_data() if df is None or df.empty: st.error("Failed to load BTC data from both sources.") return pd.DataFrame()

df['RSI'] = RSIIndicator(df['close'], window=14).rsi()
df['EMA20'] = EMAIndicator(df['close'], window=20).ema_indicator()
df['MACD'] = MACD(df['close']).macd()
df['Future_Close'] = df['close'].shift(-10)
df['Target'] = (df['Future_Close'] > df['close']).astype(int)
df.dropna(inplace=True)
return df

def interpret(prob): direction = "UP" if prob > 0.5 else "DOWN" action = "BUY" if direction == "UP" else "SELL" return direction, action

@st.cache_resource def train_lstm(X, y): model = Sequential([ LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])), Dropout(0.2), LSTM(32), Dropout(0.2), Dense(1, activation='sigmoid') ]) model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) model.fit(X, y, epochs=3, batch_size=32, verbose=0) acc = model.evaluate(X, y, verbose=0)[1] return model, acc

@st.cache_resource def train_rf(X, y): model = RandomForestClassifier(n_estimators=100) model.fit(X, y) acc = model.score(X, y) return model, acc

df = load_data() if df.empty: st.stop()

features = ['close', 'RSI', 'volume', 'MACD', 'EMA20'] target = df['Target']

Chart Section

st.subheader("BTC Price, RSI, MACD, Volume") chart_data = df.iloc[-30:] fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True) axs[0].plot(chart_data.index, chart_data['close'], color='orange') axs[0].set_title("BTC Price") axs[1].plot(chart_data.index, chart_data['RSI'], color='purple') axs[1].axhline(70, color='red', linestyle='--') axs[1].axhline(30, color='green', linestyle='--') axs[1].set_title("RSI") axs[2].bar(chart_data.index, chart_data['volume'], width=0.0008) axs[2].set_title("Volume") axs[3].plot(chart_data.index, chart_data['MACD'], color='blue') axs[3].set_title("MACD") st.pyplot(fig)

AI Section

st.header("AI Prediction + Accuracy") scaler = MinMaxScaler() X_scaled = scaler.fit_transform(df[features]) X_rf = X_scaled y_rf = target.values rf_model, rf_acc = train_rf(X_rf, y_rf)

LSTM reshape

seq_len = 30 X_lstm, y_lstm = [], [] for i in range(seq_len, len(X_scaled)): X_lstm.append(X_scaled[i-seq_len:i]) y_lstm.append(target.iloc[i]) X_lstm = np.array(X_lstm) y_lstm = np.array(y_lstm) lstm_model, lstm_acc = train_lstm(X_lstm, y_lstm)

Predict

latest_rf = scaler.transform(df[features].iloc[[-1]]) rf_prob = rf_model.predict_proba(latest_rf)[0][1] latest_lstm = np.expand_dims(X_scaled[-seq_len:], axis=0) lstm_prob = lstm_model.predict(latest_lstm)[0][0] dir_rf, act_rf = interpret(rf_prob) dir_lstm, act_lstm = interpret(lstm_prob)

col1, col2 = st.columns(2) with col1: st.markdown("### Random Forest") st.write(f"Prediction: {dir_rf}  ") st.write(f"Confidence: {rf_prob:.2f}") st.info(f"Action: {act_rf}") st.caption(f"Train Accuracy: {rf_acc:.2f}")

with col2: st.markdown("### LSTM") st.write(f"Prediction: {dir_lstm}  ") st.write(f"Confidence: {lstm_prob:.2f}") st.info(f"Action: {act_lstm}") st.caption(f"Train Accuracy: {lstm_acc:.2f}")

st.success("Auto-updating every 10 minutes with latest data. Stay tuned.") st.caption("Uses Binance data first. Falls back to YFinance if needed.")
