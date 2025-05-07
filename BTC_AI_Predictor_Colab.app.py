{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Nikhilkumar9117/Bitcoin-predictionapp.py/blob/main/BTC_AI_Predictor_Colab.app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "SSRmTaX9fSp6",
        "outputId": "b1b4c52f-4d74-4a54-d688-26d7e5360d74"
      },
      "source": [
        "# Install all required libraries\n",
        "!pip install -q yfinance ta scikit-learn tensorflow matplotlib streamlit pyngrok\n"
      ],
      "id": "SSRmTaX9fSp6",
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "  Preparing metadata (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m44.3/44.3 kB\u001b[0m \u001b[31m2.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m9.9/9.9 MB\u001b[0m \u001b[31m68.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m6.9/6.9 MB\u001b[0m \u001b[31m75.8 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m79.1/79.1 kB\u001b[0m \u001b[31m6.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25h  Building wheel for ta (setup.py) ... \u001b[?25l\u001b[?25hdone\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "N8GrBH_AfSp_",
        "outputId": "02df944a-2c68-4a82-d01c-cf9d8be0f7d5"
      },
      "source": [
        "%%writefile app.py\n",
        "import streamlit as st\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import yfinance as yf\n",
        "from ta.momentum import RSIIndicator\n",
        "from ta.trend import MACD, EMAIndicator\n",
        "from sklearn.preprocessing import MinMaxScaler\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from tensorflow.keras.models import Sequential\n",
        "from tensorflow.keras.layers import LSTM, Dense, Dropout\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "st.set_page_config(page_title=\"BTC AI Predictor Pro\", layout=\"wide\")\n",
        "\n",
        "@st.cache_resource\n",
        "def load_data():\n",
        "    df = yf.download('BTC-USD', period='1d', interval='1m')\n",
        "    if df.empty or 'Close' not in df.columns:\n",
        "        st.error(\"Failed to load BTC-USD data. Try again later.\")\n",
        "        return pd.DataFrame()\n",
        "    df.dropna(inplace=True)\n",
        "    try:\n",
        "        df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()\n",
        "        df['EMA20'] = EMAIndicator(df['Close'], window=20).ema_indicator()\n",
        "        macd = MACD(df['Close'])\n",
        "        df['MACD'] = macd.macd()\n",
        "    except Exception as e:\n",
        "        st.error(f\"Error calculating indicators: {e}\")\n",
        "        return pd.DataFrame()\n",
        "    df['Future_Close'] = df['Close'].shift(-10)\n",
        "    df['Target'] = (df['Future_Close'] > df['Close']).astype(int)\n",
        "    df.dropna(inplace=True)\n",
        "    return df\n",
        "\n",
        "@st.cache_resource\n",
        "def train_lstm(X, y):\n",
        "    model = Sequential([\n",
        "        LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),\n",
        "        Dropout(0.2),\n",
        "        LSTM(32),\n",
        "        Dropout(0.2),\n",
        "        Dense(1, activation='sigmoid')\n",
        "    ])\n",
        "    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])\n",
        "    model.fit(X, y, epochs=3, batch_size=32, verbose=0)\n",
        "    acc = model.evaluate(X, y, verbose=0)[1]\n",
        "    return model, acc\n",
        "\n",
        "@st.cache_resource\n",
        "def train_rf(X, y):\n",
        "    model = RandomForestClassifier(n_estimators=100)\n",
        "    model.fit(X, y)\n",
        "    acc = model.score(X, y)\n",
        "    return model, acc\n",
        "\n",
        "df = load_data()\n",
        "if df.empty:\n",
        "    st.stop()\n",
        "\n",
        "features = ['Close', 'RSI', 'Volume', 'MACD', 'EMA20']\n",
        "target = df['Target']\n",
        "\n",
        "st.title(\"BTC Dashboard with RSI, MACD, Volume\")\n",
        "chart_data = df.iloc[-30:]\n",
        "fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)\n",
        "axs[0].plot(chart_data.index, chart_data['Close'], color='orange')\n",
        "axs[0].set_title(\"BTC Price\")\n",
        "axs[1].plot(chart_data.index, chart_data['RSI'], color='purple')\n",
        "axs[1].axhline(70, color='red', linestyle='--')\n",
        "axs[1].axhline(30, color='green', linestyle='--')\n",
        "axs[1].set_title(\"RSI\")\n",
        "axs[2].bar(chart_data.index, chart_data['Volume'], width=0.0008)\n",
        "axs[2].set_title(\"Volume\")\n",
        "axs[3].plot(chart_data.index, chart_data['MACD'], color='blue')\n",
        "axs[3].set_title(\"MACD\")\n",
        "st.pyplot(fig)\n",
        "\n",
        "st.markdown(\"---\")\n",
        "st.header(\"AI Prediction + Accuracy Comparison\")\n",
        "scaler = MinMaxScaler()\n",
        "X_scaled = scaler.fit_transform(df[features])\n",
        "X_rf = X_scaled\n",
        "y_rf = target.values\n",
        "rf_model, rf_acc = train_rf(X_rf, y_rf)\n",
        "seq_len = 30\n",
        "X_lstm, y_lstm = [], []\n",
        "for i in range(seq_len, len(X_scaled)):\n",
        "    X_lstm.append(X_scaled[i-seq_len:i])\n",
        "    y_lstm.append(target.iloc[i])\n",
        "X_lstm = np.array(X_lstm)\n",
        "y_lstm = np.array(y_lstm)\n",
        "lstm_model, lstm_acc = train_lstm(X_lstm, y_lstm)\n",
        "latest_rf = scaler.transform(df[features].iloc[[-1]])\n",
        "rf_prob = rf_model.predict_proba(latest_rf)[0][1]\n",
        "latest_lstm = np.expand_dims(X_scaled[-seq_len:], axis=0)\n",
        "lstm_prob = lstm_model.predict(latest_lstm)[0][0]\n",
        "st.subheader(\"Prediction Results\")\n",
        "col1, col2 = st.columns(2)\n",
        "def interpret(prob):\n",
        "    direction = \"UP\" if prob > 0.5 else \"DOWN\"\n",
        "    action = \"BUY\" if direction == \"UP\" else \"SELL\"\n",
        "    return direction, action\n",
        "dir_rf, act_rf = interpret(rf_prob)\n",
        "dir_lstm, act_lstm = interpret(lstm_prob)\n",
        "with col1:\n",
        "    st.markdown(\"### Random Forest\")\n",
        "    st.write(f\"Prediction: **{dir_rf}**  \\nConfidence: {rf_prob:.2f}\")\n",
        "    st.info(f\"Suggested Action: **{act_rf}**\")\n",
        "    st.caption(f\"Train Accuracy: {rf_acc:.2f}\")\n",
        "with col2:\n",
        "    st.markdown(\"### LSTM (Deep Learning)\")\n",
        "    st.write(f\"Prediction: **{dir_lstm}**  \\nConfidence: {lstm_prob:.2f}\")\n",
        "    st.info(f\"Suggested Action: **{act_lstm}**\")\n",
        "    st.caption(f\"Train Accuracy: {lstm_acc:.2f}\")\n",
        "st.markdown(\"> Models use **Close, RSI, Volume, MACD, EMA** to predict BTC's next 10-minute move.\")\n"
      ],
      "id": "N8GrBH_AfSp_",
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Writing app.py\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import streamlit as st import pandas as pd import numpy as np import requests import yfinance as yf from ta.momentum import RSIIndicator from ta.trend import MACD, EMAIndicator from sklearn.preprocessing import MinMaxScaler from sklearn.ensemble import RandomForestClassifier from tensorflow.keras.models import Sequential from tensorflow.keras.layers import LSTM, Dense, Dropout import matplotlib.pyplot as plt import time\n",
        "\n",
        "st.set_page_config(page_title=\"BTC AI Predictor Pro\", layout=\"wide\") st.title(\"BTC Live Predictor with Auto-Refresh (10 min)\")\n",
        "\n",
        "@st.cache_data(ttl=600) def get_binance_data(): try: url = \"https://api.binance.com/api/v3/klines\" params = {\"symbol\": \"BTCUSDT\", \"interval\": \"1m\", \"limit\": 1440}  # 1 day = 1440 mins response = requests.get(url, params=params, timeout=10) data = response.json() df = pd.DataFrame(data, columns=[ 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'trades', 'tbbav', 'tbqav', 'ignore']) df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms') df.set_index('timestamp', inplace=True) df = df[['open', 'high', 'low', 'close', 'volume']].astype(float) return df except: return None\n",
        "\n",
        "@st.cache_data(ttl=600) def get_yfinance_data(): df = yf.download('BTC-USD', period='1d', interval='1m') if not df.empty: return df return None\n",
        "\n",
        "@st.cache_data(ttl=600) def load_data(): df = get_binance_data() if df is None or df.empty: st.warning(\"Binance data failed. Trying YFinance...\") df = get_yfinance_data() if df is None or df.empty: st.error(\"Failed to load BTC data from both sources.\") return pd.DataFrame()\n",
        "\n",
        "df['RSI'] = RSIIndicator(df['close'], window=14).rsi()\n",
        "df['EMA20'] = EMAIndicator(df['close'], window=20).ema_indicator()\n",
        "df['MACD'] = MACD(df['close']).macd()\n",
        "df['Future_Close'] = df['close'].shift(-10)\n",
        "df['Target'] = (df['Future_Close'] > df['close']).astype(int)\n",
        "df.dropna(inplace=True)\n",
        "return df\n",
        "\n",
        "def interpret(prob): direction = \"UP\" if prob > 0.5 else \"DOWN\" action = \"BUY\" if direction == \"UP\" else \"SELL\" return direction, action\n",
        "\n",
        "@st.cache_resource def train_lstm(X, y): model = Sequential([ LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])), Dropout(0.2), LSTM(32), Dropout(0.2), Dense(1, activation='sigmoid') ]) model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) model.fit(X, y, epochs=3, batch_size=32, verbose=0) acc = model.evaluate(X, y, verbose=0)[1] return model, acc\n",
        "\n",
        "@st.cache_resource def train_rf(X, y): model = RandomForestClassifier(n_estimators=100) model.fit(X, y) acc = model.score(X, y) return model, acc\n",
        "\n",
        "df = load_data() if df.empty: st.stop()\n",
        "\n",
        "features = ['close', 'RSI', 'volume', 'MACD', 'EMA20'] target = df['Target']\n",
        "\n",
        "Chart Section\n",
        "\n",
        "st.subheader(\"BTC Price, RSI, MACD, Volume\") chart_data = df.iloc[-30:] fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True) axs[0].plot(chart_data.index, chart_data['close'], color='orange') axs[0].set_title(\"BTC Price\") axs[1].plot(chart_data.index, chart_data['RSI'], color='purple') axs[1].axhline(70, color='red', linestyle='--') axs[1].axhline(30, color='green', linestyle='--') axs[1].set_title(\"RSI\") axs[2].bar(chart_data.index, chart_data['volume'], width=0.0008) axs[2].set_title(\"Volume\") axs[3].plot(chart_data.index, chart_data['MACD'], color='blue') axs[3].set_title(\"MACD\") st.pyplot(fig)\n",
        "\n",
        "AI Section\n",
        "\n",
        "st.header(\"AI Prediction + Accuracy\") scaler = MinMaxScaler() X_scaled = scaler.fit_transform(df[features]) X_rf = X_scaled y_rf = target.values rf_model, rf_acc = train_rf(X_rf, y_rf)\n",
        "\n",
        "LSTM reshape\n",
        "\n",
        "seq_len = 30 X_lstm, y_lstm = [], [] for i in range(seq_len, len(X_scaled)): X_lstm.append(X_scaled[i-seq_len:i]) y_lstm.append(target.iloc[i]) X_lstm = np.array(X_lstm) y_lstm = np.array(y_lstm) lstm_model, lstm_acc = train_lstm(X_lstm, y_lstm)\n",
        "\n",
        "Predict\n",
        "\n",
        "latest_rf = scaler.transform(df[features].iloc[[-1]]) rf_prob = rf_model.predict_proba(latest_rf)[0][1] latest_lstm = np.expand_dims(X_scaled[-seq_len:], axis=0) lstm_prob = lstm_model.predict(latest_lstm)[0][0] dir_rf, act_rf = interpret(rf_prob) dir_lstm, act_lstm = interpret(lstm_prob)\n",
        "\n",
        "col1, col2 = st.columns(2) with col1: st.markdown(\"### Random Forest\") st.write(f\"Prediction: {dir_rf}  \") st.write(f\"Confidence: {rf_prob:.2f}\") st.info(f\"Action: {act_rf}\") st.caption(f\"Train Accuracy: {rf_acc:.2f}\")\n",
        "\n",
        "with col2: st.markdown(\"### LSTM\") st.write(f\"Prediction: {dir_lstm}  \") st.write(f\"Confidence: {lstm_prob:.2f}\") st.info(f\"Action: {act_lstm}\") st.caption(f\"Train Accuracy: {lstm_acc:.2f}\")\n",
        "\n",
        "st.success(\"Auto-updating every 10 minutes with latest data. Stay tuned.\") st.caption(\"Uses Binance data first. Falls back to YFinance if needed.\")"
      ],
      "metadata": {
        "id": "HFmZ7E7Fn2pJ"
      },
      "id": "HFmZ7E7Fn2pJ",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "oGImvcBEfSqC",
        "outputId": "ee4b1d7c-142d-4d1e-9af1-2543cc106465"
      },
      "source": [
        "# Start Streamlit with ngrok (paste your token below)\n",
        "from pyngrok import ngrok\n",
        "import os\n",
        "# Start Streamlit with ngrok (paste your token below)\n",
        "from pyngrok import ngrok\n",
        "import os\n",
        "\n",
        "# Set your ngrok token\n",
        "os.environ[\"NGROK_AUTHTOKEN\"] = \"2wjX77p9lS4QC24z7Xq31k8P2kK_3ipyyRidJLB3MSWMjA2xi\"\n",
        "\n",
        "# Kill any previous Streamlit app\n",
        "!pkill streamlit\n",
        "\n",
        "# Start tunnel\n",
        "public_url = ngrok.connect(8501)\n",
        "print(\"Streamlit app running at:\", public_url)\n",
        "\n",
        "# Run Streamlit\n",
        "!streamlit run app.py &\n",
        "\n",
        "# Kill any previous Streamlit app\n",
        "!pkill streamlit\n",
        "\n",
        "# Start tunnel\n",
        "public_url = ngrok.connect(8501)\n",
        "print(\"Streamlit app running at:\", public_url)\n",
        "\n",
        "# Run Streamlit\n",
        "!streamlit run app.py &"
      ],
      "id": "oGImvcBEfSqC",
      "execution_count": null,
      "outputs": [
        {
          "metadata": {
            "tags": null
          },
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Streamlit app running at: NgrokTunnel: \"https://c585-34-16-199-239.ngrok-free.app\" -> \"http://localhost:8501\"\n",
            "\n",
            "Collecting usage statistics. To deactivate, set browser.gatherUsageStats to false.\n",
            "\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m\u001b[1m  You can now view your Streamlit app in your browser.\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m  Local URL: \u001b[0m\u001b[1mhttp://localhost:8501\u001b[0m\n",
            "\u001b[34m  Network URL: \u001b[0m\u001b[1mhttp://172.28.0.12:8501\u001b[0m\n",
            "\u001b[34m  External URL: \u001b[0m\u001b[1mhttp://34.16.199.239:8501\u001b[0m\n",
            "\u001b[0m\n",
            "2025-05-07 02:08:41.876580: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered\n",
            "WARNING: All log messages before absl::InitializeLog() is called are written to STDERR\n",
            "E0000 00:00:1746583721.907286    1458 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered\n",
            "E0000 00:00:1746583721.916451    1458 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered\n",
            "2025-05-07 02:08:41.944789: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.\n",
            "To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.\n",
            "YF.download() has changed argument auto_adjust default to True\n",
            "[*********************100%***********************]  1 of 1 completed\n",
            "\n",
            "1 Failed download:\n",
            "['BTC-USD']: YFRateLimitError('Too Many Requests. Rate limited. Try after a while.')\n"
          ]
        }
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "version": "3.11"
    },
    "colab": {
      "provenance": [],
      "include_colab_link": true
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}