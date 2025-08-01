import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib, os
from .models import CandleData, PredictionLog
from .binance import get_latest_kline
from django.utils import timezone

MODEL_DIR = "models/"


def preprocess_data(df):
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['macd'] = ta.macd(df['close'])['MACD_12_26_9']
    df['ema12'] = ta.ema(df['close'], length=12)
    df['ema26'] = ta.ema(df['close'], length=26)
    df.dropna(inplace=True)
    return df


def load_model_and_scaler(interval):
    model = load_model(os.path.join(MODEL_DIR, f"model_{interval}.h5"))
    scaler = joblib.load(os.path.join(MODEL_DIR, f"scaler_{interval}.save"))
    return model, scaler


def update_candle_data(interval):
    kline = get_latest_kline("BTCUSDT", interval)
    if not kline:
        return
    ts = int(kline[0])
    if not CandleData.objects.filter(timestamp=ts, interval=interval).exists():
        CandleData.objects.create(
            interval=interval,
            timestamp=ts,
            open=float(kline[1]),
            high=float(kline[2]),
            low=float(kline[3]),
            close=float(kline[4]),
            volume=float(kline[5]),
        )


def predict_next(interval):
    update_candle_data(interval)
    candles = CandleData.objects.filter(interval=interval).order_by('-timestamp')[:100][::-1]
    df = pd.DataFrame(list(candles.values('open', 'high', 'low', 'close', 'volume')))
    df = preprocess_data(df)
    model, scaler = load_model_and_scaler(interval)
    scaled = scaler.transform(df)
    X = np.expand_dims(scaled[-50:], axis=0)
    prediction = model.predict(X)[0][0]

    last_row = df.iloc[-1]
    real_close = last_row['close']
    error = abs(real_close - prediction)
    timestamp = candles.last().timestamp

    PredictionLog.objects.create(
        interval=interval,
        predicted_close=prediction,
        real_close=real_close,
        error=error,
        timestamp=timestamp,
    )

    return {
        "predicted_close": prediction,
        "real_close": real_close,
        "error": error,
        "timestamp": timestamp
    }


def train_model_with_latest_data(interval):
    update_candle_data(interval)
    candles = CandleData.objects.filter(interval=interval).order_by('timestamp')
    df = pd.DataFrame(list(candles.values('open', 'high', 'low', 'close', 'volume')))
    df = preprocess_data(df)

    from sklearn.model_selection import train_test_split
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(50, len(scaled)):
        X.append(scaled[i-50:i])
        y.append(scaled[i][3])

    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    model.save(os.path.join(MODEL_DIR, f"model_{interval}.h5"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"scaler_{interval}.save"))