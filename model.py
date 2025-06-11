import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def prepare_data_for_lstm(df: pd.DataFrame, look_back: int = 30):
    df_price = df.set_index('timestamp')['price'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df_price)

    X, y = [], []
    for i in range(look_back, len(df_scaled)):
        X.append(df_scaled[i - look_back:i, 0])
        y.append(df_scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

def train_and_predict_lstm(df: pd.DataFrame, days_to_predict: int = 7, look_back: int = 30):
    X, y, scaler = prepare_data_for_lstm(df, look_back)

    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=16, verbose=0)

    last_sequence = X[-1]
    predictions = []
    for _ in range(days_to_predict):
        pred = model.predict(last_sequence.reshape(1, look_back, 1), verbose=0)
        predictions.append(pred[0][0])
        last_sequence = np.append(last_sequence[1:], pred[0][0])
        last_sequence = last_sequence.reshape(look_back, 1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    future_dates = pd.date_range(start=df['timestamp'].iloc[-1] + pd.Timedelta(days=1), periods=days_to_predict)
    prediction_df = pd.DataFrame({'timestamp': future_dates, 'predicted_price': predictions.flatten()})
    return prediction_df
