import streamlit as st
import requests
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Final
from model import train_and_predict_lstm

# --- CONSTANTS ---
BASE_URL: Final[str] = 'https://api.coingecko.com/api/v3/coins/markets'
HISTORY_URL: Final[str] = 'https://api.coingecko.com/api/v3/coins/{id}/market_chart'

# --- DATA CLASS ---
@dataclass()
class Coin:
    id: str
    name: str
    symbol: str
    current_price: float
    high_24h: float
    low_24h: float
    price_change_24h: float
    price_change_percentage_24h: float

# --- GET COINS ---
def get_coins(vs_currency: str = 'USD') -> list[Coin]:
    payload = {
        'vs_currency': vs_currency,
        'order': 'market_cap_desc',
        'page': 1
    }
    try:
        response = requests.get(BASE_URL, params=payload)
        response.raise_for_status()
        data = response.json()
        return [
            Coin(
                id=item['id'],
                name=item['name'],
                symbol=item['symbol'],
                current_price=item['current_price'],
                high_24h=item['high_24h'],
                low_24h=item['low_24h'],
                price_change_24h=item['price_change_24h'],
                price_change_percentage_24h=item['price_change_percentage_24h']
            ) for item in data
        ]
    except Exception as e:
        st.error(f"Error fetching coins: {e}")
        return []

# --- GET PRICE HISTORY ---
def get_price_history(coin_id: str, vs_currency: str = 'USD', days: int = 90) -> pd.DataFrame:
    url = HISTORY_URL.format(id=coin_id)
    params = {'vs_currency': vs_currency, 'days': days}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        prices = response.json()['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        st.error(f"Could not fetch history: {e}")
        return pd.DataFrame()

# --- STREAMLIT UI ---
st.set_page_config(page_title="Crypto Tracker with LSTM", layout="wide")
st.title("📈 Real-time Crypto Tracker + LSTM Forecast")

currency = st.sidebar.selectbox("Select Currency", ['USD', 'EUR'])
coins = get_coins(currency)

if coins:
    coin_names = [f"{coin.name} ({coin.symbol.upper()})" for coin in coins]
    selected_coin = st.sidebar.selectbox("Select Coin", coin_names)
    coin_obj = coins[coin_names.index(selected_coin)]

    st.subheader(f"{coin_obj.name} ({coin_obj.symbol.upper()})")
    st.metric("Current Price", f"{coin_obj.current_price:,.2f} {currency.upper()}")

    col1, col2, col3 = st.columns(3)
    col1.metric("24h High", f"{coin_obj.high_24h:,.2f}")
    col2.metric("24h Low", f"{coin_obj.low_24h:,.2f}")
    col3.metric("Change (24h)", f"{coin_obj.price_change_percentage_24h:,.2f} %")

    df_history = get_price_history(coin_obj.id, vs_currency=currency)

    if not df_history.empty:
        st.subheader("📊 Price History (Last 90 Days)")
        st.line_chart(df_history.set_index('timestamp')['price'])

        with st.expander("🔮 Show LSTM Forecast (Next 7 Days)"):
            with st.spinner("Training LSTM model..."):
                pred_df = train_and_predict_lstm(df_history)
                st.success("Prediction Completed!")

                st.subheader("📈 Forecast + History (Combined)")
                combined_df = pd.concat([
                    df_history[['timestamp', 'price']],
                    pred_df.rename(columns={'predicted_price': 'price'})
                ])
                st.line_chart(combined_df.set_index('timestamp'))

                st.subheader("📉 Forecast Only (Next 7 Days)")
                st.line_chart(pred_df.set_index('timestamp')['predicted_price'])
    else:
        st.info("No historical data available.")
else:
    st.error("No coins data available.")
