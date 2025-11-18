import numpy as np
import pandas as pd
import yfinance as yf
import websocket, json
from os import getenv
from time import sleep
from datetime import datetime, timezone

FINNHUB_API_KEY = getenv("FINNHUB_API_KEY")

TICKER = "AAPL"
TRADING_DAYS_PER_YEAR = 252
N = 100000 # number of simulations
mu = 0.00414 # annualised drift (risk-free rate)
T = 1.0 # time horizon (years)
LAMBDA = 0.94 # how reactive is sigma

def get_span(lambda_):
    return 2 / (1 - lambda_) - 1

def calculate_sigma(): # annualised volatility
    data = yf.download(TICKER, period="1y")
    
    if isinstance(data.columns, pd.MultiIndex):
        close = data[("Close", TICKER)]
    else:
        close = data["Close"]

    logr = np.log(close / close.shift(1)).dropna() # r_t = ln P_t/P_{t-1}
    sigma_daily = logr.ewm(span=get_span(LAMBDA)).std().iloc[-1]
    sigma = sigma_daily * np.sqrt(TRADING_DAYS_PER_YEAR) 
    return sigma

sigma = calculate_sigma()
    

def calculate_S_T(S_0: float):
    Z = np.random.normal(0,1, size=N)
    W_T = np.sqrt(T) * Z

    return S_0 * np.exp((mu - 0.5* sigma**2)*T + sigma*W_T)

def main():
    print(FINNHUB_API_KEY)
    print_prices()

def print_prices():
    def on_message(ws, message):
        msg = json.loads(message)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{now}: Data received")

        if msg.get("type") == "trade" and "data" in msg:
            trades = msg["data"]
            if trades:
                S_0 = trades[-1]["p"] # last available price
                S_T = calculate_S_T(S_0)
                now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                print(f"{now}: S_0 = {S_0}, S_T = {S_T}")

    def on_open(ws):
        ws.send(json.dumps({"type": "subscribe", "symbol": "AAPL"}))

    def on_close(ws):
        print("closed connection")

    socket = f"wss://ws.finnhub.io?token={FINNHUB_API_KEY}"
    ws = websocket.WebSocketApp(socket, on_message=on_message, on_open=on_open, on_close=on_close)

    while True:
        try:
            ws.run_forever(ping_interval=5)    
        except Exception as e:
            print("Error: ", e)
            sleep(5)
            continue

if __name__ == "__main__":
    main()