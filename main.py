import numpy as np
import pandas as pd
import yfinance as yf
import websocket, json
from os import getenv
from time import sleep
from datetime import datetime, timezone

FINNHUB_API_KEY = getenv("FINNHUB_API_KEY")
FINNHUB_TICKER = "BINANCE:BTCUSDT"
YF_TICKER = "BTC-USD"

TRADING_DAYS_PER_YEAR = 252
N = 1000000 # number of simulations
r = 0.00414 # annualised drift (risk-free rate)
K = 100000 # strike price (USD)
expiry = datetime(2026,1,1, tzinfo=timezone.utc)
LAMBDA = 0.94 # how reactive is sigma
Z = np.random.normal(0, 1, N)

def time_to_maturity():
    now = datetime.now(timezone.utc)
    days = (expiry-now).days
    return max(days, 0) / TRADING_DAYS_PER_YEAR

def get_span(lambda_):
    return 2 / (1 - lambda_) - 1

def calculate_sigma(): # annualised volatility
    data = yf.download(YF_TICKER, period="1y")
    
    if isinstance(data.columns, pd.MultiIndex):
        close = data[("Close", YF_TICKER)]
    else:
        close = data["Close"]

    logr = np.log(close / close.shift(1)).dropna() # r_t = ln P_t/P_{t-1}
    sigma_daily = logr.ewm(span=get_span(LAMBDA)).std().iloc[-1]
    return sigma_daily * np.sqrt(TRADING_DAYS_PER_YEAR)
sigma = calculate_sigma()
    
# N simulated GBM future prices
def simulate_terminal_prices(S_0, T, sigma_):
    W_T = np.sqrt(T) * Z
    drift = (r - 0.5 * sigma_**2) * T
    return S_0 * np.exp(drift + sigma_*W_T)

def price_from_paths(S_T, T):
    payoffs = np.maximum(S_T - K, 0.0)
    discounted = np.exp(-r * T) * payoffs # risk-neutral valuation formula
    price = discounted.mean() # estimate of option fair price
    se = payoffs.std(ddof=1) / np.sqrt(N)
    ci_95 = (price - 1.96*se, price + 1.96*se)
    return price, se, ci_95

def mc_price(S_0, T, sigma_):
    S_T = simulate_terminal_prices(S_0, T, sigma_)
    price, _, _ = price_from_paths(S_T, T)
    return price

def delta_gamma(S_0, T, sigma_, h=None):
    # choose a relative step if not provided
    if h is None:
        h = max(1e-4 * S_0, 0.01)   # relative 0.01% or at least 0.01 USD
    p_plus  = mc_price(S_0 + h, T, sigma_)
    p_minus = mc_price(S_0 - h, T, sigma_)
    p0      = mc_price(S_0, T, sigma_) 
    delta = (p_plus - p_minus) / (2 * h)
    gamma = (p_plus - 2 * p0 + p_minus) / (h * h)
    return delta, gamma

def vega(S0, T, sigma_, h_sigma=None):
    # h_sigma in absolute vol (e.g. 0.01 = 1 vol point)
    if h_sigma is None:
        h_sigma = max(1e-4, 0.01 * sigma_)
    p_plus  = mc_price(S0, T, sigma_ + h_sigma)
    p_minus = mc_price(S0, T, sigma_ - h_sigma)
    vega = (p_plus - p_minus) / (2 * h_sigma)
    return vega

def main():
    print(FINNHUB_API_KEY)
    print_prices()

def print_prices():
    def on_message(ws, message):
        msg = json.loads(message)
        if msg.get("type") == "trade" and "data" in msg:
            trades = msg["data"]
            if trades:
                S_0 = trades[-1]["p"] # last available price
                now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                price, se, ci_95 = mc_price(S_0, T=time_to_maturity())
                l, r = ci_95
                print(f"{now}: ${price:.2f} +- ${1.96*se:.2f} (95% confidence interval: ${l:.2f}, ${r:.2f})")

    def on_open(ws):
        ws.send(json.dumps({"type": "subscribe", "symbol": FINNHUB_TICKER}))

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