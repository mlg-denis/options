import numpy as np, pandas as pd, yfinance as yf, websocket, json, threading, queue
from os import getenv
from time import sleep
from datetime import datetime, timezone

FINNHUB_API_KEY = getenv("FINNHUB_API_KEY")
FINNHUB_TICKER = "BINANCE:BTCUSDT"
YF_TICKER = "BTC-USD"

TRADING_DAYS_PER_YEAR = 252
N = 25000 # number of simulations
r = 0.00414 # annualised drift (risk-free rate)
K = 100000 # strike price (USD)
expiry = datetime(2026,1,1, tzinfo=timezone.utc)
LAMBDA = 0.94 # how reactive is sigma

price_queue = queue.Queue(maxsize=1)

def time_to_maturity():
    now = datetime.now(timezone.utc)
    days = (expiry-now).total_seconds() / 86400.0
    return max(days, 0.0) / TRADING_DAYS_PER_YEAR

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

def european_payoffs(paths):
    S_T = paths[:, -1]
    return np.maximum(S_T - K, 0.0)

def mc_price(price, T, sigma):
    steps = 50
    paths = simulate_paths(price, T, sigma, r, steps)
    payoffs = european_payoffs(paths)
    discounted = np.exp(-r * T) * payoffs
    return discounted.mean()

rng = np.random.default_rng()

def simulate_paths(S0, T, sigma, r, steps):      
    dt = T / steps
    sqrt_dt = np.sqrt(dt)

    # Generate random shocks (N paths × steps increments)
    Z = rng.standard_normal(size=(N, steps))

    # Precompute drift/vol terms
    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * sqrt_dt

    # Build log-increments
    increments = drift + vol * Z       # shape (N, steps)

    # Cumulative log-price
    log_paths = np.cumsum(increments, axis=1)   # shape (N, steps)

    # Allocate result array
    paths = np.empty((N, steps+1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(log_paths)

    return paths

def websocket_thread():
    def on_message(ws, message):
        msg = json.loads(message)
        if msg.get("type") == "trade" and "data" in msg:
            trades = msg["data"]
            if trades:
                price = trades[-1]["p"] # last available price
                if price_queue.full():
                    try:
                        price_queue.get_nowait()
                    except queue.Empty:
                        pass
                price_queue.put(price)        

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

def pricing_thread():
    while True:
        price = price_queue.get()
        T = time_to_maturity()
        est_price = mc_price(price, T, sigma)
        print(f"MC price = {est_price:.2f}")

if __name__ == "__main__":
    threading.Thread(target=websocket_thread, daemon=True).start()
    threading.Thread(target=pricing_thread, daemon=True).start()

    threading.Event().wait() # block forever to keep daemons alive