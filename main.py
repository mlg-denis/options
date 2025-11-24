import numpy as np, pandas as pd, yfinance as yf, websocket, json, threading, queue, math
from os import getenv
from time import sleep
from datetime import datetime, timezone

FINNHUB_API_KEY = getenv("FINNHUB_API_KEY")
FINNHUB_TICKER = "BINANCE:BTCUSDT"
YF_TICKER = "BTC-USD"

TRADING_DAYS_PER_YEAR = 365 # this is the case for BTC. change for instruments on NYSE for example, using pandas_market_calendars
N = 50000 # number of simulations
BATCH_SIZE = 50000
STEPS = 50
r = 0.00414 # annualised drift (risk-free rate)
K = 100000 # strike price
EXPIRY = datetime(2026,1,1, tzinfo=timezone.utc)

OPTION_TYPE = "EUROPEAN CALL"

price_queue = queue.Queue(maxsize=1)

def time_to_maturity():
    now = datetime.now(timezone.utc)
    seconds = max((EXPIRY - now).total_seconds(), 0.0)
    return seconds / (TRADING_DAYS_PER_YEAR * 24 * 3600)

def calculate_sigma(): # annualised volatility
    data = yf.download(YF_TICKER, period="1y", progress=False)
    LAMBDA = 0.94 # how reactive is sigma
    
    if isinstance(data.columns, pd.MultiIndex):
        close = data[("Close", YF_TICKER)]
    else:
        close = data["Close"]

    span = 2 / (1 - LAMBDA) - 1
    logr = np.log(close / close.shift(1)).dropna() # r_t = ln P_t/P_{t-1}
    sigma_daily = logr.ewm(span=span).std().iloc[-1]
    return sigma_daily * np.sqrt(TRADING_DAYS_PER_YEAR)

def payoff_european_call(ST):
    return np.maximum(ST - K, 0.0)

def payoff_european_put(ST):
    return np.maximum(K - ST, 0.0)

def payoff_asian_call(paths):
    avg_price = paths[:, 1:].mean(axis=1)
    return np.maximum(avg_price - K, 0.0)

def payoff_asian_call_geometric(paths):
    # using the exp(log(_)) trick guards against underflow/overflow
    log_avg_price = np.log(paths[:,1:]).mean(axis=1)
    return np.maximum(np.exp(log_avg_price) - K, 0.0)

def payoff_asian_put(paths):
    avg_price = paths[:, 1:].mean(axis=1)
    return np.maximum(K - avg_price, 0.0)

def payoff_asian_put_geometric(paths):
    # using the exp(log(_)) trick guards against underflow/overflow
    log_avg_price = np.log(paths[:,1:]).mean(axis=1)
    return np.maximum(K - np.exp(log_avg_price), 0.0)

rng = np.random.default_rng(seed=0)

def simulate_paths(S0, drift, vol, Z_batch):
    increments = drift + vol * Z_batch
    log_paths = np.cumsum(increments, axis=1)
    # allocate result array
    paths = np.empty((Z_batch.shape[0], STEPS+1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(log_paths)
    return paths

def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def black_scholes(S0, T, sigma, option_type):
    sqrtT = math.sqrt(T)
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    Nd1, Nd2 = norm_cdf(d1), norm_cdf(d2)

    if option_type == "EUROPEAN CALL":
        return S0 * Nd1 - K * math.exp(-r * T) * Nd2
    elif option_type == "EUROPEAN PUT":
        return K * math.exp(-r * T) * (1 - Nd2) - S0 * (1 - Nd1)
    else:
        raise ValueError("Option type must be EUROPEAN CALL or EUROPEAN PUT to use Black-Scholes closed form.")

def black_scholes_greeks(S0, T, sigma, option_type):
    sqrtT = math.sqrt(T)
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    Nd1, Nd2 = norm_cdf(d1), norm_cdf(d2)

    delta_c = Nd1
    gamma = math.exp(-0.5 * d1 * d1) / (S0 * sigma * sqrtT * math.sqrt(2 * math.pi))
    vega = S0 * math.exp(-0.5 * d1 * d1) * sqrtT / math.sqrt(2 * math.pi)
    theta_c = -(S0 * sigma * math.exp(-0.5 * d1 * d1)) / (2 * sqrtT * math.sqrt(2 * math.pi)) - r * K * math.exp(-r * T) * Nd2
    rho_c = K * T * math.exp(-r * T) * Nd2

    if option_type == "EUROPEAN CALL":
        return (delta_c, gamma, vega, theta_c, rho_c)
    elif option_type == "EUROPEAN PUT":
        delta_p = delta_c - 1
        theta_p = theta_c + r * K * math.exp(-r * T)
        rho_p = rho_c - T * K * math.exp(-r * T)
        return (delta_p, gamma, vega, theta_p, rho_p)
    else:
        raise ValueError("Option type must be EUROPEAN CALL or EUROPEAN PUT to use Black-Scholes closed form.")

def geometric_asian_price(S0, T, sigma, option_type):
    sigma_ = sigma * math.sqrt((2*STEPS + 1) / (6*(STEPS + 1)))
    mu_ = 0.5 * (r - 0.5*sigma*sigma) + 0.5*sigma_*sigma_

    sqrtT = math.sqrt(T)
    d1 = (math.log(S0 / K) + (mu_ + 0.5*sigma_*sigma_) * T) / (sigma_ * sqrtT)
    d2 = d1 - sigma_ * sqrtT

    Nd1, Nd2 = norm_cdf(d1), norm_cdf(d2)

    discount = math.exp(-r * T)

    if option_type == "ASIAN CALL":
        return S0 * math.exp(mu_*T) * Nd1 - K * discount * Nd2
    elif option_type == "ASIAN PUT":
        return K * discount * (1-Nd2) - S0 * math.exp(mu_*T) * (1-Nd1)
    else:
        raise ValueError("Option type must be ASIAN CALL or ASIAN PUT to use geometric Asian control variate")        

def mc_path_dependent(S0, T, sigma, payoff_X, payoff_Y = None, E_Y = None):
    discount = np.exp(-r * T)
    total = 0.0
    total_sq = 0.0
    total_N = 0
   
    dt = T / STEPS
    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)

    use_cv = payoff_Y is not None and E_Y is not None # cv - control variate

    if use_cv:
        # run a smaller MC to estimate control variate correlation coefficient
        m = max(5000, BATCH_SIZE // 2)
        half = (m+1)//2
        Z_half = rng.standard_normal((half, STEPS))
        Z = np.vstack((Z_half, -Z_half))[:m]
        paths = simulate_paths(S0, drift, vol, Z) # Z is small enough to reasonably calculate all its paths in one go

        X = payoff_X(paths)
        Y = payoff_Y(paths)
        var_Y = np.var(Y, ddof=1)
        
        beta = np.cov(X, Y, ddof=1)[0,1] / var_Y
    else:
        beta = 0.0    


    while total_N < N:
        m = min(BATCH_SIZE, N-total_N)

        # antithetic variates
        half = (m+1) // 2
        Z_half = rng.standard_normal((half, STEPS))
        Zb = np.vstack((Z_half, -Z_half))[:m]

        paths = simulate_paths(S0, drift, vol, Zb)
        X = payoff_X(paths)

        if use_cv:
            Y = payoff_Y(paths)
            adjusted = X - beta*(Y - E_Y)
        else:
            adjusted = X    

        total += adjusted.sum()
        total_sq += (adjusted*adjusted).sum()
        total_N += m

    mean = total / total_N
    var = max(0.0, (total_sq / total_N) - mean**2) # to clamp floating point errors
    se = np.sqrt(var / total_N)

    price = discount * mean
    se_price = discount * se
    return price, se_price

def mc_terminal_only(S0, T, sigma, payoff_X):
    discount = np.exp(-r * T)
    total_payoff = 0.0
    total_sq = 0.0
    total_N = 0

    drift = (r - 0.5 * sigma**2) * T
    vol = sigma * np.sqrt(T)

    while total_N < N:
        m = min(BATCH_SIZE, N-total_N)

        # antithetic variates
        half = (m+1) // 2
        Z_half = rng.standard_normal(half)
        Zb = np.concatenate([Z_half, -Z_half])[:m]

        x = drift + vol * Zb
        ST = S0 * np.exp(x)
        payoffs = payoff_X(ST)

        total_payoff += payoffs.sum()
        total_sq += (payoffs ** 2).sum()
        total_N += m

    mean_payoff = total_payoff / total_N
    var_payoff = max(0.0, (total_sq / total_N) - mean_payoff**2)
    se = np.sqrt(var_payoff / total_N)

    price = discount * mean_payoff
    se_price = discount * se
    return price, se_price

def mc_option_price(S0, T, sigma, option_type):
    o = option_type.upper()
    match o:
        case "EUROPEAN CALL":
            price, se = black_scholes(S0, T, sigma, "EUROPEAN CALL"), 0.0
            greeks = black_scholes_greeks(S0, T, sigma, "EUROPEAN CALL")
            return price, se, greeks
        case "EUROPEAN PUT":
            price, se = black_scholes(S0, T, sigma, "EUROPEAN PUT"), 0.0
            greeks = black_scholes_greeks(S0, T, sigma, "EUROPEAN PUT")
            return price, se, greeks
        case "ASIAN CALL":
            E_Y = geometric_asian_price(S0, T, sigma, o)
            return mc_path_dependent(S0, T, sigma, payoff_asian_call, payoff_asian_call_geometric, E_Y)
        case "ASIAN PUT":
            E_Y = geometric_asian_price(S0, T, sigma, o)
            return mc_path_dependent(S0, T, sigma, payoff_asian_put, payoff_asian_put_geometric, E_Y)
        case _:
            raise ValueError("Option type not recognised")     

def websocket_thread():
    def on_message(ws, message):
        msg = json.loads(message)
        if msg.get("type") == "trade" and "data" in msg:
            trades = msg["data"]
            if trades:
                S0 = trades[-1]["p"] # last available price
                if price_queue.full():
                    try:
                        price_queue.get_nowait()
                    except queue.Empty:
                        pass
                price_queue.put(S0)        

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
    sigma = calculate_sigma()
    last_sigma_update = datetime.now(timezone.utc)
    UPDATE_INTERVAL_SECONDS = 60

    while True:
        S0 = price_queue.get()

        # update volatility periodically
        now = datetime.now(timezone.utc)
        if (now - last_sigma_update).total_seconds() >= UPDATE_INTERVAL_SECONDS:
            try:
                sigma = calculate_sigma()
                print(f"{now.strftime("%Y-%m-%d %H-%M-%S")}: [Sigma updated to {sigma:.4f}]")
                last_sigma_update = now
            except Exception as e: # if yfinance doesn't work as expected
                print(f"{now} [Sigma update failed]")       

        T = time_to_maturity()
        est_price, est_se, greeks = mc_option_price(S0, T, sigma, OPTION_TYPE)
        (d, g, v, t, r) = greeks
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{now}: MC price = {est_price:.2f}, MC SE = {est_se:.4f}")
        print(f"Delta = {d:.4f}, Gamma = {g:.2e}, Vega = {v/100:.2f} per 1% vol, Theta = {t/365:.2f} per day, Rho = {r/100:.2f} per 1% rate")

if __name__ == "__main__":
    threading.Thread(target=websocket_thread, daemon=True).start()
    threading.Thread(target=pricing_thread, daemon=True).start()

    threading.Event().wait() # block forever to keep daemons alive