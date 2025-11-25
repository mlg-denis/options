import yfinance as yf, pandas as pd
from config import YF_TICKER, TRADING_DAYS_PER_YEAR
from numpy import log, sqrt

def calculate_sigma(): # annualised volatility
    data = yf.download(YF_TICKER, period="1y", progress=False, auto_adjust=True)
    print("yFinance data downloaded")
    LAMBDA = 0.94 # how reactive is sigma
    
    if isinstance(data.columns, pd.MultiIndex):
        close = data[("Close", YF_TICKER)]
    else:
        close = data["Close"]

    span = 2 / (1 - LAMBDA) - 1
    logr = log(close / close.shift(1)).dropna() # r_t = ln P_t/P_{t-1}
    sigma_daily = logr.ewm(span=span).std().iloc[-1]
    return sigma_daily * sqrt(TRADING_DAYS_PER_YEAR)