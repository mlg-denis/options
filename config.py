from datetime import datetime, timezone
from os import getenv

TRADING_DAYS_PER_YEAR = 365 # this is the case for BTC. change for instruments on NYSE for example, using pandas_market_calendars
N = 250000 # number of simulations
BATCH_SIZE = 50000
STEPS = 50
r = 0.0404 # annualised drift (risk-free rate)
K = 90000 # strike price
EXPIRY = datetime(2025,12,24, tzinfo=timezone.utc)

FINNHUB_API_KEY = getenv("FINNHUB_API_KEY")
FINNHUB_TICKER = "BINANCE:BTCUSDT"
YF_TICKER = "BTC-USD"

OPTION_TYPE = "ASIAN CALL" # "EUROPEAN CALL", "EUROPEAN PUT", "ASIAN CALL", "ASIAN PUT"