from numpy import maximum, exp, log
from config import K

def payoff_asian_call(paths):
    avg_price = paths[:, 1:].mean(axis=1)
    return maximum(avg_price - K, 0.0)

def payoff_asian_call_geometric(paths):
    # using the exp(log(_)) trick guards against underflow/overflow
    log_avg_price = log(paths[:,1:]).mean(axis=1)
    return maximum(exp(log_avg_price) - K, 0.0)

def payoff_asian_put(paths):
    avg_price = paths[:, 1:].mean(axis=1)
    return maximum(K - avg_price, 0.0)

def payoff_asian_put_geometric(paths):
    log_avg_price = log(paths[:,1:]).mean(axis=1)
    return maximum(K - exp(log_avg_price), 0.0)