from scipy.stats import norm
from math import log, sqrt, exp
from config import K, r

def black_scholes(S0, T, sigma, option_type):
    d1 = (log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option_type == "EUROPEAN CALL":
        price = S0 * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    elif option_type == "EUROPEAN PUT":
        price = K * exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    else:
        raise ValueError("Black-Scholes model only supports EUROPEAN CALL and EUROPEAN PUT options.")

    return price

def black_scholes_greeks(S0, T, sigma, option_type):
    d1 = (log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option_type == "EUROPEAN CALL":
        delta = norm.cdf(d1)
        theta = (-(S0 * norm.pdf(d1) * sigma) / (2 * sqrt(T)) - r * K * exp(-r * T) * norm.cdf(d2))
        rho = K * T * exp(-r * T) * norm.cdf(d2)
    elif option_type == "EUROPEAN PUT":
        delta = norm.cdf(d1) - 1
        theta = (-(S0 * norm.pdf(d1) * sigma) / (2 * sqrt(T)) + r * K * exp(-r * T) * norm.cdf(-d2))
        rho = K * T * exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("Black-Scholes model only supports EUROPEAN CALL and EUROPEAN PUT options.")
    vega = S0 * norm.pdf(d1) * sqrt(T)
    gamma = norm.pdf(d1) / (S0 * sigma * sqrt(T))

    return delta, gamma, vega, theta, rho