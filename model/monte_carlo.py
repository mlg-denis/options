import numpy as np
from config import STEPS, r, N, BATCH_SIZE
from model.black_scholes import black_scholes, black_scholes_greeks
from model.geometric_asian import geometric_asian_price
from model.payoffs import payoff_asian_call, payoff_asian_call_geometric, payoff_asian_put, payoff_asian_put_geometric

def simulate_paths(S0, drift, vol, Z_batch):
    increments = drift + vol * Z_batch
    log_paths = np.cumsum(increments, axis=1)
    # allocate result array
    paths = np.empty((Z_batch.shape[0], STEPS+1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(log_paths)
    return paths

def mc_path_dependent(S0, T, sigma, payoff_X, payoff_Y = None, E_Y = None):
    discount = np.exp(-r * T)
    total = 0.0
    total_sq = 0.0
    total_N = 0
   
    dt = T / STEPS
    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)

    use_cv = payoff_Y is not None and E_Y is not None # cv - control variate

    rng = np.random.default_rng()
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
        rng = np.random.default_rng()
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
            price = black_scholes(S0, T, sigma, "EUROPEAN CALL")
            greeks = black_scholes_greeks(S0, T, sigma, "EUROPEAN CALL")
            return price, None, greeks # deterministic price, no SE
        case "EUROPEAN PUT":
            price = black_scholes(S0, T, sigma, "EUROPEAN PUT")
            greeks = black_scholes_greeks(S0, T, sigma, "EUROPEAN PUT")
            return price, None, greeks
        case "ASIAN CALL":
            E_Y = geometric_asian_price(S0, T, sigma, o)
            price, se = mc_path_dependent(S0, T, sigma, payoff_asian_call, payoff_asian_call_geometric, E_Y)
            return price, se, None
        case "ASIAN PUT":
            E_Y = geometric_asian_price(S0, T, sigma, o)
            price, se = mc_path_dependent(S0, T, sigma, payoff_asian_put, payoff_asian_put_geometric, E_Y)
            return price, se, None
        case _:
            raise ValueError("Option type not recognised")     
