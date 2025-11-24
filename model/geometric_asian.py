from math import sqrt, log, exp, erf
from config import r, STEPS, K

def geometric_asian_price(S0, T, sigma, option_type):
    sigma_ = sigma * sqrt((2*STEPS + 1) / (6*(STEPS + 1)))
    mu_ = 0.5 * (r - 0.5*sigma*sigma) + 0.5*sigma_*sigma_

    sqrtT = sqrt(T)
    d1 = (log(S0 / K) + (mu_ + 0.5*sigma_*sigma_) * T) / (sigma_ * sqrtT)
    d2 = d1 - sigma_ * sqrtT

    def norm_cdf(x):
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    Nd1, Nd2 = norm_cdf(d1), norm_cdf(d2)

    discount = exp(-r * T)

    if option_type == "ASIAN CALL":
        return S0 * exp(mu_*T) * Nd1 - K * discount * Nd2
    elif option_type == "ASIAN PUT":
        return K * discount * (1-Nd2) - S0 * exp(mu_*T) * (1-Nd1)
    else:
        raise ValueError("Option type must be ASIAN CALL or ASIAN PUT to use geometric Asian control variate")        
