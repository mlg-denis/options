from datetime import datetime, timezone
from config import EXPIRY, TRADING_DAYS_PER_YEAR, K

def time_to_maturity():
    now = datetime.now(timezone.utc)
    seconds = max((EXPIRY - now).total_seconds(), 0.0)
    return seconds / (TRADING_DAYS_PER_YEAR * 24 * 3600)

def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def vega_to_percent(vega):
    return vega * 0.01

def theta_per_day(theta):
    return theta / 365

def rho_to_percent(rho):
    return rho * 0.01

def format_greeks(delta, gamma, vega, theta, rho):
    theta = abs(theta_per_day(theta))
    return (f"Δ = {delta:.4f}  "
            f"Γ = {gamma:.2e}  "
            f"V = ${vega_to_percent(vega):.2f}/1%  "
            f"Θ = -${theta:.2f}/day  "
            f"ρ = ${rho_to_percent(rho):.2f}/1%")

def format_output_str(option_type, price, se, S0):
    option_info = f" with Spot = ${S0}, Strike = ${K}, Expiry = {EXPIRY.date()}"
    if se is None:
        out = f"\n{now()}: {option_type} Price = ${price:.2f}"
    else:
        out = f"\n{now()}: {option_type} MC price = ${price:.2f}, MC SE = {se:.4f}"
    return out + option_info