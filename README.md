# Option Pricer (Monte Carlo + Black-Scholes)

This project prices European and Asian (arithmetic & geometric) options in real time using:
- **Black-Scholes** closed form for European options
- **Monte Carlo simulation** with
  - antithetic variates
  - geometric-Asian control variate
  - batching to optimise memory usage
- Real-time volatility estimation updates using the RiskMetrics model
 
It subscribes to live prices through the **Finnhub API**, updates **σ** value periodically using **yfinance**, and prints option prices and Greeks (where applicable).

## Features

**Real time pricing**
- Live spot price feed via WebSocket
- Background pricing thread
- Thread-safe communication through a bounded queue

**European options**
- Closed-form Black Scholes
- 5 main Greeks (Δ, Γ, V, Θ, ρ)
- Deterministic output

**Asian options**
- Arithmetic Asian priced with Monte Carlo
- Control variate using geometric-Asian closed form
- Standard error calculation

## Output

There is continuous output as and when data comes in from Finnhub.

For options with closed form formulas (i.e. European options), output looks like:

<img width="1007" height="52" alt="image" src="https://github.com/user-attachments/assets/afa38516-b743-4316-b752-101afbddaf9a" />

For options calculated through Monte Carlo (i.e. Asian options), outputs looks like:

<img width="1155" height="26" alt="image" src="https://github.com/user-attachments/assets/4318bebe-f648-46a0-9554-d42b68771079" />



## Configuration

**config.py** sets:
- **TRADING_DAYS_PER_YEAR**: The number of days per year where the asset is traded. For most crypto, this is 365, for stocks this varies.
- **N**: Number of Monte Carlo simulations
- **STEPS**: Number of Monte Carlo steps
- **BATCH_SIZE**: Batch size for Z batching
- **K**: Strike price
- **r**: Risk-free rate
- **EXPIRY**: Expiry date of the option
- Ticker symbols for Finnhub and yfinance
- **OPTION_TYPE**: "EUROPEAN CALL" | "EUROPEAN PUT" | "ASIAN CALL" | "ASIAN PUT" (case-insensitive)
 
## Requirements

- Python 3.10+
- numpy
- pandas
- yfinance
- websocket-client

Finnhub requires an API key.

