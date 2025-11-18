import numpy as np
import websocket, json, os

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

def main():
    N = 100000 # number of simulations
    S_0 = 100 # initial price
    mu = 0.05 # annualised drift
    sigma = 0.2 # annualised volatlity
    T = 1.0 # time horizon (years)

    Z = np.random.normal(0,1, size=N)
    W_T = np.sqrt(T) * Z

    S_T = S_0 * np.exp((mu - 0.5* sigma**2)*T + sigma*W_T)
    print(S_T)
    
    print_prices()

def print_prices():
    def on_message(ws, message):
        data = json.loads(message)
        print(data)

    def on_open(ws):
        ws.send(json.dumps({"type": "subscribe", "symbol": "AAPL"}))

    def on_close(ws):
        print("closed connection")

    socket = f"wss://ws.finnhub.io?token={FINNHUB_API_KEY}"
    ws = websocket.WebSocketApp(socket, on_message=on_message, on_open=on_open, on_close=on_close)

    ws.run_forever()    


if __name__ == "__main__":
    main()