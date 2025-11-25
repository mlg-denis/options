import json
import websocket
from time import sleep
from threading import Thread

class PriceFeed:
    def __init__(self, api_key: str, symbol: str, on_price):
        self.api_key = api_key
        self.symbol = symbol
        self.on_price = on_price
        self._ws = None

    def _on_message(self, ws, message):
        msg = json.loads(message)
        if msg.get("type") == "trade":
            data = msg.get("data", [])
            if data:
                self.on_price(data[-1]["p"])

    def _on_open(self, ws):
        print("WebSocket connected")
        sub = {"type": "subscribe", "symbol": self.symbol}
        ws.send(json.dumps(sub))

    def _on_close(self, ws):
        print("WebSocket closed")

    def run_forever(self):
        url = f"wss://ws.finnhub.io?token={self.api_key}"
        while True:
            try:
                self._ws = websocket.WebSocketApp(
                    url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_close=self._on_close,
                )
                self._ws.run_forever(ping_interval=5)
            except Exception as e:
                print("WS error:", e)
                sleep(5)

    def start(self):
        Thread(target=self.run_forever, daemon=True).start()
