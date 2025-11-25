import threading
from queue import Queue
from datetime import datetime, timezone
from time import sleep
from config import FINNHUB_API_KEY, FINNHUB_TICKER, OPTION_TYPE
from market_data.websocket_client import PriceFeed
from market_data.yf_client import calculate_sigma
from model.monte_carlo import option_price
from utils import time_to_maturity, now, format_greeks, format_output_str

def pricing_thread(price_queue: Queue):
    sigma = calculate_sigma()
    last_sigma_update = datetime.now(timezone.utc)
    UPDATE_INTERVAL_SECONDS = 60

    while True:
        S0 = price_queue.get()

        # update volatility periodically
        now = datetime.now(timezone.utc)
        if (now - last_sigma_update).total_seconds() >= UPDATE_INTERVAL_SECONDS:
            try:
                sigma = calculate_sigma()
                print(f"{now.strftime("%Y-%m-%d %H-%M-%S")}: [Sigma updated to {sigma:.4f}]")
                last_sigma_update = now
            except Exception as e: # if yfinance doesn't work as expected
                print(f"{now} [Sigma update failed]")

        T = time_to_maturity()

        price, se, greeks = option_price(S0, T, sigma, OPTION_TYPE)

        output_str = format_output_str(OPTION_TYPE, price, se, S0)
        print(output_str)
        # if se is None:
        #     print(f"{timestamp}: Price = ${price:.2f} (deterministic)")
        # else:
        #     print(f"{timestamp}: MC price = ${price:.2f}, MC SE = {se:.4f}")

        if greeks is not None:
            d, g, v, t, r = greeks
            print(format_greeks(d, g, v, t, r))


def main():
    price_queue = Queue(maxsize=1)

    def on_price(price: float):
        if price_queue.full():
            price_queue.get_nowait()
        price_queue.put(price)

    # start websocket feed
    feed = PriceFeed(FINNHUB_API_KEY, FINNHUB_TICKER, on_price)
    feed.start()

    # start pricing thread
    threading.Thread(target=pricing_thread, args=(price_queue,), daemon=True).start()

    # block forever
    threading.Event().wait()


if __name__ == "__main__":
    main()