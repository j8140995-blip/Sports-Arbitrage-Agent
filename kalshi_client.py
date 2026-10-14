"""
kalshi_client.py
Kalshi API client — RSA-PSS auth + market helpers.
"""

import os
import time
import base64
import requests
from urllib.parse import urlparse

BASE = "https://api.elections.kalshi.com/trade-api/v2"


def make_headers(method: str, path: str) -> dict:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding as apad
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    with open("private_key.pem", "rb") as f:
        key = load_pem_private_key(f.read(), password=None)

    ts_ms = str(int(time.time() * 1000))
    msg = (ts_ms + method.upper() + urlparse(BASE + path).path).encode()
    sig = base64.b64encode(
        key.sign(
            msg,
            apad.PSS(
                mgf=apad.MGF1(hashes.SHA256()),
                salt_length=apad.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
    ).decode()

    return {
        "KALSHI-ACCESS-KEY": os.environ.get("KALSHI_KEY_ID", ""),
        "KALSHI-ACCESS-TIMESTAMP": ts_ms,
        "KALSHI-ACCESS-SIGNATURE": sig,
    }


def get(path: str, params: dict = None) -> dict:
    headers = make_headers("GET", path)
    r = requests.get(BASE + path, headers=headers, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def post(path: str, body: dict) -> dict:
    headers = make_headers("POST", path)
    headers["Content-Type"] = "application/json"
    r = requests.post(BASE + path, headers=headers, json=body, timeout=15)
    r.raise_for_status()
    return r.json()


def get_markets(category: str = None, limit: int = 200, cursor: str = None) -> dict:
    params = {"limit": limit}
    if category:
        params["category"] = category
    if cursor:
        params["cursor"] = cursor
    return get("/markets", params)


def get_market(ticker: str) -> dict:
    return get(f"/markets/{ticker}")


def get_orderbook(ticker: str, depth: int = 5) -> dict:
    return get(f"/markets/{ticker}/orderbook", {"depth": depth})


def get_balance() -> float:
    data = get("/portfolio/balance")
    return data.get("balance", 0) / 100


def place_order(ticker: str, side: str, count: int, price_cents: int,
                order_type: str = "limit") -> dict:
    body = {
        "ticker": ticker,
        "side": side,
        "count": count,
        "action": "buy",
        "type": order_type,
        "yes_price" if side == "yes" else "no_price": price_cents,
    }
    return post("/portfolio/orders", body)


def get_positions() -> list:
    data = get("/portfolio/positions")
    return data.get("market_positions", [])


def mid_price(ticker: str) -> float | None:
    try:
        ob = get_orderbook(ticker, depth=1)
        yes_bids = ob.get("orderbook", {}).get("yes", [])
        yes_asks = ob.get("orderbook", {}).get("no", [])

        best_bid = yes_bids[0][0] if yes_bids else None
        best_ask = (100 - yes_asks[0][0]) if yes_asks else None

        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2 / 100
        elif best_bid is not None:
            return best_bid / 100
        elif best_ask is not None:
            return best_ask / 100
        else:
            mkt = get_market(ticker)
            lp = mkt.get("market", {}).get("last_price")
            return lp / 100 if lp else None
    except Exception as e:
        print(f"[mid_price] {ticker} error: {e}")
        return None