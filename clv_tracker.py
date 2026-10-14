"""
clv_tracker.py
Closing Line Value (CLV) tracking — the gold standard for model validation.

Per the research report: "Consistently positive CLV is the key factor in
determining whether a bettor has genuine edge or is burning through variance."

How it works:
- When we place a trade, we log our entry price
- After market resolution, we fetch the final/closing price from Kalshi
- CLV = our_price vs closing_price (did we get a better price than the market settled at?)
- Track rolling CLV over 250 and 1000 trade windows

Benchmarks (from report):
- +1-2% CLV consistently = strong edge
- +5% CLV = excellent
- Negative CLV after 1,000 bets = investigate immediately
"""

import json
import time
import datetime
from pathlib import Path
from collections import defaultdict

import kalshi_client as kc

PAPER_LOG = Path("paper_trades.json")
CLV_LOG   = Path("clv_log.json")


def fetch_closing_price(ticker: str) -> float | None:
    """
    Fetch the final/closing price for a resolved market.
    Returns the YES price at close (0-1), or None if unavailable.
    """
    try:
        resp    = kc.get(f"/markets/{ticker}")
        market  = resp.get("market", resp)
        result  = market.get("result")

        # If settled YES, closing price = 1.0; NO = 0.0
        if result == "yes":
            return 1.0
        elif result == "no":
            return 0.0

        # For still-open markets, use current mid
        yes_bid = market.get("yes_bid_dollars")
        yes_ask = market.get("yes_ask_dollars")
        if yes_bid is not None and yes_ask is not None:
            return (float(yes_bid) + float(yes_ask)) / 2
        return None
    except Exception:
        return None


def calculate_clv(trade: dict) -> float | None:
    """
    Calculate CLV for a single trade.

    CLV > 0: we got a better price than the market eventually offered
    CLV < 0: we paid more than the market's closing price

    For YES trades: CLV = closing_price - our_entry_price
    For NO trades:  CLV = (1-closing_price) - our_entry_price
    """
    entry_cents  = trade.get("entry_price_cents")
    side         = trade.get("side", "yes")
    closing_price = trade.get("closing_price")

    if entry_cents is None or closing_price is None:
        return None

    entry = entry_cents / 100

    if side == "yes":
        return closing_price - entry
    else:
        # For NO bets, our entry is the NO price
        return (1 - closing_price) - entry


def update_clv_for_resolved_trades():
    """
    For all resolved trades missing a closing_price, fetch and store it.
    """
    if not PAPER_LOG.exists():
        return

    trades  = json.loads(PAPER_LOG.read_text())
    updated = 0

    for trade in trades:
        if trade.get("outcome") not in ("win", "loss"):
            continue
        if trade.get("closing_price") is not None:
            continue

        ticker        = trade.get("ticker", "")
        closing_price = fetch_closing_price(ticker)
        if closing_price is not None:
            trade["closing_price"] = closing_price
            clv = calculate_clv(trade)
            trade["clv"] = round(clv, 4) if clv is not None else None
            updated += 1
        time.sleep(0.1)

    if updated:
        PAPER_LOG.write_text(json.dumps(trades, indent=2))
        print(f"[CLV] Updated closing prices for {updated} trades.")


def compute_clv_summary(trades: list[dict]) -> dict:
    """Compute CLV statistics across all resolved trades."""
    clv_trades = [t for t in trades
                  if t.get("outcome") in ("win", "loss")
                  and t.get("clv") is not None]

    if not clv_trades:
        return {"n": 0, "mean_clv": None, "status": "insufficient data"}

    clvs      = [t["clv"] for t in clv_trades]
    mean_clv  = sum(clvs) / len(clvs)
    positive  = sum(1 for c in clvs if c > 0)

    # Rolling windows
    last_250  = clvs[-250:] if len(clvs) >= 250 else clvs
    last_1000 = clvs[-1000:] if len(clvs) >= 1000 else clvs

    # By sport
    by_sport = defaultdict(list)
    for t in clv_trades:
        sport = t.get("sport", "UNKNOWN")
        by_sport[sport].append(t["clv"])

    # Status assessment
    if len(clvs) < 250:
        status = f"insufficient data ({len(clvs)} trades, need 250+)"
    elif mean_clv > 0.02:
        status = "strong edge ✅"
    elif mean_clv > 0.01:
        status = "positive edge ✅"
    elif mean_clv > -0.01:
        status = "marginal — monitor closely ⚠️"
    else:
        status = "negative CLV — investigate immediately 🚨"

    return {
        "n":            len(clvs),
        "mean_clv":     round(mean_clv * 100, 2),
        "positive_pct": round(positive / len(clvs) * 100, 1),
        "rolling_250":  round(sum(last_250) / len(last_250) * 100, 2) if last_250 else None,
        "rolling_1000": round(sum(last_1000) / len(last_1000) * 100, 2) if last_1000 else None,
        "by_sport":     {s: round(sum(v)/len(v)*100, 2) for s, v in by_sport.items()},
        "status":       status,
    }


def print_clv_report(trades: list[dict]):
    summary = compute_clv_summary(trades)

    print(f"\n{'='*60}")
    print(f"  CLOSING LINE VALUE (CLV) REPORT")
    print(f"{'='*60}")

    if summary["n"] == 0:
        print(f"  No resolved trades with CLV data yet.")
        print(f"  CLV tracking will populate as markets settle.")
        print(f"{'='*60}")
        return

    print(f"  Trades with CLV data: {summary['n']}")
    print(f"  Mean CLV:             {summary['mean_clv']:+.2f}%")
    print(f"  Positive CLV rate:    {summary['positive_pct']:.1f}%")
    if summary["rolling_250"]:
        print(f"  Rolling 250:          {summary['rolling_250']:+.2f}%")
    if summary["rolling_1000"]:
        print(f"  Rolling 1000:         {summary['rolling_1000']:+.2f}%")
    print(f"  Status:               {summary['status']}")

    if summary["by_sport"]:
        print(f"\n  By sport:")
        for sport, clv in sorted(summary["by_sport"].items(),
                                  key=lambda x: x[1], reverse=True):
            print(f"    {sport:<20} {clv:+.2f}%")

    print(f"\n  Benchmarks: +1-2% = strong edge | +5% = excellent")
    print(f"  Need 250+ trades for meaningful signal.")
    print(f"{'='*60}")