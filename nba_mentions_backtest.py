"""
nba_mentions_backtest.py
Scrapes all settled Kalshi NBA mention markets and calculates
real hit rates for each word. Replaces the hardcoded BASE_RATES
in nba_mentions.py with empirical data.
"""

import json
import time
import kalshi_client as kc
from pathlib import Path
from collections import defaultdict

BACKTEST_CACHE = Path("nba_mentions_backtest.json")


def fetch_all_settled_events() -> list[dict]:
    """Fetch all settled KXNBAMENTION events with market outcomes."""
    all_events = []
    cursor     = None

    print("[backtest] Fetching settled NBA mention events...")
    while True:
        params = {"series_ticker": "KXNBAMENTION", "limit": 100, "status": "settled"}
        if cursor:
            params["cursor"] = cursor
        resp   = kc.get("/events", params)
        events = resp.get("events", [])
        all_events.extend(events)
        cursor = resp.get("cursor")
        if not cursor or not events:
            break
        time.sleep(0.2)

    print(f"[backtest] Found {len(all_events)} settled events.")
    return all_events


def fetch_market_outcomes(events: list[dict]) -> list[dict]:
    """Pull market results for each event."""
    records = []

    for i, event in enumerate(events):
        ticker = event.get("event_ticker", "")
        try:
            markets = kc.get(f"/events/{ticker}").get("markets", [])
            for m in markets:
                word   = (m.get("yes_sub_title") or "").strip()
                result = m.get("result")
                if not word or result not in ("yes", "no"):
                    continue
                if "does not qualify" in word.lower():
                    continue
                records.append({
                    "event":  ticker,
                    "word":   word.lower(),
                    "result": result,
                })
            time.sleep(0.12)
        except Exception as e:
            print(f"[backtest] Error fetching {ticker}: {e}")

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(events)}] fetched {len(records)} market outcomes")

    return records


def calculate_hit_rates(records: list[dict]) -> dict[str, dict]:
    """
    Calculate empirical hit rate for each word.
    Returns {word: {hit_rate, yes_count, total, confidence}}
    """
    word_stats = defaultdict(lambda: {"yes": 0, "total": 0})

    for r in records:
        word = r["word"]
        word_stats[word]["total"] += 1
        if r["result"] == "yes":
            word_stats[word]["yes"] += 1

    hit_rates = {}
    for word, stats in word_stats.items():
        total     = stats["total"]
        yes_count = stats["yes"]
        hit_rate  = yes_count / total if total else 0
        # Laplace smoothed
        smoothed  = (yes_count + 1) / (total + 2)
        hit_rates[word] = {
            "hit_rate":   round(hit_rate, 4),
            "smoothed":   round(smoothed, 4),
            "yes_count":  yes_count,
            "total":      total,
            "confidence": "high" if total >= 20 else "medium" if total >= 10 else "low",
        }

    return hit_rates


def run_backtest(force_refresh: bool = False) -> dict[str, dict]:
    if not force_refresh and BACKTEST_CACHE.exists():
        data = json.loads(BACKTEST_CACHE.read_text())
        print(f"[backtest] Loaded from cache: {len(data)} words.")
        return data

    events  = fetch_all_settled_events()
    records = fetch_market_outcomes(events)

    print(f"\n[backtest] Total market outcomes: {len(records)}")
    hit_rates = calculate_hit_rates(records)

    BACKTEST_CACHE.write_text(json.dumps(hit_rates, indent=2))
    print(f"[backtest] Saved hit rates for {len(hit_rates)} words.")
    return hit_rates


def print_hit_rates(hit_rates: dict[str, dict], min_games: int = 10):
    print(f"\n{'='*70}")
    print(f"  {'WORD':<42} {'HIT%':>6} {'N':>5} {'CONF'}")
    print(f"{'='*70}")
    sorted_words = sorted(hit_rates.items(), key=lambda x: x[1]["hit_rate"], reverse=True)
    for word, stats in sorted_words:
        if stats["total"] < min_games:
            continue
        print(f"  {word:<42} {stats['hit_rate']*100:>5.1f}%  {stats['total']:>4}  {stats['confidence']}")
    print(f"{'='*70}")


def compare_to_model(hit_rates: dict[str, dict]):
    """Compare empirical rates to our BASE_RATES in nba_mentions.py."""
    from nba_mentions import BASE_RATES

    print(f"\n{'='*75}")
    print(f"  EMPIRICAL vs MODEL BASE RATES")
    print(f"{'='*75}")
    print(f"  {'WORD':<42} {'EMPIRICAL':>10} {'MODEL':>8} {'DIFF':>8}")
    print(f"  {'-'*65}")

    for model_key, model_rate in sorted(BASE_RATES.items()):
        # Find matching empirical word
        empirical = None
        for word in hit_rates:
            if any(v.strip() in word for v in model_key.lower().split("/")):
                empirical = hit_rates[word]
                break

        if empirical and empirical["total"] >= 10:
            diff = empirical["hit_rate"] - model_rate
            flag = "⚠️ " if abs(diff) > 0.15 else "  "
            print(f"  {flag}{model_key:<40} {empirical['hit_rate']*100:>8.1f}%  "
                  f"{model_rate*100:>6.1f}%  {diff*100:>+6.1f}%  (n={empirical['total']})")
        else:
            n = empirical["total"] if empirical else 0
            print(f"  {'  '}{model_key:<40} {'?':>8}   {model_rate*100:>6.1f}%  {'n/a':>8}  (n={n})")

    print(f"{'='*75}")


if __name__ == "__main__":
    hit_rates = run_backtest(force_refresh=False)
    print_hit_rates(hit_rates, min_games=5)
    compare_to_model(hit_rates)