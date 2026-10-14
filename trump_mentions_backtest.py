"""
trump_mentions_backtest.py
Scrapes all settled Kalshi Trump mention markets and calculates
real hit rates per word per series type (event/week/month).
"""

import json
import time
from pathlib import Path
from collections import defaultdict
import kalshi_client as kc

BACKTEST_CACHE = Path("trump_mentions_backtest.json")

TRUMP_SERIES = {
    "KXTRUMPMENTION":  "event",
    "KXTRUMPSAY":      "week",
    "KXTRUMPSAYMONTH": "month",
}


def fetch_all_settled(series: str) -> list[dict]:
    all_events = []
    cursor     = None
    while True:
        params = {"series_ticker": series, "limit": 100, "status": "settled"}
        if cursor:
            params["cursor"] = cursor
        resp   = kc.get("/events", params)
        events = resp.get("events", [])
        all_events.extend(events)
        cursor = resp.get("cursor")
        if not cursor or not events:
            break
        time.sleep(0.2)
    return all_events


def fetch_outcomes(events: list[dict], series_label: str) -> list[dict]:
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
                    "series": series_label,
                    "word":   word.lower(),
                    "result": result,
                })
            time.sleep(0.1)
        except Exception as e:
            print(f"[trump_bt] Error {ticker}: {e}")
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(events)}] {len(records)} outcomes so far")
    return records


def calculate_rates(records: list[dict]) -> dict:
    """Calculate hit rates per word per series."""
    stats = defaultdict(lambda: defaultdict(lambda: {"yes": 0, "total": 0}))
    for r in records:
        word   = r["word"]
        series = r["series"]
        stats[series][word]["total"] += 1
        if r["result"] == "yes":
            stats[series][word]["yes"] += 1

    rates = {}
    for series, words in stats.items():
        rates[series] = {}
        for word, s in words.items():
            total = s["total"]
            yes   = s["yes"]
            rates[series][word] = {
                "hit_rate":   round(yes / total, 4) if total else 0,
                "smoothed":   round((yes + 1) / (total + 2), 4),
                "yes_count":  yes,
                "total":      total,
                "confidence": "high" if total >= 15 else "medium" if total >= 7 else "low",
            }
    return rates


def run_backtest(force_refresh: bool = False) -> dict:
    if not force_refresh and BACKTEST_CACHE.exists():
        data = json.loads(BACKTEST_CACHE.read_text())
        total = sum(len(v) for v in data.values())
        print(f"[trump_bt] Loaded from cache: {total} words across {len(data)} series.")
        return data

    all_records = []
    for series, label in TRUMP_SERIES.items():
        print(f"\n[trump_bt] Fetching {series} ({label})...")
        events  = fetch_all_settled(series)
        print(f"  {len(events)} settled events")
        records = fetch_outcomes(events, label)
        all_records.extend(records)
        print(f"  {len(records)} outcomes")

    rates = calculate_rates(all_records)
    BACKTEST_CACHE.write_text(json.dumps(rates, indent=2))
    print(f"\n[trump_bt] Saved backtest cache.")
    return rates


def print_rates(rates: dict, min_games: int = 5):
    for series, words in rates.items():
        print(f"\n{'='*70}")
        print(f"  {series.upper()} — empirical hit rates")
        print(f"{'='*70}")
        print(f"  {'WORD':<44} {'HIT%':>6} {'N':>5} {'CONF'}")
        print(f"  {'-'*60}")
        sorted_w = sorted(words.items(), key=lambda x: x[1]["hit_rate"], reverse=True)
        for word, s in sorted_w:
            if s["total"] < min_games:
                continue
            print(f"  {word:<44} {s['hit_rate']*100:>5.1f}%  {s['total']:>4}  {s['confidence']}")


def compare_to_claude(rates: dict):
    """
    Compare empirical rates to what Claude API was estimating.
    Useful for calibrating the prompt or adding context.
    """
    event_rates = rates.get("event", {})
    print(f"\n{'='*70}")
    print(f"  HIGH CONFIDENCE EMPIRICAL RATES (event, n≥15)")
    print(f"{'='*70}")
    for word, s in sorted(event_rates.items(),
                          key=lambda x: x[1]["hit_rate"], reverse=True):
        if s["total"] < 15:
            continue
        bar = "█" * int(s["hit_rate"] * 20)
        print(f"  {word:<40} {s['hit_rate']*100:>5.1f}%  {bar}")


if __name__ == "__main__":
    rates = run_backtest(force_refresh=False)
    print_rates(rates, min_games=5)
    compare_to_claude(rates)