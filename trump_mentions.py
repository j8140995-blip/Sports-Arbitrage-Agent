"""
trump_mentions.py
Model for Trump mention markets on Kalshi.

Uses Claude API to estimate word probabilities based on:
- Trump's known speech patterns and vocabulary
- Event context (speech type, topic, audience)
- Current news environment
- Historical base rates baked into the prompt

Covers: KXTRUMPMENTION, KXTRUMPSAY, KXTRUMPSAYMONTH
"""

import re
import json
import time
import requests
from pathlib import Path

CACHE_FILE     = Path("trump_mentions_cache.json")
CACHE_TTL      = 3 * 3600   # 3 hours
EDGE_THRESHOLD = 0.10

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
BACKTEST_CACHE = Path("trump_mentions_backtest.json")

TRUMP_SERIES = {
    "KXTRUMPMENTION":  "event",
    "KXTRUMPSAY":      "week",
    "KXTRUMPSAYMONTH": "month",
}


# ── Empirical data lookup ─────────────────────────────────────────────────────

def load_empirical_rates() -> dict:
    """Load empirical hit rates from backtest cache."""
    if BACKTEST_CACHE.exists():
        return json.loads(BACKTEST_CACHE.read_text())
    return {}


def find_empirical_rate(word: str, series_label: str, empirical: dict) -> tuple[float, int] | None:
    """
    Look up empirical hit rate for a word in the given series.
    Returns (smoothed_rate, n) or None if not found.
    Min 5 samples required.
    """
    series_data = empirical.get(series_label, {})
    word_lower  = word.lower()

    # Exact match
    if word_lower in series_data:
        s = series_data[word_lower]
        if s["total"] >= 5:
            return s["smoothed"], s["total"]

    # Variant match (split on /)
    for key, s in series_data.items():
        if s["total"] < 5:
            continue
        variants = [v.strip() for v in key.split("/")]
        word_variants = [v.strip().lower() for v in word_lower.split("/")]
        if any(wv in variants or any(wv in v for v in variants)
               for wv in word_variants):
            return s["smoothed"], s["total"]

        # Substring match
        if word_lower in key or key in word_lower:
            return s["smoothed"], s["total"]

    return None


# ── Claude API probability estimation ────────────────────────────────────────

def estimate_batch_probabilities(
    words: list[str],
    event_title: str,
    window: str,
    event_context: str = "",
) -> dict[str, float]:
    """
    Ask Claude to estimate probability for each word/phrase being said by Trump
    in the given event context. Returns {word: probability} dict.
    """
    window_desc = {
        "event": "during this specific speech/event",
        "week":  "at least once in the next 7 days across all public statements",
        "month": "at least once in the next 30 days across all public statements",
    }.get(window, "during this event")

    words_list = "\n".join(f"- {w}" for w in words)

    prompt = f"""You are an expert analyst of Donald Trump's speech patterns with deep knowledge of his vocabulary, rhetorical style, and recurring themes.

Event: {event_title}
Window: {window_desc}
{f"Additional context: {event_context}" if event_context else ""}

For each word/phrase below, estimate the probability (0.00 to 1.00) that Trump says it {window_desc}.

Base your estimates on:
- Trump's known vocabulary and catchphrases (e.g., "Fake News", "Witch Hunt", "Biden" are very high)
- Whether the word fits the event topic and audience
- How often Trump uses this word in general vs. niche contexts
- Current news environment (tariffs, Iran, economy are hot topics in early 2026)
- For weekly/monthly windows: more chances = higher probability

Words to evaluate:
{words_list}

Respond ONLY with a JSON object mapping each word exactly as given to a probability float.
Example format: {{"Biden": 0.92, "Tariff": 0.75}}
No other text, no markdown, just the JSON object."""

    try:
        import os
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        resp = requests.post(
            ANTHROPIC_URL,
            headers={
                "Content-Type":      "application/json",
                "x-api-key":         api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model":      "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "messages":   [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        if resp.status_code != 200:
            print(f"[trump] API error: {resp.status_code}")
            return {}

        text = resp.json()["content"][0]["text"].strip()
        # Clean any markdown fences
        text = re.sub(r"```json|```", "", text).strip()
        probs = json.loads(text)
        return {k: max(0.02, min(0.98, float(v))) for k, v in probs.items()}

    except Exception as e:
        print(f"[trump] Claude API error: {e}")
        return {}


def match_prob(word: str, probs: dict) -> float | None:
    """Find probability for a word, trying exact match then fuzzy."""
    if word in probs:
        return probs[word]
    word_lower = word.lower()
    for k, v in probs.items():
        if k.lower() == word_lower:
            return v
        # Match on first variant
        first_variant = k.split("/")[0].strip().lower()
        if first_variant in word_lower or word_lower in first_variant:
            return v
    return None


# ── Cache ─────────────────────────────────────────────────────────────────────

def load_cache() -> dict:
    if CACHE_FILE.exists():
        data = json.loads(CACHE_FILE.read_text())
        if time.time() - data.get("timestamp", 0) < CACHE_TTL:
            return data.get("probs", {})
    return {}


def save_cache(probs: dict):
    CACHE_FILE.write_text(json.dumps({
        "probs":     probs,
        "timestamp": time.time(),
    }, indent=2))


# ── Kalshi scanner ────────────────────────────────────────────────────────────

def scan_trump_mentions(force_refresh: bool = False) -> list[dict]:
    import kalshi_client as kc

    cached_probs = {} if force_refresh else load_cache()
    all_probs    = dict(cached_probs)
    signals      = []

    empirical = load_empirical_rates()

    for series, window in TRUMP_SERIES.items():
        resp   = kc.get("/events", {"series_ticker": series, "limit": 10, "status": "open"})
        events = resp.get("events", [])

        for event in events:
            event_ticker = event.get("event_ticker", "")
            event_title  = event.get("title", "")
            print(f"\n[trump] Scanning {event_ticker}: {event_title}")

            markets = kc.get(f"/events/{event_ticker}").get("markets", [])
            time.sleep(0.1)

            # Separate words into empirical vs need Claude estimation
            words_need_claude = []
            market_map        = {}

            for m in markets:
                if m.get("status") != "active":
                    continue
                word = (m.get("yes_sub_title") or "").strip()
                if not word or "does not qualify" in word.lower():
                    continue
                yes_bid = m.get("yes_bid_dollars")
                yes_ask = m.get("yes_ask_dollars")
                if yes_bid is None or yes_ask is None:
                    continue
                bid, ask = float(yes_bid), float(yes_ask)
                if ask >= 0.99 or bid <= 0.01:
                    continue

                cache_key = f"{event_ticker}:{word}"
                market_map[word] = {
                    "bid": bid, "ask": ask,
                    "ticker": m.get("ticker", ""),
                    "cache_key": cache_key,
                }

                # Check empirical data first
                emp = find_empirical_rate(word, window, empirical)
                if emp:
                    all_probs[cache_key] = emp[0]
                    all_probs[cache_key + ":source"] = f"empirical(n={emp[1]})"
                elif cache_key not in all_probs:
                    words_need_claude.append(word)

            # Only call Claude for words without empirical data
            if words_need_claude:
                print(f"  [trump] Estimating {len(words_need_claude)} words via Claude API (no empirical data)...")
                probs = estimate_batch_probabilities(words_need_claude, event_title, window)
                for word, p in probs.items():
                    cache_key = f"{event_ticker}:{word}"
                    all_probs[cache_key] = p
                    all_probs[cache_key + ":source"] = "claude"
                save_cache(all_probs)
                time.sleep(0.5)

            # Generate signals
            for word, mkt in market_map.items():
                cache_key = mkt["cache_key"]
                model_p   = all_probs.get(cache_key)
                source    = all_probs.get(cache_key + ":source", "unknown")
                if model_p is None:
                    continue

                bid, ask = mkt["bid"], mkt["ask"]
                mid      = (bid + ask) / 2
                edge     = model_p - mid

                if abs(edge) < EDGE_THRESHOLD:
                    continue

                side = "yes" if edge > 0 else "no"
                signals.append({
                    "ticker":     mkt["ticker"],
                    "event":      event_ticker,
                    "series":     series,
                    "window":     window,
                    "word":       word,
                    "model_prob": round(model_p, 4),
                    "market_mid": round(mid, 4),
                    "edge":       round(edge, 4),
                    "side":       side,
                    "bid":        bid,
                    "ask":        ask,
                    "spread":     round(ask - bid, 4),
                    "source":     source,
                })

    signals.sort(key=lambda s: abs(s["edge"]), reverse=True)
    return signals


def print_trump_signals(signals: list[dict]):
    if not signals:
        print("[trump] No signals above threshold.")
        return

    current_event = None
    for s in signals:
        if s["event"] != current_event:
            current_event = s["event"]
            print(f"\n{'='*85}")
            print(f"  {current_event} [{s['window']}]")
            print(f"{'='*85}")
            print(f"  {'WORD':<44} {'MODEL':>7} {'MID':>7} {'EDGE':>7} {'SIDE':<5} {'SOURCE'}")
            print(f"  {'-'*78}")
        src = s.get("source", "")
        print(
            f"  {s['word']:<44} "
            f"{s['model_prob']*100:>6.1f}% "
            f"{s['market_mid']*100:>6.1f}% "
            f"{s['edge']*100:>+6.1f}% "
            f"{s['side']:<5} "
            f"{src}"
        )
    print(f"\n  Total Trump signals: {len(signals)}")


# Keep load_trump_data for backward compat with main.py
def load_trump_data(force_refresh: bool = False) -> dict:
    return {"force_refresh": force_refresh}


def log_trump_trade(signal: dict):
    """Log a Trump mention signal to paper_trades.json."""
    import json
    import datetime
    from pathlib import Path

    log_file = Path("paper_trades.json")
    trades   = json.loads(log_file.read_text()) if log_file.exists() else []

    if any(t.get("ticker") == signal["ticker"] for t in trades):
        return False

    entry_price = signal["ask"] if signal["side"] == "yes" else 1 - signal["bid"]
    contracts   = max(1, int(5 / entry_price))
    cost        = contracts * entry_price

    trades.append({
        "ticker":            signal["ticker"],
        "title":             signal.get("word", ""),
        "sport":             "TRUMP_MENTION",
        "side":              signal["side"],
        "market_mid":        signal["market_mid"],
        "model_prob":        signal["model_prob"],
        "edge":              signal["edge"],
        "kelly_fraction":    0.0,
        "half_kelly":        0.0,
        "contracts":         contracts,
        "cost_dollars":      round(cost, 2),
        "entry_price_cents": round(entry_price * 100),
        "timestamp":         datetime.datetime.utcnow().isoformat(),
        "mode":              "paper",
        "status":            "open",
        "outcome":           None,
        "pnl":               None,
        "game_date":         datetime.date.today().isoformat(),
        "home_team":         "",
        "away_team":         "",
        "source":            signal.get("source", ""),
        "event":             signal.get("event", ""),
        "word":              signal.get("word", ""),
        "window":            signal.get("window", ""),
        "series":            signal.get("series", ""),
    })

    log_file.write_text(json.dumps(trades, indent=2))
    return True


if __name__ == "__main__":
    print("=== Scanning Trump mention markets ===")
    signals = scan_trump_mentions(force_refresh=False)
    print_trump_signals(signals)

    logged = 0
    for s in signals:
        if log_trump_trade(s):
            logged += 1
    if logged:
        print(f"\n[trump] Logged {logged} new trades to paper_trades.json")