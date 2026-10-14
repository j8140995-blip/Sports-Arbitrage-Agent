"""
position_sizing.py
Quarter-Kelly position sizing with bankroll cap.

Design philosophy:
- Full Kelly is mathematically optimal but catastrophic with model uncertainty
- Quarter Kelly: sacrifices ~43% of theoretical growth for ~93% less ruin risk
- Hard cap at 5% of bankroll: protects against model blowups on any single trade
- Edge floor of 5%: filters noise; sub-5% edges are usually model error, not edge

Assumes fixed notional bankroll of $100 so sizing is stable regardless of
actual Kalshi balance. Translates directly when scaling to live capital.
"""

# ── Config ────────────────────────────────────────────────────────────────────
NOTIONAL_BANKROLL = 100.0   # fixed reference bankroll for sizing math
KELLY_FRACTION    = 0.25    # quarter Kelly
MAX_BET_PCT       = 0.05    # 5% of bankroll hard cap per trade
MAX_SPORT_PCT     = 0.15    # 15% of bankroll hard cap per sport per day
MIN_EDGE          = 0.05    # skip signals below 5% edge (filters model noise)
MIN_CONTRACTS     = 1       # at least 1 contract if we take the bet


def kelly_fraction(edge: float, entry_price: float) -> float:
    """
    Full Kelly optimal fraction for a binary outcome.

    Formula: f* = edge / (price * (1 - price))

    edge = model_prob - market_price (for YES side)
    entry_price = the price we're paying (0-1 range)

    Returns the optimal fraction of bankroll to bet.
    """
    if entry_price <= 0 or entry_price >= 1:
        return 0.0
    variance = entry_price * (1 - entry_price)
    if variance == 0:
        return 0.0
    return edge / variance


def size_bet(edge: float, entry_price: float,
             bankroll: float = NOTIONAL_BANKROLL,
             sport_used_pct: float = 0.0) -> dict:
    """
    Calculate contracts to buy for a trade using quarter-Kelly + cap.

    Args:
      edge:           absolute edge magnitude (positive number)
      entry_price:    price paid per contract (0-1 range)
      bankroll:       notional bankroll (default $100)
      sport_used_pct: fraction of bankroll already allocated to this sport today
                      (0.08 = 8% already used → only 7% more available before 15% cap)

    Returns dict with:
      - 'contracts': integer number of contracts to buy
      - 'dollars':   dollar amount wagered
      - 'pct_bank':  percentage of bankroll used
      - 'kelly_raw': theoretical full Kelly fraction
      - 'kelly_q':   quarter-Kelly fraction
      - 'capped':    whether the bet was capped by per-trade limit
      - 'sport_capped': whether the bet was capped by per-sport daily limit
      - 'skip':      True if signal should be skipped
      - 'reason':    human-readable explanation

    Edge is always the absolute edge magnitude (positive number).
    Entry price is the price you pay per contract (0-1 range).
    """
    abs_edge = abs(edge)

    # Minimum edge filter
    if abs_edge < MIN_EDGE:
        return {
            "contracts": 0,
            "dollars":   0.0,
            "pct_bank":  0.0,
            "kelly_raw": 0.0,
            "kelly_q":   0.0,
            "capped":    False,
            "sport_capped": False,
            "skip":      True,
            "reason":    f"edge {abs_edge*100:.1f}% below min {MIN_EDGE*100:.0f}% — skipped",
        }

    # Sport-level daily cap: how much is left in the sport's daily allocation?
    sport_remaining_pct = MAX_SPORT_PCT - sport_used_pct
    if sport_remaining_pct <= 0.001:   # effectively exhausted
        return {
            "contracts": 0,
            "dollars":   0.0,
            "pct_bank":  0.0,
            "kelly_raw": 0.0,
            "kelly_q":   0.0,
            "capped":    False,
            "sport_capped": True,
            "skip":      True,
            "reason":    f"sport daily cap of {MAX_SPORT_PCT*100:.0f}% already used "
                         f"({sport_used_pct*100:.1f}%) — skipped",
        }

    # Full Kelly
    kelly = kelly_fraction(abs_edge, entry_price)

    # Quarter Kelly
    kelly_q = kelly * KELLY_FRACTION

    # Apply per-trade cap
    trade_capped = kelly_q > MAX_BET_PCT
    final_pct    = min(kelly_q, MAX_BET_PCT)

    # Apply per-sport daily cap (further constrain)
    sport_capped = final_pct > sport_remaining_pct
    if sport_capped:
        final_pct = sport_remaining_pct

    final_pct = max(0.0, final_pct)

    # Convert to dollars and contracts
    dollars   = bankroll * final_pct
    contracts = max(MIN_CONTRACTS, int(dollars / entry_price)) if entry_price > 0 else 0
    actual_dollars = contracts * entry_price

    reason = f"quarter-Kelly {kelly_q*100:.1f}%"
    if trade_capped and not sport_capped:
        reason += f" → capped at {MAX_BET_PCT*100:.0f}% per-trade"
    elif sport_capped:
        reason += f" → capped at {sport_remaining_pct*100:.1f}% (sport daily remainder)"

    return {
        "contracts":    contracts,
        "dollars":      round(actual_dollars, 2),
        "pct_bank":     round(final_pct * 100, 2),
        "kelly_raw":    round(kelly, 4),
        "kelly_q":      round(kelly_q, 4),
        "capped":       trade_capped,
        "sport_capped": sport_capped,
        "skip":         False,
        "reason":       reason,
    }


def sport_used_today(trades: list, sport: str, today: str = None) -> float:
    """
    Calculate fraction of bankroll already allocated to a sport today.

    Args:
      trades: list of trade records from paper_trades.json
      sport:  sport name (e.g. "MLB", "NBA", "SOCCER")
      today:  ISO date string (default: today)

    Returns: fraction (0.0-1.0) of bankroll used. e.g. 0.08 = 8% used.
    """
    import datetime
    if today is None:
        today = datetime.date.today().isoformat()

    used_pct = 0.0
    for t in trades:
        if t.get("sport") != sport:
            continue
        if t.get("game_date") != today and t.get("timestamp", "")[:10] != today:
            continue
        # Prefer explicit pct_bankroll if present (new format)
        if "pct_bankroll" in t:
            used_pct += t["pct_bankroll"] / 100.0
        else:
            # Fallback: derive from cost_dollars assuming $100 bankroll
            used_pct += t.get("cost_dollars", 0) / NOTIONAL_BANKROLL

    return used_pct


if __name__ == "__main__":
    # Per-trade sanity checks
    cases = [
        ("BAL/KC at 45¢", 0.145, 0.45),
        ("ORL/DET at 76¢", 0.09, 0.76),
        ("Barcelona at 80¢", 0.12, 0.80),
        ("Marginal 5% edge at 50¢", 0.05, 0.50),
        ("Marginal 3% edge at 50¢", 0.03, 0.50),
        ("Huge 20% edge at 20¢", 0.20, 0.20),
    ]

    print(f"\nBankroll: ${NOTIONAL_BANKROLL}  |  Quarter Kelly  |  Trade cap: {MAX_BET_PCT*100:.0f}%  |  Sport cap: {MAX_SPORT_PCT*100:.0f}%\n")
    print(f"{'Scenario':<32} {'Kelly':>8} {'Q.Kelly':>8} {'%Bank':>7} {'$':>7} {'Notes':<30}")
    print("-" * 95)
    for label, edge, price in cases:
        r = size_bet(edge, price)
        flag = "⚠️ CAPPED" if r["capped"] else ("⊘ SKIP" if r["skip"] else "")
        print(f"{label:<32} {r['kelly_raw']*100:>7.1f}% {r['kelly_q']*100:>7.1f}% "
              f"{r['pct_bank']:>6.1f}% ${r['dollars']:>5.2f}  {flag}")

    # Per-sport daily cap demo — 4 MLB signals throughout the day
    print("\n\nPer-sport daily cap demo (4 MLB signals in sequence):\n")
    print(f"{'Signal':<25} {'Used before':>12} {'Cap remain':>12} {'% this bet':>12} {'$':>7} {'Flag':<20}")
    print("-" * 95)
    used = 0.0
    for i, (label, edge, price) in enumerate([
        ("Signal 1: +15% at 45¢",  0.15, 0.45),
        ("Signal 2: +12% at 60¢",  0.12, 0.60),
        ("Signal 3: +14% at 55¢",  0.14, 0.55),
        ("Signal 4: +18% at 40¢",  0.18, 0.40),
    ]):
        r = size_bet(edge, price, sport_used_pct=used)
        flag = ""
        if r["sport_capped"]:
            flag = "🛑 SPORT CAP"
        elif r["skip"]:
            flag = "⊘ SKIP"
        elif r["capped"]:
            flag = "⚠️ TRADE CAP"
        print(f"{label:<25} {used*100:>11.1f}% {(MAX_SPORT_PCT-used)*100:>11.1f}% "
              f"{r['pct_bank']:>11.1f}% ${r['dollars']:>5.2f}  {flag}")
        used += r["pct_bank"] / 100.0