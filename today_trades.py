"""
today_trades.py
Show all trades for games taking place on a specific date.

Filters by the actual game date parsed from the Kalshi ticker
(e.g. KXEPLGAME-26APR25... = April 25, 2026), NOT by when the trade
was logged.

Usage:
    python today_trades.py              # games today
    python today_trades.py 2026-04-25   # games on April 25
"""

import json
import re
import sys
import datetime
from pathlib import Path
from collections import defaultdict

PAPER_LOG = Path("paper_trades.json")

MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def game_date_from_ticker(ticker: str) -> str:
    """
    Parse the game date from a Kalshi ticker.
    Ticker format: KX<SPORT>GAME-<YY><MON><DD>...
    e.g. KXEPLGAME-26APR25WOLTOT-TOT → 2026-04-25
    e.g. KXMLBGAME-26APR231545LADSF-LAD → 2026-04-23
    e.g. KXNBAGAME-26APR22PHXOKC-OKC → 2026-04-22
    """
    if not ticker:
        return ""
    # Find -YYMONDD pattern after the sport prefix
    m = re.search(r'-(\d{2})([A-Z]{3})(\d{2})', ticker)
    if not m:
        return ""
    yy, mon, dd = m.group(1), m.group(2), m.group(3)
    month = MONTH_MAP.get(mon)
    if not month:
        return ""
    try:
        return f"20{yy}-{month:02d}-{int(dd):02d}"
    except ValueError:
        return ""


def main():
    if not PAPER_LOG.exists():
        print("No paper_trades.json found.")
        return

    # Parse date arg or default to today
    if len(sys.argv) > 1:
        target_date = sys.argv[1]
    else:
        target_date = datetime.date.today().isoformat()

    trades = json.loads(PAPER_LOG.read_text())

    # Filter by GAME date parsed from ticker
    target_trades = []
    for t in trades:
        gdate = game_date_from_ticker(t.get("ticker", ""))
        if gdate == target_date:
            target_trades.append(t)
        elif not gdate and t.get("game_date") == target_date:
            # Fallback for trades where ticker doesn't parse (mentions etc.)
            target_trades.append(t)

    if not target_trades:
        print(f"No trades for games on {target_date}.")
        return

    # Group by sport
    by_sport = defaultdict(list)
    for t in target_trades:
        by_sport[t.get("sport", "?")].append(t)

    print(f"\n{'='*100}")
    print(f"  Trades for games on {target_date}  |  {len(target_trades)} total")
    print(f"{'='*100}")

    total_dollars = 0.0
    total_resolved = 0
    total_wins = 0
    total_pnl = 0.0

    for sport in sorted(by_sport.keys()):
        sport_trades = by_sport[sport]
        sport_dollars = sum(t.get("cost_dollars", 0) for t in sport_trades)
        total_dollars += sport_dollars

        print(f"\n  {sport}  ({len(sport_trades)} trades, ${sport_dollars:.2f} deployed)")
        print(f"  {'-'*96}")

        for t in sport_trades:
            ticker    = t.get("ticker", "")
            side      = t.get("side", "").upper()
            edge      = t.get("edge", 0) * 100
            contracts = t.get("contracts", 0)
            entry     = t.get("entry_price_cents", 0)
            cost      = t.get("cost_dollars", 0)
            status    = t.get("status", "?")
            outcome   = t.get("outcome", "")
            pnl       = t.get("pnl", None)
            model     = t.get("model_prob", 0) * 100
            consens   = t.get("consensus", t.get("market_mid", 0)) * 100

            # Status flag and P&L aggregation
            if status == "open":
                flag = "⏳ OPEN"
            elif outcome == "win":
                flag = "✅ WIN"
                total_wins += 1
                total_resolved += 1
                if pnl is not None:
                    total_pnl += pnl
            elif outcome == "loss":
                flag = "❌ LOSS"
                total_resolved += 1
                if pnl is not None:
                    total_pnl += pnl
            else:
                flag = "?"

            pnl_str = f"P&L: ${pnl:+.2f}" if pnl is not None else "             "

            print(f"    {ticker:<42} {side:<3}  {contracts:>3}c@{entry:>2}¢  ${cost:>5.2f}  "
                  f"model={model:>5.1f}%  consensus={consens:>5.1f}%  edge={edge:>+5.1f}%  "
                  f"{flag:<8}  {pnl_str}")

    # Summary footer
    print(f"\n{'='*100}")
    print(f"  TOTAL deployed on {target_date} games: ${total_dollars:.2f}")
    if total_resolved > 0:
        win_pct = total_wins / total_resolved * 100
        print(f"  Resolved: {total_resolved}/{len(target_trades)}  "
              f"|  Wins: {total_wins} ({win_pct:.0f}%)  |  P&L: ${total_pnl:+.2f}")
    else:
        print(f"  All {len(target_trades)} trades still open.")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()