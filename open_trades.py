"""
open_trades.py
Show all unresolved trades grouped by sport, sorted by game date.

Useful for seeing your live exposure across upcoming events at a glance.
Unlike today_trades.py (which filters to today only), this shows everything
still open regardless of when the game is scheduled.

Usage:
    python open_trades.py              # all open trades, all sports
    python open_trades.py --sport NBA  # filter to one sport
    python open_trades.py --upcoming   # only games in the future
"""

import json
import re
import sys
from pathlib import Path
from datetime import date, datetime
from collections import defaultdict

PAPER_LOG = Path("paper_trades.json")

MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def game_date_from_ticker(ticker: str) -> date | None:
    """Parse YYMMM (e.g. 26APR26) from a Kalshi ticker. Returns None if unparseable."""
    if not ticker:
        return None
    # Tickers look like KXEPLGAME-26APR26ARSNEW-ARS, KXMLBGAME-26APR261910COLNYM-COL, etc.
    m = re.search(r'-(\d{2})([A-Z]{3})(\d{2})', ticker)
    if not m:
        return None
    yy, mon, dd = m.groups()
    month = MONTH_MAP.get(mon)
    if not month:
        return None
    try:
        return date(2000 + int(yy), month, int(dd))
    except ValueError:
        return None


def parse_args():
    args = {"sport": None, "upcoming": False}
    for i, a in enumerate(sys.argv[1:]):
        if a == "--sport" and i + 1 < len(sys.argv) - 1:
            args["sport"] = sys.argv[i + 2].upper()
        elif a == "--upcoming":
            args["upcoming"] = True
    return args


def main():
    args = parse_args()

    if not PAPER_LOG.exists():
        print("No paper_trades.json found.")
        return

    trades = json.loads(PAPER_LOG.read_text())
    today = date.today()

    # Filter: unresolved only
    open_trades = [t for t in trades if t.get("pnl") is None]

    if args["sport"]:
        open_trades = [t for t in open_trades if t.get("sport") == args["sport"]]

    if args["upcoming"]:
        # Only games today or later
        def is_upcoming(t):
            d = game_date_from_ticker(t.get("ticker", ""))
            return d is None or d >= today
        open_trades = [t for t in open_trades if is_upcoming(t)]

    if not open_trades:
        msg = "No open trades"
        if args["sport"]:
            msg += f" for {args['sport']}"
        if args["upcoming"]:
            msg += " (upcoming)"
        print(msg + ".")
        return

    # Group by sport
    by_sport = defaultdict(list)
    for t in open_trades:
        by_sport[t.get("sport", "?")].append(t)

    # Sort sports by name for consistent output
    sports = sorted(by_sport.keys())

    print(f"\n{'='*110}")
    print(f"  OPEN TRADES  —  {len(open_trades)} total"
          + (f"  (sport: {args['sport']})" if args['sport'] else "")
          + ("  (upcoming only)" if args["upcoming"] else ""))
    print(f"{'='*110}")

    grand_total_cost = 0.0

    for sport in sports:
        trades_in_sport = by_sport[sport]

        # Sort by game date (None values sort first)
        trades_in_sport.sort(key=lambda t: (
            game_date_from_ticker(t.get("ticker", "")) or date(1900, 1, 1),
            t.get("ticker", "")
        ))

        sport_cost = sum(t.get("cost_dollars", 0) for t in trades_in_sport)
        grand_total_cost += sport_cost

        print(f"\n  {sport}  —  {len(trades_in_sport)} open trades, ${sport_cost:.2f} deployed")
        print(f"  {'-'*106}")
        print(f"  {'Game Date':<11} {'Ticker':<48} {'Side':<4} "
              f"{'Size':<10} {'Cost':>7}  {'Model':>7} {'Cons':>7} {'Edge':>7}")
        print(f"  {'-'*106}")

        for t in trades_in_sport:
            ticker      = t.get("ticker", "")[:48]
            side        = t.get("side", "?").upper()
            contracts   = t.get("contracts", 0)
            price       = int(t.get("entry_price_cents", 0))
            cost        = t.get("cost_dollars", 0)
            model_p     = t.get("model_prob", 0) * 100
            consensus   = t.get("consensus_prob", 0) * 100
            edge        = t.get("edge", 0) * 100
            game_date   = game_date_from_ticker(t.get("ticker", ""))
            game_date_s = game_date.isoformat() if game_date else "?"

            size_str = f"{contracts:>3}c@{price:>2}¢"
            print(f"  {game_date_s:<11} {ticker:<48} {side:<4} "
                  f"{size_str:<10} ${cost:>5.2f}  {model_p:>5.1f}%  {consensus:>5.1f}% "
                  f"{edge:>+5.1f}%")

    print(f"\n{'='*110}")
    print(f"  TOTAL deployed across all open trades: ${grand_total_cost:.2f}")
    print(f"{'='*110}")


if __name__ == "__main__":
    main()