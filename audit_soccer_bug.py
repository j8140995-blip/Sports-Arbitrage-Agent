"""
audit_soccer_bug.py
Analyze past soccer trades for the substring-matching bug.

The bug in soccer_scanner.py caused markets to be mislabeled when:
1. The market ticker's suffix didn't match the home team's 3-letter abbrev, AND
2. The home team's first word appeared as a substring in the away team's name
   (e.g. "real" in "villarreal"), OR vice versa

This script identifies historical trades that may have been affected.

Usage:
    python audit_soccer_bug.py
"""

import json
import re
from pathlib import Path
from collections import Counter

PAPER_LOG = Path("paper_trades.json")


def first_word_collision(home: str, away: str) -> tuple[bool, str]:
    """Check if either team's first word is a substring of the other's full name."""
    if not home or not away:
        return False, ""
    home_first = home.lower().split()[0] if home else ""
    away_first = away.lower().split()[0] if away else ""
    if len(home_first) < 3 or len(away_first) < 3:
        return False, ""
    home_full = home.lower()
    away_full = away.lower()
    # Real substring collision (not equal, not same team)
    if home_first != away_first:
        if home_first in away_full:
            return True, f"'{home_first}' substring-matched in '{away_full}'"
        if away_first in home_full:
            return True, f"'{away_first}' substring-matched in '{home_full}'"
    return False, ""


def extract_ticker_suffix(ticker: str) -> str:
    """Get the last part after the final dash."""
    if "-" not in ticker:
        return ""
    return ticker.split("-")[-1].upper()


def main():
    if not PAPER_LOG.exists():
        print("No paper_trades.json found.")
        return

    trades = json.loads(PAPER_LOG.read_text())
    soccer = [t for t in trades if t.get("sport") == "SOCCER"]

    print(f"\n{'='*110}")
    print(f"  Soccer trade audit — checking for substring-matching bug")
    print(f"  Total soccer trades: {len(soccer)}")
    print(f"{'='*110}\n")

    # Flag any trade where the home/away teams have a substring collision
    affected = []
    clean = []
    unknown = []
    for t in soccer:
        home = t.get("home_team", "")
        away = t.get("away_team", "")
        if not home or not away:
            unknown.append(t)
            continue
        collides, reason = first_word_collision(home, away)
        if collides:
            affected.append((t, reason))
        else:
            clean.append(t)

    # Print suspect trades
    if affected:
        print(f"⚠️  POTENTIALLY AFFECTED TRADES ({len(affected)}):\n")
        print(f"{'Ticker':<42} {'Home':<22} {'Away':<22} {'Side':<5} "
              f"{'Status':<8} {'P&L':>8}  Why")
        print("-" * 130)
        for t, reason in affected:
            ticker = t.get("ticker", "")
            home   = t.get("home_team", "")[:22]
            away   = t.get("away_team", "")[:22]
            side   = t.get("side", "")
            status = t.get("outcome") or t.get("status", "?")
            pnl    = t.get("pnl")
            pnl_s  = f"${pnl:+.2f}" if pnl is not None else "  —  "
            print(f"{ticker:<42} {home:<22} {away:<22} {side:<5} "
                  f"{status:<8} {pnl_s:>8}  {reason}")
    else:
        print("✅ No trades found with potential substring collisions.")

    # Summary stats
    print(f"\n{'='*110}")
    print(f"  SUMMARY:")
    print(f"    Total soccer trades:           {len(soccer)}")
    print(f"    Potentially affected:          {len(affected)}")
    print(f"    Clean (no collision possible): {len(clean)}")
    if unknown:
        print(f"    Missing home/away data:        {len(unknown)}")

    # Performance of affected vs clean
    def stats(trade_list):
        resolved = [t for t in trade_list if t.get("pnl") is not None]
        if not resolved:
            return 0, 0, 0, 0.0, 0.0
        wins  = sum(1 for t in resolved if t.get("outcome") == "win")
        pnl   = sum(t.get("pnl", 0) for t in resolved)
        cost  = sum(t.get("cost_dollars", 0) for t in resolved)
        roi   = (pnl / cost * 100) if cost else 0
        winpct = (wins / len(resolved) * 100) if resolved else 0
        return len(resolved), wins, winpct, pnl, roi

    if affected:
        n, w, wp, pnl, roi = stats([t for t, _ in affected])
        print(f"\n  Affected performance:  {n} resolved, {w} wins ({wp:.0f}%), "
              f"P&L ${pnl:+.2f}, ROI {roi:+.1f}%")

    if clean:
        n, w, wp, pnl, roi = stats(clean)
        print(f"  Clean performance:     {n} resolved, {w} wins ({wp:.0f}%), "
              f"P&L ${pnl:+.2f}, ROI {roi:+.1f}%")

    print(f"{'='*110}\n")

    # Also list unique team pairings that flagged, for sanity check
    if affected:
        pairs = Counter((t.get("home_team",""), t.get("away_team","")) for t, _ in affected)
        print("  Unique affected matchups:")
        for (h, a), count in pairs.most_common():
            print(f"    {h} vs {a}  ×{count}")


if __name__ == "__main__":
    main()