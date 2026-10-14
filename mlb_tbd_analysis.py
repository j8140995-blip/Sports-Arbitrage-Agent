"""
mlb_tbd_analysis.py
Analyze whether MLB trades with TBD starters perform differently from
trades where both starters are known.

Detects TBD by checking the trade's edge magnitude — TBD-flagged trades
in the scanner showed FIP 5.00 for unknown starters, which usually
correlates with extreme edges. But the cleanest signal is checking
the trade record's pct_bankroll, edge, and consensus values.

Since the trade log doesn't store TBD status directly, we infer it:
  - The MLB scanner appended ⚠️TBD when home_sp_known=False or away_sp_known=False
  - That info isn't preserved in paper_trades.json (yet)

So this script can only do an indirect analysis until we add a `tbd_flag`
field to logged trades. For now, we show a comparison by edge magnitude
and discuss whether the data supports a TBD penalty.

Usage:
    python mlb_tbd_analysis.py
"""

import json
from pathlib import Path

PAPER_LOG = Path("paper_trades.json")


def main():
    if not PAPER_LOG.exists():
        print("No paper_trades.json found.")
        return

    trades = json.loads(PAPER_LOG.read_text())
    mlb = [t for t in trades if t.get("sport") == "MLB" and t.get("pnl") is not None]

    if not mlb:
        print("No resolved MLB trades.")
        return

    print(f"\n{'='*80}")
    print(f"  MLB TRADE ANALYSIS — {len(mlb)} resolved trades")
    print(f"{'='*80}\n")

    # Bucket by absolute edge magnitude
    # Heuristic: TBD-flagged trades often produced larger absolute edges
    # because FIP 5.00 substitution skewed model probabilities away from Vegas.
    buckets = {
        "Small (<8%)":   [],
        "Medium (8-12%)":[],
        "Large (12-18%)":[],
        "Huge (≥18%)":   [],
    }
    for t in mlb:
        edge = abs(t.get("edge", 0)) * 100
        if edge < 8:
            buckets["Small (<8%)"].append(t)
        elif edge < 12:
            buckets["Medium (8-12%)"].append(t)
        elif edge < 18:
            buckets["Large (12-18%)"].append(t)
        else:
            buckets["Huge (≥18%)"].append(t)

    print(f"  {'Edge band':<16} {'N':>4}  {'Wins':>5}  {'Win%':>6}  "
          f"{'Total P&L':>11}  {'Avg P&L':>9}  {'ROI':>7}")
    print("-" * 80)

    for label, ts in buckets.items():
        n = len(ts)
        if n == 0:
            print(f"  {label:<16} {n:>4}")
            continue
        wins   = sum(1 for t in ts if t.get("outcome") == "win")
        pnl    = sum(t.get("pnl", 0) for t in ts)
        cost   = sum(t.get("cost_dollars", 0) for t in ts)
        avg    = pnl / n
        roi    = pnl / cost * 100 if cost else 0
        print(f"  {label:<16} {n:>4}  {wins:>5}  {wins/n*100:>5.0f}%  "
              f"${pnl:>+8.2f}  ${avg:>+7.2f}  {roi:>+5.1f}%")

    print(f"\n{'='*80}")
    print(f"  HUGE EDGES (≥15%) DETAIL — these are mostly TBD games")
    print(f"{'='*80}\n")

    huge = [t for t in mlb if abs(t.get("edge", 0)) >= 0.15]
    if huge:
        print(f"  {'Date':<10} {'Ticker':<35} {'Edge':>6} {'Side':<3} "
              f"{'Outcome':<6} {'P&L':>8}")
        print("-" * 85)
        for t in sorted(huge, key=lambda x: x.get("game_date", "")):
            edge = t.get("edge", 0) * 100
            print(f"  {t.get('game_date',''):<10} {t.get('ticker','')[:35]:<35} "
                  f"{edge:>+5.1f}% {t.get('side',''):<3} "
                  f"{t.get('outcome',''):<6} ${t.get('pnl', 0):>+6.2f}")

        wins = sum(1 for t in huge if t.get("outcome") == "win")
        pnl  = sum(t.get("pnl", 0) for t in huge)
        cost = sum(t.get("cost_dollars", 0) for t in huge)
        print(f"\n  Summary: {len(huge)} trades, {wins} wins ({wins/len(huge)*100:.0f}%), "
              f"P&L ${pnl:+.2f}, ROI {pnl/cost*100 if cost else 0:+.1f}%")


if __name__ == "__main__":
    main()