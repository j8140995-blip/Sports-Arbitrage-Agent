"""
clv_by_prob_bucket.py
Analyze whether the 75% win-prob filter on NBA signals is justified.

For each sport, groups resolved trades by the model's predicted win probability
on the side actually bet, and reports CLV + win rate per bucket.

If the lower-confidence buckets show strong positive CLV, the 75% filter
is leaving money on the table. If they're negative/zero, the filter is doing
its job.

Usage:
    python clv_by_prob_bucket.py
"""

import json
from pathlib import Path
from collections import defaultdict

PAPER_LOG = Path("paper_trades.json")


def bet_prob(trade: dict) -> float:
    """Return the model's probability on the side actually bet."""
    model_p = trade.get("model_prob", 0.5)
    side = trade.get("side", "yes")
    return model_p if side == "yes" else 1.0 - model_p


def bucket_label(p: float) -> str:
    """Return a 5-pct-point bucket label, e.g. '55-60%'."""
    lo = int(p * 100 // 5 * 5)
    hi = lo + 5
    return f"{lo:>2d}-{hi:<2d}%"


def compute_clv(trade: dict) -> float:
    """Compute CLV as (close_price - entry_price), expressed as %pts."""
    # Different logs use different field names. Try several.
    entry = trade.get("entry_price_cents")
    close = trade.get("close_price_cents") or trade.get("closing_price_cents")
    if entry is None or close is None:
        return trade.get("clv", 0) or 0
    return close - entry


def main():
    if not PAPER_LOG.exists():
        print("No paper_trades.json found.")
        return

    trades = json.loads(PAPER_LOG.read_text())

    by_sport_bucket = defaultdict(list)
    for t in trades:
        sport = t.get("sport", "?")
        pnl   = t.get("pnl")
        if pnl is None:
            continue  # unresolved
        outcome = t.get("outcome", "")
        if outcome not in ("win", "loss"):
            continue
        p     = bet_prob(t)
        label = bucket_label(p)
        clv   = compute_clv(t)
        by_sport_bucket[(sport, label)].append({
            "pnl":     pnl,
            "clv":     clv,
            "win":     outcome == "win",
            "cost":    t.get("cost_dollars", 0),
            "prob":    p,
        })

    # Group by sport, show buckets sorted by probability
    sports = sorted(set(k[0] for k in by_sport_bucket.keys()))

    for sport in sports:
        sport_buckets = [(b, v) for (s, b), v in by_sport_bucket.items() if s == sport]
        if not sport_buckets:
            continue
        # Sort by bucket (numeric)
        sport_buckets.sort(key=lambda x: int(x[0].split("-")[0]))

        print(f"\n{'='*95}")
        print(f"  {sport}  —  CLV by model win-probability bucket")
        print(f"{'='*95}")
        print(f"  {'Bucket':<10} {'N':>4}  {'Wins':>4}  {'Win%':>6}  "
              f"{'Mean CLV':>10}  {'Total P&L':>11}  {'ROI':>7}")
        print(f"  {'-'*85}")

        all_trades = []
        for bucket, trades_in_bucket in sport_buckets:
            all_trades.extend(trades_in_bucket)
            n       = len(trades_in_bucket)
            wins    = sum(1 for t in trades_in_bucket if t["win"])
            win_pct = wins / n * 100 if n else 0
            mean_clv = sum(t["clv"] for t in trades_in_bucket) / n if n else 0
            pnl     = sum(t["pnl"] for t in trades_in_bucket)
            cost    = sum(t["cost"] for t in trades_in_bucket)
            roi     = pnl / cost * 100 if cost else 0
            clv_str = f"{mean_clv:>+7.1f}%"
            pnl_str = f"${pnl:>+8.2f}"
            roi_str = f"{roi:>+5.1f}%"

            # Flag low-prob buckets
            lo_pct = int(bucket.split("-")[0])
            flag = " 🚩" if lo_pct < 75 and sport == "NBA" else ""
            print(f"  {bucket:<10} {n:>4}  {wins:>4}  {win_pct:>5.0f}%  "
                  f"{clv_str:>10}  {pnl_str:>11}  {roi_str:>7}{flag}")

        # Overall totals
        n = len(all_trades)
        wins = sum(1 for t in all_trades if t["win"])
        mean_clv = sum(t["clv"] for t in all_trades) / n if n else 0
        pnl = sum(t["pnl"] for t in all_trades)
        cost = sum(t["cost"] for t in all_trades)
        roi = pnl / cost * 100 if cost else 0
        print(f"  {'-'*85}")
        print(f"  {'TOTAL':<10} {n:>4}  {wins:>4}  {wins/n*100 if n else 0:>5.0f}%  "
              f"{mean_clv:>+7.1f}%  ${pnl:>+8.2f}  {roi:>+5.1f}%")

    # NBA-specific: under-75 vs over-75 comparison
    print(f"\n{'='*95}")
    print(f"  NBA FILTER JUSTIFICATION CHECK")
    print(f"{'='*95}")
    nba_trades = [t for t in trades if t.get("sport") == "NBA" and t.get("pnl") is not None]
    under_75 = [t for t in nba_trades if bet_prob(t) < 0.75]
    over_75  = [t for t in nba_trades if bet_prob(t) >= 0.75]

    def stats(trades_):
        n = len(trades_)
        if not n:
            return None
        wins = sum(1 for t in trades_ if t.get("outcome") == "win")
        pnl = sum(t.get("pnl", 0) for t in trades_)
        cost = sum(t.get("cost_dollars", 0) for t in trades_)
        clvs = [compute_clv(t) for t in trades_]
        mean_clv = sum(clvs) / len(clvs) if clvs else 0
        return n, wins, pnl, cost, mean_clv

    u = stats(under_75)
    o = stats(over_75)

    if u:
        print(f"  Under 75%:  {u[0]:>3} trades, {u[1]} wins ({u[1]/u[0]*100:.0f}%), "
              f"P&L ${u[2]:+.2f}, ROI {u[2]/u[3]*100 if u[3] else 0:+.1f}%, "
              f"mean CLV {u[4]:+.1f}%")
    if o:
        print(f"  Over  75%:  {o[0]:>3} trades, {o[1]} wins ({o[1]/o[0]*100:.0f}%), "
              f"P&L ${o[2]:+.2f}, ROI {o[2]/o[3]*100 if o[3] else 0:+.1f}%, "
              f"mean CLV {o[4]:+.1f}%")

    if u and o:
        print(f"\n  Verdict:")
        if u[4] > 0 and u[4] >= o[4] - 2:
            print(f"  ⚠️  Under-75% bucket has healthy CLV ({u[4]:+.1f}%). "
                  f"The 75% filter may be leaving money on the table.")
        elif u[4] < -2 and o[4] > 0:
            print(f"  ✅ Under-75% bucket has negative CLV ({u[4]:+.1f}%). "
                  f"The 75% filter is doing its job.")
        else:
            print(f"  🤷 Results inconclusive. Under-75: CLV {u[4]:+.1f}%, "
                  f"Over-75: CLV {o[4]:+.1f}%. More data needed.")


if __name__ == "__main__":
    main()