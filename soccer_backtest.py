"""
soccer_backtest.py
Backtesting framework for the soccer Poisson model.

Replays the current season match by match, building team strengths
only from games BEFORE each target date (no lookahead bias).

Metrics:
- Accuracy: % of games where model picked the winner
- Brier score: probability calibration (lower = better)
- ROI simulation: using Kalshi mid as market proxy
- Results by league

Usage:
  python soccer_backtest.py
  python soccer_backtest.py --league EPL
  python soccer_backtest.py --edge-threshold 8
"""

import sys
import json
import math
import time
import datetime
from pathlib import Path
from collections import defaultdict

CACHE_FILE = Path("soccer_model_cache.json")


def load_cached_results() -> dict:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text()).get("leagues", {})
    return {}


def build_snapshot(results_before: list[dict]) -> dict:
    """Build team strengths from results before a given date."""
    from soccer_model import build_team_strengths, league_avg_goals
    strengths = build_team_strengths(results_before)
    avg       = league_avg_goals(results_before)
    return {"strengths": strengths, "avg_goals": avg}


def predict_snapshot(home_team: str, away_team: str, snapshot: dict) -> dict:
    """Predict using a model snapshot."""
    from soccer_model import predict_match
    return predict_match(
        home_team, away_team,
        strengths=snapshot["strengths"],
        league_avg=snapshot["avg_goals"],
    )


def brier_score(predictions: list[tuple[float, int]]) -> float:
    if not predictions:
        return 0.0
    return sum((p - o) ** 2 for p, o in predictions) / len(predictions)


def run_soccer_backtest(league: str = None,
                        edge_threshold: float = 0.08,
                        min_prob: float = 0.38,
                        warmup_matches: int = 10) -> dict:
    leagues_data = load_cached_results()

    if not leagues_data:
        print("[backtest] No cached data. Run: python soccer_model.py first.")
        return {}

    leagues_to_test = [league] if league else list(leagues_data.keys())
    all_results     = {}

    for lg in leagues_to_test:
        if lg not in leagues_data:
            print(f"[backtest] {lg} not in cache.")
            continue

        results = leagues_data[lg].get("results", [])
        if not results:
            continue

        print(f"\n[backtest] Testing {lg} ({len(results)} matches)...")

        # Get unique dates
        dates       = sorted(set(r["date"] for r in results))
        team_count  = defaultdict(int)  # games played per team
        seen        = set()

        total      = 0
        correct    = 0
        bets       = 0
        total_pnl  = 0.0
        total_wag  = 0.0
        brier_data = []
        by_edge    = defaultdict(lambda: {"bets": 0, "wins": 0, "pnl": 0.0})

        for date in dates:
            results_before = [r for r in results if r["date"] < date]
            if len(results_before) < warmup_matches * 2:
                # Update counts
                for r in results:
                    if r["date"] == date:
                        team_count[r["home_team"]] += 1
                        team_count[r["away_team"]] += 1
                continue

            snapshot = build_snapshot(results_before)
            strengths = snapshot["strengths"]

            todays = [r for r in results if r["date"] == date]
            for r in todays:
                key = f"{r['home_team']}_{r['away_team']}_{date}"
                if key in seen:
                    continue
                seen.add(key)

                home = r["home_team"]
                away = r["away_team"]

                # Skip if not enough data for either team
                if team_count[home] < warmup_matches or team_count[away] < warmup_matches:
                    team_count[home] += 1
                    team_count[away] += 1
                    continue

                team_count[home] += 1
                team_count[away] += 1

                if home not in strengths or away not in strengths:
                    continue

                try:
                    pred = predict_snapshot(home, away, snapshot)
                except Exception:
                    continue

                # Actual result
                if r["home_goals"] > r["away_goals"]:
                    actual = "home_win"
                elif r["home_goals"] == r["away_goals"]:
                    actual = "draw"
                else:
                    actual = "away_win"

                total += 1

                # Accuracy (pick the most likely outcome)
                probs     = {"home_win": pred["home_win"], "draw": pred["draw"], "away_win": pred["away_win"]}
                predicted = max(probs, key=probs.get)
                if predicted == actual:
                    correct += 1

                # Brier on home win probability
                brier_data.append((pred["home_win"], 1 if actual == "home_win" else 0))

                # Simulate bet — find best edge vs uniform baseline
                # Use draw-adjusted market proxy: home 45%, draw 25%, away 30%
                market_proxy = {"home_win": 0.45, "draw": 0.25, "away_win": 0.30}

                best_outcome = None
                best_edge    = 0.0
                for outcome, model_p in probs.items():
                    if outcome == "draw":
                        continue  # skip draw bets
                    market_p = market_proxy[outcome]
                    edge     = model_p - market_p
                    bet_prob = model_p if edge > 0 else 1 - model_p
                    if abs(edge) > best_edge and abs(edge) >= edge_threshold and bet_prob >= min_prob:
                        best_edge    = abs(edge)
                        best_outcome = outcome

                if best_outcome:
                    model_p  = probs[best_outcome]
                    entry    = min(0.99, model_p + 0.01)
                    won      = actual == best_outcome
                    pnl      = (1 - entry) if won else -entry
                    bets    += 1
                    total_pnl += pnl
                    total_wag += entry

                    bucket = f"{int(best_edge*100//5)*5}-{int(best_edge*100//5)*5+5}%"
                    by_edge[bucket]["bets"] += 1
                    by_edge[bucket]["pnl"]  += pnl
                    if won:
                        by_edge[bucket]["wins"] += 1

        accuracy = correct / total if total else 0
        roi      = total_pnl / total_wag if total_wag else 0
        brier    = brier_score(brier_data)

        print(f"\n{'='*55}")
        print(f"  BACKTEST: {lg}")
        print(f"{'='*55}")
        print(f"  Matches evaluated: {total}")
        print(f"  Accuracy:          {accuracy*100:.1f}%")
        print(f"  Brier score:       {brier:.4f}  (random=0.25)")
        print(f"  Bets placed:       {bets}")
        if bets > 0:
            print(f"  Total P&L:         ${total_pnl:.2f}")
            print(f"  ROI:               {roi*100:.1f}%")
            print(f"\n  Edge breakdown:")
            for bucket in sorted(by_edge.keys()):
                b  = by_edge[bucket]
                wr = b["wins"] / b["bets"] * 100 if b["bets"] else 0
                print(f"    {bucket:<12} {b['bets']:>4} bets  {wr:.0f}% win  ${b['pnl']:.2f}")
        print(f"{'='*55}")

        all_results[lg] = {"accuracy": accuracy, "brier": brier, "bets": bets, "roi": roi}

    return all_results


if __name__ == "__main__":
    league         = None
    edge_threshold = 0.08
    min_prob       = 0.38

    for arg in sys.argv[1:]:
        if arg.startswith("--league="):
            league = arg.split("=")[1].upper()
        elif arg.startswith("--edge-threshold="):
            edge_threshold = float(arg.split("=")[1]) / 100
        elif arg.startswith("--min-prob="):
            min_prob = float(arg.split("=")[1]) / 100

    run_soccer_backtest(league=league,
                        edge_threshold=edge_threshold,
                        min_prob=min_prob)