"""
backtest.py
Backtesting framework for the NBA model.

Replays the 2024-25 season using real historical odds from the-odds-api.com
as the market proxy instead of the ELO component.

Usage:
  python backtest.py
  python backtest.py --edge-threshold=8
  python backtest.py --min-prob=40

Requires: ODDS_API_KEY in Replit Secrets
Get a free key at: https://the-odds-api.com (500 requests/month free)
"""

import os
import sys
import json
import time
import datetime
from pathlib import Path
from collections import defaultdict

BACKTEST_SEASON  = "2024-25"
CACHE_FILE       = Path("backtest_cache.json")
ODDS_CACHE_FILE  = Path("backtest_odds_cache.json")


# ── Game log fetching ─────────────────────────────────────────────────────────

def fetch_season_logs(season: str) -> list[dict]:
    try:
        from nba_api.stats.endpoints import leaguegamelog
    except ImportError:
        print("nba_api not installed.")
        return []

    print(f"[backtest] Fetching {season} game logs...")
    time.sleep(0.6)
    log = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star="Regular Season",
        direction="ASC",
    )
    df = log.get_data_frames()[0]
    games = []
    for _, row in df.iterrows():
        games.append({
            "team":       row["TEAM_NAME"],
            "date":       row["GAME_DATE"],
            "wl":         row["WL"],
            "matchup":    row["MATCHUP"],
            "home":       "vs." in row["MATCHUP"],
            "pts":        int(row.get("PTS", 0) or 0),
            "fga":        int(row.get("FGA", 0) or 0),
            "fta":        int(row.get("FTA", 0) or 0),
            "oreb":       int(row.get("OREB", 0) or 0),
            "tov":        int(row.get("TOV", 0) or 0),
            "plus_minus": int(row.get("PLUS_MINUS", 0) or 0),
        })
    games.sort(key=lambda g: g["date"])
    print(f"[backtest] Loaded {len(games)} game records.")
    return games


def load_or_fetch_games(season: str) -> list[dict]:
    if CACHE_FILE.exists():
        data = json.loads(CACHE_FILE.read_text())
        if data.get("season") == season:
            print(f"[backtest] Loaded {len(data['games'])} games from cache.")
            return data["games"]
    games = fetch_season_logs(season)
    CACHE_FILE.write_text(json.dumps({"season": season, "games": games}))
    return games


# ── Historical odds fetching ──────────────────────────────────────────────────

def fetch_historical_odds(api_key: str) -> dict[str, dict]:
    """
    Fetch historical NBA odds from the-odds-api.com.
    Returns {game_key: {home_prob, away_prob}} where
    game_key = f"{away_team}_{home_team}_{date}"

    Uses the /scores endpoint which includes completed games with scores,
    then maps to moneyline odds from the /historical/odds endpoint.

    Note: Free tier = 500 requests/month. We batch by week to minimize calls.
    """
    import requests

    if ODDS_CACHE_FILE.exists():
        data = json.loads(ODDS_CACHE_FILE.read_text())
        print(f"[backtest] Loaded {len(data)} historical odds from cache.")
        return data

    print("[backtest] Fetching historical odds from the-odds-api.com...")
    print("  This uses ~50 API requests from your 500/month free quota.")

    odds_by_game = {}

    # The odds API historical endpoint lets you fetch odds at a specific date
    # We'll sample ~weekly through the 2024-25 season (Oct 2024 - Apr 2025)
    import datetime

    dates_to_fetch = []
    start = datetime.date(2024, 10, 22)
    end   = datetime.date(2025, 4, 13)
    current = start
    while current <= end:
        dates_to_fetch.append(current.strftime("%Y-%m-%dT12:00:00Z"))
        current += datetime.timedelta(days=1)

    print(f"  Fetching odds for {len(dates_to_fetch)} dates...")
    fetched = 0

    for date_str in dates_to_fetch:
        try:
            url = "https://api.the-odds-api.com/v4/historical/sports/basketball_nba/odds"
            params = {
                "apiKey":    api_key,
                "regions":   "us",
                "markets":   "h2h",
                "date":      date_str,
                "oddsFormat": "decimal",
            }
            resp = requests.get(url, params=params, timeout=15)

            if resp.status_code == 401:
                print("  ❌ Invalid API key. Check ODDS_API_KEY in Replit Secrets.")
                return {}
            if resp.status_code == 422:
                continue  # No games on this date
            if resp.status_code != 200:
                continue

            data  = resp.json()
            games = data.get("data", [])

            for game in games:
                home_team  = game.get("home_team", "")
                away_team  = game.get("away_team", "")
                commence   = game.get("commence_time", "")[:10]  # YYYY-MM-DD

                # Get average market probability across bookmakers
                home_probs = []
                away_probs = []

                for bookmaker in game.get("bookmakers", []):
                    for market in bookmaker.get("markets", []):
                        if market.get("key") != "h2h":
                            continue
                        for outcome in market.get("outcomes", []):
                            dec_odds = outcome.get("price", 2.0)
                            prob     = 1 / dec_odds
                            if outcome["name"] == home_team:
                                home_probs.append(prob)
                            elif outcome["name"] == away_team:
                                away_probs.append(prob)

                if home_probs and away_probs:
                    # Average and normalize to remove vig
                    raw_home = sum(home_probs) / len(home_probs)
                    raw_away = sum(away_probs) / len(away_probs)
                    total    = raw_home + raw_away
                    norm_home = raw_home / total
                    norm_away = raw_away / total

                    key = f"{away_team}_{home_team}_{commence}"
                    odds_by_game[key] = {
                        "home_prob": round(norm_home, 4),
                        "away_prob": round(norm_away, 4),
                        "home_team": home_team,
                        "away_team": away_team,
                        "date":      commence,
                    }

            fetched += 1
            remaining = resp.headers.get("x-requests-remaining", "?")
            if fetched % 30 == 0:
                print(f"  Progress: {fetched}/{len(dates_to_fetch)} dates  |  API requests remaining: {remaining}")

            time.sleep(0.1)  # be polite to the API

        except Exception as e:
            print(f"  Error fetching {date_str}: {e}")
            continue

    print(f"[backtest] Fetched odds for {len(odds_by_game)} games.")
    ODDS_CACHE_FILE.write_text(json.dumps(odds_by_game))
    return odds_by_game


def normalize_team_name(name: str) -> str:
    """Map odds API team names to our model's team names."""
    mapping = {
        "Los Angeles Clippers": "Los Angeles Clippers",
        "LA Clippers":          "Los Angeles Clippers",
        "Los Angeles Lakers":   "Los Angeles Lakers",
        "LA Lakers":            "Los Angeles Lakers",
        "Golden State Warriors":"Golden State Warriors",
        "New Orleans Pelicans": "New Orleans Pelicans",
        "Oklahoma City Thunder":"Oklahoma City Thunder",
        "San Antonio Spurs":    "San Antonio Spurs",
        "New York Knicks":      "New York Knicks",
        "Brooklyn Nets":        "Brooklyn Nets",
        "Portland Trail Blazers":"Portland Trail Blazers",
    }
    return mapping.get(name, name)


# ── Model snapshot builder ────────────────────────────────────────────────────

def build_model_snapshot(games_before: list[dict]):
    from nba_model import build_elo_ratings, build_team_ratings, build_recent_form
    elos    = build_elo_ratings(games_before)
    ratings = build_team_ratings(games_before)
    form    = build_recent_form(games_before, elos)
    return elos, ratings, form


def predict_game(home_team: str, away_team: str,
                 elos: dict, ratings: dict, form: dict,
                 games_before: list[dict]) -> dict:
    from nba_model import (
        normalize_team, expected_score, rating_win_prob,
        recent_form_win_prob, rest_days, get_home_court,
        BASE_ELO, B2B_PENALTY, W_ELO, W_RATING, W_RECENT
    )

    home = normalize_team(home_team)
    away = normalize_team(away_team)

    home_elo     = elos.get(home, BASE_ELO)
    away_elo     = elos.get(away, BASE_ELO)
    adj_home_elo = home_elo + get_home_court(home)

    home_rest = rest_days(home, games_before)
    away_rest = rest_days(away, games_before)
    if home_rest == 0:
        adj_home_elo -= B2B_PENALTY * 1.5
    elif home_rest == 1:
        adj_home_elo -= B2B_PENALTY
    if away_rest == 0:
        away_elo -= B2B_PENALTY * 1.5
    elif away_rest == 1:
        away_elo -= B2B_PENALTY

    p_elo    = expected_score(adj_home_elo, away_elo)
    p_rating = rating_win_prob(home, away, ratings)
    p_recent = recent_form_win_prob(home, away, form)
    p_home   = W_ELO * p_elo + W_RATING * p_rating + W_RECENT * p_recent

    return {"home": home, "away": away, "p_home": round(p_home, 4)}


# ── Brier score ───────────────────────────────────────────────────────────────

def brier_score(predictions: list[tuple[float, int]]) -> float:
    if not predictions:
        return 0.0
    return sum((p - o) ** 2 for p, o in predictions) / len(predictions)


# ── Main backtest ─────────────────────────────────────────────────────────────

def run_backtest(season: str = BACKTEST_SEASON,
                 edge_threshold: float = 0.08,
                 min_prob: float = 0.40,
                 warmup_games: int = 50) -> dict:

    api_key = os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        print("⚠️  ODDS_API_KEY not set in Replit Secrets.")
        print("   Get a free key at https://the-odds-api.com")
        print("   Falling back to ELO-proxy backtest (less accurate).\n")

    games     = load_or_fetch_games(season)
    odds_data = fetch_historical_odds(api_key) if api_key else {}
    has_odds  = len(odds_data) > 0

    if has_odds:
        print(f"[backtest] Using real market odds for {len(odds_data)} games.")
    else:
        print("[backtest] Using ELO-proxy (no real odds). Results are approximate.")

    dates        = sorted(set(g["date"] for g in games))
    game_count   = defaultdict(int)
    matchups_seen = set()

    total_games   = 0
    correct       = 0
    bets_made     = 0
    total_pnl     = 0.0
    total_wagered = 0.0
    brier_data    = []
    by_edge       = defaultdict(lambda: {"bets": 0, "wins": 0, "pnl": 0.0})
    odds_matched  = 0

    print(f"\n[backtest] Running {season} season...")
    print(f"  Edge: {edge_threshold*100:.0f}%  |  Min prob: {min_prob*100:.0f}%  |  Warmup: {warmup_games}\n")

    for i, date in enumerate(dates):
        games_before = [g for g in games if g["date"] < date]
        if len(games_before) < 10:
            continue

        try:
            elos, ratings, form = build_model_snapshot(games_before)
        except Exception:
            continue

        todays_games = [g for g in games if g["date"] == date]

        for g in todays_games:
            from nba_model import normalize_team
            parts    = g["matchup"].split()
            opponent = normalize_team(parts[-1]) if parts else None
            if not opponent:
                continue

            if "vs." in g["matchup"]:
                home_team, away_team = g["team"], opponent
            else:
                away_team, home_team = g["team"], opponent

            key = "|".join(sorted([home_team, away_team])) + date
            if key in matchups_seen:
                continue
            matchups_seen.add(key)

            if game_count[home_team] < warmup_games or game_count[away_team] < warmup_games:
                game_count[home_team] += 1
                game_count[away_team] += 1
                continue

            game_count[home_team] += 1
            game_count[away_team] += 1

            try:
                pred = predict_game(home_team, away_team, elos, ratings, form, games_before)
            except Exception:
                continue

            home_game = next((g2 for g2 in todays_games
                             if g2["team"] == home_team and "vs." in g2["matchup"]), None)
            if not home_game:
                continue

            home_won = home_game["wl"] == "W"
            total_games += 1

            if pred["p_home"] > 0.5:
                correct += 1 if home_won else 0
            else:
                correct += 1 if not home_won else 0

            brier_data.append((pred["p_home"], 1 if home_won else 0))

            # ── Get market probability ────────────────────────────────────────
            if has_odds:
                # Try to find matching game in odds data
                odds_key = f"{away_team}_{home_team}_{date}"
                # Also try normalized names
                away_norm = normalize_team_name(away_team)
                home_norm = normalize_team_name(home_team)
                odds_key2 = f"{away_norm}_{home_norm}_{date}"

                odds_entry = odds_data.get(odds_key) or odds_data.get(odds_key2)
                if odds_entry:
                    market_home = odds_entry["home_prob"]
                    odds_matched += 1
                else:
                    market_home = pred["p_home"] * 0.95 + 0.5 * 0.05  # fallback
            else:
                # ELO-only as proxy (less accurate)
                from nba_model import expected_score, get_home_court, BASE_ELO
                h_elo = elos.get(pred["home"], BASE_ELO) + get_home_court(pred["home"])
                a_elo = elos.get(pred["away"], BASE_ELO)
                market_home = expected_score(h_elo, a_elo)

            # ── Determine edge and bet ────────────────────────────────────────
            edge = pred["p_home"] - market_home
            if edge > 0:
                side     = "yes"
                bet_prob = pred["p_home"]
                entry    = min(0.99, market_home + 0.01)
            else:
                side     = "no"
                bet_prob = 1 - pred["p_home"]
                entry    = min(0.99, (1 - market_home) + 0.01)

            if abs(edge) < edge_threshold or bet_prob < min_prob:
                continue

            won = (side == "yes" and home_won) or (side == "no" and not home_won)
            pnl = (1 - entry) if won else -entry

            bets_made     += 1
            total_pnl     += pnl
            total_wagered += entry

            bucket = f"{int(abs(edge)*100 // 5)*5}-{int(abs(edge)*100 // 5)*5+5}%"
            by_edge[bucket]["bets"] += 1
            by_edge[bucket]["pnl"]  += pnl
            if won:
                by_edge[bucket]["wins"] += 1

        if i % 20 == 0 and total_games > 0:
            acc = correct / total_games * 100
            roi = total_pnl / total_wagered * 100 if total_wagered else 0
            print(f"  [{date}] Games: {total_games}  Acc: {acc:.1f}%  Bets: {bets_made}  ROI: {roi:.1f}%")

    accuracy = correct / total_games if total_games else 0
    roi      = total_pnl / total_wagered if total_wagered else 0
    brier    = brier_score(brier_data)

    print(f"\n{'='*60}")
    print(f"  BACKTEST RESULTS — {season}")
    if has_odds:
        print(f"  Market: Real odds ({odds_matched} games matched)")
    else:
        print(f"  Market: ELO proxy (approximate)")
    print(f"{'='*60}")
    print(f"  Total games:    {total_games}")
    print(f"  Accuracy:       {accuracy*100:.1f}%")
    print(f"  Brier score:    {brier:.4f}  (random=0.25, perfect=0.00)")
    print(f"  Total bets:     {bets_made}")
    if bets_made > 0:
        print(f"  Total P&L:      ${total_pnl:.2f}")
        print(f"  Total wagered:  ${total_wagered:.2f}")
        print(f"  ROI:            {roi*100:.1f}%")
        print(f"\n  Edge bucket breakdown:")
        for bucket in sorted(by_edge.keys()):
            b  = by_edge[bucket]
            wr = b["wins"] / b["bets"] * 100 if b["bets"] else 0
            print(f"    {bucket:<12} {b['bets']:>4} bets  {wr:.0f}% win  ${b['pnl']:.2f} P&L")
    print(f"{'='*60}")

    return {
        "season": season, "accuracy": accuracy, "brier": brier,
        "bets": bets_made, "pnl": total_pnl, "roi": roi,
    }


if __name__ == "__main__":
    edge_threshold = 0.08
    min_prob       = 0.40

    for arg in sys.argv[1:]:
        if arg.startswith("--edge-threshold="):
            edge_threshold = float(arg.split("=")[1]) / 100
        elif arg.startswith("--min-prob="):
            min_prob = float(arg.split("=")[1]) / 100

    run_backtest(
        season=BACKTEST_SEASON,
        edge_threshold=edge_threshold,
        min_prob=min_prob,
    )