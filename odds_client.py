"""
odds_client.py
Fetches current NBA Vegas lines from the-odds-api.com free tier.
Uses these as the market consensus instead of Kalshi mid prices.

Why Vegas > Kalshi mid:
- Vegas lines are set by professional oddsmakers with massive liquidity
- Kalshi mid prices can be stale or reflect retail sentiment
- When our model disagrees with Vegas, that's a stronger signal

Free tier: 500 requests/month
Each bot run uses 1 request to get all NBA games for the day.

Now stores commence_time so downstream scanners can filter out live games.
"""

import os
import json
import time
import requests
from pathlib import Path
from typing import Optional

ODDS_CACHE_FILE = Path("today_odds_cache.json")
CACHE_TTL_MIN   = 30  # refresh odds every 30 minutes


# ── Name normalization ────────────────────────────────────────────────────────
# The odds API uses full team names but sometimes different from nba_api

ODDS_TO_MODEL = {
    "Los Angeles Clippers":   "Los Angeles Clippers",
    "LA Clippers":            "Los Angeles Clippers",
    "Los Angeles Lakers":     "Los Angeles Lakers",
    "LA Lakers":              "Los Angeles Lakers",
    "Golden State Warriors":  "Golden State Warriors",
    "New Orleans Pelicans":   "New Orleans Pelicans",
    "Oklahoma City Thunder":  "Oklahoma City Thunder",
    "San Antonio Spurs":      "San Antonio Spurs",
    "New York Knicks":        "New York Knicks",
    "Brooklyn Nets":          "Brooklyn Nets",
    "Portland Trail Blazers": "Portland Trail Blazers",
    "Utah Jazz":              "Utah Jazz",
    "Memphis Grizzlies":      "Memphis Grizzlies",
    "Minnesota Timberwolves": "Minnesota Timberwolves",
    "Charlotte Hornets":      "Charlotte Hornets",
    "Miami Heat":             "Miami Heat",
    "Milwaukee Bucks":        "Milwaukee Bucks",
    "Detroit Pistons":        "Detroit Pistons",
    "Indiana Pacers":         "Indiana Pacers",
    "Chicago Bulls":          "Chicago Bulls",
    "Cleveland Cavaliers":    "Cleveland Cavaliers",
    "Atlanta Hawks":          "Atlanta Hawks",
    "Washington Wizards":     "Washington Wizards",
    "Orlando Magic":          "Orlando Magic",
    "Toronto Raptors":        "Toronto Raptors",
    "Philadelphia 76ers":     "Philadelphia 76ers",
    "Boston Celtics":         "Boston Celtics",
    "Denver Nuggets":         "Denver Nuggets",
    "Phoenix Suns":           "Phoenix Suns",
    "Sacramento Kings":       "Sacramento Kings",
    "Dallas Mavericks":       "Dallas Mavericks",
    "Houston Rockets":        "Houston Rockets",
}


def normalize_odds_team(name: str) -> str:
    return ODDS_TO_MODEL.get(name, name)


def _load_odds_cache() -> Optional[dict]:
    if ODDS_CACHE_FILE.exists():
        data = json.loads(ODDS_CACHE_FILE.read_text())
        age_min = (time.time() - data.get("timestamp", 0)) / 60
        if age_min < CACHE_TTL_MIN:
            return data.get("odds", {})
    return None


def _save_odds_cache(odds: dict):
    ODDS_CACHE_FILE.write_text(json.dumps({
        "timestamp": time.time(),
        "odds":      odds,
    }))


def fetch_today_odds(force_refresh: bool = False) -> dict[str, dict]:
    """
    Fetch today's NBA moneyline odds from the-odds-api.com.

    Returns {game_key: {home_team, away_team, home_prob, away_prob, bookmakers, commence_time}}
    where game_key = f"{away_team}_{home_team}"

    Probabilities are vig-removed (sum to 1.0).
    commence_time is ISO 8601 UTC string, used by scanners to skip live games.
    """
    api_key = os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        print("[odds] ODDS_API_KEY not set — skipping Vegas odds.")
        return {}

    if not force_refresh:
        cached = _load_odds_cache()
        if cached is not None:
            return cached

    try:
        resp = requests.get(
            "https://api.the-odds-api.com/v4/sports/basketball_nba/odds/",
            params={
                "apiKey":     api_key,
                "regions":    "us",
                "markets":    "h2h",
                "oddsFormat": "decimal",
            },
            timeout=15,
        )

        remaining = resp.headers.get("x-requests-remaining", "?")

        if resp.status_code == 401:
            print("[odds] Invalid API key.")
            return {}
        if resp.status_code != 200:
            print(f"[odds] API error {resp.status_code}")
            return {}

        games    = resp.json()
        odds_out = {}

        for game in games:
            home_team = normalize_odds_team(game.get("home_team", ""))
            away_team = normalize_odds_team(game.get("away_team", ""))

            home_probs = []
            away_probs = []

            for bookmaker in game.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    for outcome in market.get("outcomes", []):
                        dec_odds = outcome.get("price", 2.0)
                        prob     = 1 / dec_odds
                        raw_name = outcome.get("name", "")
                        norm_name = normalize_odds_team(raw_name)
                        if norm_name == home_team:
                            home_probs.append(prob)
                        elif norm_name == away_team:
                            away_probs.append(prob)

            if home_probs and away_probs:
                raw_home = sum(home_probs) / len(home_probs)
                raw_away = sum(away_probs) / len(away_probs)
                total    = raw_home + raw_away
                norm_home = round(raw_home / total, 4)
                norm_away = round(raw_away / total, 4)

                key = f"{away_team}_{home_team}"
                odds_out[key] = {
                    "home_team":     home_team,
                    "away_team":     away_team,
                    "home_prob":     norm_home,
                    "away_prob":     norm_away,
                    "bookmakers":    len(game.get("bookmakers", [])),
                    "commence_time": game.get("commence_time", ""),
                }

        _save_odds_cache(odds_out)
        print(f"[odds] Fetched Vegas lines for {len(odds_out)} NBA games. "
              f"Requests remaining: {remaining}")
        return odds_out

    except Exception as e:
        print(f"[odds] Error fetching odds: {e}")
        return {}


def get_game_odds(away_team: str, home_team: str,
                  odds: dict = None) -> Optional[dict]:
    """
    Look up Vegas odds for a specific game.
    Tries multiple key formats to handle name variations.
    Returned dict includes commence_time for live-game filtering.
    """
    if odds is None:
        odds = fetch_today_odds()

    # Direct lookup
    key = f"{away_team}_{home_team}"
    if key in odds:
        return odds[key]

    # Try normalized names
    from nba_model import normalize_team
    away_norm = normalize_team(away_team)
    home_norm = normalize_team(home_team)
    key2 = f"{away_norm}_{home_norm}"
    if key2 in odds:
        return odds[key2]

    # Fuzzy last-word match
    away_last = away_team.split()[-1].lower()
    home_last = home_team.split()[-1].lower()
    for k, v in odds.items():
        k_away = v["away_team"].split()[-1].lower()
        k_home = v["home_team"].split()[-1].lower()
        if k_away == away_last and k_home == home_last:
            return v

    return None


def print_today_odds():
    """Print all today's NBA Vegas lines."""
    odds = fetch_today_odds(force_refresh=True)
    if not odds:
        print("No odds available.")
        return
    print(f"\n{'='*80}")
    print(f"  {'GAME':<40} {'AWAY':>7} {'HOME':>7} {'BOOKS':>6} {'START':>16}")
    print(f"{'='*80}")
    for key, o in sorted(odds.items()):
        game = f"{o['away_team']} @ {o['home_team']}"
        start = o.get("commence_time", "")[:16].replace("T", " ")
        print(f"  {game:<40} {o['away_prob']*100:>6.1f}% {o['home_prob']*100:>6.1f}% "
              f"{o['bookmakers']:>6} {start:>16}")
    print(f"{'='*80}")


if __name__ == "__main__":
    print_today_odds()