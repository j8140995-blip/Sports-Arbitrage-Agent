"""
soccer_model.py  v2
Soccer win/draw/loss probability model — improvements:
  - xG as Poisson λ input (biggest single improvement per research)
  - EMA weighting (half-life ~15 games) replaces step-function weights
  - League-specific home advantage (not global constant)
  - Improved Dixon-Coles rho (-0.13 from literature vs -0.10)
  - Falls back to actual goals when xG unavailable
"""

import math
import json
import time
import datetime
import requests
from pathlib import Path
from typing import Optional

CACHE_FILE      = Path("soccer_model_cache.json")
CACHE_TTL_HOURS = 12

LEAGUES = {
    "EPL":        {"espn_id": "eng.1",           "name": "English Premier League",   "season_start": "20250815"},
    "LALIGA":     {"espn_id": "esp.1",           "name": "La Liga",                  "season_start": "20250816"},
    "BUNDESLIGA": {"espn_id": "ger.1",           "name": "Bundesliga",               "season_start": "20250823"},
    "UCL":        {"espn_id": "uefa.champions",  "name": "Champions League",         "season_start": "20250917"},
    "MLS":        {"espn_id": "usa.1",           "name": "MLS",                      "season_start": "20260222"},
    "LIGUE1":     {"espn_id": "fra.1",           "name": "Ligue 1",                  "season_start": "20250816"},
    "SERIEA":     {"espn_id": "ita.1",           "name": "Serie A",                  "season_start": "20250823"},
    "FACUP":      {"espn_id": "eng.fa",          "name": "FA Cup",                   "season_start": "20251101"},
    "UEL":        {"espn_id": "uefa.europa",     "name": "Europa League",            "season_start": "20250918"},
    "UECL":       {"espn_id": "uefa.conference", "name": "Conference League",        "season_start": "20250918"},
    "CLUBWC":     {"espn_id": "fifa.cwc",        "name": "Club World Cup",           "season_start": "20260614"},
    "FIFAQUAL":   {"espn_id": "fifa.worldq",     "name": "FIFA World Cup Qualifiers","season_start": "20250901"},
}

KALSHI_SERIES = {
    "EPL":        "KXEPLGAME",
    "LALIGA":     "KXLALIGAGAME",
    "BUNDESLIGA": "KXBUNDESLIGAGAME",
    "UCL":        "KXUCLGAME",
    "MLS":        "KXMLSGAME",
    "SERIEA":     "KXSERIEAGAME",
    "FACUP":      "KXFACUPGAME",
    "UEL":        "KXUELGAME",
    "UECL":       "KXUECLGAME",
    "CLUBWC":     "KXCLUBWCGAME",
    "FIFAQUAL":   "KXFIFAGAME",
}

# League-specific home advantage in expected goals (post-COVID calibrated)
# From dashee87 analysis + Springer 2025 crowd effect study
LEAGUE_HOME_ADV = {
    "EPL":        0.22,
    "LALIGA":     0.28,
    "BUNDESLIGA": 0.25,
    "SERIEA":     0.30,
    "UCL":        0.20,
    "UEL":        0.22,
    "UECL":       0.22,
    "LIGUE1":     0.28,
    "MLS":        0.30,
    "FACUP":      0.25,
    "CLUBWC":     0.20,
    "FIFAQUAL":   0.25,
    "DEFAULT":    0.25,
}

DC_RHO         = -0.13  # improved from literature (was -0.10)
REST_PENALTY   = 0.08
EMA_HALF_LIFE  = 15     # games; optimal from Beat the Bookie EMA-15 research


def ema_weight(game_index: int, half_life: float = EMA_HALF_LIFE) -> float:
    """Exponential decay weight. game_index=0 is most recent."""
    return math.exp(-math.log(2) * game_index / half_life)


def fetch_league_results(espn_id: str, season_start: str) -> list[dict]:
    results = []
    seen    = set()
    start   = datetime.datetime.strptime(season_start, "%Y%m%d").date()
    today   = datetime.date.today()
    current = start

    while current <= today:
        date_str = current.strftime("%Y%m%d")
        try:
            resp = requests.get(
                f"https://site.api.espn.com/apis/site/v2/sports/soccer/{espn_id}/scoreboard",
                params={"dates": date_str},
                timeout=10,
            )
            if resp.status_code != 200:
                current += datetime.timedelta(days=3)
                continue

            events = resp.json().get("events", [])
            for event in events:
                status = event.get("status", {}).get("type", {}).get("name", "")
                if "FULL_TIME" not in status and "FINAL" not in status and status != "STATUS_FINAL":
                    continue

                comps       = event.get("competitions", [{}])[0]
                competitors = comps.get("competitors", [])
                if len(competitors) != 2:
                    continue

                home = next((c for c in competitors if c.get("homeAway") == "home"), None)
                away = next((c for c in competitors if c.get("homeAway") == "away"), None)
                if not home or not away:
                    continue

                home_name  = home.get("team", {}).get("displayName", "")
                away_name  = away.get("team", {}).get("displayName", "")
                home_goals = int(home.get("score", 0) or 0)
                away_goals = int(away.get("score", 0) or 0)
                game_date  = event.get("date", "")[:10]

                key = f"{home_name}_{away_name}_{game_date}"
                if key in seen:
                    continue
                seen.add(key)

                # Extract xG — try ESPN xG field first, fall back to shots-based proxy
                # Proxy formula: xG ≈ shots_on_target * 0.35 + off_target * 0.05
                # (derived from league-average conversion rates)
                home_xg = 0.0
                away_xg = 0.0
                try:
                    def parse_stats(competitor):
                        stats = {s.get("name","").lower(): s.get("value") for s in competitor.get("statistics", [])}
                        # Try true xG first
                        for key in ("expectedgoals", "xg", "expected goals"):
                            v = stats.get(key)
                            if v and float(v) > 0:
                                return float(v)
                        # Fall back to shots-based proxy
                        sot   = float(stats.get("shotsonTarget", stats.get("shotsontarget", 0)) or 0)
                        shots = float(stats.get("totalShots",    stats.get("totalshots",    0)) or 0)
                        off_target = max(0, shots - sot)
                        if shots > 0:
                            return round(sot * 0.35 + off_target * 0.05, 2)
                        return 0.0
                    home_xg = parse_stats(home)
                    away_xg = parse_stats(away)
                except Exception:
                    pass

                results.append({
                    "home_team":  home_name,
                    "away_team":  away_name,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "date":       game_date,
                    "home_xg":    home_xg,
                    "away_xg":    away_xg,
                })

        except Exception:
            pass

        current += datetime.timedelta(days=1)
        time.sleep(0.05)

    return sorted(results, key=lambda r: r["date"])


def fetch_injury_report_soccer(espn_id: str) -> dict[str, list[str]]:
    try:
        resp = requests.get(
            f"https://site.api.espn.com/apis/site/v2/sports/soccer/{espn_id}/injuries",
            timeout=10
        )
        if resp.status_code != 200:
            return {}
        data   = resp.json()
        result = {}
        for team in data.get("injuries", []):
            team_name = team.get("displayName", "")
            players   = [
                p.get("athlete", {}).get("displayName", "")
                for p in team.get("injuries", [])
                if p.get("status") in ("Out", "Doubtful")
            ]
            if players:
                result[team_name] = players
        return result
    except Exception:
        return {}


def build_team_strengths(results: list[dict], league_key: str = "DEFAULT") -> dict:
    """
    Build attack/defense ratings using:
    - xG as primary metric (falls back to actual goals if xG unavailable)
    - EMA weighting with half-life of 15 games
    - Separate home/away attack strengths
    - League-specific home advantage
    """
    if not results:
        return {}

    team_home_games: dict[str, list] = {}
    team_away_games: dict[str, list] = {}
    for r in results:
        team_home_games.setdefault(r["home_team"], []).append(r)
        team_away_games.setdefault(r["away_team"], []).append(r)

    all_teams = set(list(team_home_games.keys()) + list(team_away_games.keys()))

    # League average xG (prefer xG, fall back to goals)
    total_xg  = 0.0
    total_cnt = 0
    for r in results:
        hxg = r.get("home_xg", 0)
        axg = r.get("away_xg", 0)
        if hxg > 0 or axg > 0:
            total_xg += hxg + axg
        else:
            total_xg += r["home_goals"] + r["away_goals"]
        total_cnt += 2
    league_avg_xg = total_xg / total_cnt if total_cnt > 0 else 1.3

    home_adv = LEAGUE_HOME_ADV.get(league_key, LEAGUE_HOME_ADV["DEFAULT"])
    strengths = {}

    for team in all_teams:
        home_games = sorted(team_home_games.get(team, []), key=lambda r: r["date"], reverse=True)
        away_games = sorted(team_away_games.get(team, []), key=lambda r: r["date"], reverse=True)
        all_games  = sorted(
            [(r, "home") for r in home_games] + [(r, "away") for r in away_games],
            key=lambda x: x[0]["date"], reverse=True
        )

        w_scored   = 0.0
        w_conceded = 0.0
        w_total    = 0.0
        w_home_scored = 0.0
        w_home_total  = 0.0
        w_away_scored = 0.0
        w_away_total  = 0.0

        for i, (r, venue) in enumerate(all_games):
            weight = ema_weight(i)

            if venue == "home":
                xg_for     = r.get("home_xg", 0)
                xg_against = r.get("away_xg", 0)
                goals_for     = r["home_goals"]
                goals_against = r["away_goals"]
            else:
                xg_for     = r.get("away_xg", 0)
                xg_against = r.get("home_xg", 0)
                goals_for     = r["away_goals"]
                goals_against = r["home_goals"]

            # Use xG if available; fall back to goals
            scored   = xg_for     if xg_for     > 0 else goals_for
            conceded = xg_against if xg_against > 0 else goals_against

            w_scored   += scored   * weight
            w_conceded += conceded * weight
            w_total    += weight

            if venue == "home":
                w_home_scored += scored * weight
                w_home_total  += weight
            else:
                w_away_scored += scored * weight
                w_away_total  += weight

        if w_total == 0:
            continue

        avg_scored   = w_scored   / w_total
        avg_conceded = w_conceded / w_total
        home_attack_avg = (w_home_scored / w_home_total) if w_home_total > 0 else avg_scored
        away_attack_avg = (w_away_scored / w_away_total) if w_away_total > 0 else avg_scored

        # Recent form win rate (last 6 games)
        recent_games = all_games[:6]
        wins = sum(
            1 for r, venue in recent_games
            if (venue == "home" and r["home_goals"] > r["away_goals"]) or
               (venue == "away" and r["away_goals"] > r["home_goals"])
        )
        recent_form = round(wins / len(recent_games), 3) if recent_games else 0.5

        strengths[team] = {
            "attack":         round(avg_scored   / league_avg_xg, 4),
            "defense":        round(avg_conceded / league_avg_xg, 4),
            "home_attack":    round(home_attack_avg / league_avg_xg, 4),
            "away_attack":    round(away_attack_avg / league_avg_xg, 4),
            "games":          len(all_games),
            "avg_scored":     round(avg_scored, 3),
            "avg_conceded":   round(avg_conceded, 3),
            "recent_form":    recent_form,
            "home_adv":       home_adv,
            "league_avg_xg":  round(league_avg_xg, 3),
        }

    return strengths


def build_h2h_table(results: list[dict]) -> dict:
    h2h = {}
    for r in results:
        key = f"{r['home_team']}_{r['away_team']}"
        if key not in h2h:
            h2h[key] = {"home_wins": 0, "away_wins": 0, "draws": 0}
        if r["home_goals"] > r["away_goals"]:
            h2h[key]["home_wins"] += 1
        elif r["home_goals"] < r["away_goals"]:
            h2h[key]["away_wins"] += 1
        else:
            h2h[key]["draws"] += 1
    return h2h


def get_last_game_date(team: str, results: list[dict]) -> Optional[datetime.date]:
    team_games = [r for r in results if r["home_team"] == team or r["away_team"] == team]
    if not team_games:
        return None
    latest = max(team_games, key=lambda r: r["date"])
    return datetime.date.fromisoformat(latest["date"])


def rest_days(team: str, results: list[dict]) -> int:
    last = get_last_game_date(team, results)
    if not last:
        return 7
    return (datetime.date.today() - last).days


def league_avg_goals(results: list[dict]) -> float:
    if not results:
        return 1.3
    total = sum(r["home_goals"] + r["away_goals"] for r in results)
    return total / (2 * len(results))


def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def dc_correction(hg: int, ag: int, mu_h: float, mu_a: float, rho: float) -> float:
    if hg == 0 and ag == 0:
        return 1 - mu_h * mu_a * rho
    elif hg == 1 and ag == 0:
        return 1 + mu_a * rho
    elif hg == 0 and ag == 1:
        return 1 + mu_h * rho
    elif hg == 1 and ag == 1:
        return 1 - rho
    return 1.0


def predict_match(home_team: str, away_team: str,
                  strengths: dict,
                  h2h: dict = None,
                  injuries: dict = None,
                  results: list = None,
                  league_avg: float = 1.3,
                  league_key: str = "DEFAULT",
                  max_goals: int = 10) -> dict:
    """
    Full prediction using xG-based Dixon-Coles with EMA-weighted strengths.
    """
    home_s = strengths.get(home_team)
    away_s = strengths.get(away_team)

    # League-specific home advantage
    home_adv = LEAGUE_HOME_ADV.get(league_key, LEAGUE_HOME_ADV["DEFAULT"])
    if home_s:
        home_adv = home_s.get("home_adv", home_adv)

    # Use xG-based league average if stored in strengths
    if home_s and "league_avg_xg" in home_s:
        league_avg = home_s["league_avg_xg"]

    home_attack  = home_s["home_attack"] if home_s else 1.0
    home_defense = home_s["defense"]     if home_s else 1.0
    away_attack  = away_s["away_attack"] if away_s else 1.0
    away_defense = away_s["defense"]     if away_s else 1.0

    mu_home = home_attack * away_defense * league_avg * (1 + home_adv)
    mu_away = away_attack * home_defense * league_avg

    if results:
        home_rest = rest_days(home_team, results)
        away_rest = rest_days(away_team, results)
        if home_rest < 4:
            mu_home *= (1 - REST_PENALTY * (4 - home_rest))
        if away_rest < 4:
            mu_away *= (1 - REST_PENALTY * (4 - away_rest))

    if injuries:
        home_out = len(injuries.get(home_team, []))
        away_out = len(injuries.get(away_team, []))
        mu_home *= max(0.7, 1 - home_out * 0.04)
        mu_away *= max(0.7, 1 - away_out * 0.04)

    if h2h:
        key      = f"{home_team}_{away_team}"
        h2h_data = h2h.get(key)
        if h2h_data:
            total_h2h = sum(h2h_data.values())
            if total_h2h >= 3:
                h2h_home_rate = h2h_data["home_wins"] / total_h2h
                h2h_adj       = (h2h_home_rate - 0.5) * 0.1
                mu_home *= (1 + h2h_adj)

    mu_home = max(0.1, mu_home)
    mu_away = max(0.1, mu_away)

    home_win = draw = away_win = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            prob = poisson_pmf(h, mu_home) * poisson_pmf(a, mu_away) * \
                   dc_correction(h, a, mu_home, mu_away, DC_RHO)
            if h > a:
                home_win += prob
            elif h == a:
                draw     += prob
            else:
                away_win += prob

    total = home_win + draw + away_win
    if total > 0:
        home_win /= total
        draw     /= total
        away_win /= total

    return {
        "home_team":         home_team,
        "away_team":         away_team,
        "home_win":          round(home_win, 4),
        "draw":              round(draw, 4),
        "away_win":          round(away_win, 4),
        "home_xg":           round(mu_home, 2),
        "away_xg":           round(mu_away, 2),
        "home_recent_form":  home_s["recent_form"] if home_s else 0.5,
        "away_recent_form":  away_s["recent_form"] if away_s else 0.5,
        "home_rest_days":    rest_days(home_team, results) if results else 7,
        "away_rest_days":    rest_days(away_team, results) if results else 7,
    }


def _load_cache() -> Optional[dict]:
    if CACHE_FILE.exists():
        data = json.loads(CACHE_FILE.read_text())
        if (time.time() - data.get("timestamp", 0)) / 3600 < CACHE_TTL_HOURS:
            return data
    return None


def _save_cache(data: dict):
    data["timestamp"] = time.time()
    CACHE_FILE.write_text(json.dumps(data, indent=2))


class SoccerModel:
    def __init__(self):
        self.leagues: dict[str, dict] = {}
        self._loaded = False

    def load(self, force_refresh: bool = False):
        cache = None if force_refresh else _load_cache()
        if cache:
            self.leagues = cache.get("leagues", {})
            print(f"[SoccerModel] Loaded {len(self.leagues)} leagues from cache.")
        else:
            print("[SoccerModel] Fetching results from ESPN...")
            self.leagues = {}
            for key, config in LEAGUES.items():
                print(f"  Fetching {config['name']}...")
                results = fetch_league_results(config["espn_id"], config["season_start"])
                if not results:
                    print(f"  ⚠️  No results for {config['name']}")
                    continue
                # Pass league_key for league-specific home advantage
                strengths = build_team_strengths(results, league_key=key)
                h2h       = build_h2h_table(results)
                avg       = league_avg_goals(results)
                injuries  = fetch_injury_report_soccer(config["espn_id"])
                self.leagues[key] = {
                    "strengths": strengths,
                    "h2h":       h2h,
                    "avg_goals": avg,
                    "results":   results,
                    "injuries":  injuries,
                    "name":      config["name"],
                    "league_key": key,
                }
                xg_used = sum(1 for r in results if r.get("home_xg", 0) > 0)
                print(f"  ✅ {config['name']}: {len(results)} matches, "
                      f"{len(strengths)} teams, avg {avg:.2f} G/game, "
                      f"{xg_used}/{len(results)} with xG data")
                time.sleep(0.3)
            _save_cache({"leagues": self.leagues})
            print(f"[SoccerModel] Built model for {len(self.leagues)} leagues.")
        self._loaded = True

    def predict(self, home_team: str, away_team: str,
                league: str = None) -> Optional[dict]:
        leagues_to_search = [league] if league else list(self.leagues.keys())
        for lg in leagues_to_search:
            if lg not in self.leagues:
                continue
            data      = self.leagues[lg]
            strengths = data["strengths"]
            home_matched = self._fuzzy_match(home_team, strengths)
            away_matched = self._fuzzy_match(away_team, strengths)
            if not home_matched and not away_matched:
                continue
            home_use = home_matched or home_team
            away_use = away_matched or away_team
            result = predict_match(
                home_use, away_use,
                strengths=strengths,
                h2h=data.get("h2h"),
                injuries=data.get("injuries"),
                results=data.get("results"),
                league_avg=data["avg_goals"],
                league_key=data.get("league_key", lg),
            )
            result["league"] = lg
            return result
        return None

    def _fuzzy_match(self, name: str, strengths: dict) -> Optional[str]:
        name_lower = name.lower().strip()
        for team in strengths:
            if team.lower() == name_lower:
                return team
        for team in strengths:
            if name_lower in team.lower() or team.lower() in name_lower:
                return team
        last = name_lower.split()[-1]
        matches = [t for t in strengths if last in t.lower()]
        if len(matches) == 1:
            return matches[0]
        return None

    def top_teams(self, league: str, n: int = 10) -> list[dict]:
        if league not in self.leagues:
            return []
        s = self.leagues[league]["strengths"]
        return [
            {"team": t, **v}
            for t, v in sorted(s.items(), key=lambda x: x[1]["attack"] - x[1]["defense"], reverse=True)[:n]
        ]


if __name__ == "__main__":
    model = SoccerModel()
    model.load(force_refresh=True)
    for league in ["EPL", "LALIGA", "BUNDESLIGA"]:
        if league not in model.leagues:
            continue
        print(f"\n=== {league} Top 8 Teams ===")
        for t in model.top_teams(league, 8):
            print(f"  {t['team']:<30} ATT:{t['attack']:.2f}  DEF:{t['defense']:.2f}  "
                  f"Form:{t['recent_form']:.0%}  xG:{t['avg_scored']:.2f}")
    print("\n=== Sample: Man United vs Bournemouth ===")
    result = model.predict("Manchester United", "AFC Bournemouth")
    if result:
        for k, v in result.items():
            print(f"  {k}: {v}")