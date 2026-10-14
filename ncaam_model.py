"""
ncaam_model.py
Men's college basketball win-probability model.

Components:
1. ELO ratings (margin-of-victory + SOS-adjusted, current season only)
2. Pace-adjusted offensive/defensive efficiency ratings
3. Weighted recent form (last 6 games, tournament-weighted)
4. Team stats: 3PT%, assist/turnover ratio, rebounding
5. Star player identification + injury status from ESPN roster
6. Seed adjustment for tournament context

Neutral site only — no home court applied.
Data source: ESPN public API (no key required)
"""

import math
import json
import time
import datetime
import requests
from pathlib import Path
from typing import Optional

CACHE_FILE      = Path("ncaam_model_cache.json")
CACHE_TTL_HOURS = 12
CURRENT_SEASON  = "2025-26"

# ELO constants
BASE_ELO   = 1500
K_FACTOR   = 24
MIN_GAMES  = 8

# Weights
W_ELO    = 0.38
W_RATING = 0.37
W_RECENT = 0.25

# Injury penalty per star player out (in ELO points)
STAR_INJURY_PENALTY   = 55
ROSTER_INJURY_PENALTY = 20

# Conference strength multipliers (verified ESPN conf IDs)
ESPN_CONF_ID = {
    "2":  "ACC",         "4":  "Big East",    "7":  "Big Ten",
    "8":  "Big 12",      "23": "SEC",         "29": "WCC",
    "44": "Mountain West","3":  "Atlantic 10", "18": "Missouri Valley",
    "62": "American",    "14": "MAC",          "27": "Sun Belt",
    "11": "Conference USA","6": "Big South",   "10": "CAA",
    "24": "SoCon",       "22": "Patriot",      "45": "Horizon",
    "49": "Summit",      "5":  "Big Sky",      "25": "Southland",
    "20": "OVC",         "16": "MEAC",         "26": "SWAC",
    "9":  "Big West",    "1":  "America East", "12": "Ivy",
    "30": "WAC",
}

CONF_STRENGTH = {
    "Big 12":         1.32,  "SEC":            1.30,  "Big Ten":        1.28,
    "ACC":            1.24,  "Big East":       1.20,  "American":       1.08,
    "Mountain West":  1.06,  "Atlantic 10":    1.04,  "WCC":            1.02,
    "Missouri Valley":0.98,  "CAA":            0.97,  "SoCon":          0.95,
    "MAC":            0.93,  "Conference USA": 0.93,  "Sun Belt":       0.90,
    "Horizon":        0.90,  "Big South":      0.88,  "Summit":         0.88,
    "Big Sky":        0.87,  "OVC":            0.86,  "WAC":            0.85,
    "Patriot":        0.85,  "Ivy":            0.84,  "America East":   0.83,
    "Southland":      0.82,  "Big West":       0.82,  "MEAC":           0.78,
    "SWAC":           0.78,
}
DEFAULT_CONF_STRENGTH = 0.88

# ELO cap for low-major conference teams to prevent inflation
# Power conferences can reach any ELO; low-majors are capped
POWER_CONFERENCES = {"Big 12", "SEC", "Big Ten", "ACC", "Big East", "American"}
LOW_MAJOR_ELO_CAP = 1720  # ~equivalent to a 9-10 seed power conf team

# Seed ELO adjustments for tournament context
SEED_ELO_ADJ = {
    1: +40, 2: +25, 3: +15, 4: +8,  5: +3,
    6:   0, 7:   0, 8:  -3, 9: -3, 10: -5,
    11: -8, 12: -12, 13: -18, 14: -25, 15: -35, 16: -50,
}

# All ESPN group IDs covering every D1 conference
ALL_GROUPS = [3,7,8,13,15,21,23,24,27,33,34,37,38,39,40,41,42,45,47,48,49,50,52,62,63,64,72,73]


# ── Conference strength ───────────────────────────────────────────────────────

def get_conf_strength(conf_id: str) -> float:
    name = ESPN_CONF_ID.get(str(conf_id), "")
    return CONF_STRENGTH.get(name, DEFAULT_CONF_STRENGTH)


# ── ESPN data fetching ────────────────────────────────────────────────────────

def fetch_season_results(season_slug: str = "2025-26") -> list[dict]:
    results = []
    seen    = set()
    year    = int(season_slug.split("-")[0])
    start   = datetime.date(year, 11, 1)
    today   = datetime.date.today()
    current = start

    print(f"[NCAAModel] Fetching games from ESPN ({start} to {today})...")

    while current <= today:
        date_str = current.strftime("%Y%m%d")
        try:
            all_events_today = {}
            for group in ALL_GROUPS:
                resp = requests.get(
                    "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard",
                    params={"dates": date_str, "limit": 500, "groups": group},
                    timeout=10,
                )
                if resp.status_code != 200:
                    continue
                for e in resp.json().get("events", []):
                    all_events_today[e.get("id", "")] = e
                time.sleep(0.02)

            events = list(all_events_today.values())
            for event in events:
                status = event.get("status", {}).get("type", {}).get("name", "")
                if status not in ("STATUS_FINAL", "STATUS_FULL_TIME"):
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
                home_score = int(home.get("score", 0) or 0)
                away_score = int(away.get("score", 0) or 0)
                game_date  = event.get("date", "")[:10]

                if not home_name or not away_name:
                    continue

                key = f"{home_name}_{away_name}_{game_date}"
                if key in seen:
                    continue
                seen.add(key)

                results.append({
                    "home_team":   home_name,
                    "away_team":   away_name,
                    "home_score":  home_score,
                    "away_score":  away_score,
                    "date":        game_date,
                    "neutral":     comps.get("neutralSite", False),
                    "home_conf":   home.get("team", {}).get("conferenceId", ""),
                    "away_conf":   away.get("team", {}).get("conferenceId", ""),
                    "home_id":     home.get("team", {}).get("id", ""),
                    "away_id":     away.get("team", {}).get("id", ""),
                    "tournament":  "NCAA" in event.get("name", "") or
                                   "Tournament" in event.get("name", ""),
                })

        except Exception:
            pass

        current += datetime.timedelta(days=1)
        time.sleep(0.05)

    return sorted(results, key=lambda r: r["date"])


def fetch_team_stats(team_id: str) -> dict:
    """Fetch team-level stats: pace, 3PT%, A/TO ratio, rebounding."""
    try:
        resp = requests.get(
            f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams/{team_id}/statistics",
            timeout=8,
        )
        if resp.status_code != 200:
            return {}

        stats_out = {}
        cats = resp.json().get("results", {}).get("stats", {}).get("categories", [])
        for cat in cats:
            for s in cat.get("stats", []):
                stats_out[s["name"]] = s.get("value", 0)
        return stats_out
    except Exception:
        return {}


def fetch_team_roster_injuries(team_id: str) -> tuple[list[str], list[str]]:
    """
    Returns (star_players, injured_players) for a team.
    Stars = players averaging significant minutes/scoring based on status.
    Injured = players with non-active status.
    """
    try:
        resp = requests.get(
            f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams/{team_id}/roster",
            timeout=8,
        )
        if resp.status_code != 200:
            return [], []

        athletes = resp.json().get("athletes", [])
        injured  = []
        active   = []

        for a in athletes:
            name   = a.get("displayName", "")
            status = a.get("status", {}).get("type", "active")
            exp    = a.get("experience", {}).get("years", 0) or 0

            if status != "active":
                injured.append(name)
            else:
                active.append((name, exp))

        # Stars = seniors/juniors who are likely key contributors
        # (We can't get PPG without stats endpoint, use experience as proxy)
        stars = [name for name, exp in active if exp >= 2][:5]

        return stars, injured
    except Exception:
        return [], []


def build_team_profiles(results: list[dict]) -> dict[str, dict]:
    """
    Build team profiles with stats and injury info for all tournament teams.
    Only fetches for teams we've seen in results to save API calls.
    """
    # Build team_name -> team_id mapping
    team_ids: dict[str, str] = {}
    for r in results:
        if r.get("home_id"):
            team_ids[r["home_team"]] = r["home_id"]
        if r.get("away_id"):
            team_ids[r["away_team"]] = r["away_id"]

    profiles = {}
    total    = len(team_ids)
    print(f"[NCAAModel] Fetching team stats for {total} teams...")

    for i, (team, tid) in enumerate(team_ids.items()):
        if not tid:
            continue
        stats            = fetch_team_stats(tid)
        stars, injured   = fetch_team_roster_injuries(tid)
        profiles[team]   = {
            "team_id":      tid,
            "three_pct":    stats.get("avgThreePointFieldGoalsMade", 0) /
                            max(stats.get("avgThreePointFieldGoalsAttempted", 1), 1),
            "ft_pct":       stats.get("freeThrowPct", 70) / 100,
            "avg_pts":      stats.get("avgPoints", 70),
            "avg_ast":      stats.get("avgAssists", 12),
            "avg_tov":      stats.get("assistTurnoverRatio", 1.2),
            "avg_reb":      stats.get("avgRebounds", 35),
            "avg_fouls":    stats.get("avgFouls", 16),
            "stars":        stars,
            "injured":      injured,
        }
        time.sleep(0.05)

    return profiles


# ── ELO with MoV + SOS ───────────────────────────────────────────────────────

def mov_multiplier(point_diff: int, winner_elo: float, loser_elo: float) -> float:
    elo_diff = max(1, winner_elo - loser_elo)
    return (abs(point_diff) + 3) ** 0.8 / (7.5 + 0.006 * elo_diff)


def expected_score(elo_a: float, elo_b: float) -> float:
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def build_elo_ratings(results: list[dict]) -> tuple[dict, dict]:
    elos:       dict[str, float] = {}
    game_count: dict[str, int]   = {}
    team_conf:  dict[str, str]   = {}

    def get_elo(team):
        return elos.setdefault(team, float(BASE_ELO))

    for r in results:
        home, away = r["home_team"], r["away_team"]
        if r.get("home_conf"):
            team_conf[home] = r["home_conf"]
        if r.get("away_conf"):
            team_conf[away] = r["away_conf"]

        h_elo = get_elo(home)
        a_elo = get_elo(away)
        h_adj = h_elo + (50 if not r.get("neutral") else 0)
        exp   = expected_score(h_adj, a_elo)
        diff  = r["home_score"] - r["away_score"]
        mov   = mov_multiplier(abs(diff), h_elo if diff > 0 else a_elo,
                                a_elo if diff > 0 else h_elo)

        # SOS multiplier: beating a strong conference team is worth more
        loser_conf  = team_conf.get(away if diff > 0 else home, "")
        sos_mult    = get_conf_strength(loser_conf)
        k = K_FACTOR * mov * sos_mult

        if diff > 0:
            elos[home] = h_elo + k * (1 - exp)
            elos[away] = a_elo + k * (0 - (1 - exp))
        elif diff < 0:
            elos[home] = h_elo + k * (0 - exp)
            elos[away] = a_elo + k * (1 - (1 - exp))

        game_count[home] = game_count.get(home, 0) + 1
        game_count[away] = game_count.get(away, 0) + 1

    # Apply ELO cap for low-major teams
    for team in list(elos.keys()):
        conf_id  = team_conf.get(team, "")
        conf_name = ESPN_CONF_ID.get(str(conf_id), "")
        if conf_name not in POWER_CONFERENCES and elos[team] > LOW_MAJOR_ELO_CAP:
            elos[team] = LOW_MAJOR_ELO_CAP

    return elos, game_count


# ── Pace-adjusted efficiency ──────────────────────────────────────────────────

def build_efficiency_ratings(results: list[dict]) -> dict[str, dict]:
    team_stats: dict[str, dict] = {}

    for r in results:
        home, away = r["home_team"], r["away_team"]
        # Estimate possessions from scoring
        poss = max(1, (r["home_score"] + r["away_score"]) / 2 / 1.05)

        for team, scored, conceded in [
            (home, r["home_score"], r["away_score"]),
            (away, r["away_score"], r["home_score"]),
        ]:
            if team not in team_stats:
                team_stats[team] = {"pts_scored": 0, "pts_allowed": 0,
                                     "poss": 0, "games": 0}
            team_stats[team]["pts_scored"]  += scored
            team_stats[team]["pts_allowed"] += conceded
            team_stats[team]["poss"]        += poss
            team_stats[team]["games"]       += 1

    ratings = {}
    for team, s in team_stats.items():
        if s["poss"] == 0:
            continue
        ortg = s["pts_scored"]  / s["poss"] * 100
        drtg = s["pts_allowed"] / s["poss"] * 100
        ratings[team] = {
            "ortg":    round(ortg, 2),
            "drtg":    round(drtg, 2),
            "net_rtg": round(ortg - drtg, 2),
            "games":   s["games"],
        }
    return ratings


def rating_win_prob(team_a: str, team_b: str, ratings: dict) -> float:
    a_net = ratings.get(team_a, {}).get("net_rtg", 0.0)
    b_net = ratings.get(team_b, {}).get("net_rtg", 0.0)
    diff  = a_net - b_net
    prob  = 1 / (1 + 10 ** (-diff / 12))
    return max(0.05, min(0.95, prob))


# ── Weighted recent form ──────────────────────────────────────────────────────

def build_recent_form(results: list[dict], elos: dict, n: int = 6) -> dict[str, float]:
    by_team: dict[str, list] = {}
    for r in results:
        by_team.setdefault(r["home_team"], []).append((r, "home"))
        by_team.setdefault(r["away_team"], []).append((r, "away"))

    form           = {}
    league_avg_elo = sum(elos.values()) / len(elos) if elos else BASE_ELO

    for team, games in by_team.items():
        recent = sorted(games, key=lambda x: x[0]["date"], reverse=True)[:n]
        w_wins = 0.0
        w_tot  = 0.0

        for i, (r, venue) in enumerate(recent):
            recency_w  = 3.0 if i < 3 else 1.5
            # Extra weight for tournament games — recent tournament wins matter most
            if r.get("tournament"):
                recency_w *= 2.0
            opponent   = r["away_team"] if venue == "home" else r["home_team"]
            opp_elo    = elos.get(opponent, league_avg_elo)
            strength_w = max(0.7, min(1.5, opp_elo / league_avg_elo))
            weight     = recency_w * strength_w
            w_tot     += weight

            won = (venue == "home" and r["home_score"] > r["away_score"]) or \
                  (venue == "away" and r["away_score"] > r["home_score"])
            if won:
                w_wins += weight

        form[team] = w_wins / w_tot if w_tot else 0.5

    return form


def form_win_prob(team_a: str, team_b: str, form: dict) -> float:
    pa    = form.get(team_a, 0.5)
    pb    = form.get(team_b, 0.5)
    num   = pa * (1 - pb)
    denom = pa * (1 - pb) + pb * (1 - pa)
    return max(0.05, min(0.95, num / denom)) if denom else 0.5


# ── Injury adjustment ────────────────────────────────────────────────────────

def injury_elo_penalty(team: str, profiles: dict) -> float:
    """Returns negative ELO adjustment for injured players."""
    profile = profiles.get(team, {})
    injured = profile.get("injured", [])
    stars   = set(profile.get("stars", []))
    penalty = 0.0
    for player in injured:
        if player in stars:
            penalty += STAR_INJURY_PENALTY
        else:
            penalty += ROSTER_INJURY_PENALTY
    return -penalty


# ── Fuzzy name matching ───────────────────────────────────────────────────────

def fuzzy_match(name: str, known_teams: set,
                game_count: dict = None) -> Optional[str]:
    """
    Match team name to known teams. Prefers:
    1. Exact match
    2. Exact last-word match with fewest extra words (most specific)
    3. Contains match — only if one result, preferring teams with more games
    """
    name_lower = name.lower().strip()
    name_words = name_lower.split()

    # 1. Exact match
    for t in known_teams:
        if t.lower() == name_lower:
            return t

    # 2. All words in name appear in team name (order-sensitive subset)
    # Score by how many extra words the team name has (fewer = better match)
    subset_matches = []
    for t in known_teams:
        t_lower = t.lower()
        t_words = t_lower.split()
        # Check if all query words appear in team name words
        if all(w in t_words for w in name_words):
            extra = len(t_words) - len(name_words)
            games = game_count.get(t, 0) if game_count else 0
            subset_matches.append((extra, -games, t))

    if subset_matches:
        subset_matches.sort()
        # Return best match only if it's clearly better (fewer extra words)
        best_extra = subset_matches[0][0]
        best = [m for m in subset_matches if m[0] == best_extra]
        if len(best) == 1:
            return best[0][2]
        # Tie on extra words — pick team with most games
        best.sort(key=lambda x: x[1])
        return best[0][2]

    # 3. Last word exact match — only safe for unique mascots
    last = name_words[-1]
    last_matches = [t for t in known_teams if t.lower().split()[-1] == last]
    if len(last_matches) == 1:
        return last_matches[0]
    if len(last_matches) > 1 and game_count:
        # Pick the one with most games (most likely the major program)
        return max(last_matches, key=lambda t: game_count.get(t, 0))

    return None


# ── Seed adjustment ───────────────────────────────────────────────────────────

def seed_adjustment(seed: Optional[int]) -> float:
    if seed is None:
        return 0.0
    return SEED_ELO_ADJ.get(seed, 0)


# ── Cache ─────────────────────────────────────────────────────────────────────

def _load_cache() -> Optional[dict]:
    if CACHE_FILE.exists():
        data = json.loads(CACHE_FILE.read_text())
        if (time.time() - data.get("timestamp", 0)) / 3600 < CACHE_TTL_HOURS:
            return data
    return None


def _save_cache(data: dict):
    data["timestamp"] = time.time()
    CACHE_FILE.write_text(json.dumps(data, indent=2))


# ── Main model ────────────────────────────────────────────────────────────────

class NCAAMModel:
    def __init__(self):
        self.elos:       dict[str, float] = {}
        self.game_count: dict[str, int]   = {}
        self.ratings:    dict[str, dict]  = {}
        self.form:       dict[str, float] = {}
        self.profiles:   dict[str, dict]  = {}
        self.results:    list[dict]       = []
        self._loaded = False

    def load(self, force_refresh: bool = False):
        cache = None if force_refresh else _load_cache()
        if cache:
            self.elos       = cache["elos"]
            self.game_count = cache.get("game_count", {})
            self.ratings    = cache.get("ratings", {})
            self.form       = cache.get("form", {})
            self.profiles   = cache.get("profiles", {})
            self.results    = cache.get("results", [])
            print(f"[NCAAModel] Loaded {len(self.elos)} teams from cache.")
        else:
            self.results = fetch_season_results(CURRENT_SEASON)
            if not self.results:
                print("[NCAAModel] No results loaded.")
                return
            self.elos, self.game_count = build_elo_ratings(self.results)
            self.ratings  = build_efficiency_ratings(self.results)
            self.form     = build_recent_form(self.results, self.elos)
            self.profiles = build_team_profiles(self.results)
            _save_cache({
                "elos":       self.elos,
                "game_count": self.game_count,
                "ratings":    self.ratings,
                "form":       self.form,
                "profiles":   self.profiles,
                "results":    self.results,
            })
            print(f"[NCAAModel] Built model for {len(self.elos)} teams "
                  f"from {len(self.results)} games.")
        self._loaded = True

    def win_probability(self, team_a: str, team_b: str,
                        seed_a: int = None, seed_b: int = None,
                        apply_injuries: bool = True) -> dict:
        known = set(self.elos.keys())
        a     = fuzzy_match(team_a, known, self.game_count) or team_a
        b     = fuzzy_match(team_b, known, self.game_count) or team_b

        # Base ELO with seed adjustment
        elo_a = self.elos.get(a, BASE_ELO) + seed_adjustment(seed_a)
        elo_b = self.elos.get(b, BASE_ELO) + seed_adjustment(seed_b)

        # Injury adjustment
        inj_a = injury_elo_penalty(a, self.profiles) if apply_injuries else 0
        inj_b = injury_elo_penalty(b, self.profiles) if apply_injuries else 0
        elo_a += inj_a
        elo_b += inj_b

        # Neutral site — no home court
        p_elo    = expected_score(elo_a, elo_b)
        p_rating = rating_win_prob(a, b, self.ratings)
        p_recent = form_win_prob(a, b, self.form)

        p_a = W_ELO * p_elo + W_RATING * p_rating + W_RECENT * p_recent

        profile_a = self.profiles.get(a, {})
        profile_b = self.profiles.get(b, {})

        return {
            "team_a":        a,
            "team_b":        b,
            "p_a":           round(p_a, 4),
            "p_b":           round(1 - p_a, 4),
            "p_elo":         round(p_elo, 4),
            "p_rating":      round(p_rating, 4),
            "p_recent":      round(p_recent, 4),
            "elo_a":         round(self.elos.get(a, BASE_ELO), 1),
            "elo_b":         round(self.elos.get(b, BASE_ELO), 1),
            "inj_penalty_a": round(inj_a, 1),
            "inj_penalty_b": round(inj_b, 1),
            "injured_a":     profile_a.get("injured", []),
            "injured_b":     profile_b.get("injured", []),
            "games_a":       self.game_count.get(a, 0),
            "games_b":       self.game_count.get(b, 0),
            "reliable":      self.game_count.get(a, 0) >= MIN_GAMES and
                             self.game_count.get(b, 0) >= MIN_GAMES,
        }

    def top_teams(self, n: int = 25) -> list[tuple]:
        return sorted(
            [(t, e) for t, e in self.elos.items()
             if self.game_count.get(t, 0) >= MIN_GAMES],
            key=lambda x: x[1], reverse=True
        )[:n]


if __name__ == "__main__":
    model = NCAAMModel()
    model.load(force_refresh=True)

    print("\n=== Top 25 Teams by ELO ===")
    for i, (team, elo) in enumerate(model.top_teams(25), 1):
        r = model.ratings.get(team, {})
        p = model.profiles.get(team, {})
        inj_str = f" ⚠️ {p.get('injured')}" if p.get("injured") else ""
        print(f"  {i:>2}. {team:<35} ELO:{elo:.0f}  NET:{r.get('net_rtg','?')}{inj_str}")

    print("\n=== Arizona vs Utah State (Round of 32) ===")
    result = model.win_probability("Arizona Wildcats", "Utah State Aggies", seed_a=2, seed_b=10)
    for k, v in result.items():
        print(f"  {k}: {v}")