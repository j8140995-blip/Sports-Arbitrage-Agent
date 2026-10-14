"""
nba_model.py  v3
NBA win-probability model — improvements:
  - DARKO DPM integration for quantified injury impact
  - Injury status weighting (Out=100%, Doubtful=75%, Day-To-Day=50%, Questionable=25%)
  - Rest/travel adjustments unchanged
  - Platt scaling calibration

v3 FIX: Previously only "Out" and "Suspension" triggered injury penalties.
Stars listed as "Day-To-Day" (Wemby 4/24/26 e.g.) were ignored, causing the
model to overrate teams whose stars were actually sitting. Now uses partial
penalties for softer injury labels.
"""

import time
import json
import datetime
from pathlib import Path
from typing import Optional

K_FACTOR        = 20
BASE_ELO        = 1500
B2B_PENALTY     = 65
CACHE_FILE      = Path("nba_elo_cache.json")
CACHE_TTL_HOURS = 12
CURRENT_SEASON  = "2025-26"
RECENT_N_GAMES  = 15

W_ELO    = 0.30
W_RATING = 0.50
W_RECENT = 0.20

HOME_COURT = {
    "Oklahoma City Thunder":      90,
    "Boston Celtics":             88,
    "Sacramento Kings":           87,
    "Utah Jazz":                  85,
    "Denver Nuggets":             85,
    "Indiana Pacers":             83,
    "Memphis Grizzlies":          83,
    "Milwaukee Bucks":            82,
    "Cleveland Cavaliers":        82,
    "New York Knicks":            80,
    "San Antonio Spurs":          80,
    "Houston Rockets":            80,
    "Golden State Warriors":      78,
    "Minnesota Timberwolves":     78,
    "Chicago Bulls":              76,
    "Philadelphia 76ers":         76,
    "Detroit Pistons":            75,
    "Atlanta Hawks":              75,
    "Charlotte Hornets":          74,
    "Toronto Raptors":            74,
    "Portland Trail Blazers":     73,
    "New Orleans Pelicans":       73,
    "Washington Wizards":         72,
    "Orlando Magic":              72,
    "Dallas Mavericks":           72,
    "Phoenix Suns":               71,
    "Miami Heat":                 70,
    "Los Angeles Lakers":         70,
    "Los Angeles Clippers":       68,
    "Brooklyn Nets":              67,
}
DEFAULT_HOME_COURT = 75

TEAM_ALIASES = {
    "lakers": "Los Angeles Lakers", "lal": "Los Angeles Lakers",
    "celtics": "Boston Celtics", "bos": "Boston Celtics",
    "warriors": "Golden State Warriors", "gsw": "Golden State Warriors",
    "nuggets": "Denver Nuggets", "den": "Denver Nuggets",
    "heat": "Miami Heat", "mia": "Miami Heat",
    "bucks": "Milwaukee Bucks", "mil": "Milwaukee Bucks",
    "76ers": "Philadelphia 76ers", "sixers": "Philadelphia 76ers", "phi": "Philadelphia 76ers",
    "suns": "Phoenix Suns", "phx": "Phoenix Suns",
    "clippers": "Los Angeles Clippers", "lac": "Los Angeles Clippers",
    "la clippers": "Los Angeles Clippers",
    "knicks": "New York Knicks", "nyk": "New York Knicks",
    "nets": "Brooklyn Nets", "bkn": "Brooklyn Nets",
    "bulls": "Chicago Bulls", "chi": "Chicago Bulls",
    "cavaliers": "Cleveland Cavaliers", "cavs": "Cleveland Cavaliers", "cle": "Cleveland Cavaliers",
    "mavericks": "Dallas Mavericks", "mavs": "Dallas Mavericks", "dal": "Dallas Mavericks",
    "pistons": "Detroit Pistons", "det": "Detroit Pistons",
    "rockets": "Houston Rockets", "hou": "Houston Rockets",
    "pacers": "Indiana Pacers", "ind": "Indiana Pacers",
    "grizzlies": "Memphis Grizzlies", "mem": "Memphis Grizzlies",
    "timberwolves": "Minnesota Timberwolves", "wolves": "Minnesota Timberwolves", "min": "Minnesota Timberwolves",
    "pelicans": "New Orleans Pelicans", "nop": "New Orleans Pelicans",
    "thunder": "Oklahoma City Thunder", "okc": "Oklahoma City Thunder",
    "magic": "Orlando Magic", "orl": "Orlando Magic",
    "trail blazers": "Portland Trail Blazers", "blazers": "Portland Trail Blazers", "por": "Portland Trail Blazers",
    "kings": "Sacramento Kings", "sac": "Sacramento Kings",
    "spurs": "San Antonio Spurs", "sas": "San Antonio Spurs",
    "raptors": "Toronto Raptors", "tor": "Toronto Raptors",
    "jazz": "Utah Jazz", "uta": "Utah Jazz",
    "wizards": "Washington Wizards", "was": "Washington Wizards",
    "hawks": "Atlanta Hawks", "atl": "Atlanta Hawks",
    "hornets": "Charlotte Hornets", "cha": "Charlotte Hornets",
}

ESPN_TO_MODEL = {
    "LA Clippers":  "Los Angeles Clippers",
    "LA Lakers":    "Los Angeles Lakers",
    "GS Warriors":  "Golden State Warriors",
    "SA Spurs":     "San Antonio Spurs",
    "NY Knicks":    "New York Knicks",
    "OKC Thunder":  "Oklahoma City Thunder",
    "NO Pelicans":  "New Orleans Pelicans",
    "NJ Nets":      "Brooklyn Nets",
}

KNOWN_STARS = {
    "Stephen Curry", "LeBron James", "Nikola Jokic", "Giannis Antetokounmpo",
    "Joel Embiid", "Luka Doncic", "Kevin Durant", "Jayson Tatum",
    "Shai Gilgeous-Alexander", "Damian Lillard", "Kawhi Leonard", "Anthony Davis",
    "Tyrese Haliburton", "Ja Morant", "Zion Williamson", "Devin Booker",
    "Trae Young", "Donovan Mitchell", "Bam Adebayo", "Draymond Green",
    "Kyrie Irving", "Jimmy Butler", "Jimmy Butler III", "Tyrese Maxey",
    "Paul George", "Jaylen Brown", "De'Aaron Fox", "Lauri Markkanen",
    "Jaren Jackson Jr.", "Jalen Williams", "Domantas Sabonis", "Zach LaVine",
    "Cooper Flagg", "Victor Wembanyama", "Paolo Banchero", "Cade Cunningham",
    "Evan Mobley", "Franz Wagner", "Anthony Edwards",
}

INJURY_PENALTIES = {"star": 60, "starter": 25}
PACE_ADVANTAGE_FACTOR = 0.015

# Partial penalties for softer injury labels.
# Stars listed Day-To-Day often end up sitting, and Vegas prices this in immediately
# — our model needs to react similarly.
INJURY_STATUS_WEIGHTS = {
    "out":                 1.00,
    "suspension":          1.00,
    "doubtful":            0.75,
    "day-to-day":          0.50,
    "day to day":          0.50,
    "game time decision":  0.50,
    "gtd":                 0.50,
    "questionable":        0.25,
    "probable":            0.00,
    "available":           0.00,
}

_auto_stars:     set[str]         = set()
_player_minutes: dict[str, float] = {}
_darko_players:  dict             = {}


def normalize_team(name: str) -> str:
    return TEAM_ALIASES.get(name.lower().strip(), name.strip())


def get_home_court(team: str) -> int:
    return HOME_COURT.get(team, DEFAULT_HOME_COURT)


def _load_darko():
    global _darko_players
    if _darko_players:
        return _darko_players
    try:
        from darko_client import fetch_darko_dpm
        raw = fetch_darko_dpm()
        # Cache file has nested structure: {'players': {...}, 'timestamp': ...}
        # Unwrap if needed so we have a flat dict of player_name → {dpm, mpg}
        if isinstance(raw, dict) and 'players' in raw and isinstance(raw.get('players'), dict):
            _darko_players = raw['players']
        else:
            _darko_players = raw or {}
        if _darko_players:
            print(f"[NBAModel] DARKO/EPM loaded: {len(_darko_players)} players.")
    except Exception as e:
        print(f"[NBAModel] DARKO load failed: {e}")
        _darko_players = {}
    return _darko_players


def fetch_player_data() -> tuple[set[str], dict[str, float]]:
    global _auto_stars, _player_minutes
    if _auto_stars:
        return _auto_stars | KNOWN_STARS, _player_minutes
    try:
        from nba_api.stats.endpoints import leaguedashplayerstats
        time.sleep(0.6)
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=CURRENT_SEASON,
            per_mode_detailed="PerGame",
            season_type_all_star="Regular Season",
        )
        df = stats.get_data_frames()[0]
        if "MIN" in df.columns:
            for _, row in df.iterrows():
                _player_minutes[row["PLAYER_NAME"]] = float(row.get("MIN", 0) or 0)
        top = df[df["GP"] >= 20].nlargest(65, "PTS")
        _auto_stars = set(top["PLAYER_NAME"].tolist())
        print(f"[NBAModel] Auto-detected {len(_auto_stars)} star players. "
              f"Minutes loaded for {len(_player_minutes)} players.")
    except Exception as e:
        print(f"[NBAModel] Player data fetch failed: {e}. Using known list.")
        _auto_stars = set()
    return _auto_stars | KNOWN_STARS, _player_minutes


def expected_score(elo_a: float, elo_b: float) -> float:
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def mov_multiplier(point_diff: int, winner_elo: float, loser_elo: float) -> float:
    elo_diff = winner_elo - loser_elo
    return (abs(point_diff) + 3) ** 0.8 / (7.5 + 0.006 * elo_diff)


def update_elo_with_mov(winner_elo: float, loser_elo: float,
                         point_diff: int) -> tuple[float, float]:
    exp        = expected_score(winner_elo, loser_elo)
    mov        = mov_multiplier(abs(point_diff), winner_elo, loser_elo)
    k          = K_FACTOR * mov
    new_winner = winner_elo + k * (1 - exp)
    new_loser  = loser_elo  + k * (0 - (1 - exp))
    return new_winner, new_loser


def _load_cache() -> Optional[dict]:
    if CACHE_FILE.exists():
        data = json.loads(CACHE_FILE.read_text())
        if (time.time() - data.get("timestamp", 0)) / 3600 < CACHE_TTL_HOURS:
            return data
    return None


def _save_cache(data: dict):
    data["timestamp"] = time.time()
    CACHE_FILE.write_text(json.dumps(data, indent=2))


def fetch_game_logs(seasons: list[str] = None) -> list[dict]:
    try:
        from nba_api.stats.endpoints import leaguegamelog
    except ImportError:
        print("[nba_model] nba_api not installed.")
        return []
    if seasons is None:
        seasons = [CURRENT_SEASON]
    all_games = []
    for season in seasons:
        try:
            time.sleep(0.6)
            log = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star="Regular Season",
                direction="ASC",
            )
            df = log.get_data_frames()[0]
            for _, row in df.iterrows():
                all_games.append({
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
        except Exception as e:
            print(f"[nba_model] Error fetching {season}: {e}")
    all_games.sort(key=lambda g: g["date"])
    return all_games


def build_elo_ratings(games: list[dict]) -> dict[str, float]:
    elos: dict[str, float] = {}
    seen: set[str] = set()

    def get_elo(team):
        return elos.setdefault(team, float(BASE_ELO))

    for g in games:
        team     = g["team"]
        parts    = g["matchup"].split()
        opponent = normalize_team(parts[-1]) if parts else None
        if not opponent or opponent == team:
            continue
        key = "|".join(sorted([team, opponent])) + g["date"]
        if key in seen:
            continue
        seen.add(key)
        team_elo     = get_elo(team)
        opponent_elo = get_elo(opponent)
        point_diff   = abs(g.get("plus_minus", 0))
        if g["wl"] == "W":
            elos[team], elos[opponent] = update_elo_with_mov(team_elo, opponent_elo, point_diff)
        elif g["wl"] == "L":
            elos[opponent], elos[team] = update_elo_with_mov(opponent_elo, team_elo, point_diff)
    return elos


def estimate_possessions(pts, fga, fta, oreb, tov) -> float:
    return fga - oreb + tov + 0.44 * fta


def build_team_ratings(games: list[dict]) -> dict[str, dict]:
    game_lookup: dict[str, list[dict]] = {}
    for g in games:
        parts    = g["matchup"].split()
        opponent = normalize_team(parts[-1]) if parts else None
        if not opponent or opponent == g["team"]:
            continue
        key = "|".join(sorted([g["team"], opponent])) + g["date"]
        if key not in game_lookup:
            game_lookup[key] = []
        game_lookup[key].append(g)

    team_stats: dict[str, dict] = {}
    for key, pair in game_lookup.items():
        if len(pair) != 2:
            continue
        t1, t2 = pair[0], pair[1]
        poss1  = estimate_possessions(t1["pts"], t1["fga"], t1["fta"], t1["oreb"], t1["tov"])
        poss2  = estimate_possessions(t2["pts"], t2["fga"], t2["fta"], t2["oreb"], t2["tov"])
        poss   = max((poss1 + poss2) / 2, 1)
        for scorer, conceder in [(t1, t2), (t2, t1)]:
            name = scorer["team"]
            if name not in team_stats:
                team_stats[name] = {"pts_scored": 0, "pts_allowed": 0, "poss": 0, "games": 0}
            team_stats[name]["pts_scored"]  += scorer["pts"]
            team_stats[name]["pts_allowed"] += conceder["pts"]
            team_stats[name]["poss"]        += poss
            team_stats[name]["games"]       += 1

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
        }
    return ratings


def build_pace_ratings(games: list[dict]) -> dict[str, float]:
    team_poss  = {}
    team_games = {}
    for g in games:
        parts    = g["matchup"].split()
        opponent = parts[-1] if parts else None
        if not opponent:
            continue
        pts  = g.get("pts", 0) or 0
        tov  = g.get("tov", 0) or 0
        fga  = g.get("fga", 0) or 0
        fta  = g.get("fta", 0) or 0
        oreb = g.get("oreb", 0) or 0
        poss = fga - oreb + tov + 0.44 * fta if fga > 0 else pts / 1.05
        team = g["team"]
        team_poss[team]  = team_poss.get(team, 0) + poss
        team_games[team] = team_games.get(team, 0) + 1
    pace = {}
    for team in team_poss:
        if team_games[team] >= 5:
            pace[team] = team_poss[team] / team_games[team]
    return pace


def pace_win_prob_adjustment(home: str, away: str, pace: dict) -> float:
    home_pace = pace.get(home, 100.0)
    away_pace = pace.get(away, 100.0)
    diff      = home_pace - away_pace
    adj = -(diff / 5) * PACE_ADVANTAGE_FACTOR
    return max(-0.04, min(0.04, adj))


def rating_win_prob(home: str, away: str, ratings: dict) -> float:
    home_net = ratings.get(home, {}).get("net_rtg", 0.0)
    away_net = ratings.get(away, {}).get("net_rtg", 0.0)
    diff     = (home_net - away_net) + 3.0
    prob     = 1 / (1 + 10 ** (-diff / 10))
    return max(0.05, min(0.95, prob))


def build_recent_form(games: list[dict], elos: dict[str, float],
                      n: int = RECENT_N_GAMES) -> dict[str, float]:
    by_team: dict[str, list[dict]] = {}
    for g in games:
        by_team.setdefault(g["team"], []).append(g)
    form           = {}
    league_avg_elo = sum(elos.values()) / len(elos) if elos else BASE_ELO
    for team, tgames in by_team.items():
        recent = sorted(tgames, key=lambda g: g["date"], reverse=True)[:n]
        weighted_wins  = 0.0
        weighted_total = 0.0
        for i, g in enumerate(recent):
            recency_w = 3.0 if i < 5 else (2.0 if i < 10 else 1.0)
            parts     = g["matchup"].split()
            opponent  = normalize_team(parts[-1]) if parts else None
            opp_elo   = elos.get(opponent, league_avg_elo) if opponent else league_avg_elo
            strength_w = max(0.7, min(1.5, opp_elo / league_avg_elo))
            weight         = recency_w * strength_w
            weighted_total += weight
            if g["wl"] == "W":
                weighted_wins += weight
        form[team] = weighted_wins / weighted_total if weighted_total else 0.5
    return form


def recent_form_win_prob(home: str, away: str, form: dict) -> float:
    p_home = form.get(home, 0.5)
    p_away = form.get(away, 0.5)
    p_home = min(0.95, p_home + 0.03)
    num    = p_home * (1 - p_away)
    denom  = p_home * (1 - p_away) + p_away * (1 - p_home)
    if denom == 0:
        return 0.5
    return max(0.05, min(0.95, num / denom))


def get_team_game_dates(team: str, games: list[dict]) -> list[datetime.date]:
    team_games = sorted(
        [g for g in games if g["team"] == team],
        key=lambda g: g["date"], reverse=True
    )
    return [datetime.date.fromisoformat(g["date"]) for g in team_games]


def rest_days(team: str, games: list[dict],
              reference_date: datetime.date = None) -> int:
    dates = get_team_game_dates(team, games)
    if not dates:
        return 7
    ref = reference_date or datetime.date.today()
    return (ref - dates[0]).days


def is_back_to_back(team: str, games: list[dict],
                    reference_date: datetime.date = None) -> bool:
    return rest_days(team, games, reference_date) == 1


def fetch_injury_report() -> dict[str, list[dict]]:
    try:
        import requests
        resp = requests.get(
            "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries",
            timeout=10
        )
        data   = resp.json()
        result = {}
        for team in data.get("injuries", []):
            team_name = ESPN_TO_MODEL.get(
                team.get("displayName", ""), team.get("displayName", ""))
            players = []
            for p in team.get("injuries", []):
                players.append({
                    "player": p.get("athlete", {}).get("displayName", ""),
                    "status": p.get("status", ""),
                })
            result[team_name] = players
        return result
    except Exception as e:
        print(f"[injuries] Failed to fetch: {e}")
        return {}


def injury_elo_adjustment(team: str, injury_report: dict,
                           star_players: set[str] = None,
                           player_minutes: dict = None) -> float:
    """
    Returns negative Elo adjustment for injured players.
    Uses DARKO DPM if available; falls back to minutes-based heuristic.

    Status weighting (see INJURY_STATUS_WEIGHTS):
      Out/Suspension = 100%
      Doubtful       =  75%
      Day-To-Day     =  50%
      Questionable   =  25%
    """
    if star_players is None:
        star_players = KNOWN_STARS
    if player_minutes is None:
        player_minutes = {}

    darko   = _load_darko()
    penalty = 0.0
    REPLACEMENT_DPM    = -3.0
    AVG_STARTER_MINS   = 28.0
    TEAM_MINUTES       = 240.0
    ELO_PER_POINT      = 28.0

    for p in injury_report.get(team, []):
        status        = (p.get("status") or "").strip().lower()
        status_weight = INJURY_STATUS_WEIGHTS.get(status, 0.0)
        if status_weight == 0.0:
            continue

        name = p["player"]
        mpg  = player_minutes.get(name, AVG_STARTER_MINS)

        if darko:
            player_dpm = None
            impact_data = darko.get(name)
            if impact_data:
                player_dpm = impact_data["dpm"]
                mpg        = impact_data.get("mpg", mpg)
            else:
                name_lower = name.lower()
                for pname, pdata in darko.items():
                    if name_lower in pname.lower() or pname.lower() in name_lower:
                        player_dpm = pdata["dpm"]
                        mpg        = pdata.get("mpg", mpg)
                        break

            if player_dpm is not None:
                minutes_share = min(mpg / TEAM_MINUTES, 0.20)
                point_swing   = (player_dpm - REPLACEMENT_DPM) * minutes_share

                # Superstar boost: players with DPM >= 4 have outsized team impact
                # (they raise teammate play, change defensive schemes, etc.).
                # Linear (dpm - replacement) * minutes share understates this.
                if player_dpm >= 4.0:
                    point_swing *= 1.6   # ~60% extra impact for elite players

                # When a superstar is merely "Day-To-Day" but Vegas has moved the line
                # significantly, it's usually because they're actually sitting. We
                # scale partial penalties less aggressively for stars to reflect this.
                effective_weight = status_weight
                if player_dpm >= 4.0 and status_weight < 1.0:
                    effective_weight = min(1.0, status_weight + 0.25)

                penalty      += point_swing * ELO_PER_POINT * effective_weight
                continue

        # Fallback: star/starter heuristic
        if name in star_players:
            base_penalty = INJURY_PENALTIES["star"]
        else:
            base_penalty = INJURY_PENALTIES["starter"]
        minutes_scale = mpg / AVG_STARTER_MINS
        penalty += base_penalty * minutes_scale * status_weight

    return -penalty


def fetch_star_players() -> set[str]:
    stars, _ = fetch_player_data()
    return stars


class NBAModel:
    def __init__(self):
        self.elos:           dict[str, float] = {}
        self.ratings:        dict[str, dict]  = {}
        self.form:           dict[str, float] = {}
        self.games:          list[dict]       = []
        self.stars:          set[str]         = KNOWN_STARS
        self.player_minutes: dict[str, float] = {}
        self._loaded = False

    def load(self, force_refresh: bool = False):
        cache = None if force_refresh else _load_cache()
        if cache:
            self.elos    = cache["elos"]
            self.ratings = cache.get("ratings", {})
            self.form    = cache.get("form", {})
            self.pace    = cache.get("pace", {})
            self.games   = cache["games"]
            print(f"[NBAModel] Loaded {len(self.elos)} teams from cache.")
        else:
            print("[NBAModel] Fetching game logs from stats.nba.com...")
            self.games = fetch_game_logs()
            if not self.games:
                print("[NBAModel] No games loaded.")
                return
            print("[NBAModel] Fetching player data...")
            self.stars, self.player_minutes = fetch_player_data()
            self.elos    = build_elo_ratings(self.games)
            self.ratings = build_team_ratings(self.games)
            self.form    = build_recent_form(self.games, self.elos)
            self.pace    = build_pace_ratings(self.games)
            _save_cache({
                "elos":    self.elos,
                "ratings": self.ratings,
                "form":    self.form,
                "pace":    self.pace,
                "games":   self.games,
            })
            print(f"[NBAModel] Built model for {len(self.elos)} teams from {len(self.games)} games.")
        self._loaded = True

    def win_probability(self, home_team: str, away_team: str,
                        apply_rest: bool = True,
                        injury_report: dict = None) -> dict:
        home = normalize_team(home_team)
        away = normalize_team(away_team)

        home_elo     = self.elos.get(home, BASE_ELO)
        away_elo     = self.elos.get(away, BASE_ELO)
        home_court   = get_home_court(home)
        adj_home_elo = home_elo + home_court

        if apply_rest and self.games:
            ref_date  = datetime.date.today()
            home_rest = rest_days(home, self.games, ref_date)
            away_rest = rest_days(away, self.games, ref_date)

            if home_rest == 0:
                adj_home_elo -= B2B_PENALTY * 1.5
            elif home_rest == 1:
                adj_home_elo -= B2B_PENALTY

            if away_rest == 0:
                away_elo -= B2B_PENALTY * 1.8
            elif away_rest == 1:
                away_elo -= B2B_PENALTY * 1.2

            try:
                from travel import away_travel_penalty, home_travel_penalty
                away_travel = away_travel_penalty(away, home, self.games)
                home_travel = home_travel_penalty(home, self.games)
                away_elo     -= away_travel
                adj_home_elo -= home_travel
            except ImportError:
                pass

        # Load player data (stars + minutes) if not already done, so injuries work
        if not self.player_minutes:
            self.stars, self.player_minutes = fetch_player_data()

        if injury_report:
            adj_home_elo += injury_elo_adjustment(
                home, injury_report, self.stars, self.player_minutes)
            away_elo += injury_elo_adjustment(
                away, injury_report, self.stars, self.player_minutes)

        p_elo = expected_score(adj_home_elo, away_elo)

        p_rating = rating_win_prob(home, away, self.ratings)

        if injury_report:
            home_inj   = injury_elo_adjustment(home, injury_report, self.stars, self.player_minutes)
            away_inj   = injury_elo_adjustment(away, injury_report, self.stars, self.player_minutes)
            # Convert Elo penalty diff back to net rating points (~28 Elo per point),
            # then to win-prob shift (~3.5 pct points per net rating point).
            # Previous code used * 0.05 which nearly zeroed this out.
            net_rating_shift = (home_inj - away_inj) / 28.0   # net rating points
            p_rating_shift   = net_rating_shift * 0.035        # win prob shift
            p_rating         = max(0.05, min(0.95, p_rating + p_rating_shift))

        p_recent  = recent_form_win_prob(home, away, self.form)
        pace_adj  = pace_win_prob_adjustment(home, away, self.pace) if self.pace else 0.0

        p_home = W_ELO * p_elo + W_RATING * p_rating + W_RECENT * p_recent + pace_adj
        p_home = max(0.05, min(0.95, p_home))

        home_b2b = is_back_to_back(home, self.games) if self.games else False
        away_b2b = is_back_to_back(away, self.games) if self.games else False

        return {
            "home_team":   home,
            "away_team":   away,
            "home_elo":    round(home_elo, 1),
            "away_elo":    round(away_elo, 1),
            "home_court":  home_court,
            "p_home":      round(p_home, 4),
            "p_away":      round(1 - p_home, 4),
            "p_elo":       round(p_elo, 4),
            "p_rating":    round(p_rating, 4),
            "p_recent":    round(p_recent, 4),
            "pace_adj":    round(pace_adj, 4),
            "home_b2b":    home_b2b,
            "away_b2b":    away_b2b,
        }

    def top_teams(self, n: int = 30) -> list[tuple[str, float]]:
        return sorted(self.elos.items(), key=lambda x: x[1], reverse=True)[:n]

    def team_report(self, team: str) -> dict:
        t = normalize_team(team)
        return {
            "team":       t,
            "elo":        round(self.elos.get(t, BASE_ELO), 1),
            "home_court": get_home_court(t),
            "ortg":       self.ratings.get(t, {}).get("ortg", "?"),
            "drtg":       self.ratings.get(t, {}).get("drtg", "?"),
            "net_rtg":    self.ratings.get(t, {}).get("net_rtg", "?"),
            "recent":     f"{self.form.get(t, 0.5)*100:.0f}% L{RECENT_N_GAMES}",
            "b2b":        is_back_to_back(t, self.games) if self.games else False,
        }


if __name__ == "__main__":
    model = NBAModel()
    model.load(force_refresh=True)
    print(f"\n=== {len(model.stars)} star players detected ===")
    injuries = fetch_injury_report()
    print("\n=== PHX at SAS ===")
    result = model.win_probability("San Antonio Spurs", "Phoenix Suns", injury_report=injuries)
    for k, v in result.items():
        print(f"  {k}: {v}")


# ── Standings + playoff motivation ───────────────────────────────────────────

def fetch_nba_standings() -> dict[str, dict]:
    try:
        from nba_api.stats.endpoints import leaguestandingsv3
        time.sleep(0.6)
        s  = leaguestandingsv3.LeagueStandingsV3(season=CURRENT_SEASON)
        df = s.get_data_frames()[0]
        standings = {}
        for _, row in df.iterrows():
            full_name = normalize_team(f"{row['TeamCity']} {row['TeamName']}")
            standings[full_name] = {
                "wins":      int(row["WINS"]),
                "losses":    int(row["LOSSES"]),
                "conf_seed": int(row["PlayoffRank"]),
            }
        return standings
    except Exception as e:
        print(f"[standings] Failed: {e}")
        return {}


def playoff_motivation_adjustment(team: str, standings: dict) -> float:
    if not standings or team not in standings:
        return 0.0
    seed = standings[team].get("conf_seed", 8)
    gb   = standings[team].get("games_back", 0)
    if 7 <= seed <= 10 and gb <= 3:
        return 25.0
    if seed <= 3 and gb < -5:
        return -10.0
    if seed >= 11 and gb > 8:
        return -15.0
    return 0.0


# ── Line movement detection ───────────────────────────────────────────────────

_line_history: dict[str, list[tuple[float, float]]] = {}


def record_line(ticker: str, price: float):
    import time as _time
    if ticker not in _line_history:
        _line_history[ticker] = []
    _line_history[ticker].append((_time.time(), price))
    cutoff = _time.time() - 10800
    _line_history[ticker] = [(t, p) for t, p in _line_history[ticker] if t > cutoff]


def line_movement(ticker: str, current_price: float) -> float:
    import time as _time
    history = _line_history.get(ticker, [])
    cutoff  = _time.time() - 7200
    recent  = [(t, p) for t, p in history if t > cutoff]
    if not recent:
        return 0.0
    return round((current_price - recent[0][1]) * 100, 1)


def is_suspicious_line_movement(ticker: str, current_price: float,
                                 threshold: float = 5.0) -> bool:
    return abs(line_movement(ticker, current_price)) >= threshold