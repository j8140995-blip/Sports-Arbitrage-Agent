"""
mlb_model.py  v4
MLB win probability model — adds pitcher priors from prior season:
  - Fetches 2025 full-season FIP as prior for each pitcher
  - Bayesian shrinkage now uses pitcher's 2025 FIP as prior (not league avg)
  - Rookies without 2025 data still regress toward league average
  - Full-career context: Yamamoto/Luzardo/etc. get their real skill level
    blended with current-season small samples
"""

import json, time, datetime, math
from pathlib import Path
from collections import defaultdict

CACHE_FILE     = Path("mlb_model_cache.json")
CACHE_TTL      = 6 * 3600

K_FACTOR            = 6
HOME_ADV            = 24
FIP_WEIGHT          = 50
LEAGUE_AVG_FIP      = 4.15
# When a pitcher is unknown (TBD/rookie/no data), use league average rather
# than a "replacement level" 5.00. The 5.00 value was overly pessimistic and
# created artificial edges on TBD games — analysis showed huge-edge MLB bets
# (mostly TBD-driven) ran -21% ROI vs +34% on medium-edge known-pitcher bets.
REPLACEMENT_FIP     = 4.15
PRIOR_WEIGHT_IP     = 40
BULLPEN_WEIGHT      = 35
LEAGUE_AVG_BULL     = 3.90
OFFENSE_WEIGHT      = 8
LEAGUE_AVG_RUNS     = 4.5
DEFAULT_ELO         = 1500
PLATT_A, PLATT_B    = 0.82, 0.04

PARK_FACTORS = {
    "COL": 1.18, "CIN": 1.08, "BOS": 1.06, "TEX": 1.05, "PHI": 1.04,
    "CHC": 1.03, "MIL": 1.02, "HOU": 1.01, "NYY": 1.01,
    "LAD": 0.99, "NYM": 0.99, "SF":  0.97, "OAK": 0.97,
    "SEA": 0.96, "SD":  0.96, "MIA": 0.96, "TB":  0.95,
}

TEAM_ABBR = {
    "Arizona Diamondbacks":"ARI","Atlanta Braves":"ATL","Baltimore Orioles":"BAL",
    "Boston Red Sox":"BOS","Chicago Cubs":"CHC","Chicago White Sox":"CWS",
    "Cincinnati Reds":"CIN","Cleveland Guardians":"CLE","Colorado Rockies":"COL",
    "Detroit Tigers":"DET","Houston Astros":"HOU","Kansas City Royals":"KC",
    "Los Angeles Angels":"LAA","Los Angeles Dodgers":"LAD","Miami Marlins":"MIA",
    "Milwaukee Brewers":"MIL","Minnesota Twins":"MIN","New York Mets":"NYM",
    "New York Yankees":"NYY","Oakland Athletics":"OAK","Philadelphia Phillies":"PHI",
    "Pittsburgh Pirates":"PIT","San Diego Padres":"SD","San Francisco Giants":"SF",
    "Seattle Mariners":"SEA","St. Louis Cardinals":"STL","Tampa Bay Rays":"TB",
    "Texas Rangers":"TEX","Toronto Blue Jays":"TOR","Washington Nationals":"WSH",
    "Athletics":"OAK","Diamondbacks":"ARI","Guardians":"CLE","AZ":"ARI","ATH":"OAK",
}

def normalize_team(name):
    if not name: return name
    name = name.strip()
    if name in TEAM_ABBR.values(): return name
    for full, abbr in TEAM_ABBR.items():
        if full.lower() in name.lower() or name.lower() in full.lower():
            return abbr
    return name.upper()[:3]

def _logit(p): return math.log(max(0.001,min(0.999,p)) / (1-max(0.001,min(0.999,p))))
def _sigmoid(x): return 1/(1+math.exp(-x))


def compute_fip(stat):
    ip = float(stat.get("inningsPitched", 0) or 0)
    if ip <= 0:
        return None, 0.0
    hr  = float(stat.get("homeRuns", 0) or 0)
    bb  = float(stat.get("baseOnBalls", 0) or 0)
    hbp = float(stat.get("hitByPitch", 0) or 0)
    k   = float(stat.get("strikeOuts", 0) or 0)
    fip = (13*hr + 3*(bb+hbp) - 2*k) / ip + 3.2
    return max(1.0, min(9.0, fip)), ip


def regress_with_prior(current_fip: float, current_ip: float,
                       prior_fip: float, prior_weight: float = PRIOR_WEIGHT_IP) -> float:
    """Bayesian shrinkage: (current_ip * current_fip + prior_weight * prior_fip) / (current_ip + prior_weight)"""
    if current_ip <= 0:
        return prior_fip
    return (current_ip * current_fip + prior_weight * prior_fip) / (current_ip + prior_weight)


class MLBModel:
    def __init__(self):
        self.elos          = {}
        self.pitcher_fips  = {}   # name -> {"fip": cur_fip, "ip": cur_ip, "prior_fip": ..., "prior_ip": ...}
        self.bullpen_eras  = {}
        self.team_runs_pg  = {}
        self.games_played  = defaultdict(int)
        self.loaded        = False

    def load(self, force_refresh=False):
        if not force_refresh and CACHE_FILE.exists():
            data = json.loads(CACHE_FILE.read_text())
            if time.time() - data.get("timestamp",0) < CACHE_TTL:
                self.elos         = data.get("elos",{})
                self.pitcher_fips = data.get("pitcher_fips",{})
                self.bullpen_eras  = data.get("bullpen_eras",{})
                self.team_runs_pg  = data.get("team_runs_pg",{})
                self.games_played = defaultdict(int, data.get("games_played",{}))
                self.loaded = True
                print(f"[MLBModel] Loaded {len(self.elos)} teams from cache.")
                return
        print("[MLBModel] Building model from scratch...")
        self._fetch_pitcher_priors()
        self._fetch_pitcher_fips()
        self._build_elos_from_schedule()
        self._save_cache()
        self.loaded = True

    def _fetch_pitcher_priors(self):
        """Fetch 2025 season FIPs as priors. Uses playerPool=ALL to capture
        full league (873+ pitchers) instead of the default 52 qualified starters."""
        try:
            import statsapi
            print("[MLBModel] Fetching 2025 prior-season FIPs...")
            data   = statsapi.get("stats", {
                "stats":      "season",
                "group":      "pitching",
                "season":     2025,
                "sportId":    1,
                "playerPool": "ALL",   # CRITICAL: without this only ~52 pitchers return
                "limit":      1500,
            })
            splits = data.get("stats", [{}])[0].get("splits", [])
            count = 0
            for s in splits:
                stat = s.get("stat", {})
                name = s.get("player", {}).get("fullName", "")
                ip   = float(stat.get("inningsPitched", 0) or 0)
                if ip < 1:   # capture call-ups and brief stints
                    continue
                fip, _ = compute_fip(stat)
                if fip is None:
                    continue
                self.pitcher_fips[name] = {
                    "fip":       None,
                    "ip":        0.0,
                    "prior_fip": round(fip, 2),
                    "prior_ip":  round(ip, 1),
                }
                count += 1
            print(f"[MLBModel] Loaded 2025 priors for {count} pitchers.")
        except Exception as e:
            print(f"[MLBModel] Prior FIP fetch failed: {e}")

    def _fetch_pitcher_fips(self):
        """Fetch 2026 current-season FIPs. Merges with priors. Uses playerPool=ALL
        to get the full league instead of only qualified starters."""
        try:
            import statsapi
            print("[MLBModel] Fetching 2026 pitcher stats...")
            data   = statsapi.get("stats", {
                "stats":      "season",
                "group":      "pitching",
                "season":     2026,
                "sportId":    1,
                "playerPool": "ALL",   # CRITICAL: without this only ~52 pitchers return
                "limit":      1500,
            })
            splits = data.get("stats", [{}])[0].get("splits", [])
            updated = 0
            new_rookies = 0
            for s in splits:
                stat = s.get("stat", {})
                name = s.get("player", {}).get("fullName", "")
                ip   = float(stat.get("inningsPitched", 0) or 0)
                if ip < 1:
                    continue
                fip, _ = compute_fip(stat)
                if fip is None:
                    continue
                if name in self.pitcher_fips:
                    self.pitcher_fips[name]["fip"] = round(fip, 2)
                    self.pitcher_fips[name]["ip"]  = round(ip, 1)
                    updated += 1
                else:
                    self.pitcher_fips[name] = {
                        "fip":       round(fip, 2),
                        "ip":        round(ip, 1),
                        "prior_fip": None,
                        "prior_ip":  0.0,
                    }
                    new_rookies += 1
            total = len(self.pitcher_fips)
            print(f"[MLBModel] Current FIP: {updated} with priors, "
                  f"{new_rookies} new/rookies. Total: {total}.")
        except Exception as e:
            print(f"[MLBModel] Current FIP fetch error: {e}")

    def _build_elos_from_schedule(self):
        import statsapi
        for abbr in set(TEAM_ABBR.values()):
            self.elos[abbr]         = DEFAULT_ELO
            self.bullpen_eras[abbr] = LEAGUE_AVG_BULL
            self.team_runs_pg[abbr] = LEAGUE_AVG_RUNS

        season_start = "2026-03-20"
        today        = datetime.date.today().isoformat()
        print(f"[MLBModel] Fetching 2026 results ({season_start} to {today})...")
        try:
            schedule = statsapi.schedule(start_date=season_start, end_date=today, sportId=1)
        except Exception as e:
            print(f"[MLBModel] Schedule fetch error: {e}")
            return

        team_runs  = defaultdict(float)
        team_gp    = defaultdict(int)
        games_proc = 0

        for game in schedule:
            if game.get("status") != "Final": continue
            home       = normalize_team(game.get("home_name",""))
            away       = normalize_team(game.get("away_name",""))
            home_score = game.get("home_score",0)
            away_score = game.get("away_score",0)
            if not home or not away: continue
            for t in [home,away]:
                if t not in self.elos: self.elos[t] = DEFAULT_ELO
            p_home = self._raw_prob(home,away)
            actual = 1.0 if home_score > away_score else 0.0
            self.elos[home] += K_FACTOR*(actual-p_home)
            self.elos[away] += K_FACTOR*((1-actual)-(1-p_home))
            self.games_played[home] += 1
            self.games_played[away] += 1
            team_runs[home]  += home_score
            team_runs[away]  += away_score
            team_gp[home]    += 1
            team_gp[away]    += 1
            games_proc += 1

        for team, runs in team_runs.items():
            gp = team_gp.get(team,1)
            self.team_runs_pg[team] = round(runs/gp,2) if gp > 0 else LEAGUE_AVG_RUNS

        self._fetch_bullpen_eras()
        print(f"[MLBModel] Processed {games_proc} completed games. Teams: {len(self.elos)}")

    def _fetch_bullpen_eras(self):
        try:
            import statsapi
            teams_data = statsapi.get("teams", {"sportId": 1, "season": 2026})
            teams      = teams_data.get("teams", [])
            count      = 0
            for team in teams:
                team_id   = team["id"]
                team_abbr = normalize_team(team.get("name", ""))
                try:
                    data   = statsapi.get("stats", {
                        "stats": "season", "group": "pitching",
                        "season": 2026, "teamId": team_id,
                        "playerPool": "ALL", "limit": 40,
                    })
                    splits = data.get("stats", [{}])[0].get("splits", [])
                    bull_er = bull_ip = 0.0
                    for s in splits:
                        stat = s.get("stat", {})
                        gs   = int(stat.get("gamesStarted", 0) or 0)
                        ip   = float(stat.get("inningsPitched", 0) or 0)
                        er   = float(stat.get("earnedRuns", 0) or 0)
                        if gs == 0 and ip >= 2:
                            bull_er += er
                            bull_ip += ip
                    if bull_ip > 0:
                        era = (bull_er / bull_ip) * 9
                        self.bullpen_eras[team_abbr] = round(max(1.5, min(8.0, era)), 2)
                        count += 1
                except Exception:
                    pass
            print(f"[MLBModel] Fetched bullpen ERA for {count} teams.")
        except Exception as e:
            print(f"[MLBModel] Bullpen ERA fetch failed: {e}")

    def _raw_prob(self,home,away):
        h = self.elos.get(home,DEFAULT_ELO)+HOME_ADV
        a = self.elos.get(away,DEFAULT_ELO)
        return 1.0/(1.0+10**((a-h)/400))

    def _calibrate(self,p_raw):
        return _sigmoid(PLATT_A*_logit(p_raw)+PLATT_B)

    def get_pitcher_fip(self, name):
        """
        Returns (projected_fip, is_known, raw_current_fip, current_ip, prior_fip, prior_ip).

        Projection logic:
          - Both current + prior: blend current_fip with prior_fip
          - Only prior: use prior_fip (pitcher hasn't thrown in 2026)
          - Only current: regress toward league avg (rookie)
          - Neither: replacement level
        """
        if not name or name.strip() in ("", "TBD", "Unknown"):
            return REPLACEMENT_FIP, False, REPLACEMENT_FIP, 0.0, None, 0.0

        import unicodedata
        def norm(s): return unicodedata.normalize("NFD",s).encode("ascii","ignore").decode().lower().strip()
        nn = norm(name)

        def project(data):
            if isinstance(data, (int, float)):
                return regress_with_prior(float(data), 50.0, LEAGUE_AVG_FIP), True, float(data), 50.0, None, 0.0

            current_fip = data.get("fip")
            current_ip  = data.get("ip", 0.0)
            prior_fip   = data.get("prior_fip")
            prior_ip    = data.get("prior_ip", 0.0)

            # Regress the PRIOR itself toward league average if it's a
            # small-sample number. A pitcher's 14-IP debut shouldn't be
            # treated as their true talent. Use weight = 60 IP — a pitcher
            # with 60+ IP from last year has a reasonably stable estimate.
            # This was the bug that caused the BOS/TOR (+18.4%) and TB/CLE
            # type signals: tiny prior samples got passed through as if
            # they were stable estimates.
            if prior_fip is not None and prior_ip > 0:
                stable_prior = regress_with_prior(
                    prior_fip, prior_ip, LEAGUE_AVG_FIP, prior_weight=60.0
                )
            else:
                stable_prior = None

            if current_fip is not None and stable_prior is not None:
                projected = regress_with_prior(current_fip, current_ip, stable_prior)
            elif current_fip is None and stable_prior is not None:
                # No 2026 data yet — use the regressed prior, not the raw prior
                projected = stable_prior
            elif current_fip is not None and stable_prior is None:
                projected = regress_with_prior(current_fip, current_ip, LEAGUE_AVG_FIP)
            else:
                return REPLACEMENT_FIP, False, REPLACEMENT_FIP, 0.0, None, 0.0

            return projected, True, current_fip if current_fip else LEAGUE_AVG_FIP, current_ip, prior_fip, prior_ip

        for p, data in self.pitcher_fips.items():
            if norm(p) == nn:
                return project(data)
        last = nn.split()[-1] if nn else ""
        for p, data in self.pitcher_fips.items():
            if last and last == norm(p).split()[-1]:
                return project(data)
        for p, data in self.pitcher_fips.items():
            if last and last in norm(p):
                return project(data)

        return REPLACEMENT_FIP, False, REPLACEMENT_FIP, 0.0, None, 0.0

    def win_probability(self, home, away, home_sp="", away_sp=""):
        home = normalize_team(home)
        away = normalize_team(away)

        home_elo = self.elos.get(home, DEFAULT_ELO)
        away_elo = self.elos.get(away, DEFAULT_ELO)

        home_fip, home_known, home_raw_fip, home_ip, home_prior_fip, home_prior_ip = self.get_pitcher_fip(home_sp)
        away_fip, away_known, away_raw_fip, away_ip, away_prior_fip, away_prior_ip = self.get_pitcher_fip(away_sp)
        home_p_adj = (LEAGUE_AVG_FIP - home_fip) * FIP_WEIGHT
        away_p_adj = (LEAGUE_AVG_FIP - away_fip) * FIP_WEIGHT

        home_bull = self.bullpen_eras.get(home, LEAGUE_AVG_BULL)
        away_bull = self.bullpen_eras.get(away, LEAGUE_AVG_BULL)
        home_b_adj = (LEAGUE_AVG_BULL - home_bull) * BULLPEN_WEIGHT
        away_b_adj = (LEAGUE_AVG_BULL - away_bull) * BULLPEN_WEIGHT

        home_rpg = self.team_runs_pg.get(home, LEAGUE_AVG_RUNS)
        away_rpg = self.team_runs_pg.get(away, LEAGUE_AVG_RUNS)
        home_o_adj = (home_rpg - LEAGUE_AVG_RUNS) * OFFENSE_WEIGHT
        away_o_adj = (away_rpg - LEAGUE_AVG_RUNS) * OFFENSE_WEIGHT

        park = PARK_FACTORS.get(home, 1.0)
        park_adj = (park - 1.0) * 8

        h_adj = home_elo + HOME_ADV + home_p_adj + home_b_adj + home_o_adj + park_adj
        a_adj = away_elo + away_p_adj + away_b_adj + away_o_adj

        p_raw = 1.0 / (1.0 + 10 ** ((a_adj - h_adj) / 400))
        p_cal = self._calibrate(p_raw)

        return {
            "p_home":         round(p_cal, 4),
            "p_away":         round(1 - p_cal, 4),
            "home_elo":       round(home_elo),
            "away_elo":       round(away_elo),
            "home_fip":       round(home_fip, 2),
            "away_fip":       round(away_fip, 2),
            "home_raw_fip":   round(home_raw_fip, 2),
            "away_raw_fip":   round(away_raw_fip, 2),
            "home_ip":        round(home_ip, 1),
            "away_ip":        round(away_ip, 1),
            "home_prior_fip": round(home_prior_fip, 2) if home_prior_fip else None,
            "away_prior_fip": round(away_prior_fip, 2) if away_prior_fip else None,
            "home_prior_ip":  round(home_prior_ip, 1),
            "away_prior_ip":  round(away_prior_ip, 1),
            "home_sp":        home_sp or "TBD",
            "away_sp":        away_sp or "TBD",
            "home_sp_known":  home_known,
            "away_sp_known":  away_known,
            "home_bull_era":  round(home_bull, 2),
            "away_bull_era":  round(away_bull, 2),
            "home_rpg":       round(home_rpg, 2),
            "away_rpg":       round(away_rpg, 2),
            "park_factor":    park,
        }

    def _save_cache(self):
        CACHE_FILE.write_text(json.dumps({
            "elos":         self.elos,
            "pitcher_fips": self.pitcher_fips,
            "bullpen_eras": self.bullpen_eras,
            "team_runs_pg": self.team_runs_pg,
            "games_played": dict(self.games_played),
            "timestamp":    time.time(),
        }, indent=2))

    def get_todays_games(self):
        import statsapi
        today = datetime.date.today().strftime("%m/%d/%Y")
        try:
            schedule = statsapi.schedule(date=today, sportId=1)
            games = []
            for g in schedule:
                if g.get("status") in ("Final","Game Over","Completed Early"): continue
                games.append({
                    "game_id":   g.get("game_id"),
                    "home":      normalize_team(g.get("home_name","")),
                    "away":      normalize_team(g.get("away_name","")),
                    "home_name": g.get("home_name",""),
                    "away_name": g.get("away_name",""),
                    "home_sp":   g.get("home_probable_pitcher",""),
                    "away_sp":   g.get("away_probable_pitcher",""),
                    "game_time": g.get("game_datetime",""),
                    "status":    g.get("status",""),
                    "venue":     g.get("venue_name",""),
                })
            return games
        except Exception as e:
            print(f"[MLBModel] Schedule error: {e}")
            return []