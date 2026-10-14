"""
nba_mentions.py
Model for NBA announcer mention markets on Kalshi.

Improvements v2:
1. Player name cross-check — detects if a player named "Jordan" etc is playing
2. Fixed nba_api per_mode parameter
3. Arena names validated against correct home team
4. Jordan/player-name word gets boosted when matching player is on roster
5. Backtest split by player-name-present context (TODO: needs more data)
6. Spread filter tightened to 25 cents
"""

import re
import json
import time
import requests
from pathlib import Path

CACHE_FILE     = Path("nba_mentions_cache.json")
CACHE_TTL      = 4 * 3600
EDGE_THRESHOLD = 0.12
MAX_SPREAD     = 0.25   # tightened from 0.40

HEADERS = {"User-Agent": "Mozilla/5.0"}

# ── Arena → home team mapping ─────────────────────────────────────────────────
ARENA_HOME_TEAM = {
    "spectrum":              "CHA",
    "target center":         "MIN",
    "ball arena":            "DEN",
    "msg / madison square garden": "NYK",
    "intuit":                "LAC",
    "crypto / crypto.com":   "MIA",
    "chase":                 "GSW",
    "paycom":                "OKC",
    "frost bank":            "SAS",
    "fiserv":                "MIL",
    "toyota":                "HOU",
    "american airlines":     "DAL",
    "xfinity":               "PHI",
    "little caesars":        "DET",
    "kia":                   "ATL",
    "rocket":                "CLE",
    "gainbridge":            "IND",
    "capital one":           "WAS",
    "fedexforum":            "MEM",
    "smoothie king":         "NOP",
    "footprint":             "PHX",
    "moda":                  "POR",
    "vivint":                "UTA",
    "golden 1":              "SAC",
    "scotiabank":            "TOR",
    "td garden":             "BOS",
    "united center":         "CHI",
    "barclays":              "BKN",
    "amway":                 "ORL",
    "wells fargo":           "PHI",
}

# Empirically calibrated from 219 settled Kalshi NBA mention markets
BASE_RATES = {
    "injury / injured":               0.97,
    "rookie":                         0.88,
    "all-star":                       0.87,
    "crowd / crowded":                0.85,
    "elbow":                          0.77,
    "playoff":                        0.76,
    "trade / trades / traded":        0.72,
    "tech / technical":               0.63,
    "mvp":                            0.63,
    "draft / drafted":                0.62,
    "airball / airballs / airballed": 0.60,
    "ankle":                          0.58,
    "jordan":                         0.57,
    "triple double":                  0.41,
    "alley-oop":                      0.39,
    "buzzer":                         0.35,
    "overtime":                       0.29,
    "retire / retired / retirement":  0.14,
    "buzzer beater":                  0.04,
    # Arena sponsors
    "spectrum":              0.85,
    "target center":         0.85,
    "ball arena":            0.91,
    "msg / madison square garden": 0.83,
    "intuit":                0.92,
    "crypto / crypto.com":   0.83,
    "chase":                 0.75,
    "paycom":                0.60,
    "frost bank":            0.67,
    "fiserv":                0.80,
    "toyota":                0.80,
    "american airlines":     0.67,
    "xfinity":               0.67,
    "little caesars":        0.67,
    "kia":                   0.75,
    "rocket":                0.77,
}


# ── NBA data fetching ─────────────────────────────────────────────────────────

def fetch_player_stats() -> dict:
    """Fetch current season player stats from NBA API."""
    try:
        from nba_api.stats.endpoints import leaguedashplayerstats
        # Fix: use per_mode_simple as positional kwarg with correct value
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season="2025-26",
            per_mode_detailed="PerGame",
        ).get_data_frames()[0]

        players = {}
        for _, row in stats.iterrows():
            name = row.get("PLAYER_NAME", row.get("playerName", ""))
            team = row.get("TEAM_ABBREVIATION", row.get("teamAbbreviation", ""))
            players[name] = {
                "pts":  float(row.get("PTS", 0)),
                "reb":  float(row.get("REB", 0)),
                "ast":  float(row.get("AST", 0)),
                "team": team,
                "gp":   int(row.get("GP", 0)),
            }
        return players
    except Exception as e:
        print(f"[nba_mentions] Player stats error: {e}")
        return {}


def get_roster_with_experience() -> dict:
    """Get team rosters with rookie status and full player names."""
    try:
        from nba_api.stats.endpoints import commonteamroster
        from nba_api.stats.static import teams
        roster_data = {}
        for team in teams.get_teams():
            tid  = team["id"]
            abbr = team["abbreviation"]
            try:
                data     = commonteamroster.CommonTeamRoster(
                    team_id=tid, season="2025-26"
                ).get_data_frames()[0]
                rookies  = data[data["EXP"] == "R"]["PLAYER"].tolist()
                all_p    = data["PLAYER"].tolist()
                roster_data[abbr] = {
                    "players": all_p,
                    "rookies": rookies,
                }
                time.sleep(0.3)
            except Exception:
                pass
        return roster_data
    except Exception as e:
        print(f"[nba_mentions] Roster error: {e}")
        return {}


def load_nba_mention_data(force_refresh: bool = False) -> dict:
    if not force_refresh and CACHE_FILE.exists():
        data = json.loads(CACHE_FILE.read_text())
        if time.time() - data.get("timestamp", 0) < CACHE_TTL:
            print(f"[nba_mentions] Loaded from cache.")
            return data

    print("[nba_mentions] Fetching NBA player/roster data...")
    players = fetch_player_stats()
    rosters = get_roster_with_experience()

    triple_double_threats = []

    for name, s in players.items():
        cats = sorted([s["pts"], s["reb"], s["ast"]], reverse=True)
        if cats[0] >= 10 and cats[1] >= 7 and cats[2] >= 5:
            triple_double_threats.append(name)

    # Hardcoded MVP contenders for 2025-26 — SGA, Luka, Jokic are top 3
    MVP_CONTENDERS = [
        "Shai Gilgeous-Alexander",
        "Luka Doncic",
        "Nikola Jokic",
        "Giannis Antetokounmpo",
        "Jayson Tatum",
        "LeBron James",
        "Kevin Durant",
        "Stephen Curry",
    ]
    mvp_candidates = [name for name in MVP_CONTENDERS
                      if any(name.lower() in p.lower() or p.lower() in name.lower()
                             for p in players.keys())]

    data = {
        "players":               players,
        "rosters":               rosters,
        "triple_double_threats": triple_double_threats,
        "mvp_candidates":        mvp_candidates,
        "timestamp":             time.time(),
    }
    CACHE_FILE.write_text(json.dumps(data, indent=2))
    print(f"[nba_mentions] TD threats: {triple_double_threats[:5]}")
    print(f"[nba_mentions] MVP candidates: {mvp_candidates[:5]}")
    return data


# ── Player name detection ─────────────────────────────────────────────────────

def players_matching_word(word: str, home_team: str, away_team: str,
                           rosters: dict) -> list[str]:
    """
    Find players on either team whose name contains the word.
    e.g. word="Jordan" → finds Jordan Clarkson, DeAndre Jordan, etc.
    """
    word_lower   = word.lower().split("/")[0].strip()  # use first variant
    home_players = rosters.get(home_team, {}).get("players", [])
    away_players = rosters.get(away_team, {}).get("players", [])
    all_players  = home_players + away_players

    matches = []
    for player in all_players:
        # Check if any word in player name matches
        player_words = player.lower().split()
        if word_lower in player_words or word_lower in player.lower():
            matches.append(player)
    return matches


# ── Word probability model ────────────────────────────────────────────────────

def load_empirical_rates() -> dict:
    cache = Path("nba_mentions_backtest.json")
    if cache.exists():
        return json.loads(cache.read_text())
    return {}


def find_base_rate(word_lower: str, empirical: dict) -> tuple[float, str]:
    # Try empirical first
    for key, stats in empirical.items():
        if stats.get("total", 0) < 10:
            continue
        if any(v.strip() in word_lower for v in key.split("/")):
            return stats["smoothed"], f"empirical(n={stats['total']})"
        if word_lower in key or key in word_lower:
            return stats["smoothed"], f"empirical(n={stats['total']})"
    # Fall back to BASE_RATES
    for key, rate in BASE_RATES.items():
        if any(v.strip() in word_lower for v in key.split("/")):
            return rate, "base_rate"
    return None, "unknown"


def estimate_word_probability(
    word: str,
    home_team: str,
    away_team: str,
    nba_data: dict,
    nba_model=None,
    empirical: dict = None,
) -> dict:
    if empirical is None:
        empirical = load_empirical_rates()

    word_lower = word.lower()
    base_rate, source = find_base_rate(word_lower, empirical)

    if base_rate is None:
        return {"prob": 0.5, "base_rate": 0.5, "source": "unknown", "confidence": "low"}

    prob     = base_rate
    rosters  = nba_data.get("rosters", {})
    players  = nba_data.get("players", {})
    mvp_cands  = set(nba_data.get("mvp_candidates", []))
    td_threats = set(nba_data.get("triple_double_threats", []))

    home_roster = rosters.get(home_team, {})
    away_roster = rosters.get(away_team, {})
    all_players = (home_roster.get("players", []) + away_roster.get("players", []))
    all_rookies = (home_roster.get("rookies", []) + away_roster.get("rookies", []))

    # ── Improvement 1: Player name detection ──────────────────────────────────
    # Check if any player's name matches the word — massive impact on probability
    first_variant = word_lower.split("/")[0].strip()
    matching_players = players_matching_word(first_variant, home_team, away_team, rosters)

    if matching_players and first_variant not in [
        "playoff", "trade", "draft", "injury", "injured", "rookie",
        "crowd", "elbow", "ankle", "tech", "technical", "overtime",
        "buzzer", "alley", "airball", "triple", "mvp", "retire"
    ]:
        # Word matches a player name — very high probability of mention
        prob   = min(0.97, base_rate + 0.30)
        source = f"player_name:{','.join(matching_players[:2])}"

    # ── Overtime/Buzzer: use game model ───────────────────────────────────────
    elif "overtime" in word_lower or "buzzer" in word_lower:
        if nba_model:
            try:
                result    = nba_model.win_probability(home_team, away_team)
                p_home    = result["p_home"]
                closeness = 1 - abs(p_home - 0.5) * 2
                if "overtime" in word_lower:
                    prob = 0.18 + closeness * 0.25
                else:
                    prob = 0.22 + closeness * 0.20
                source = "game_model"
            except Exception:
                pass

    # ── Triple Double: check if TD threat is playing ──────────────────────────
    elif "triple double" in word_lower:
        has_threat = any(t in all_players for t in td_threats)
        if has_threat:
            prob   = min(0.75, base_rate + 0.20)
            source = "td_threat"

    # ── MVP: check if MVP candidate is playing ────────────────────────────────
    elif "mvp" in word_lower:
        has_mvp = any(m in all_players for m in mvp_cands)
        if has_mvp:
            prob   = min(0.82, base_rate + 0.12)
            source = "mvp_candidate"
        else:
            prob   = max(0.45, base_rate - 0.10)
            source = "no_mvp_candidate"

    # ── Rookie ────────────────────────────────────────────────────────────────
    elif "rookie" in word_lower:
        if all_rookies:
            prob   = min(0.95, base_rate + 0.08)
            source = "has_rookies"
        else:
            prob   = max(0.60, base_rate - 0.15)
            source = "no_rookies"

    # ── Alley-oop: assist tendency ────────────────────────────────────────────
    elif "alley" in word_lower:
        team_ast = [players[p]["ast"] for p in all_players if p in players]
        if team_ast:
            avg_ast = sum(team_ast) / len(team_ast)
            if avg_ast > 4.5:
                prob   = min(0.65, base_rate + 0.15)
                source = "high_assist_team"
            elif avg_ast < 2.5:
                prob   = max(0.25, base_rate - 0.10)
                source = "low_assist_team"

    # ── Improvement 3: Arena name — validate correct home team ───────────────
    else:
        for arena_key, arena_team in ARENA_HOME_TEAM.items():
            if arena_key in word_lower or any(
                v.strip() in word_lower for v in arena_key.split("/")
            ):
                if home_team == arena_team:
                    prob   = min(0.95, base_rate + 0.08)
                    source = "arena_name"
                elif away_team == arena_team:
                    # Visiting team's arena mentioned — still happens but less
                    prob   = max(0.20, base_rate - 0.20)
                    source = "wrong_arena"
                else:
                    prob   = 0.15
                    source = "wrong_arena"
                break

    return {
        "prob":       round(max(0.02, min(0.98, prob)), 4),
        "base_rate":  round(base_rate, 4),
        "source":     source,
        "confidence": "medium",
    }


# ── Kalshi scanner ────────────────────────────────────────────────────────────

def parse_teams_from_ticker(ticker: str) -> tuple[str, str]:
    suffix = re.sub(r'KXNBAMENTION-\d{2}[A-Z]{3}\d{2}', '', ticker)
    if not suffix:
        return "", ""

    NBA_TEAMS = [
        "ATL","BOS","BKN","CHA","CHI","CLE","DAL","DEN","DET","GSW",
        "HOU","IND","LAC","LAL","MEM","MIA","MIL","MIN","NOP","NYK",
        "OKC","ORL","PHI","PHX","POR","SAC","SAS","TOR","UTA","WAS",
    ]

    found     = []
    remaining = suffix
    while remaining and len(found) < 2:
        matched = False
        for length in [3, 2]:
            candidate = remaining[:length].upper()
            if candidate in NBA_TEAMS:
                found.append(candidate)
                remaining = remaining[length:]
                matched   = True
                break
        if not matched:
            remaining = remaining[1:]

    if len(found) >= 2:
        return found[0], found[1]
    mid = len(suffix) // 2
    return suffix[:mid], suffix[mid:]


def scan_nba_mentions(nba_data: dict, nba_model=None) -> list[dict]:
    import kalshi_client as kc
    signals  = []
    empirical = load_empirical_rates()

    resp   = kc.get("/events", {"series_ticker": "KXNBAMENTION", "limit": 20, "status": "open"})
    events = resp.get("events", [])
    print(f"[nba_mentions] Found {len(events)} open NBA mention events.")

    for event in events:
        event_ticker        = event.get("event_ticker", "")
        away_team, home_team = parse_teams_from_ticker(event_ticker)

        print(f"\n[nba_mentions] {event_ticker}: {away_team} at {home_team}")
        markets = kc.get(f"/events/{event_ticker}").get("markets", [])
        time.sleep(0.1)

        for m in markets:
            if m.get("status") != "active":
                continue
            word = (m.get("yes_sub_title") or "").strip()
            if not word or "does not qualify" in word.lower():
                continue
            yes_bid = m.get("yes_bid_dollars")
            yes_ask = m.get("yes_ask_dollars")
            if yes_bid is None or yes_ask is None:
                continue

            bid, ask = float(yes_bid), float(yes_ask)
            if ask >= 0.95 or bid <= 0.02:
                continue
            # Improvement 6: tighter spread filter
            if ask - bid > MAX_SPREAD:
                continue

            mid    = (bid + ask) / 2
            result = estimate_word_probability(
                word, home_team, away_team, nba_data, nba_model, empirical
            )

            if result["confidence"] == "low":
                continue

            edge = result["prob"] - mid
            if abs(edge) < EDGE_THRESHOLD:
                continue

            side = "yes" if edge > 0 else "no"
            signals.append({
                "ticker":     m.get("ticker", ""),
                "event":      event_ticker,
                "home":       home_team,
                "away":       away_team,
                "word":       word,
                "model_prob": round(result["prob"], 4),
                "market_mid": round(mid, 4),
                "edge":       round(edge, 4),
                "side":       side,
                "bid":        bid,
                "ask":        ask,
                "spread":     round(ask - bid, 4),
                "source":     result["source"],
                "confidence": result["confidence"],
            })

    signals.sort(key=lambda s: abs(s["edge"]), reverse=True)
    return signals


def print_nba_mention_signals(signals: list[dict]):
    if not signals:
        print("[nba_mentions] No signals above threshold.")
        return

    current_event = None
    for s in signals:
        if s["event"] != current_event:
            current_event = s["event"]
            print(f"\n{'='*90}")
            print(f"  {current_event}: {s['away']} at {s['home']}")
            print(f"{'='*90}")
            print(f"  {'WORD':<42} {'MODEL':>7} {'MID':>7} {'EDGE':>7} {'SIDE':<5} {'SOURCE'}")
            print(f"  {'-'*80}")
        print(
            f"  {s['word']:<42} "
            f"{s['model_prob']*100:>6.1f}% "
            f"{s['market_mid']*100:>6.1f}% "
            f"{s['edge']*100:>+6.1f}% "
            f"{s['side']:<5} "
            f"{s['source']}"
        )
    print(f"\n  Total NBA mention signals: {len(signals)}")


def log_mention_trade(signal: dict, sport: str = "NBA_MENTION"):
    import datetime
    log_file = Path("paper_trades.json")
    trades   = json.loads(log_file.read_text()) if log_file.exists() else []

    if any(t.get("ticker") == signal["ticker"] for t in trades):
        return False

    entry_price = signal["ask"] if signal["side"] == "yes" else 1 - signal["bid"]
    contracts   = max(1, int(5 / entry_price))
    cost        = contracts * entry_price

    trades.append({
        "ticker":            signal["ticker"],
        "title":             signal.get("word", ""),
        "sport":             sport,
        "side":              signal["side"],
        "market_mid":        signal["market_mid"],
        "model_prob":        signal["model_prob"],
        "edge":              signal["edge"],
        "kelly_fraction":    0.0,
        "half_kelly":        0.0,
        "contracts":         contracts,
        "cost_dollars":      round(cost, 2),
        "entry_price_cents": round(entry_price * 100),
        "timestamp":         datetime.datetime.utcnow().isoformat(),
        "mode":              "paper",
        "status":            "open",
        "outcome":           None,
        "pnl":               None,
        "game_date":         datetime.date.today().isoformat(),
        "home_team":         signal.get("home", ""),
        "away_team":         signal.get("away", ""),
        "source":            signal.get("source", ""),
        "event":             signal.get("event", ""),
        "word":              signal.get("word", ""),
    })

    log_file.write_text(json.dumps(trades, indent=2))
    return True


if __name__ == "__main__":
    data = load_nba_mention_data(force_refresh=False)
    print("\n=== Scanning Kalshi NBA mention markets ===")
    signals = scan_nba_mentions(data)
    print_nba_mention_signals(signals)
    logged = sum(1 for s in signals if log_mention_trade(s, "NBA_MENTION"))
    if logged:
        print(f"\n[nba_mentions] Logged {logged} new trades to paper_trades.json")