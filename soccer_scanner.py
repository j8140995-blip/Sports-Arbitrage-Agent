"""
soccer_scanner.py
Scans Kalshi soccer markets and generates signals using the Poisson model.

Output now shows:
- Explicit SIDE (YES/NO) for the ticker
- Human-readable bet description (e.g. "Bet ON Barcelona" or "Bet AGAINST Tottenham")
"""

import os
import re
import time
import datetime
import requests
from dataclasses import dataclass
from typing import Optional
import kalshi_client as kc

EDGE_THRESHOLD  = 0.08
MIN_WIN_PROB    = 0.38
MIN_GAMES       = 8
MAX_XG          = 7.0
SKIP_DRAWS      = False

MIN_VOLUME = {
    "EPL":        10_000,
    "LALIGA":     10_000,
    "BUNDESLIGA":  5_000,
    "UCL":         5_000,
    "LIGUE1":      3_000,
    "MLS":         3_000,
    "SERIEA":      5_000,
    "FACUP":       3_000,
    "UEL":         5_000,
    "UECL":        3_000,
    "CLUBWC":      3_000,
    "FIFAQUAL":    2_000,
}

LEAGUE_EDGE_THRESHOLD = {
    "EPL":        0.15,
    "LALIGA":     0.08,
    "BUNDESLIGA": 0.08,
    "UCL":        0.10,
    "MLS":        0.07,
    "LIGUE1":     None,
    "SERIEA":     0.09,
    "FACUP":      0.10,
    "UEL":        0.10,
    "UECL":       0.10,
    "CLUBWC":     0.10,
    "FIFAQUAL":   0.10,
}

LEAGUE_SERIES = {
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

ODDS_SPORT_KEYS = {
    "EPL":        "soccer_epl",
    "LALIGA":     "soccer_spain_la_liga",
    "BUNDESLIGA": "soccer_germany_bundesliga",
    "UCL":        "soccer_uefa_champs_league",
    "LIGUE1":     "soccer_france_ligue_one",
    "MLS":        "soccer_usa_mls",
    "SERIEA":     "soccer_italy_serie_a",
    "FACUP":      "soccer_fa_cup",
    "UEL":        "soccer_uefa_europa_league",
    "UECL":       "soccer_uefa_europa_conference_league",
    "CLUBWC":     "soccer_fifa_club_world_cup",
    "FIFAQUAL":   "soccer_fifa_world_cup_qualification_intercontinental",
}


@dataclass
class SoccerSignal:
    ticker:        str
    title:         str
    league:        str
    home_team:     str
    away_team:     str
    outcome:       str          # "home_win", "draw", or "away_win"
    market_mid:    float
    vegas_prob:    float
    consensus:     float
    model_prob:    float
    edge:          float
    side:          str           # "yes" or "no"
    volume:        int
    home_xg:       float
    away_xg:       float
    market_source: str = "kalshi"
    notes:         str = ""


_soccer_odds_cache: dict = {}
_soccer_odds_ts:    float = 0


def fetch_soccer_vegas_odds(force_refresh: bool = False) -> dict:
    global _soccer_odds_cache, _soccer_odds_ts

    api_key = os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        return {}

    age_min = (time.time() - _soccer_odds_ts) / 60
    if not force_refresh and age_min < 30 and _soccer_odds_cache:
        return _soccer_odds_cache

    all_odds = {}
    for league_key, sport_key in ODDS_SPORT_KEYS.items():
        try:
            resp = requests.get(
                "https://api.the-odds-api.com/v4/sports/{}/odds/".format(sport_key),
                params={
                    "apiKey":     api_key,
                    "regions":    "eu",
                    "markets":    "h2h",
                    "oddsFormat": "decimal",
                },
                timeout=15,
            )
            if resp.status_code != 200:
                continue

            games = resp.json()
            for game in games:
                home_team = game.get("home_team", "")
                away_team = game.get("away_team", "")

                home_probs = []
                draw_probs = []
                away_probs = []

                for bookmaker in game.get("bookmakers", []):
                    for market in bookmaker.get("markets", []):
                        if market.get("key") != "h2h":
                            continue
                        for outcome in market.get("outcomes", []):
                            p    = 1 / outcome.get("price", 2.0)
                            name = outcome.get("name", "")
                            if name == home_team:
                                home_probs.append(p)
                            elif name == away_team:
                                away_probs.append(p)
                            elif name.lower() == "draw":
                                draw_probs.append(p)

                if home_probs and away_probs and draw_probs:
                    raw_h = sum(home_probs) / len(home_probs)
                    raw_d = sum(draw_probs) / len(draw_probs)
                    raw_a = sum(away_probs) / len(away_probs)
                    total = raw_h + raw_d + raw_a

                    key = f"{away_team}_{home_team}"
                    all_odds.setdefault(league_key, {})[key] = {
                        "home_prob": round(raw_h / total, 4),
                        "draw_prob": round(raw_d / total, 4),
                        "away_prob": round(raw_a / total, 4),
                        "home_team": home_team,
                        "away_team": away_team,
                        "commence_time": game.get("commence_time", ""),
                    }

            time.sleep(0.2)

        except Exception:
            continue

    _soccer_odds_cache = all_odds
    _soccer_odds_ts    = time.time()

    total_games = sum(len(v) for v in all_odds.values())
    if total_games:
        print(f"[soccer_scanner] Vegas odds loaded for {total_games} soccer games.")

    return all_odds


def get_soccer_vegas(league: str, home_team: str, away_team: str,
                     odds: dict) -> Optional[dict]:
    league_odds = odds.get(league, {})
    if not league_odds:
        return None

    key = f"{away_team}_{home_team}"
    if key in league_odds:
        return league_odds[key]

    away_last = away_team.split()[-1].lower()
    home_last = home_team.split()[-1].lower()
    for k, v in league_odds.items():
        k_away = v["away_team"].split()[-1].lower()
        k_home = v["home_team"].split()[-1].lower()
        if k_away == away_last and k_home == home_last:
            return v

    return None


def today_ct() -> datetime.date:
    import zoneinfo
    return datetime.datetime.now(zoneinfo.ZoneInfo("America/Chicago")).date()


def is_today_or_future(event: dict) -> bool:
    today = today_ct()
    sub   = event.get("sub_title", "")
    match = re.search(r"\((\w+ \d+)\)", sub)
    if match:
        try:
            parsed = datetime.datetime.strptime(
                match.group(1) + f" {today.year}", "%b %d %Y").date()
            return parsed >= today
        except ValueError:
            pass
    return True


def fetch_events_for_series(series_ticker: str, max_pages: int = 5) -> list[dict]:
    all_events = []
    cursor     = None
    for page in range(max_pages):
        try:
            params = {"series_ticker": series_ticker, "limit": 100, "status": "open"}
            if cursor:
                params["cursor"] = cursor
            resp   = kc.get("/events", params)
            events = resp.get("events", [])
            all_events.extend(events)
            cursor = resp.get("cursor")
            if not cursor or not events:
                break
            time.sleep(0.3)
        except Exception as e:
            print(f"[soccer_scanner] Error: {e}")
            break
    return all_events


def fetch_event_markets(event_ticker: str) -> list[dict]:
    try:
        return kc.get(f"/events/{event_ticker}").get("markets", [])
    except Exception:
        return []


def market_mid(market: dict) -> float:
    yes_bid = market.get("yes_bid_dollars")
    yes_ask = market.get("yes_ask_dollars")
    if yes_bid is not None and yes_ask is not None:
        return (float(yes_bid) + float(yes_ask)) / 2
    last = market.get("last_price_dollars")
    return float(last) if last else 0.5


def extract_volume(market: dict) -> int:
    v = market.get("volume_fp") or market.get("volume_24h_fp") or "0"
    return int(float(v))


def parse_teams_from_event(event: dict) -> tuple[str, str]:
    title = event.get("title", "")
    match = re.match(r"(.+?)\s+vs\.?\s+(.+?)(?:\s*[-–(]|$)", title, re.IGNORECASE)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "", ""


def is_model_reliable(home_team: str, away_team: str,
                      strengths: dict, min_games: int = MIN_GAMES) -> bool:
    home_s = strengths.get(home_team)
    away_s = strengths.get(away_team)
    if not home_s or not away_s:
        return False
    return home_s.get("games", 0) >= min_games and away_s.get("games", 0) >= min_games


def generate_soccer_signals(soccer_model,
                             edge_threshold: float = EDGE_THRESHOLD) -> list[SoccerSignal]:
    signals    = []
    vegas_odds = fetch_soccer_vegas_odds()

    for league_key, series_ticker in LEAGUE_SERIES.items():
        league_threshold = LEAGUE_EDGE_THRESHOLD.get(league_key, edge_threshold)
        if league_threshold is None:
            print(f"[soccer_scanner] {league_key}: DISABLED (model miscalibrated) — skipping.")
            continue

        effective_threshold = max(edge_threshold, league_threshold)

        if league_key not in soccer_model.leagues:
            continue

        league_data = soccer_model.leagues[league_key]
        strengths   = league_data["strengths"]
        min_vol     = MIN_VOLUME.get(league_key, 2_000)

        events       = fetch_events_for_series(series_ticker)
        today_events = [e for e in events if is_today_or_future(e)]

        if not today_events:
            continue

        print(f"[soccer_scanner] {league_key}: {len(today_events)} upcoming games "
              f"(edge threshold: {effective_threshold*100:.0f}%).")

        for event in today_events:
            event_ticker         = event.get("event_ticker", "")
            home_team, away_team = parse_teams_from_event(event)
            if not home_team or not away_team:
                continue

            pred = soccer_model.predict(home_team, away_team, league=league_key)
            if not pred:
                continue

            if pred["home_xg"] > MAX_XG or pred["away_xg"] > MAX_XG:
                print(f"  [skip] {home_team} vs {away_team} — xG out of range "
                      f"({pred['home_xg']:.1f}-{pred['away_xg']:.1f})")
                continue

            home_matched = soccer_model._fuzzy_match(home_team, strengths) or home_team
            away_matched = soccer_model._fuzzy_match(away_team, strengths) or away_team
            if not is_model_reliable(home_matched, away_matched, strengths):
                continue

            game_vegas = get_soccer_vegas(league_key, home_matched, away_matched, vegas_odds)

            # Skip live/in-progress games
            from live_game_filter import is_live_game
            is_live, minutes_in = is_live_game(game_vegas)
            if is_live:
                print(f"  [skip-live] {home_matched} vs {away_matched} "
                      f"— match started {minutes_in}m ago")
                continue

            markets = fetch_event_markets(event_ticker)
            time.sleep(0.15)

            # Parse event ticker to extract home/away abbreviations.
            # Format: KXLEAGUEGAME-YYMONDD[HHMM]AAABBB where AAA=home, BBB=away.
            # Example: KXLALIGAGAME-26APR23OVIVIL → OVI (home), VIL (away)
            #          KXEPLGAME-26APR25WOLTOT → WOL (home), TOT (away)
            event_home_abbr = None
            event_away_abbr = None
            event_suffix    = event_ticker.split("-")[-1].upper()
            # Strip leading digits (date/time) to get pure team codes
            import re as _re
            team_codes = _re.sub(r'^\d+', '', event_suffix)
            if len(team_codes) >= 6:
                # Take last 6 chars = AAA + BBB (3-letter codes each)
                event_home_abbr = team_codes[-6:-3]
                event_away_abbr = team_codes[-3:]

            game_candidates = []

            for m in markets:
                if m.get("status") != "active":
                    continue
                volume = extract_volume(m)
                if volume < min_vol:
                    continue

                ticker  = m.get("ticker", "")
                suffix  = ticker.split("-")[-1].upper()
                yes_sub = (m.get("yes_sub_title") or "").lower()
                mid     = market_mid(m)

                if suffix == "TIE" or "tie" in yes_sub or "draw" in yes_sub:
                    if SKIP_DRAWS:
                        continue
                    outcome = "draw"
                    model_p = pred["draw"]
                    vegas_p = game_vegas["draw_prob"] if game_vegas else 0.0
                else:
                    # Match market suffix against ticker-encoded home abbreviation.
                    # This is reliable because Kalshi encodes both teams in the event ticker.
                    # Fallback to name prefix only if ticker parsing failed.
                    if event_home_abbr and suffix == event_home_abbr:
                        is_home = True
                    elif event_away_abbr and suffix == event_away_abbr:
                        is_home = False
                    else:
                        # Fallback: compare first word of team name (guard against
                        # common substrings like "Real" being in "Villarreal")
                        home_first = home_matched.lower().split()[0]
                        away_first = away_matched.lower().split()[0]
                        home_abbr  = home_matched.replace(" ", "")[:3].upper()
                        away_abbr  = away_matched.replace(" ", "")[:3].upper()
                        if suffix == home_abbr:
                            is_home = True
                        elif suffix == away_abbr:
                            is_home = False
                        else:
                            # Last-resort yes_sub scan (weak signal, may be wrong)
                            if yes_sub and home_first in yes_sub.split():
                                is_home = True
                            elif yes_sub and away_first in yes_sub.split():
                                is_home = False
                            else:
                                # Can't determine — skip this market to avoid bad bet
                                print(f"  [skip-ambig] {ticker} — can't determine home/away "
                                      f"(suffix={suffix}, expected home={event_home_abbr} "
                                      f"or away={event_away_abbr})")
                                continue

                    if is_home:
                        outcome = "home_win"
                        model_p = pred["home_win"]
                        vegas_p = game_vegas["home_prob"] if game_vegas else 0.0
                    else:
                        outcome = "away_win"
                        model_p = pred["away_win"]
                        vegas_p = game_vegas["away_prob"] if game_vegas else 0.0

                if game_vegas and vegas_p > 0:
                    consensus  = vegas_p
                    mkt_source = "vegas"
                else:
                    consensus  = mid
                    mkt_source = "kalshi"

                edge = model_p - consensus
                if abs(edge) < effective_threshold:
                    continue

                side     = "yes" if edge > 0 else "no"
                bet_prob = model_p if side == "yes" else 1 - model_p

                if bet_prob < MIN_WIN_PROB:
                    continue

                game_candidates.append((abs(edge), SoccerSignal(
                    ticker=ticker,
                    title=f"{home_matched} vs {away_matched}",
                    league=league_key,
                    home_team=home_matched,
                    away_team=away_matched,
                    outcome=outcome,
                    market_mid=round(mid, 4),
                    vegas_prob=round(vegas_p, 4),
                    consensus=round(consensus, 4),
                    model_prob=round(model_p, 4),
                    edge=round(edge, 4),
                    side=side,
                    volume=volume,
                    home_xg=pred["home_xg"],
                    away_xg=pred["away_xg"],
                    market_source=mkt_source,
                    notes=f"{home_matched} vs {away_matched} ({outcome})  ← verify ticker matches this game",
                )))

            if game_candidates:
                game_candidates.sort(key=lambda x: x[0], reverse=True)
                signals.append(game_candidates[0][1])

    signals.sort(key=lambda s: abs(s.edge), reverse=True)
    return signals


def _describe_bet(signal: SoccerSignal) -> str:
    """Human-readable description of what team the signal is betting on/against."""
    # Which team does the market pay out on?
    if signal.outcome == "home_win":
        team = signal.home_team
    elif signal.outcome == "away_win":
        team = signal.away_team
    else:
        team = "Draw"

    action = "ON" if signal.side == "yes" else "AGAINST"
    return f"Bet {action} {team}"


def print_soccer_signals(signals: list[SoccerSignal]):
    if not signals:
        print("[soccer_scanner] No signals above threshold.")
        return
    print(f"\n{'='*100}")
    print(f"  {'TICKER':<35} {'LEAGUE':<12} {'SRC':<8} {'SIDE':<5} {'CONSENSUS':>10} {'MODEL':>7} {'EDGE':>7} {'VOL':>8}")
    print(f"{'='*100}")
    for s in signals:
        src = f"[{s.market_source}]"
        print(
            f"  {s.ticker:<35} {s.league:<12} {src:<8} "
            f"{s.side.upper():<5} "
            f"{s.consensus*100:>9.1f}% "
            f"{s.model_prob*100:>6.1f}% "
            f"{s.edge*100:>+6.1f}% "
            f"${s.volume:>7,}"
        )
        print(f"    └─ 🎯 {_describe_bet(s)}  |  "
              f"{s.home_team} vs {s.away_team} ({s.outcome})  |  "
              f"xG: {s.home_xg}-{s.away_xg}")
    print(f"{'='*100}")
    print(f"  Total soccer signals: {len(signals)}")