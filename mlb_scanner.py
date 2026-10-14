"""
mlb_scanner.py  v2
Scans Kalshi MLB markets. Now uses Vegas consensus (same as NBA).
Flags TBD starters and shows bullpen/offense metrics.
"""

import re, time, datetime
from dataclasses import dataclass
from typing import Optional

import kalshi_client as kc
from mlb_model import MLBModel, normalize_team, REPLACEMENT_FIP

MLB_SERIES     = "KXMLBGAME"
EDGE_THRESHOLD = 0.08
MIN_WIN_PROB   = 0.40
MIN_VOLUME     = 0

ODDS_API_KEY = None  # Set via env var ODDS_API_KEY if available

@dataclass
class MLBSignal:
    ticker:         str
    title:          str
    home:           str
    away:           str
    home_sp:        str
    away_sp:        str
    home_sp_known:  bool
    away_sp_known:  bool
    market_mid:     float
    vegas_prob:     float
    model_prob:     float
    consensus:      float
    edge:           float
    side:           str
    volume:         int
    home_fip:       float
    away_fip:       float
    home_raw_fip:   float
    away_raw_fip:   float
    home_ip:        float
    away_ip:        float
    home_bull_era:  float
    away_bull_era:  float
    home_rpg:       float
    away_rpg:       float
    park_factor:    float
    market_source:  str
    # Optional prior-season info (added last so defaults work)
    home_prior_fip: float = None
    away_prior_fip: float = None
    home_prior_ip:  float = 0.0
    away_prior_ip:  float = 0.0


def fetch_mlb_vegas_odds() -> dict:
    """
    Fetch MLB moneyline odds from The Odds API.
    Returns {away_home_key: {home_prob, away_prob}}
    """
    import os
    api_key = os.environ.get("ODDS_API_KEY", ODDS_API_KEY)
    if not api_key:
        return {}
    try:
        import requests
        r = requests.get(
            "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds",
            params={
                "apiKey":   api_key,
                "regions":  "us",
                "markets":  "h2h",
                "oddsFormat": "american",
                "bookmakers": "draftkings,fanduel,betmgm",
            },
            timeout=8,
        )
        if r.status_code != 200:
            return {}

        odds = {}
        for game in r.json():
            home_team = normalize_team(game.get("home_team",""))
            away_team = normalize_team(game.get("away_team",""))
            if not home_team or not away_team:
                continue

            # Average implied prob across bookmakers
            home_probs, away_probs = [], []
            for bk in game.get("bookmakers",[]):
                for mkt in bk.get("markets",[]):
                    if mkt.get("key") != "h2h":
                        continue
                    for outcome in mkt.get("outcomes",[]):
                        price = outcome.get("price",0)
                        team  = normalize_team(outcome.get("name",""))
                        if price > 0:
                            imp = 100/(100+price)
                        else:
                            imp = abs(price)/(abs(price)+100)
                        if team == home_team:
                            home_probs.append(imp)
                        elif team == away_team:
                            away_probs.append(imp)

            if home_probs and away_probs:
                h = sum(home_probs)/len(home_probs)
                a = sum(away_probs)/len(away_probs)
                total = h + a
                key = f"{away_team}_{home_team}"
                odds[key] = {
                    "home_prob": round(h/total,4),
                    "away_prob": round(a/total,4),
                    "commence_time": game.get("commence_time", ""),
                }
        return odds
    except Exception as e:
        print(f"[mlb_odds] Fetch error: {e}")
        return {}


def parse_teams_from_ticker(ticker):
    suffix = re.sub(r'KXMLBGAME-\d{2}[A-Z]{3}\d{4}', '', ticker)
    if not suffix: return "",""
    MLB_ABBRS = [
        "ARI","ATL","BAL","BOS","CHC","CWS","CIN","CLE","COL","DET",
        "HOU","KC","LAA","LAD","MIA","MIL","MIN","NYM","NYY","OAK",
        "PHI","PIT","SD","SF","SEA","STL","TB","TEX","TOR","WSH",
        "AZ","NY","LA",
    ]
    ALIAS = {"AZ":"ARI","NY":"NYY","LA":"LAD","ATH":"OAK"}
    found, remaining = [], suffix.upper()
    while remaining and len(found) < 2:
        matched = False
        for length in [3,2]:
            candidate = remaining[:length]
            if candidate in MLB_ABBRS:
                found.append(candidate)
                remaining = remaining[length:]
                matched = True
                break
        if not matched: remaining = remaining[1:]
    if len(found) >= 2:
        return ALIAS.get(found[0],found[0]), ALIAS.get(found[1],found[1])
    return "",""


def scan_mlb(model: MLBModel, vegas_odds: dict = None) -> list[MLBSignal]:
    import zoneinfo
    signals = []
    ct_now  = datetime.datetime.now(zoneinfo.ZoneInfo("America/Chicago"))
    ct_date = ct_now.date()

    # Fetch Vegas if not provided
    if vegas_odds is None:
        vegas_odds = fetch_mlb_vegas_odds()
        if vegas_odds:
            print(f"[mlb_scanner] Fetched Vegas odds for {len(vegas_odds)} MLB games.")

    resp   = kc.get("/events",{"series_ticker":MLB_SERIES,"limit":50,"status":"open"})
    events = resp.get("events",[])

    upcoming = []
    for e in events:
        ticker = e.get("event_ticker","")
        match  = re.search(r'-(\d{2})([A-Z]{3})(\d{2})',ticker)
        if match:
            try:
                gdate = datetime.datetime.strptime(
                    f"20{match.group(1)}-{match.group(2)}-{match.group(3)}","%Y-%b-%d"
                ).date()
                if gdate == ct_date: upcoming.append(e)
            except: upcoming.append(e)
        else:
            upcoming.append(e)

    print(f"[mlb_scanner] Found {len(events)} open MLB events, {len(upcoming)} today.")

    schedule_cache = {}
    for g in model.get_todays_games():
        key = f"{g['away']}_{g['home']}"
        schedule_cache[key] = g

    seen_games = set()

    for event in upcoming:
        event_ticker         = event.get("event_ticker","")
        away_abbr, home_abbr = parse_teams_from_ticker(event_ticker)

        home_sp, away_sp = "",""
        g = schedule_cache.get(f"{away_abbr}_{home_abbr}",{})
        if g:
            home_sp = g.get("home_sp","")
            away_sp = g.get("away_sp","")

        result = model.win_probability(home_abbr, away_abbr, home_sp, away_sp)
        p_home = result["p_home"]

        # Determine consensus: Vegas > model
        game_key_v = f"{away_abbr}_{home_abbr}"
        vegas_game = vegas_odds.get(game_key_v,{})
        has_vegas  = bool(vegas_game)

        # Skip live/in-progress games
        from live_game_filter import is_live_game
        is_live, minutes_in = is_live_game(vegas_game)
        if is_live:
            print(f"  [skip-live] {away_abbr} @ {home_abbr} — game started {minutes_in}m ago")
            continue

        markets = kc.get(f"/events/{event_ticker}").get("markets",[])
        time.sleep(0.1)

        best_signal = None
        best_edge   = 0.0

        for m in markets:
            if m.get("status") != "active": continue
            ticker  = m.get("ticker","")
            title   = m.get("title","")
            yes_bid = m.get("yes_bid_dollars")
            yes_ask = m.get("yes_ask_dollars")
            volume  = int(m.get("volume",0) or 0)
            if yes_bid is None or yes_ask is None: continue
            bid, ask = float(yes_bid), float(yes_ask)
            if ask >= 0.97 or bid <= 0.02: continue
            if ask - bid > 0.25: continue
            mid = (bid+ask)/2

            suffix = ticker.split("-")[-1].upper() if "-" in ticker else ""
            if suffix == home_abbr:
                model_p   = p_home
                vegas_p   = vegas_game.get("home_prob",0)
            elif suffix == away_abbr:
                model_p   = 1-p_home
                vegas_p   = vegas_game.get("away_prob",0)
            else:
                if home_abbr.lower() in title.lower()[:20]:
                    model_p = p_home
                    vegas_p = vegas_game.get("home_prob",0)
                else:
                    model_p = 1-p_home
                    vegas_p = vegas_game.get("away_prob",0)

            # Use Vegas as consensus if available
            if has_vegas and vegas_p > 0:
                consensus = vegas_p
                src       = "vegas"
            else:
                consensus = model_p
                src       = "model"

            edge     = model_p - consensus
            bet_prob = model_p if edge > 0 else 1-model_p

            if abs(edge) < EDGE_THRESHOLD: continue
            if bet_prob < MIN_WIN_PROB: continue
            # Skip TBD-both: signal is unreliable when neither starter is known
            if not result["home_sp_known"] and not result["away_sp_known"]: continue

            if abs(edge) > best_edge:
                best_edge   = abs(edge)
                best_signal = MLBSignal(
                    ticker=ticker, title=title,
                    home=home_abbr, away=away_abbr,
                    home_sp=home_sp, away_sp=away_sp,
                    home_sp_known=result["home_sp_known"],
                    away_sp_known=result["away_sp_known"],
                    market_mid=round(mid,4),
                    vegas_prob=round(vegas_p,4) if has_vegas else 0.0,
                    model_prob=round(model_p,4),
                    consensus=round(consensus,4),
                    edge=round(edge,4),
                    side="yes" if edge > 0 else "no",
                    volume=volume,
                    home_fip=result["home_fip"],
                    away_fip=result["away_fip"],
                    home_raw_fip=result.get("home_raw_fip", result["home_fip"]),
                    away_raw_fip=result.get("away_raw_fip", result["away_fip"]),
                    home_ip=result.get("home_ip", 0.0),
                    away_ip=result.get("away_ip", 0.0),
                    home_prior_fip=result.get("home_prior_fip"),
                    away_prior_fip=result.get("away_prior_fip"),
                    home_prior_ip=result.get("home_prior_ip", 0.0),
                    away_prior_ip=result.get("away_prior_ip", 0.0),
                    home_bull_era=result["home_bull_era"],
                    away_bull_era=result["away_bull_era"],
                    home_rpg=result["home_rpg"],
                    away_rpg=result["away_rpg"],
                    park_factor=result["park_factor"],
                    market_source=src,
                )

        gk = f"{away_abbr}{home_abbr}"
        if best_signal and gk not in seen_games:
            seen_games.add(gk)
            signals.append(best_signal)

    signals.sort(key=lambda s: abs(s.edge), reverse=True)
    return signals


def print_mlb_signals(signals):
    if not signals:
        print("[mlb_scanner] No signals above threshold.")
        return
    print(f"\n{'='*105}")
    print(f"  {'TICKER':<42} {'SRC':<8} {'CONSENSUS':>10} {'MODEL':>7} {'EDGE':>7} {'SIDE':<5} {'VOL':>8}")
    print(f"{'='*105}")
    for s in signals:
        tbd_flag = " ⚠️TBD" if not s.home_sp_known or not s.away_sp_known else ""
        print(f"  {s.ticker:<42} [{s.market_source}]  "
              f"{s.consensus*100:>9.1f}% "
              f"{s.model_prob*100:>6.1f}% "
              f"{s.edge*100:>+6.1f}% "
              f"{s.side:<5} "
              f"${s.volume:>7,}{tbd_flag}")
        def fip_str(fip, raw_fip, ip, prior_fip, prior_ip, known):
            if not known:
                return f"FIP {fip:.2f}?"
            parts = [f"FIP {fip:.2f}"]
            has_detail = False
            # Show current season info if small sample
            if ip > 0 and ip < 60:
                parts.append(f"[{ip:.0f}ip {raw_fip:.2f}")
                has_detail = True
                # Append prior if available
                if prior_fip is not None and prior_ip > 0:
                    parts[-1] += f", '25: {prior_fip:.2f}/{prior_ip:.0f}ip"
                parts[-1] += "]"
            elif prior_fip is not None and ip == 0:
                # No 2026 data yet, only prior
                parts.append(f"['25 only: {prior_fip:.2f}/{prior_ip:.0f}ip]")
            return " ".join(parts)

        print(f"    └─ {s.away} @ {s.home}  "
              f"SP: {s.away_sp or 'TBD'} ({fip_str(s.away_fip, s.away_raw_fip, s.away_ip, s.away_prior_fip, s.away_prior_ip, s.away_sp_known)}) "
              f"vs {s.home_sp or 'TBD'} ({fip_str(s.home_fip, s.home_raw_fip, s.home_ip, s.home_prior_fip, s.home_prior_ip, s.home_sp_known)})"
              f"  | Bull: {s.away_bull_era:.2f}/{s.home_bull_era:.2f}  "
              f"R/G: {s.away_rpg:.1f}/{s.home_rpg:.1f}  Park: {s.park_factor:.2f}")
    print(f"{'='*105}")
    print(f"  Total MLB signals: {len(signals)}")