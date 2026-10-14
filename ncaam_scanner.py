"""
ncaam_scanner.py
Scans Kalshi NCAAM tournament markets for edges.
Uses Vegas odds as primary consensus when available.
Only fires when model probability > 40% and edge >= 8%.
One signal per game.
"""

import re
import time
import datetime
from dataclasses import dataclass
from typing import Optional
import kalshi_client as kc

NCAAM_SERIES    = "KXNCAAMBGAME"
EDGE_THRESHOLD  = 0.08
MIN_WIN_PROB    = 0.40
MIN_VOLUME      = 2_000


@dataclass
class NCAAMSignal:
    ticker:        str
    title:         str
    sport:         str
    team_a:        str
    team_b:        str
    market_mid:    float
    vegas_prob:    float
    consensus_prob: float
    model_prob:    float
    edge:          float
    side:          str
    volume:        int
    spread:        float = 0.0
    market_source: str = "kalshi"
    notes:         str = ""


def fetch_ncaam_odds(api_key: str) -> dict:
    """Fetch NCAAM tournament odds from the-odds-api.com."""
    import os, requests
    if not api_key:
        return {}
    try:
        resp = requests.get(
            "https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds/",
            params={
                "apiKey":     api_key,
                "regions":    "us",
                "markets":    "h2h",
                "oddsFormat": "decimal",
            },
            timeout=15,
        )
        if resp.status_code != 200:
            return {}

        odds_out = {}
        for game in resp.json():
            home_team  = game.get("home_team", "")
            away_team  = game.get("away_team", "")
            home_probs = []
            away_probs = []

            for bm in game.get("bookmakers", []):
                for market in bm.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    for outcome in market.get("outcomes", []):
                        p = 1 / outcome.get("price", 2.0)
                        if outcome["name"] == home_team:
                            home_probs.append(p)
                        elif outcome["name"] == away_team:
                            away_probs.append(p)

            if home_probs and away_probs:
                raw_h  = sum(home_probs) / len(home_probs)
                raw_a  = sum(away_probs) / len(away_probs)
                total  = raw_h + raw_a
                key    = f"{away_team}_{home_team}"
                key2   = f"{home_team}_{away_team}"
                prob_h = round(raw_h / total, 4)
                prob_a = round(raw_a / total, 4)
                odds_out[key]  = {"team_a": away_team, "team_b": home_team,
                                   "prob_a": prob_a, "prob_b": prob_h}
                odds_out[key2] = {"team_a": home_team, "team_b": away_team,
                                   "prob_a": prob_h, "prob_b": prob_a}

        print(f"[ncaam_scanner] Vegas odds for {len(odds_out)//2} NCAAM games.")
        return odds_out
    except Exception as e:
        print(f"[ncaam_scanner] Odds API error: {e}")
        return {}


def get_vegas_prob(team_a: str, team_b: str, odds: dict) -> Optional[float]:
    """Look up Vegas win probability for team_a."""
    key = f"{team_a}_{team_b}"
    if key in odds:
        return odds[key]["prob_a"]

    # Fuzzy match on last word
    a_last = team_a.split()[-1].lower()
    b_last = team_b.split()[-1].lower()
    for k, v in odds.items():
        ka = v["team_a"].split()[-1].lower()
        kb = v["team_b"].split()[-1].lower()
        if ka == a_last and kb == b_last:
            return v["prob_a"]
    return None


def fetch_events(max_pages: int = 5) -> list[dict]:
    all_events = []
    cursor = None
    for _ in range(max_pages):
        try:
            params = {"series_ticker": NCAAM_SERIES, "limit": 100, "status": "open"}
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
            print(f"[ncaam_scanner] Error: {e}")
            break
    return all_events


def fetch_event_markets(event_ticker: str) -> list[dict]:
    try:
        return kc.get(f"/events/{event_ticker}").get("markets", [])
    except Exception:
        return []


def market_mid(market: dict) -> tuple[float, float]:
    yes_bid = market.get("yes_bid_dollars")
    yes_ask = market.get("yes_ask_dollars")
    if yes_bid is not None and yes_ask is not None:
        bid, ask = float(yes_bid), float(yes_ask)
        return (bid + ask) / 2, ask - bid
    last = market.get("last_price_dollars")
    return float(last) if last else 0.5, 0.05


def extract_volume(market: dict) -> int:
    v = market.get("volume_fp") or market.get("volume_24h_fp") or "0"
    return int(float(v))


def parse_teams(event: dict, markets: list = None) -> tuple[str, str]:
    """
    Parse team names from event.
    First tries yes_sub_title from markets (most accurate full names).
    Falls back to event title parsing.
    """
    # Best source: yes_sub_title fields from the two markets
    if markets and len(markets) == 2:
        name_a = (markets[0].get("yes_sub_title") or "").strip()
        name_b = (markets[1].get("yes_sub_title") or "").strip()
        if name_a and name_b:
            # Figure out which is home/away from title
            title = event.get("title", "")
            match = re.match(r"(.+?)\s+at\s+(.+?)$", title, re.IGNORECASE)
            if match:
                away_part = match.group(1).lower()
                home_part = match.group(2).lower()
                # Match sub_titles to home/away
                if away_part.split()[-1] in name_a.lower():
                    return name_a, name_b
                else:
                    return name_b, name_a
            return name_a, name_b

    # Fallback: parse from title
    title = event.get("title", "")
    match = re.match(r"(.+?)\s+at\s+(.+?)$", title, re.IGNORECASE)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    sub = event.get("sub_title", "")
    match2 = re.match(r"(.+?)\s+at\s+(.+?)(?:\s*\(|$)", sub, re.IGNORECASE)
    if match2:
        return match2.group(1).strip(), match2.group(2).strip()
    return "", ""


def generate_ncaam_signals(ncaam_model, edge_threshold: float = EDGE_THRESHOLD) -> list[NCAAMSignal]:
    import os
    api_key  = os.environ.get("ODDS_API_KEY", "")
    vegas    = fetch_ncaam_odds(api_key)
    events   = fetch_events()
    signals  = []

    print(f"[ncaam_scanner] Found {len(events)} open NCAAM events.")

    for event in events:
        event_ticker = event.get("event_ticker", "")
        markets      = fetch_event_markets(event_ticker)
        time.sleep(0.15)

        away_name, home_name = parse_teams(event, markets)
        if not away_name or not home_name:
            continue

        # Get model prediction (neutral site, no seeds for now)
        pred = ncaam_model.win_probability(away_name, home_name)
        # Log matched names for debugging
        if pred["team_a"] != away_name or pred["team_b"] != home_name:
            pass  # fuzzy matched — pred contains correct names
        if not pred["reliable"]:
            continue

        game_candidates = []
        for m in markets:
            if m.get("status") != "active":
                continue
            volume = extract_volume(m)
            if volume < MIN_VOLUME:
                continue

            ticker   = m.get("ticker", "")
            suffix   = ticker.split("-")[-1].upper()
            yes_sub  = (m.get("yes_sub_title") or "").strip()
            mid, _   = market_mid(m)

            # Determine which team this market is for
            away_abbr = away_name.split()[-1][:4].upper()
            home_abbr = home_name.split()[-1][:4].upper()

            if suffix == away_abbr or (yes_sub and away_name.split()[-1].lower() in yes_sub.lower()):
                team    = away_name
                model_p = pred["p_a"]
            else:
                team    = home_name
                model_p = pred["p_b"]

            # Get Vegas consensus — prefer Vegas over Kalshi mid
            other_team = home_name if team == away_name else away_name
            vegas_p    = get_vegas_prob(team, other_team, vegas)
            # Also try matched team names
            if vegas_p is None:
                matched_team  = pred["team_a"] if team == away_name else pred["team_b"]
                matched_other = pred["team_b"] if team == away_name else pred["team_a"]
                vegas_p = get_vegas_prob(matched_team, matched_other, vegas)
            if vegas_p is not None:
                consensus  = vegas_p
                mkt_source = "vegas"
            else:
                consensus  = mid
                mkt_source = "kalshi"

            edge = model_p - consensus
            if abs(edge) < edge_threshold:
                continue

            side     = "yes" if edge > 0 else "no"
            bet_prob = model_p if side == "yes" else 1 - model_p
            if bet_prob < MIN_WIN_PROB:
                continue

            game_candidates.append((abs(edge), NCAAMSignal(
                ticker=ticker,
                title=m.get("title", f"{away_name} vs {home_name}"),
                sport="NCAAM",
                team_a=away_name,
                team_b=home_name,
                market_mid=round(mid, 4),
                vegas_prob=round(vegas_p, 4) if vegas_p else 0.0,
                consensus_prob=round(consensus, 4),
                model_prob=round(model_p, 4),
                edge=round(edge, 4),
                side=side,
                volume=volume,
                market_source=mkt_source,
                notes=f"{away_name} vs {home_name}",
            )))

        if game_candidates:
            game_candidates.sort(key=lambda x: x[0], reverse=True)
            signals.append(game_candidates[0][1])

    signals.sort(key=lambda s: abs(s.edge), reverse=True)
    return signals


def print_ncaam_signals(signals: list[NCAAMSignal]):
    if not signals:
        print("[ncaam_scanner] No signals above threshold.")
        return
    print(f"\n{'='*95}")
    print(f"  {'TICKER':<35} {'SRC':<8} {'CONSENSUS':>10} {'MODEL':>7} {'EDGE':>7} {'SIDE':<5} {'VOL':>8}")
    print(f"{'='*95}")
    for s in signals:
        src = f"[{s.market_source}]"
        print(
            f"  {s.ticker:<35} {src:<8} "
            f"{s.consensus_prob*100:>9.1f}% "
            f"{s.model_prob*100:>6.1f}% "
            f"{s.edge*100:>+6.1f}% "
            f"{s.side:<5} "
            f"${s.volume:>7,}"
        )
        print(f"    └─ {s.notes}")
    print(f"{'='*95}")
    print(f"  Total NCAAM signals: {len(signals)}")