"""
kalshi_scanner.py
Scans Kalshi NBA and golf events.
Uses Vegas moneyline odds as the primary market consensus.
Applies playoff motivation adjustment.
Flags suspicious line movement.

Only generates ONE signal per game.
Only considers games happening today (Central Time).
Only fires when model win probability > 40%.
Skips live/in-progress games (filtered via commence_time from Vegas odds).
"""

import re
import time
import datetime
from dataclasses import dataclass
from typing import Optional
import kalshi_client as kc

NBA_SERIES     = "KXNBAGAME"
GOLF_SERIES    = ["KXMASTERS", "KXPGA", "KXUSOPEN", "KXTHEOPEN", "KXPLAYERS"]
MIN_VOLUME     = 5_000
EDGE_THRESHOLD = 0.08
MIN_WIN_PROB   = 0.75   # audit: only 75%+ confidence gives 77.5% win rate
DEAD_ZONE_LOW  = 0.65
DEAD_ZONE_HIGH = 0.75


@dataclass
class MarketSignal:
    ticker:          str
    title:           str
    sport:           str
    team_or_player:  str
    market_mid:      float
    vegas_prob:      float
    consensus_prob:  float
    model_prob:      float
    edge:            float
    side:            str
    volume:          int
    spread:          float
    notes:           str = ""
    market_source:   str = "kalshi"
    line_movement:   float = 0.0
    suspicious_line: bool = False


def today_ct() -> datetime.date:
    import zoneinfo
    return datetime.datetime.now(zoneinfo.ZoneInfo("America/Chicago")).date()


def is_today_ct(event: dict) -> bool:
    today = today_ct()
    sub   = event.get("sub_title", "")
    match = re.search(r"\((\w+ \d+)\)", sub)
    if match:
        try:
            parsed = datetime.datetime.strptime(
                match.group(1) + f" {today.year}", "%b %d %Y").date()
            return parsed == today
        except ValueError:
            pass
    ticker = event.get("event_ticker", "")
    match2 = re.search(r"-(\d{2})(\w{3})(\d{2})", ticker)
    if match2:
        try:
            parsed = datetime.datetime.strptime(
                match2.group(2) + match2.group(3) + "20" + match2.group(1),
                "%b%d%Y").date()
            return parsed == today
        except ValueError:
            pass
    return True


def fetch_events_for_series(series_ticker: str, max_pages: int = 10) -> list[dict]:
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
            print(f"[scanner] Error fetching {series_ticker} page {page}: {e}")
            break
    return all_events


def fetch_event_markets(event_ticker: str) -> list[dict]:
    try:
        resp = kc.get(f"/events/{event_ticker}")
        return resp.get("markets", [])
    except Exception as e:
        print(f"[scanner] Error fetching event {event_ticker}: {e}")
        return []


def market_mid_from_dollars(market: dict) -> tuple[float, float]:
    yes_bid = market.get("yes_bid_dollars")
    yes_ask = market.get("yes_ask_dollars")
    if yes_bid is not None and yes_ask is not None:
        bid = float(yes_bid)
        ask = float(yes_ask)
        return (bid + ask) / 2, ask - bid
    last = market.get("last_price_dollars")
    if last:
        return float(last), 0.05
    return 0.50, 0.10


def extract_team(market: dict) -> str:
    return market.get("yes_sub_title", market.get("subtitle", ""))


def extract_volume(market: dict) -> int:
    v = market.get("volume_fp") or market.get("volume_24h_fp") or "0"
    return int(float(v))


def generate_nba_signals(nba_model, edge_threshold: float = EDGE_THRESHOLD) -> list[MarketSignal]:
    from nba_model import (fetch_injury_report, fetch_nba_standings,
                           playoff_motivation_adjustment,
                           record_line, is_suspicious_line_movement)
    from odds_client import fetch_today_odds, get_game_odds
    from live_game_filter import is_live_game

    print("[scanner] Fetching injury report...")
    injury_report = fetch_injury_report()
    print(f"[scanner] Loaded injuries for {len(injury_report)} teams.")

    print("[scanner] Fetching playoff standings...")
    standings = fetch_nba_standings()
    print(f"[scanner] Loaded standings for {len(standings)} teams.")

    print("[scanner] Fetching Vegas odds...")
    vegas_odds = fetch_today_odds()

    signals  = []
    events   = fetch_events_for_series(NBA_SERIES)
    today    = today_ct()
    print(f"[scanner] Found {len(events)} open NBA events. Filtering for {today} CT...")

    today_events = [e for e in events if is_today_ct(e)]
    print(f"[scanner] {len(today_events)} games today.")

    for event in today_events:
        event_ticker = event.get("event_ticker", "")
        sub_title    = event.get("sub_title", "")

        match = re.match(r"(.+?)\s+at\s+(.+?)\s*\(", sub_title)
        if not match:
            continue
        away_name = match.group(1).strip()
        home_name = match.group(2).strip()

        # ── Live-game filter ──────────────────────────────────────────────────
        # Skip games already in progress — Vegas has already priced in current
        # game state and our pre-game model can't compete with that info.
        game_odds_for_live_check = get_game_odds(away_name, home_name, vegas_odds)
        is_live, minutes_in = is_live_game(game_odds_for_live_check)
        if is_live:
            print(f"  [skip-live] {away_name} at {home_name} — game started {minutes_in}m ago")
            continue

        markets = fetch_event_markets(event_ticker)
        time.sleep(0.2)

        event_candidates = []
        for m in markets:
            if m.get("status") != "active":
                continue
            volume = extract_volume(m)
            if volume < MIN_VOLUME:
                continue

            team        = extract_team(m)
            mid, spread = market_mid_from_dollars(m)
            ticker      = m.get("ticker", "")
            suffix      = ticker.split("-")[-1].upper()
            home_ab     = home_name.split()[-1][:3].upper()

            # Record line for movement tracking
            record_line(ticker, mid)
            suspicious = is_suspicious_line_movement(ticker, mid)
            mov        = 0.0  # will be non-zero after multiple runs

            result = nba_model.win_probability(
                home_name, away_name, injury_report=injury_report)

            is_home_market = suffix in home_name.upper() or home_ab in suffix
            model_p        = result["p_home"] if is_home_market else result["p_away"]

            # Skip if ELO and ratings disagree — these are 38.9% accurate (worse than random)
            elo_says_home    = result["p_elo"] > 0.5
            rating_says_home = result["p_rating"] > 0.5
            if elo_says_home != rating_says_home:
                continue  # ELO and ratings disagree — skip this game entirely

            # Apply playoff motivation
            team_for_motivation = home_name if is_home_market else away_name
            from nba_model import normalize_team
            team_norm   = normalize_team(team_for_motivation)
            playoff_adj = playoff_motivation_adjustment(team_norm, standings)
            # Convert ELO adjustment to probability shift (~100 ELO = ~15% prob shift)
            model_p = max(0.05, min(0.95, model_p + playoff_adj / 700))

            # Use Vegas as consensus if available
            game_odds   = get_game_odds(away_name, home_name, vegas_odds)
            has_vegas   = game_odds is not None
            if has_vegas:
                consensus  = game_odds["home_prob"] if is_home_market else game_odds["away_prob"]
                mkt_source = "vegas"
            else:
                consensus  = mid
                mkt_source = "kalshi"

            edge = model_p - consensus

            # ── Favorite-longshot bias filter ─────────────────────────────────
            # Report: $0.01-$0.15 contracts systematically overpriced on Kalshi
            # Flag but don't auto-skip — let edge threshold handle it
            longshot_flag = (mid <= 0.15 and edge > 0) or (mid >= 0.85 and edge < 0)
            if longshot_flag:
                print(f"  [longshot] {ticker} mid={mid:.2f} — longshot bias may inflate edge")

            if edge > 0:
                side     = "yes"
                bet_prob = model_p
            else:
                side     = "no"
                bet_prob = 1 - model_p

            event_candidates.append((abs(edge), MarketSignal(
                ticker=ticker,
                title=m.get("title", ""),
                sport="NBA",
                team_or_player=team,
                market_mid=round(mid, 4),
                vegas_prob=round(game_odds["home_prob"] if has_vegas and is_home_market
                                 else game_odds["away_prob"] if has_vegas else 0.0, 4),
                consensus_prob=round(consensus, 4),
                model_prob=round(model_p, 4),
                edge=round(edge, 4),
                side=side,
                volume=volume,
                spread=round(spread, 4),
                notes=f"{away_name} at {home_name}",
                market_source=mkt_source,
                line_movement=mov,
                suspicious_line=suspicious,
            ), bet_prob))

        if event_candidates:
            event_candidates.sort(key=lambda x: x[0], reverse=True)
            abs_edge, best_signal, bet_prob = event_candidates[0]

            if abs_edge >= edge_threshold and bet_prob >= MIN_WIN_PROB:
                # Skip when ELO and ratings disagree — 38.9% accuracy (worse than random)
                # Skip dead zone (65-75% model confidence — miscalibrated)
                elo_pick    = result["p_elo"] > 0.5 if is_home_market else result["p_elo"] < 0.5
                rating_pick = result["p_rating"] > 0.5 if is_home_market else result["p_rating"] < 0.5
                components_agree = (elo_pick == rating_pick)

                if not components_agree:
                    print(f"  [disagree] {best_signal.ticker} ELO+ratings disagree — skipped")
                else:
                    if best_signal.suspicious_line:
                        print(f"  [caution] {best_signal.ticker} — suspicious line movement")
                    signals.append(best_signal)
            elif abs_edge >= edge_threshold:
                print(f"  [filter] {best_signal.ticker} edge={abs_edge*100:.1f}% "
                      f"but bet_prob={bet_prob*100:.1f}% < 55% — skipped")

    return signals


def generate_golf_signals(golf_model, edge_threshold: float = EDGE_THRESHOLD) -> list[MarketSignal]:
    signals = []
    for series in GOLF_SERIES:
        events = fetch_events_for_series(series)
        if not events:
            continue
        print(f"[scanner] Found {len(events)} open {series} events.")
        for event in events:
            event_ticker = event.get("event_ticker", "")
            markets      = fetch_event_markets(event_ticker)
            time.sleep(0.2)
            event_candidates = []
            for m in markets:
                if m.get("status") != "active":
                    continue
                volume = extract_volume(m)
                if volume < MIN_VOLUME:
                    continue
                player      = extract_team(m)
                mid, spread = market_mid_from_dollars(m)
                model_p     = golf_model.win_probability(player)
                if model_p is None:
                    continue
                edge     = model_p - mid
                bet_prob = model_p if edge > 0 else 1 - model_p
                event_candidates.append((abs(edge), MarketSignal(
                    ticker=m.get("ticker", ""),
                    title=m.get("title", ""),
                    sport="GOLF",
                    team_or_player=player,
                    market_mid=round(mid, 4),
                    vegas_prob=0.0,
                    consensus_prob=round(mid, 4),
                    model_prob=round(model_p, 4),
                    edge=round(edge, 4),
                    side="yes" if edge > 0 else "no",
                    volume=volume,
                    spread=round(spread, 4),
                    notes=event_ticker,
                ), bet_prob))
            if event_candidates:
                event_candidates.sort(key=lambda x: x[0], reverse=True)
                abs_edge, best_signal, bet_prob = event_candidates[0]
                if abs_edge >= edge_threshold and bet_prob >= MIN_WIN_PROB:
                    signals.append(best_signal)
    return signals


def generate_signals(markets, nba_model, golf_model,
                     edge_threshold: float = EDGE_THRESHOLD) -> list[MarketSignal]:
    signals = []
    if nba_model and nba_model._loaded:
        nba_signals = generate_nba_signals(nba_model, edge_threshold)
        signals.extend(nba_signals)
        print(f"[scanner] NBA signals: {len(nba_signals)}")
    if golf_model and golf_model._loaded:
        golf_signals = generate_golf_signals(golf_model, edge_threshold)
        signals.extend(golf_signals)
        print(f"[scanner] Golf signals: {len(golf_signals)}")
    signals.sort(key=lambda s: abs(s.edge), reverse=True)
    return signals


def print_signals(signals: list[MarketSignal]):
    if not signals:
        print("[scanner] No signals above threshold.")
        return
    print(f"\n{'='*95}")
    print(f"  {'TICKER':<35} {'SRC':<8} {'CONSENSUS':>10} {'MODEL':>7} {'EDGE':>7} {'SIDE':<5} {'VOL':>9}")
    print(f"{'='*95}")
    for s in signals:
        src  = f"[{s.market_source}]"
        flag = " ⚠️" if s.suspicious_line else ""
        print(
            f"  {s.ticker:<35} {src:<8} "
            f"{s.consensus_prob*100:>9.1f}% "
            f"{s.model_prob*100:>6.1f}% "
            f"{s.edge*100:>+6.1f}% "
            f"{s.side:<5} "
            f"${s.volume:>8,}{flag}"
        )
        print(f"    └─ {s.notes}")
    print(f"{'='*95}")
    print(f"  Total signals: {len(signals)}")