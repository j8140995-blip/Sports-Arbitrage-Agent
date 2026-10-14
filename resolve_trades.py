"""
resolve_trades.py
Automatically resolves all open trades by checking Kalshi market results.
Run this to update paper_trades.json with outcomes and P&L.

Usage: python resolve_trades.py
"""

import json
import time
from pathlib import Path
import kalshi_client as kc

PAPER_LOG = Path("paper_trades.json")


def exact_pnl(outcome: str, entry_price_cents: int, contracts: int) -> float:
    if outcome == "win":
        return round((100 - entry_price_cents) * contracts / 100, 2)
    else:
        return round(-entry_price_cents * contracts / 100, 2)


def resolve_from_kalshi(ticker: str) -> tuple[str | None, float | None]:
    """
    Fetch market result from Kalshi.
    Returns (result, closing_price) or (None, None) if not settled.
    """
    try:
        resp   = kc.get(f"/markets/{ticker}")
        market = resp.get("market", resp)
        result = market.get("result")

        if result == "yes":
            return "yes", 1.0
        elif result == "no":
            return "no", 0.0

        # Check status
        status = market.get("status", "")
        if status in ("settled", "finalized"):
            # Try event-level result
            event_ticker = "-".join(ticker.split("-")[:-1])
            try:
                event  = kc.get(f"/events/{event_ticker}")
                emkts  = event.get("markets", [])
                for m in emkts:
                    if m.get("ticker") == ticker:
                        r = m.get("result")
                        if r == "yes":
                            return "yes", 1.0
                        elif r == "no":
                            return "no", 0.0
            except Exception:
                pass

        return None, None
    except Exception as e:
        return None, None


def fetch_mlb_results() -> dict:
    """
    Fetch completed MLB game results from ESPN API.
    Returns {away_abbr_home_abbr: winner_abbr}
    """
    import requests
    TEAM_MAP = {
        "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL",
        "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
        "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
        "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
        "Colorado Rockies": "COL", "Detroit Tigers": "DET",
        "Houston Astros": "HOU", "Kansas City Royals": "KC",
        "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
        "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
        "Minnesota Twins": "MIN", "New York Mets": "NYM",
        "New York Yankees": "NYY", "Oakland Athletics": "OAK",
        "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
        "San Diego Padres": "SD", "San Francisco Giants": "SF",
        "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
        "Tampa Bay Rays": "TB", "Texas Rangers": "TEX",
        "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH",
    }
    try:
        resp = requests.get(
            "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard",
            timeout=10
        )
        results = {}
        for event in resp.json().get("events", []):
            if event.get("status", {}).get("type", {}).get("name") != "STATUS_FINAL":
                continue
            comp = event.get("competitions", [{}])[0]
            competitors = comp.get("competitors", [])
            if len(competitors) != 2:
                continue
            home = next((c for c in competitors if c.get("homeAway") == "home"), None)
            away = next((c for c in competitors if c.get("homeAway") == "away"), None)
            if not home or not away:
                continue
            home_name = home.get("team", {}).get("displayName", "")
            away_name = away.get("team", {}).get("displayName", "")
            home_abbr = TEAM_MAP.get(home_name, home_name[:3].upper())
            away_abbr = TEAM_MAP.get(away_name, away_name[:3].upper())
            home_score = int(home.get("score", 0))
            away_score = int(away.get("score", 0))
            winner = home_abbr if home_score > away_score else away_abbr
            results[f"{away_abbr}_{home_abbr}"] = winner
        return results
    except Exception as e:
        print(f"[resolve] MLB ESPN fetch error: {e}")
        return {}


def fetch_soccer_results() -> dict:
    """Fetch completed soccer results from ESPN for all major leagues."""
    import requests
    results = {}
    leagues = [
        ("eng.1", "EPL"), ("esp.1", "LALIGA"), ("ger.1", "BUNDESLIGA"),
        ("ita.1", "SERIEA"), ("usa.1", "MLS"), ("uefa.champions", "UCL"),
        ("uefa.europa", "UEL"),
    ]
    for espn_id, league in leagues:
        try:
            resp = requests.get(
                f"https://site.api.espn.com/apis/site/v2/sports/soccer/{espn_id}/scoreboard",
                timeout=8
            )
            for event in resp.json().get("events", []):
                if event.get("status", {}).get("type", {}).get("name") != "STATUS_FINAL":
                    continue
                comp = event.get("competitions", [{}])[0]
                competitors = comp.get("competitors", [])
                if len(competitors) != 2:
                    continue
                home = next((c for c in competitors if c.get("homeAway") == "home"), None)
                away = next((c for c in competitors if c.get("homeAway") == "away"), None)
                if not home or not away:
                    continue
                home_name  = home.get("team", {}).get("displayName", "")
                away_name  = away.get("team", {}).get("displayName", "")
                home_score = int(home.get("score", 0))
                away_score = int(away.get("score", 0))
                winner     = home_name if home_score > away_score else (
                             away_name if away_score > home_score else "draw")
                results[f"{away_name}_{home_name}"] = winner
            time.sleep(0.1)
        except Exception:
            pass
    return results


def resolve_all():
    if not PAPER_LOG.exists():
        print("No paper_trades.json found.")
        return

    trades  = json.loads(PAPER_LOG.read_text())
    open_t  = [t for t in trades if t.get("outcome") is None]
    print(f"Found {len(open_t)} unresolved trades out of {len(trades)} total.\n")

    # Pre-fetch ESPN results for faster resolution
    mlb_results    = fetch_mlb_results()
    soccer_results = fetch_soccer_results()
    if mlb_results:
        print(f"[resolve] Pre-fetched {len(mlb_results)} MLB results from ESPN")
    if soccer_results:
        print(f"[resolve] Pre-fetched {len(soccer_results)} soccer results from ESPN\n")

    resolved  = 0
    not_found = 0

    for t in trades:
        if t.get("outcome") is not None:
            continue

        ticker = t.get("ticker", "")
        side   = t.get("side", "yes")
        sport  = (t.get("sport") or "").upper()

        # Try ESPN pre-fetched results first for MLB and SOCCER
        result, closing_price = None, None

        if sport == "MLB" and mlb_results:
            home      = t.get("home_team", "")
            away      = t.get("away_team", "")
            key       = f"{away}_{home}"
            game_date = t.get("game_date", "")

            # Only resolve if game_date matches today (avoid cross-date false matches)
            import datetime
            today_str = datetime.date.today().isoformat()
            if key in mlb_results and game_date == today_str:
                winner     = mlb_results[key]
                # Determine if this market is for home or away win
                title_lower = ticker.lower()
                is_home = (home.lower() in title_lower or
                           ticker.upper().endswith(home.upper()))
                winning_side = "yes" if (
                    (is_home and winner == home) or
                    (not is_home and winner == away)
                ) else "no"
                result        = winning_side
                closing_price = 1.0 if result == "yes" else 0.0

        if result is None:
            result, closing_price = resolve_from_kalshi(ticker)
            time.sleep(0.1)

        if result is None:
            not_found += 1
            continue

        # Determine win/loss based on side and result
        if side == "yes":
            outcome = "win" if result == "yes" else "loss"
        else:
            outcome = "win" if result == "no" else "loss"

        entry_cents = t.get("entry_price_cents", 50)
        contracts   = t.get("contracts", 1)
        pnl         = exact_pnl(outcome, entry_cents, contracts)

        t["outcome"]       = outcome
        t["pnl"]           = pnl
        t["status"]        = "resolved"
        t["closing_price"] = closing_price

        # CLV
        entry = entry_cents / 100
        if side == "yes":
            clv = closing_price - entry
        else:
            clv = (1 - closing_price) - entry
        t["clv"] = round(clv, 4)

        resolved += 1
        icon = "✅" if outcome == "win" else "❌"
        print(f"  {icon} {ticker[:50]:<50} {side:>3} → {outcome:<4}  P&L: ${pnl:+.2f}  CLV: {clv*100:+.1f}%")

    PAPER_LOG.write_text(json.dumps(trades, indent=2))

    print(f"\n{'='*60}")
    print(f"  Resolved:  {resolved}")
    print(f"  Not found: {not_found} (still open or not settled yet)")
    print(f"{'='*60}")

    # Print updated summary
    all_resolved = [t for t in trades if t.get("outcome") in ("win", "loss")]
    wins         = [t for t in all_resolved if t["outcome"] == "win"]
    total_pnl    = sum(t.get("pnl", 0) or 0 for t in all_resolved)
    wagered      = sum(t.get("cost_dollars", 0) for t in all_resolved)

    print(f"\n  Total resolved: {len(all_resolved)}")
    if all_resolved:
        print(f"  Win rate:       {len(wins)/len(all_resolved)*100:.1f}%")
        print(f"  Total P&L:      ${total_pnl:+.2f}")
        if wagered:
            print(f"  ROI:            {total_pnl/wagered*100:.1f}%")

    # CLV summary
    clv_trades = [t for t in all_resolved if t.get("clv") is not None]
    if clv_trades:
        mean_clv = sum(t["clv"] for t in clv_trades) / len(clv_trades)
        print(f"  Mean CLV:       {mean_clv*100:+.2f}%")


if __name__ == "__main__":
    resolve_all()