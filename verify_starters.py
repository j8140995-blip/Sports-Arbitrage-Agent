"""
verify_starters.py
Cross-check today's MLB starters against ESPN's scoreboard to see if
'TBD' flags in our scanner are real or stale data.

If the scanner says TBD but ESPN has a name, our pitcher data is stale.

Usage:
    python verify_starters.py
"""

import datetime
import requests


def main():
    today = datetime.date.today().strftime("%Y%m%d")
    url   = f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={today}"

    try:
        r = requests.get(url, timeout=10)
        data = r.json()
    except Exception as e:
        print(f"Couldn't fetch ESPN: {e}")
        return

    events = data.get("events", [])
    print(f"\n{'='*100}")
    print(f"  MLB starting pitchers per ESPN — {today}")
    print(f"{'='*100}\n")

    for event in events:
        name = event.get("shortName", "")
        comp = event.get("competitions", [{}])[0]
        status = comp.get("status", {}).get("type", {}).get("description", "")

        teams = comp.get("competitors", [])
        if len(teams) < 2:
            continue

        away = next((t for t in teams if t.get("homeAway") == "away"), {})
        home = next((t for t in teams if t.get("homeAway") == "home"), {})

        away_name = away.get("team", {}).get("abbreviation", "?")
        home_name = home.get("team", {}).get("abbreviation", "?")

        # Look for probable pitchers in the event details
        away_p = "?"
        home_p = "?"

        # Check probables list (sometimes nested differently)
        probables = comp.get("probables", [])
        if probables:
            for p in probables:
                pname = p.get("athlete", {}).get("displayName", "?")
                team_id = p.get("homeAway", "")
                if team_id == "away":
                    away_p = pname
                elif team_id == "home":
                    home_p = pname

        # Fallback: check competitors
        if away_p == "?":
            for ath in away.get("probables", []):
                away_p = ath.get("athlete", {}).get("displayName", "?")
                break
        if home_p == "?":
            for ath in home.get("probables", []):
                home_p = ath.get("athlete", {}).get("displayName", "?")
                break

        print(f"  {away_name} @ {home_name}  ({status})")
        print(f"    Away SP: {away_p}")
        print(f"    Home SP: {home_p}")
        print()


if __name__ == "__main__":
    main()