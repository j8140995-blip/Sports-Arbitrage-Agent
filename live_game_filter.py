"""
live_game_filter.py
Shared utility to check if a game has already started.

Used by all scanners (NBA, MLB, soccer) to skip live/in-progress games.
Live markets mislead the model — Vegas has already priced in game state
(current score, who's hurt, etc.) which our pre-game model doesn't know.

Usage:
    from live_game_filter import is_live_game

    if is_live_game(vegas_entry):
        print(f"  [skip-live] {matchup} — game already started")
        continue
"""

import datetime


def is_live_game(vegas_entry: dict, grace_minutes: int = 0) -> tuple[bool, int]:
    """
    Check if a game has already started based on commence_time from Vegas data.

    Args:
      vegas_entry:    Dict with "commence_time" key (ISO 8601 UTC string).
                      Usually the per-game dict from the Odds API response.
      grace_minutes:  Treat games as "not yet live" until this many minutes
                      after scheduled start. Default 0 = any time past start = live.

    Returns:
      (is_live: bool, minutes_past_start: int)
      minutes_past_start is negative if game hasn't started yet.
    """
    if not vegas_entry:
        return False, 0

    commence_str = vegas_entry.get("commence_time", "")
    if not commence_str:
        return False, 0

    try:
        ct  = datetime.datetime.fromisoformat(commence_str.replace("Z", "+00:00"))
        now = datetime.datetime.now(datetime.timezone.utc)
        minutes_past = int((now - ct).total_seconds() / 60)
        return minutes_past >= grace_minutes, minutes_past
    except Exception:
        return False, 0


if __name__ == "__main__":
    # Tests
    import json

    now = datetime.datetime.now(datetime.timezone.utc)
    cases = [
        ("Future game (1h away)",    now + datetime.timedelta(hours=1)),
        ("Future game (5 min away)", now + datetime.timedelta(minutes=5)),
        ("Just started (2 min ago)", now - datetime.timedelta(minutes=2)),
        ("Mid-game (45 min ago)",    now - datetime.timedelta(minutes=45)),
        ("No commence_time",         None),
    ]
    for label, dt in cases:
        entry = {"commence_time": dt.strftime("%Y-%m-%dT%H:%M:%SZ")} if dt else {}
        live, mins = is_live_game(entry)
        status = "🛑 LIVE" if live else "✅ OK"
        print(f"  {label:<32} → {status}  ({mins:+d} min)")