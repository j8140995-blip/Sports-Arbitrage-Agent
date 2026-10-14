"""
darko_client.py  v2
Player impact ratings sourced from stats.nba.com via nba_api.

v2 improvements:
  - Persistent star archive: players with strong DPM in past runs are kept
    even if they drop out of current data (long-term injury). Prevents stars
    disappearing from the model when they're most relevant.
  - Cache TTL reduced to 6 hours (was 24) — injury info changes within a day
  - Cache validation: if the cache has weird structure, rebuild it
  - Manual override file (darko_overrides.json) for edge cases

Formula: dpm ≈ NET_RATING / 3 (approximates DARKO/EPM scale)
"""

import json
import time
from pathlib import Path
from typing import Optional

DARKO_CACHE     = Path("darko_cache.json")
DARKO_ARCHIVE   = Path("darko_archive.json")   # persistent history of stars
DARKO_OVERRIDES = Path("darko_overrides.json") # manual overrides
DARKO_CACHE_TTL = 6 * 3600                     # 6 hours

REPLACEMENT_DPM  = -3.0
AVG_STARTER_MPG  = 28.0
TEAM_MINUTES     = 240.0
MIN_GP_FOR_DPM   = 10
MIN_MPG_FOR_DPM  = 5.0

# Players who scored DPM >= this threshold historically stay in the ratings
# even if they drop below current GP/MPG filters (e.g. long-term injuries).
STAR_DPM_THRESHOLD = 3.0


def _load_archive() -> dict:
    if DARKO_ARCHIVE.exists():
        try:
            return json.loads(DARKO_ARCHIVE.read_text())
        except Exception:
            return {}
    return {}


def _save_archive(archive: dict):
    try:
        DARKO_ARCHIVE.write_text(json.dumps(archive, indent=2))
    except Exception as e:
        print(f"[DARKO] Archive save failed: {e}")


def _load_overrides() -> dict:
    if DARKO_OVERRIDES.exists():
        try:
            return json.loads(DARKO_OVERRIDES.read_text())
        except Exception:
            return {}
    return {}


def _validate_cache(data: dict) -> bool:
    if not isinstance(data, dict):
        return False
    if "players" not in data or "timestamp" not in data:
        return False
    players = data["players"]
    if not isinstance(players, dict) or len(players) < 100:
        return False
    sample = next(iter(players.values()), None)
    if not isinstance(sample, dict) or "dpm" not in sample or "mpg" not in sample:
        return False
    return True


def fetch_darko_dpm() -> dict[str, dict]:
    """
    Fetch player impact ratings from stats.nba.com.
    Returns {player_name: {"dpm": float, "mpg": float}}

    Flow:
      1. Check cache — serve if fresh and valid
      2. Fetch from nba_api
      3. Apply manual overrides
      4. Merge with persistent star archive
      5. Update archive with any new stars from this run
    """
    # 1. Cache check
    if DARKO_CACHE.exists():
        try:
            data = json.loads(DARKO_CACHE.read_text())
            if _validate_cache(data):
                age = time.time() - data.get("timestamp", 0)
                if age < DARKO_CACHE_TTL:
                    players = data["players"]
                    print(f"[DARKO] Loaded {len(players)} players from cache "
                          f"({age/3600:.1f}h old).")
                    return players
                else:
                    print(f"[DARKO] Cache stale ({age/3600:.1f}h), refetching.")
            else:
                print("[DARKO] Cache structure invalid, rebuilding.")
        except Exception as e:
            print(f"[DARKO] Cache read failed: {e}, rebuilding.")

    # 2. Fresh fetch
    players = {}
    try:
        from nba_api.stats.endpoints import leaguedashplayerstats
        time.sleep(0.6)
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season="2025-26",
            per_mode_detailed="PerGame",
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Advanced",
        )
        df = stats.get_data_frames()[0]

        for _, row in df.iterrows():
            name = row.get("PLAYER_NAME", "")
            gp   = int(row.get("GP", 0) or 0)
            mpg  = float(row.get("MIN", 0) or 0)
            if gp < MIN_GP_FOR_DPM or mpg < MIN_MPG_FOR_DPM:
                continue

            net_rtg    = float(row.get("NET_RATING", 0) or 0)
            dpm_approx = net_rtg / 3.0

            players[name] = {
                "dpm": round(dpm_approx, 2),
                "mpg": round(mpg, 1),
            }

        if players:
            print(f"[DARKO] Computed impact ratings for {len(players)} players from nba_api.")
    except Exception as e:
        print(f"[DARKO] nba_api fetch failed: {e}. Falling back to archive + overrides.")

    # 3. Apply manual overrides
    overrides = _load_overrides()
    if overrides:
        for name, data in overrides.items():
            if isinstance(data, dict) and "dpm" in data and "mpg" in data:
                players[name] = {
                    "dpm":      float(data["dpm"]),
                    "mpg":      float(data["mpg"]),
                    "override": True,
                }
        print(f"[DARKO] Applied {len(overrides)} manual overrides.")

    # 4. Merge with archive
    archive = _load_archive()
    archive_added = 0
    for name, data in archive.items():
        if name not in players:
            players[name] = {
                "dpm":      data["dpm"],
                "mpg":      data["mpg"],
                "archived": True,
            }
            archive_added += 1
    if archive_added:
        print(f"[DARKO] Merged in {archive_added} archived stars (long-term injured).")

    # 5. Update archive
    today = time.strftime("%Y-%m-%d")
    for name, data in players.items():
        if data.get("override") or data.get("archived"):
            continue
        if data.get("dpm", 0) >= STAR_DPM_THRESHOLD:
            archive[name] = {
                "dpm":       data["dpm"],
                "mpg":       data["mpg"],
                "last_seen": today,
            }
    _save_archive(archive)

    # 6. Save cache
    if players:
        try:
            DARKO_CACHE.write_text(json.dumps({
                "players":   players,
                "timestamp": time.time(),
            }, indent=2))
        except Exception as e:
            print(f"[DARKO] Cache write failed: {e}")

    return players


def quantify_injury_impact(team, injured_players, player_impacts, player_minutes):
    """Legacy helper. nba_model.injury_elo_adjustment is the primary path."""
    ELO_PER_POINT = 28.0
    total_penalty = 0.0
    for name in injured_players:
        mpg = player_minutes.get(name, AVG_STARTER_MPG)
        impact_data = player_impacts.get(name)
        if impact_data is None:
            name_lower = name.lower()
            for pname, pdata in player_impacts.items():
                if name_lower in pname.lower() or pname.lower() in name_lower:
                    impact_data = pdata
                    mpg = pdata.get("mpg", mpg)
                    break
        if impact_data:
            player_dpm = impact_data["dpm"]
            mpg = impact_data.get("mpg", mpg)
        else:
            if mpg >= 32:   player_dpm = 2.0
            elif mpg >= 25: player_dpm = 0.5
            else:           player_dpm = -1.0

        minutes_share = min(mpg / TEAM_MINUTES, 0.20)
        point_swing   = (player_dpm - REPLACEMENT_DPM) * minutes_share
        total_penalty += point_swing * ELO_PER_POINT

    return -abs(total_penalty)


if __name__ == "__main__":
    players = fetch_darko_dpm()
    print(f"\nTotal players: {len(players)}")
    if players:
        top = sorted(players.items(), key=lambda x: x[1]["dpm"], reverse=True)[:15]
        print("\nTop 15 by DPM:")
        for name, data in top:
            flag = ""
            if data.get("archived"): flag = " [archived]"
            if data.get("override"): flag = " [override]"
            print(f"  {name:<30} DPM={data['dpm']:>+5.2f}  MPG={data['mpg']:.1f}{flag}")