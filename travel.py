"""
travel.py
Travel fatigue calculator for NBA teams.

Distance between NBA arenas in miles.
Teams flying cross-country on back-to-backs are significantly more fatigued
than teams playing local back-to-backs.

Fatigue penalty tiers (added on top of B2B penalty):
  < 300 miles:   +0 (local, no extra fatigue)
  300-1000 miles: +10 ELO penalty
  1000-2000 miles: +20 ELO penalty
  2000+ miles:    +30 ELO penalty
"""

import math

# Arena coordinates (lat, lon)
ARENA_COORDS = {
    "Atlanta Hawks":            (33.7573, -84.3963),
    "Boston Celtics":           (42.3662, -71.0621),
    "Brooklyn Nets":            (40.6826, -73.9754),
    "Charlotte Hornets":        (35.2251, -80.8392),
    "Chicago Bulls":            (41.8807, -87.6742),
    "Cleveland Cavaliers":      (41.4965, -81.6882),
    "Dallas Mavericks":         (32.7905, -96.8103),
    "Denver Nuggets":           (39.7487, -105.0077),
    "Detroit Pistons":          (42.6963, -83.2096),
    "Golden State Warriors":    (37.7679, -122.3874),
    "Houston Rockets":          (29.7508, -95.3621),
    "Indiana Pacers":           (39.7639, -86.1555),
    "Los Angeles Clippers":     (34.0430, -118.2673),
    "Los Angeles Lakers":       (34.0430, -118.2673),
    "Memphis Grizzlies":        (35.1382, -90.0505),
    "Miami Heat":               (25.7814, -80.1870),
    "Milwaukee Bucks":          (43.0450, -87.9170),
    "Minnesota Timberwolves":   (44.9795, -93.2760),
    "New Orleans Pelicans":     (29.9490, -90.0821),
    "New York Knicks":          (40.7505, -73.9934),
    "Oklahoma City Thunder":    (35.4634, -97.5151),
    "Orlando Magic":            (28.5392, -81.3839),
    "Philadelphia 76ers":       (39.9012, -75.1720),
    "Phoenix Suns":             (33.4457, -112.0712),
    "Portland Trail Blazers":   (45.5316, -122.6668),
    "Sacramento Kings":         (38.5802, -121.4997),
    "San Antonio Spurs":        (29.4270, -98.4375),
    "Toronto Raptors":          (43.6435, -79.3791),
    "Utah Jazz":                (40.7683, -111.9011),
    "Washington Wizards":       (38.8981, -77.0209),
}


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in miles between two lat/lon points."""
    R = 3958.8  # Earth radius in miles
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def travel_distance(team_from: str, team_to: str) -> float:
    """
    Returns miles traveled going from team_from's arena to team_to's arena.
    Returns 0 if either team not found.
    """
    coords_from = ARENA_COORDS.get(team_from)
    coords_to   = ARENA_COORDS.get(team_to)
    if not coords_from or not coords_to:
        return 0.0
    return haversine_miles(coords_from[0], coords_from[1],
                           coords_to[0],   coords_to[1])


def travel_fatigue_penalty(miles: float) -> int:
    """
    Returns additional ELO penalty based on travel distance.
    This is applied ON TOP of the back-to-back penalty.
    """
    if miles < 300:
        return 0
    elif miles < 1000:
        return 10
    elif miles < 2000:
        return 20
    else:
        return 30


def get_last_game_location(team: str, games: list[dict]) -> str:
    """
    Find where a team played their most recent game.
    Returns the opponent team name (whose arena they were at).
    """
    from nba_model import normalize_team

    team_games = sorted(
        [g for g in games if g["team"] == team],
        key=lambda g: g["date"], reverse=True
    )

    if not team_games:
        return team  # assume they're at home

    last = team_games[0]
    parts    = last["matchup"].split()
    opponent = normalize_team(parts[-1]) if parts else None

    if "vs." in last["matchup"]:
        # Home game — they're already at their arena
        return team
    else:
        # Away game — they were at opponent's arena
        return opponent or team


def away_travel_penalty(away_team: str, home_team: str, games: list[dict]) -> int:
    """
    Calculate travel fatigue penalty for the away team.
    Looks at where they played last and how far they had to travel.
    Only applies if it was a back-to-back (rest_days == 1).
    """
    from nba_model import rest_days

    if rest_days(away_team, games) != 1:
        return 0  # Not a back-to-back, no travel penalty

    last_location = get_last_game_location(away_team, games)
    miles         = travel_distance(last_location, home_team)
    return travel_fatigue_penalty(miles)


def home_travel_penalty(home_team: str, games: list[dict]) -> int:
    """
    Calculate travel fatigue for the home team on a back-to-back.
    Home teams on B2Bs usually just played at home or traveled home.
    """
    from nba_model import rest_days

    if rest_days(home_team, games) != 1:
        return 0

    last_location = get_last_game_location(home_team, games)
    if last_location == home_team:
        return 0  # They played at home last night, no travel

    # They're returning home from a road game
    miles = travel_distance(last_location, home_team)
    # Returning home is less fatiguing than going somewhere new
    return travel_fatigue_penalty(miles) // 2


if __name__ == "__main__":
    # Test some distances
    tests = [
        ("Los Angeles Lakers", "Boston Celtics"),
        ("Dallas Mavericks", "San Antonio Spurs"),
        ("Golden State Warriors", "New York Knicks"),
        ("Chicago Bulls", "Milwaukee Bucks"),
    ]
    print("Sample travel distances:")
    for t1, t2 in tests:
        d = travel_distance(t1, t2)
        p = travel_fatigue_penalty(d)
        print(f"  {t1} → {t2}: {d:.0f} miles  penalty: -{p} ELO")