"""
nba_audit.py
Audits the current NBA model to find where it's losing.
Runs through all 2025-26 games and breaks down accuracy by:
- Edge bucket (how confident were we?)
- Favorite vs underdog
- Home vs away 
- Back-to-back games
- Component accuracy (ELO alone, ratings alone, form alone)
"""

import json
import datetime
from pathlib import Path
from collections import defaultdict

CACHE_FILE = Path("nba_elo_cache.json")

def load_games():
    # Try nba_elo_cache.json first, then fetch fresh
    if CACHE_FILE.exists():
        data = json.loads(CACHE_FILE.read_text())
        games = data.get("games", [])
        if games:
            return games
    # Fetch fresh if no cache
    print("No cache found — fetching game logs...")
    from nba_model import fetch_game_logs, _save_cache, build_elo_ratings, build_team_ratings, build_recent_form, build_pace_ratings
    games = fetch_game_logs()
    if games:
        elos    = build_elo_ratings(games)
        ratings = build_team_ratings(games)
        form    = build_recent_form(games, elos)
        pace    = build_pace_ratings(games)
        _save_cache({"elos": elos, "ratings": ratings, "form": form, "pace": pace, "games": games})
        print(f"Fetched and cached {len(games)} games.")
    return games

def run_audit():
    from nba_model import (
        NBAModel, normalize_team, expected_score, rating_win_prob,
        recent_form_win_prob, rest_days, get_home_court,
        BASE_ELO, B2B_PENALTY, W_ELO, W_RATING, W_RECENT,
        build_elo_ratings, build_team_ratings, build_recent_form,
        build_pace_ratings, pace_win_prob_adjustment,
    )

    games = load_games()
    if not games:
        print("No cached games found. Run main.py --force first.")
        return

    dates     = sorted(set(g["date"] for g in games))
    seen      = set()
    results   = []

    print(f"Auditing {len(games)} game records across {len(dates)} dates...")

    for date in dates:
        games_before = [g for g in games if g["date"] < date]
        if len(games_before) < 100:
            continue

        elos    = build_elo_ratings(games_before)
        ratings = build_team_ratings(games_before)
        form    = build_recent_form(games_before, elos)
        pace    = build_pace_ratings(games_before)

        todays = [g for g in games if g["date"] == date]
        for g in todays:
            parts    = g["matchup"].split()
            opponent = normalize_team(parts[-1]) if parts else None
            if not opponent:
                continue

            if "vs." in g["matchup"]:
                home, away = g["team"], opponent
            else:
                away, home = g["team"], opponent

            key = "|".join(sorted([home, away])) + date
            if key in seen:
                continue
            seen.add(key)

            home_game = next((x for x in todays
                             if x["team"] == home and "vs." in x["matchup"]), None)
            if not home_game:
                continue

            home_won = home_game["wl"] == "W"

            # Calculate each component
            home_elo   = elos.get(home, BASE_ELO)
            away_elo   = elos.get(away, BASE_ELO)
            hc         = get_home_court(home)
            adj_home   = home_elo + hc

            game_date_obj = datetime.date.fromisoformat(date)
            home_rest  = rest_days(home, games_before, game_date_obj)
            away_rest  = rest_days(away, games_before, game_date_obj)
            if home_rest == 1: adj_home -= B2B_PENALTY
            if home_rest == 0: adj_home -= B2B_PENALTY * 1.5
            if away_rest == 1: away_elo -= B2B_PENALTY
            if away_rest == 0: away_elo -= B2B_PENALTY * 1.5

            p_elo    = expected_score(adj_home, away_elo)
            p_rating = rating_win_prob(home, away, ratings)
            p_recent = recent_form_win_prob(home, away, form)
            p_pace   = pace_win_prob_adjustment(home, away, pace)
            p_home   = W_ELO * p_elo + W_RATING * p_rating + W_RECENT * p_recent + p_pace
            p_home   = max(0.05, min(0.95, p_home))

            results.append({
                "date":      date,
                "home":      home,
                "away":      away,
                "home_won":  home_won,
                "p_home":    p_home,
                "p_elo":     p_elo,
                "p_rating":  p_rating,
                "p_recent":  p_recent,
                "p_pace":    p_pace,
                "home_b2b":  home_rest == 1,
                "away_b2b":  away_rest == 1,
                "home_rest": home_rest,
                "away_rest": away_rest,
            })

    if not results:
        print("No results computed.")
        return

    print(f"\nAnalyzed {len(results)} games\n")

    # ── Overall accuracy ──────────────────────────────────────────────────────
    def accuracy(data, prob_key="p_home", outcome_key="home_won"):
        correct = sum(1 for r in data
                     if (r[prob_key] > 0.5) == r[outcome_key])
        return correct / len(data) if data else 0

    print(f"{'='*60}")
    print(f"  COMPONENT ACCURACY")
    print(f"{'='*60}")
    print(f"  Full model:       {accuracy(results)*100:.1f}%")
    print(f"  ELO only:         {accuracy(results,'p_elo')*100:.1f}%")
    print(f"  Ratings only:     {accuracy(results,'p_rating')*100:.1f}%")
    print(f"  Recent form only: {accuracy(results,'p_recent')*100:.1f}%")

    # ── Accuracy by confidence bucket ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ACCURACY BY MODEL CONFIDENCE")
    print(f"{'='*60}")
    buckets = defaultdict(list)
    for r in results:
        conf = max(r["p_home"], 1 - r["p_home"])
        bucket = f"{int(conf*100//5)*5}-{int(conf*100//5)*5+5}%"
        buckets[bucket].append(r)

    for bucket in sorted(buckets.keys()):
        data = buckets[bucket]
        acc  = accuracy(data)
        print(f"  {bucket:<12} {len(data):>4} games  {acc*100:.1f}% accurate")

    # ── B2B analysis ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  BACK-TO-BACK ACCURACY")
    print(f"{'='*60}")
    no_b2b    = [r for r in results if not r["home_b2b"] and not r["away_b2b"]]
    home_b2b  = [r for r in results if r["home_b2b"] and not r["away_b2b"]]
    away_b2b  = [r for r in results if r["away_b2b"] and not r["home_b2b"]]
    both_b2b  = [r for r in results if r["home_b2b"] and r["away_b2b"]]
    print(f"  No B2B:            {len(no_b2b):>4} games  {accuracy(no_b2b)*100:.1f}%")
    print(f"  Home on B2B:       {len(home_b2b):>4} games  {accuracy(home_b2b)*100:.1f}%")
    print(f"  Away on B2B:       {len(away_b2b):>4} games  {accuracy(away_b2b)*100:.1f}%")
    print(f"  Both on B2B:       {len(both_b2b):>4} games  {accuracy(both_b2b)*100:.1f}%")

    # ── Favorite accuracy ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  MODEL FAVORITE ACCURACY")
    print(f"{'='*60}")
    for threshold in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        favs = [r for r in results if max(r["p_home"], 1-r["p_home"]) >= threshold]
        if favs:
            acc = accuracy(favs)
            print(f"  Model >{threshold*100:.0f}%:    {len(favs):>4} games  {acc*100:.1f}% accurate")

    # ── Brier Score calibration (per research report: more important than accuracy) ──
    print(f"\n{'='*60}")
    print(f"  BRIER SCORE CALIBRATION")
    print(f"{'='*60}")
    # Brier score: mean((p - outcome)^2), lower is better
    # Perfect = 0.0, random = 0.25
    brier_full    = sum((r["p_home"] - (1 if r["home_won"] else 0))**2 for r in results) / len(results)
    brier_rating  = sum((r["p_rating"] - (1 if r["home_won"] else 0))**2 for r in results) / len(results)
    brier_elo     = sum((r["p_elo"] - (1 if r["home_won"] else 0))**2 for r in results) / len(results)
    brier_naive   = 0.25  # random baseline

    print(f"  Full model Brier:     {brier_full:.4f}  (vs random: {brier_naive:.4f})")
    print(f"  Ratings only Brier:   {brier_rating:.4f}")
    print(f"  ELO only Brier:       {brier_elo:.4f}")
    print(f"  Skill score:          {(1 - brier_full/brier_naive)*100:.1f}%  (% better than random)")
    print(f"  Note: lower Brier = better calibration. Skill >10% = useful model.")

    # ── Where the model is most WRONG ────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  BIGGEST MODEL FAILURES (confident but wrong)")
    print(f"{'='*60}")
    failures = [r for r in results
                if max(r["p_home"], 1-r["p_home"]) > 0.70
                and (r["p_home"] > 0.5) != r["home_won"]]
    print(f"  Confident wrong predictions: {len(failures)} of {len(results)} ({len(failures)/len(results)*100:.1f}%)")

    b2b_failures = [r for r in failures if r["home_b2b"] or r["away_b2b"]]
    print(f"  Of those, involved B2B: {len(b2b_failures)} ({len(b2b_failures)/max(len(failures),1)*100:.1f}%)")

    print(f"{'='*60}")


def run_disagreement_audit():
    """
    Find games where ELO and ratings disagree — these are likely the dead zone failures.
    """
    from nba_model import (
        normalize_team, expected_score, rating_win_prob,
        recent_form_win_prob, rest_days, get_home_court,
        BASE_ELO, B2B_PENALTY, W_ELO, W_RATING, W_RECENT,
        build_elo_ratings, build_team_ratings, build_recent_form,
        build_pace_ratings, pace_win_prob_adjustment,
    )

    games = load_games()
    if not games:
        return

    dates  = sorted(set(g["date"] for g in games))
    seen   = set()
    agree  = []   # ELO and ratings agree on winner
    disagree = [] # ELO and ratings disagree

    for date in dates:
        games_before = [g for g in games if g["date"] < date]
        if len(games_before) < 100:
            continue

        elos    = build_elo_ratings(games_before)
        ratings = build_team_ratings(games_before)
        form    = build_recent_form(games_before, elos)

        todays = [g for g in games if g["date"] == date]
        for g in todays:
            parts    = g["matchup"].split()
            opponent = normalize_team(parts[-1]) if parts else None
            if not opponent:
                continue
            if "vs." in g["matchup"]:
                home, away = g["team"], opponent
            else:
                away, home = g["team"], opponent

            key = "|".join(sorted([home, away])) + date
            if key in seen:
                continue
            seen.add(key)

            home_game = next((x for x in todays
                             if x["team"] == home and "vs." in x["matchup"]), None)
            if not home_game:
                continue

            home_won = home_game["wl"] == "W"

            game_date_obj = datetime.date.fromisoformat(date)
            home_rest = rest_days(home, games_before, game_date_obj)
            away_rest = rest_days(away, games_before, game_date_obj)

            home_elo = elos.get(home, BASE_ELO)
            away_elo = elos.get(away, BASE_ELO)
            adj_home = home_elo + get_home_court(home)
            if home_rest == 1: adj_home -= B2B_PENALTY
            if home_rest == 0: adj_home -= B2B_PENALTY * 1.5
            if away_rest == 1: away_elo -= B2B_PENALTY * 1.2
            if away_rest == 0: away_elo -= B2B_PENALTY * 1.8

            p_elo    = expected_score(adj_home, away_elo)
            p_rating = rating_win_prob(home, away, ratings)

            elo_pick    = p_elo > 0.5
            rating_pick = p_rating > 0.5

            record = {"home_won": home_won, "p_elo": p_elo, "p_rating": p_rating}
            if elo_pick == rating_pick:
                agree.append(record)
            else:
                disagree.append(record)

    def acc(data, key="p_elo"):
        if not data: return 0
        return sum(1 for r in data if (r[key] > 0.5) == r["home_won"]) / len(data)

    print(f"\n{'='*60}")
    print(f"  COMPONENT AGREEMENT ANALYSIS")
    print(f"{'='*60}")
    print(f"  ELO + Ratings AGREE:    {len(agree):>4} games  {acc(agree)*100:.1f}% accurate")
    print(f"  ELO + Ratings DISAGREE: {len(disagree):>4} games  {acc(disagree)*100:.1f}% accurate")
    print(f"\n  → Disagreement games are noise. Skip them.")
    print(f"  → If we only bet agreement games: {acc(agree)*100:.1f}% base accuracy")
    print(f"{'='*60}")


def run_filtered_simulation():
    """
    Simulate what win rate we'd get trading only when:
    1. ELO and ratings agree
    2. Model confidence > 75% (bet_prob > 0.75)
    3. Not in dead zone (65-75%)
    """
    from nba_model import (
        normalize_team, expected_score, rating_win_prob,
        recent_form_win_prob, rest_days, get_home_court,
        BASE_ELO, B2B_PENALTY, W_ELO, W_RATING, W_RECENT,
        build_elo_ratings, build_team_ratings, build_recent_form,
        build_pace_ratings, pace_win_prob_adjustment,
    )

    games = load_games()
    if not games:
        return

    dates = sorted(set(g["date"] for g in games))
    seen  = set()
    bets  = []

    for date in dates:
        games_before = [g for g in games if g["date"] < date]
        if len(games_before) < 100:
            continue

        elos    = build_elo_ratings(games_before)
        ratings = build_team_ratings(games_before)
        form    = build_recent_form(games_before, elos)
        pace    = build_pace_ratings(games_before)

        todays = [g for g in games if g["date"] == date]
        for g in todays:
            parts    = g["matchup"].split()
            opponent = normalize_team(parts[-1]) if parts else None
            if not opponent: continue
            if "vs." in g["matchup"]:
                home, away = g["team"], opponent
            else:
                away, home = g["team"], opponent

            key = "|".join(sorted([home, away])) + date
            if key in seen: continue
            seen.add(key)

            home_game = next((x for x in todays
                             if x["team"] == home and "vs." in x["matchup"]), None)
            if not home_game: continue
            home_won = home_game["wl"] == "W"

            game_date_obj = datetime.date.fromisoformat(date)
            home_rest = rest_days(home, games_before, game_date_obj)
            away_rest = rest_days(away, games_before, game_date_obj)

            home_elo = elos.get(home, BASE_ELO) + get_home_court(home)
            away_elo_adj = elos.get(away, BASE_ELO)
            if home_rest == 1: home_elo -= B2B_PENALTY
            if home_rest == 0: home_elo -= B2B_PENALTY * 1.5
            if away_rest == 1: away_elo_adj -= B2B_PENALTY * 1.2
            if away_rest == 0: away_elo_adj -= B2B_PENALTY * 1.8

            p_elo    = expected_score(home_elo, away_elo_adj)
            p_rating = rating_win_prob(home, away, ratings)
            p_recent = recent_form_win_prob(home, away, form)
            p_pace   = pace_win_prob_adjustment(home, away, pace)
            p_home   = max(0.05, min(0.95,
                          W_ELO * p_elo + W_RATING * p_rating + W_RECENT * p_recent + p_pace))

            # Apply all filters
            elo_pick    = p_elo > 0.5
            rating_pick = p_rating > 0.5
            if elo_pick != rating_pick:
                continue  # components disagree

            bet_prob = max(p_home, 1 - p_home)

            if bet_prob < 0.75:
                continue  # below 75% threshold (audit shows 77.5% win rate above this)

            side = "home" if p_home > 0.5 else "away"
            won  = (side == "home" and home_won) or (side == "away" and not home_won)
            bets.append({"won": won, "prob": bet_prob, "side": side})

    if not bets:
        print("No filtered bets found.")
        return

    win_rate = sum(1 for b in bets if b["won"]) / len(bets)
    print(f"\n{'='*60}")
    print(f"  FILTERED TRADE SIMULATION")
    print(f"  (agreement + threshold >75% — projected 77.5% win rate)")
    print(f"{'='*60}")
    print(f"  Total filtered bets:  {len(bets)}")
    print(f"  Win rate:             {win_rate*100:.1f}%")

    # Break down by confidence
    from collections import defaultdict
    by_conf = defaultdict(list)
    for b in bets:
        bucket = f"{int(b['prob']*100//5)*5}-{int(b['prob']*100//5)*5+5}%"
        by_conf[bucket].append(b)
    print(f"\n  By confidence:")
    for bucket in sorted(by_conf.keys()):
        bs  = by_conf[bucket]
        wr  = sum(1 for b in bs if b["won"]) / len(bs)
        print(f"    {bucket:<12} {len(bs):>4} bets  {wr*100:.1f}% win")
    print(f"{'='*60}")


def simulate_filtered_trades():
    """
    Simulate trade win rate with all filters applied:
    1. ELO + Ratings must agree
    2. Model confidence > 75% (bet_prob > 0.55)
    3. Dead zone excluded (65-75%)
    """
    from nba_model import (
        normalize_team, expected_score, rating_win_prob,
        recent_form_win_prob, rest_days, get_home_court,
        BASE_ELO, B2B_PENALTY, W_ELO, W_RATING, W_RECENT,
        build_elo_ratings, build_team_ratings, build_recent_form,
        build_pace_ratings, pace_win_prob_adjustment,
    )

    games = load_games()
    if not games:
        return

    dates = sorted(set(g["date"] for g in games))
    seen  = set()
    all_results = []

    for date in dates:
        games_before = [g for g in games if g["date"] < date]
        if len(games_before) < 100:
            continue

        elos    = build_elo_ratings(games_before)
        ratings = build_team_ratings(games_before)
        form    = build_recent_form(games_before, elos)
        pace    = build_pace_ratings(games_before)

        todays = [g for g in games if g["date"] == date]
        for g in todays:
            parts    = g["matchup"].split()
            opponent = normalize_team(parts[-1]) if parts else None
            if not opponent:
                continue
            if "vs." in g["matchup"]:
                home, away = g["team"], opponent
            else:
                away, home = g["team"], opponent

            key = "|".join(sorted([home, away])) + date
            if key in seen:
                continue
            seen.add(key)

            home_game = next((x for x in todays
                             if x["team"] == home and "vs." in x["matchup"]), None)
            if not home_game:
                continue

            home_won      = home_game["wl"] == "W"
            game_date_obj = datetime.date.fromisoformat(date)
            home_rest     = rest_days(home, games_before, game_date_obj)
            away_rest     = rest_days(away, games_before, game_date_obj)

            home_elo = elos.get(home, BASE_ELO)
            away_elo = elos.get(away, BASE_ELO)
            adj_home = home_elo + get_home_court(home)

            if home_rest == 1: adj_home -= B2B_PENALTY
            if home_rest == 0: adj_home -= B2B_PENALTY * 1.5
            if away_rest == 1: away_elo -= B2B_PENALTY * 1.2
            if away_rest == 0: away_elo -= B2B_PENALTY * 1.8

            p_elo    = expected_score(adj_home, away_elo)
            p_rating = rating_win_prob(home, away, ratings)
            p_recent = recent_form_win_prob(home, away, form)
            p_pace   = pace_win_prob_adjustment(home, away, pace)
            p_home   = max(0.05, min(0.95,
                       W_ELO * p_elo + W_RATING * p_rating + W_RECENT * p_recent + p_pace))

            all_results.append({
                "home_won":  home_won,
                "p_home":    p_home,
                "p_elo":     p_elo,
                "p_rating":  p_rating,
                "agree":     (p_elo > 0.5) == (p_rating > 0.5),
            })

    total = len(all_results)
    print(f"\n{'='*60}")
    print(f"  FILTERED TRADE SIMULATION")
    print(f"{'='*60}")

    scenarios = [
        ("No filter (baseline)",        lambda r: True),
        ("Agreement only",              lambda r: r["agree"]),
        ("Conf >75%",                   lambda r: max(r["p_home"], 1-r["p_home"]) > 0.75),
        ("Agreement + Conf >75%",       lambda r: r["agree"] and max(r["p_home"], 1-r["p_home"]) > 0.75),
        ("Agreement + No dead zone",    lambda r: r["agree"] and not (0.65 < max(r["p_home"], 1-r["p_home"]) < 0.75)),
        ("Agreement + Conf >75% (LIVE)",lambda r: r["agree"] and max(r["p_home"], 1-r["p_home"]) > 0.75),
    ]

    for label, filt in scenarios:
        subset = [r for r in all_results if filt(r)]
        if not subset:
            continue
        wins = sum(1 for r in subset if (r["p_home"] > 0.5) == r["home_won"])
        acc  = wins / len(subset)
        print(f"  {label:<35} {len(subset):>4} games  {acc*100:.1f}%")

    print(f"{'='*60}")



if __name__ == "__main__":
    run_audit()
    run_disagreement_audit()
    run_filtered_simulation()