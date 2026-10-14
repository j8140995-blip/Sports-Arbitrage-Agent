"""
golf_model.py
Golf win-probability model using DataGolf Strokes Gained + Monte Carlo.
Requires: DATAGOLF_API_KEY in Replit Secrets
"""

import os
import json
import time
import random
import requests
from pathlib import Path
from typing import Optional

DATAGOLF_BASE   = "https://feeds.datagolf.com"
CACHE_FILE_SG   = Path("datagolf_cache.json")
CACHE_TTL_HOURS = 4
SIM_ROUNDS      = 10_000
ROUND_STD_DEV   = 3.5


def _dg_key() -> str:
    key = os.environ.get("DATAGOLF_API_KEY", "")
    if not key:
        print("[golf_model] WARNING: DATAGOLF_API_KEY not set.")
    return key


def dg_get(endpoint: str, params: dict = None) -> dict:
    p = params or {}
    p["key"] = _dg_key()
    p["file_format"] = "json"
    r = requests.get(f"{DATAGOLF_BASE}/{endpoint}", params=p, timeout=15)
    r.raise_for_status()
    return r.json()


def get_rankings() -> list[dict]:
    return dg_get("pga-tour/get-dg-rankings").get("rankings", [])


def get_field(tour: str = "pga") -> list[dict]:
    return dg_get("field-updates", {"tour": tour}).get("field", [])


def _load_sg_cache() -> Optional[dict]:
    if CACHE_FILE_SG.exists():
        data = json.loads(CACHE_FILE_SG.read_text())
        if (time.time() - data.get("ts", 0)) / 3600 < CACHE_TTL_HOURS:
            return data
    return None


def _save_sg_cache(data: dict):
    data["ts"] = time.time()
    CACHE_FILE_SG.write_text(json.dumps(data, indent=2))


def _normalize_name(name: str) -> str:
    return name.lower().replace(".", "").replace("-", " ").strip()


def simulate_tournament(players: list[dict], n_sims: int = SIM_ROUNDS) -> dict[str, float]:
    if not players:
        return {}

    sg_values = [p["sg_total"] for p in players if p.get("sg_total") is not None]
    if not sg_values:
        return {}

    field_avg = sum(sg_values) / len(sg_values)
    for p in players:
        if p.get("sg_total") is None:
            p["sg_total"] = 0.0
        p["sg_adj"] = p["sg_total"] - field_avg

    wins = {p["player_name"]: 0 for p in players}

    for _ in range(n_sims):
        scores = {}
        for p in players:
            mean_score = -p["sg_adj"] * 4
            scores[p["player_name"]] = sum(
                random.gauss(mean_score / 4, ROUND_STD_DEV) for _ in range(4)
            )
        wins[min(scores, key=scores.get)] += 1

    return {name: count / n_sims for name, count in wins.items()}


class GolfModel:
    def __init__(self):
        self.players: list[dict] = []
        self.win_probs: dict[str, float] = {}
        self._loaded = False

    def load(self, force_refresh: bool = False):
        cache = None if force_refresh else _load_sg_cache()

        if cache:
            self.players   = cache["players"]
            self.win_probs = cache["win_probs"]
            print(f"[GolfModel] Loaded {len(self.players)} players from cache.")
        else:
            print("[GolfModel] Fetching field + rankings from DataGolf...")
            try:
                field    = get_field()
                rankings = get_rankings()
            except Exception as e:
                print(f"[GolfModel] API error: {e}")
                return

            sg_lookup = {
                _normalize_name(r.get("player_name", "")): r.get("sg_total", 0.0)
                for r in rankings
            }

            self.players = []
            for f in field:
                name = f.get("player_name", "")
                self.players.append({
                    "player_name": name,
                    "sg_total":    sg_lookup.get(_normalize_name(name), 0.0),
                    "dg_id":       f.get("dg_id"),
                })

            if not self.players:
                print("[GolfModel] No field data — check API key and whether a tournament is active.")
                return

            print(f"[GolfModel] Simulating {SIM_ROUNDS:,} tournaments for {len(self.players)} players...")
            self.win_probs = simulate_tournament(self.players, SIM_ROUNDS)
            _save_sg_cache({"players": self.players, "win_probs": self.win_probs})
            print("[GolfModel] Simulation complete.")

        self._loaded = True

    def win_probability(self, player_name: str) -> Optional[float]:
        if player_name in self.win_probs:
            return self.win_probs[player_name]

        target = _normalize_name(player_name)
        for name, prob in self.win_probs.items():
            if _normalize_name(name) == target:
                return prob

        last = target.split()[-1]
        matches = [(n, p) for n, p in self.win_probs.items() if last in _normalize_name(n)]
        if len(matches) == 1:
            return matches[0][1]

        return None

    def top_players(self, n: int = 10) -> list[tuple[str, float]]:
        return sorted(self.win_probs.items(), key=lambda x: x[1], reverse=True)[:n]


if __name__ == "__main__":
    model = GolfModel()
    model.load()
    print("\n=== Top 10 Win Probabilities ===")
    for player, prob in model.top_players(10):
        print(f"  {player:<30} {prob*100:.2f}%")