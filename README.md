# Kalshi Sports Prediction Bot

A paper-trading bot that scans [Kalshi](https://kalshi.com) sports markets, forecasts win probabilities with calibrated models, and logs trades where the model disagrees with Vegas consensus by a meaningful margin. Covers NBA, MLB, and global soccer.

The bot is designed as a research tool: it does not place real money. All P&L and CLV figures reported in logs come from simulated bets against actual Kalshi order books, resolved against real game outcomes.
To see files, go to master branch
---

## Current performance

Across 220 resolved trades on the three active sport models:

| Sport  | Trades | Win % | P&L     | ROI  | CLV     |
|--------|--------|-------|---------|------|---------|
| NBA    | 77     | 66%   | -$3.03  | -3%  | +6.10%  |
| Soccer | 72     | 64%   | +$11.62 | +7%  | +6.86%  |
| MLB    | 71     | 48%   | +$10.20 | +9%  | +1.34%  |

CLV (Closing Line Value) is the primary edge metric. +5% is the "excellent" threshold; +1-2% is "strong edge."

Legacy categories (Trump mentions, NBA mentions, NCAAM) are paused and excluded from active evaluation.

---

## Architecture

Each sport has three parts: a **model** that estimates win probability, a **scanner** that compares the model to Kalshi and Vegas prices, and a shared **paper trading** layer that logs trades, sizes positions, and resolves outcomes.

```
┌─────────────────────────────────────────────────────────────┐
│                        main.py                              │
│   CLI entry point — routes to --nba, --mlb, --soccer        │
└─────────────────────────────────────────────────────────────┘
         │                 │                   │
┌────────▼─────────┐ ┌─────▼──────────┐ ┌──────▼─────────┐
│  nba_model.py    │ │  mlb_model.py  │ │ soccer_model.py│
│  Elo + DARKO +   │ │  Elo + FIP +   │ │ xG Poisson +   │
│  injury adj      │ │  bullpen/park  │ │ league-specific│
└────────┬─────────┘ └─────┬──────────┘ └──────┬─────────┘
         │                 │                   │
┌────────▼─────────┐ ┌─────▼──────────┐ ┌──────▼─────────┐
│ kalshi_scanner.py│ │ mlb_scanner.py │ │soccer_scanner  │
│ Vegas + injuries │ │ Vegas + TBD    │ │ xG + league    │
│ 75% conf filter  │ │ pitcher checks │ │ edge thresholds│
└────────┬─────────┘ └─────┬──────────┘ └──────┬─────────┘
         │                 │                   │
         └─────────────────┼───────────────────┘
                           │
         ┌─────────────────▼──────────────────┐
         │          Shared infrastructure     │
         │  position_sizing (quarter-Kelly)   │
         │  live_game_filter                  │
         │  resolve_trades / paper_trades.json│
         └────────────────────────────────────┘
```

---

## Models

### NBA

Elo-based model with three probability components:

- **p_elo** (30% weight): Classical Elo with K=20, home advantage, and recent-form adjustment
- **p_rating** (50% weight): Net-rating differential projected to win probability
- **p_recent** (20% weight): Last-10 games weighted average

**DARKO integration** provides player-level impact (DPM) via the `nba_api` package. Injuries are quantified by converting lost DPM into Elo-equivalent penalty, scaled by the player's status (Out = 100%, Doubtful = 75%, Day-to-Day = 50%, Questionable = 25%). Superstars (DPM ≥ 4.0) get a 1.6x impact multiplier and +0.25 status weight boost.

Signals require ≥ 75% bet-side probability — a filter empirically justified by CLV-by-probability-bucket analysis showing signals under 75% confidence historically ran -31% ROI.

### MLB

Elo (K=6) blended with pitcher FIP matchups, bullpen ERA, team offense, and park factors. Pitcher FIP is the dominant signal — contributes 100-170 Elo points per matchup versus 20-40 from team strength.

**FIP calculation uses Bayesian shrinkage with two-level regression:**

1. Each pitcher's 2025 prior FIP is first regressed toward league average (4.15) using prior IP as weight. A 14 IP sample gets heavily pulled toward 4.15; a 162 IP sample barely moves.
2. Then the pitcher's current 2026 FIP is regressed toward the stabilized prior using current IP as weight.

This prevents small-sample statistics (14-IP rookie with 2.4 FIP) from being treated as true-talent estimates. League-average (4.15) is used as the default for TBD pitchers rather than the pessimistic "replacement level" (5.00) default — a change that eliminated artificial +20%+ edges on games with unknown starters.

Pitcher database loads **873+ pitchers** per season via `playerPool=ALL` on the MLB Stats API (default returns only ~52 qualified starters — a critical fix made in April 2026 after the model was found to be treating most of the league as unknown).

### Soccer

xG-based Poisson model with Dixon-Coles correction (ρ = -0.13 for low-score dependence) and league-specific priors:

- **Team strength**: Exponentially weighted moving average of attack/defense with 15-match half-life
- **Home advantage**: Varies by league (EPL 0.25, La Liga 0.30, MLS 0.35, etc.)
- **Edge thresholds**: Scaled by league efficiency — EPL requires +15% edge, La Liga/Bundesliga +8%, MLS +8%, Serie A +9%

Covers English Premier League, La Liga, Bundesliga, Champions League, MLS, Ligue 1, Serie A, FA Cup, Europa League.

---

## Position sizing

**Quarter-Kelly** with hard caps:

- Max 5% of notional bankroll per bet
- Minimum 5% edge required
- Shared `position_sizing.py` module across all sports
- $100 notional bankroll (decoupled from real Kalshi balance)

The quarter-Kelly fraction = `edge / (price × (1 - price)) × 0.25`. The 5% cap prevents extreme-price bets (model says 97%, Vegas says 80%) from eating a disproportionate share of bankroll.

---

## Daily workflow

```bash
# Scan all three sports and log signals
python main.py --nba --force && python main.py --mlb && python main.py --soccer

# Resolve yesterday's games and show summary
python resolve_trades.py && python main.py --summary

# Inspect what's currently open
python today_trades.py

# Verify MLB starters against ESPN (sanity check)
python verify_starters.py

# See CLV broken down by model confidence
python clv_by_prob_bucket.py
```

To rebuild all caches (useful after bug fixes or stale data):

```bash
rm -f mlb_model_cache.json darko_cache.json nba_elo_cache.json soccer_model_cache.json today_odds_cache.json
```

---

## Data sources

- **Kalshi API** — market prices, order books, volumes
- **The Odds API** — Vegas consensus lines across major sportsbooks
- **ESPN** — soccer match results and MLB game outcomes
- **MLB Stats API** (`statsapi`) — pitcher stats, bullpen ERA, team runs
- **NBA Stats API** (`nba_api`) — team stats, DARKO/EPM approximations, injury reports
- **FBref / Understat** — xG data (via scraped endpoints)

---

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

Create a `.env` with:

```
KALSHI_EMAIL=your_email
KALSHI_PASSWORD=your_password
ODDS_API_KEY=your_odds_api_key
```

Then run the daily workflow above.

---

## Key fixes and lessons

The bot has undergone several structural fixes that each materially improved CLV. Documenting them because the pattern is instructive:

1. **DARKO cache structure bug** — Cache stored `{"players": {...}}` but `injury_elo_adjustment` was reading top-level keys. Every NBA injury was silently falling through to a heuristic fallback for weeks.

2. **MLB pitcher database truncation** — `statsapi.get("stats")` returns only ~52 qualified starters without `playerPool="ALL"`. The model was treating 90% of the league as unknown TBD pitchers with inflated FIP 5.00 defaults.

3. **Soccer substring matching** — Team code matching used `.split()[0] in ticker`, causing "Real" (Real Oviedo) to match "Villarreal". Fixed with exact-abbreviation parsing from ticker codes.

4. **Small-sample FIP priors** — Prior FIPs from 14 IP samples were being used as true-talent estimates without regression. Added two-level Bayesian shrinkage.

5. **Live-game filter** — Model was placing bets on games already in progress. Shared `live_game_filter.py` now blocks any game with first pitch/kickoff more than 0 minutes ago.

The common thread: silent miscalibration, not crashes. The bot kept running and reporting P&L; the signals were just subtly wrong. CLV tracking per sport is what surfaced these — a model with real edge should produce positive CLV even when losing streaks happen.

---

## Why CLV matters more than P&L

Short-run P&L is dominated by variance. On a +10% edge bet, you lose 45% of the time. On 50 bets, your realized P&L has a 2x standard deviation versus expectation.

CLV is the signal underneath the noise. If your model is right, the market should move toward your prediction after you bet (because sharper money arrives after you). Positive CLV across hundreds of bets is the only reliable indicator of edge.

The bot targets +3-7% CLV per sport. +1-2% is the floor for calling an edge real; anything below is noise or model miscalibration.

---

## Status

Currently running in paper mode. Three active sport models all above +1% CLV, two above +5%. Decision to transition to real money will come after 500+ post-fix trades and 30+ days of stable positive CLV. Mention markets (Trump, NBA, NCAAM) are paused and will not be reactivated without model redesign.

---

## License

MIT. Use at your own risk. This is a research project, not financial advice.
