"""
signal_engine.py
Kelly Criterion sizing, paper trading, live order execution,
and automatic outcome tracking.

Improvements:
- Duplicate prevention: never logs the same ticker twice
- Exact P&L: calculated from actual entry price, not estimates
- Auto outcome marking: fetches completed games and resolves open trades
"""

import json
import time
import requests
import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import kalshi_client as kc
from kalshi_scanner import MarketSignal

PAPER_LOG      = Path("paper_trades.json")
LIVE_LOG       = Path("live_trades.json")
BALANCE        = 50.0
MAX_PCT        = 0.03   # hard cap 3% per trade (report: never exceed 3%)
HALF_KELLY     = False
QUARTER_KELLY  = True   # report: quarter Kelly survives in practice
MIN_CONTRACTS  = 1
EDGE_THRESHOLD = 0.08


@dataclass
class TradeDecision:
    ticker:            str
    title:             str
    sport:             str
    side:              str
    market_mid:        float
    model_prob:        float
    edge:              float
    kelly_fraction:    float
    half_kelly:        float
    contracts:         int
    cost_dollars:      float
    entry_price_cents: int
    timestamp:         str
    mode:              str
    status:            str = "pending"
    outcome:           Optional[str] = None
    pnl:               Optional[float] = None
    game_date:         Optional[str] = None
    home_team:         Optional[str] = None
    away_team:         Optional[str] = None


def load_log(path: Path) -> list[dict]:
    if path.exists():
        return json.loads(path.read_text())
    return []


def save_log(path: Path, trades: list[dict]):
    path.write_text(json.dumps(trades, indent=2))


def exact_pnl(outcome: str, entry_price_cents: int, contracts: int) -> float:
    if outcome == "win":
        return round((100 - entry_price_cents) * contracts / 100, 2)
    else:
        return round(-entry_price_cents * contracts / 100, 2)


def kelly_fraction(model_p: float, entry_price: float) -> float:
    if entry_price <= 0 or entry_price >= 1:
        return 0.0
    b = (1 - entry_price) / entry_price
    q = 1 - model_p
    f = (b * model_p - q) / b
    return max(0.0, f)


def size_trade(signal: MarketSignal, balance: float) -> TradeDecision:
    if signal.side == "yes":
        entry_price = min(signal.market_mid + 0.01, 0.99)
        model_p     = signal.model_prob
    else:
        entry_price = min((1 - signal.market_mid) + 0.01, 0.99)
        model_p     = 1 - signal.model_prob

    fk       = kelly_fraction(model_p, entry_price)
    hk       = fk * 0.25 if QUARTER_KELLY else (fk / 2 if HALF_KELLY else fk)
    fraction = min(hk, MAX_PCT)

    contracts   = max(MIN_CONTRACTS, int(balance * fraction))
    entry_cents = max(1, min(99, round(entry_price * 100)))
    cost        = contracts * entry_cents / 100

    notes     = signal.notes or ""
    home_team = None
    away_team = None
    if " at " in notes:
        parts     = notes.split(" at ")
        away_team = parts[0].strip()
        home_team = parts[1].strip()

    return TradeDecision(
        ticker=signal.ticker,
        title=signal.title,
        sport=signal.sport,
        side=signal.side,
        market_mid=signal.market_mid,
        model_prob=signal.model_prob,
        edge=signal.edge,
        kelly_fraction=round(fk, 4),
        half_kelly=round(hk, 4),
        contracts=contracts,
        cost_dollars=round(cost, 2),
        entry_price_cents=entry_cents,
        timestamp=datetime.datetime.utcnow().isoformat(),
        mode="paper",
        status="pending",
        game_date=datetime.date.today().isoformat(),
        home_team=home_team,
        away_team=away_team,
    )


def _print_trade(d: TradeDecision):
    print(f"\n  {'📊' if d.mode == 'paper' else '💰'} {d.sport} | {d.ticker}")
    print(f"     Title:      {d.title[:60]}")
    print(f"     Side:       {d.side.upper()}")
    print(f"     Market mid: {d.market_mid*100:.1f}%  →  Model: {d.model_prob*100:.1f}%  Edge: {d.edge*100:+.1f}%")
    print(f"     Kelly f*:   {d.kelly_fraction*100:.1f}%")
    print(f"     Contracts:  {d.contracts}  @  {d.entry_price_cents}¢  =  ${d.cost_dollars:.2f}")


def paper_trade(signals: list[MarketSignal], balance: float = BALANCE) -> list[TradeDecision]:
    trades    = load_log(PAPER_LOG)
    decisions = []

    for sig in signals:
        if abs(sig.edge) < EDGE_THRESHOLD:
            continue

        # Duplicate prevention
        if any(t["ticker"] == sig.ticker for t in trades):
            print(f"  [skip] Already logged: {sig.ticker}")
            continue

        decision        = size_trade(sig, balance)
        decision.mode   = "paper"
        decision.status = "open"
        trades.append(asdict(decision))
        decisions.append(decision)
        _print_trade(decision)

    save_log(PAPER_LOG, trades)
    print(f"\n[paper] Logged {len(decisions)} new trades → {PAPER_LOG}")
    return decisions


def live_trade(signals: list[MarketSignal], balance: float = None,
               dry_run: bool = True) -> list[TradeDecision]:
    if balance is None:
        balance = kc.get_balance()
        print(f"[live] Current balance: ${balance:.2f}")

    trades    = load_log(LIVE_LOG)
    decisions = []

    for sig in signals:
        if abs(sig.edge) < EDGE_THRESHOLD:
            continue

        if any(t["ticker"] == sig.ticker for t in trades):
            print(f"  [skip] Already logged: {sig.ticker}")
            continue

        decision      = size_trade(sig, balance)
        decision.mode = "live"

        if dry_run:
            decision.status = "dry_run"
            print(f"\n[DRY RUN] Would submit:")
            _print_trade(decision)
        else:
            try:
                # Maker-first execution: place limit order 1 cent better than mid
                # Report: makers earn ~1.3% more than takers on 50/50 markets
                if decision.side == "yes":
                    limit_cents = max(1, decision.entry_price_cents - 1)
                else:
                    limit_cents = min(99, decision.entry_price_cents + 1)

                resp = kc.place_order(
                    ticker=decision.ticker,
                    side=decision.side,
                    count=decision.contracts,
                    price_cents=limit_cents,
                    order_type="limit",   # maker order
                )
                decision.status = "open"
                decision.entry_price_cents = limit_cents
                print(f"[live] ✅ {decision.ticker} {decision.side} ×{decision.contracts} @ {limit_cents}¢ (limit/maker)")
            except Exception as e:
                decision.status = "error"
                decision.outcome = str(e)
                print(f"[live] ❌ {decision.ticker} — {e}")

        trades.append(asdict(decision))
        decisions.append(decision)
        time.sleep(0.5)

    save_log(LIVE_LOG, trades)
    return decisions


# ── Auto outcome marking ──────────────────────────────────────────────────────

def fetch_nba_results() -> dict[str, str]:
    try:
        resp = requests.get(
            "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
            timeout=10
        )
        data    = resp.json()
        results = {}

        for event in data.get("events", []):
            status = event.get("status", {}).get("type", {}).get("name", "")
            if status != "STATUS_FINAL":
                continue

            competitions = event.get("competitions", [])
            if not competitions:
                continue

            comp        = competitions[0]
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

            winner = home_name if home_score > away_score else away_name
            results[f"{away_name}_{home_name}"] = winner

        return results
    except Exception as e:
        print(f"[auto] Failed to fetch results: {e}")
        return {}


def auto_mark_outcomes(log_path: Path = PAPER_LOG):
    trades      = load_log(log_path)
    open_trades = [t for t in trades if t.get("status") == "open" and t.get("outcome") is None]

    if not open_trades:
        return

    results = fetch_nba_results()
    if not results:
        print("[auto] No results available yet.")
        return

    updated = 0
    for t in trades:
        if t.get("status") != "open" or t.get("outcome") is not None:
            continue

        home = t.get("home_team", "")
        away = t.get("away_team", "")
        if not home or not away:
            continue

        winner = results.get(f"{away}_{home}")
        if not winner:
            continue

        ticker    = t["ticker"]
        side      = t["side"]
        suffix    = ticker.split("-")[-1].upper()
        home_abbr = home.split()[-1][:3].upper()

        ticker_team = home if (suffix in home.upper() or home_abbr in suffix) else away

        if side == "yes":
            our_team_won = (winner == ticker_team)
        else:
            our_team_won = (winner != ticker_team)

        outcome = "win" if our_team_won else "loss"
        pnl     = exact_pnl(outcome, t["entry_price_cents"], t["contracts"])

        t["outcome"] = outcome
        t["pnl"]     = pnl
        t["status"]  = "settled"
        updated += 1

        icon = "✅" if outcome == "win" else "❌"
        print(f"  {icon} {ticker} → {outcome.upper()}  P&L: ${pnl:+.2f}  (winner: {winner})")

    if updated:
        save_log(log_path, trades)
        print(f"[auto] Resolved {updated} trade(s).")
    else:
        print("[auto] No new results to resolve yet.")


def mark_outcome(ticker: str, outcome: str, pnl: float, log_path: Path = PAPER_LOG):
    """Manual outcome marking — fallback if auto fails."""
    trades  = load_log(log_path)
    updated = 0
    for t in trades:
        if t["ticker"] == ticker and t.get("outcome") is None:
            t["outcome"] = outcome
            t["pnl"]     = pnl
            t["status"]  = "settled"
            updated += 1
    save_log(log_path, trades)
    print(f"[tracking] Marked {updated} trade(s) for {ticker}: {outcome} ${pnl:+.2f}")


def summarize_paper_trades() -> dict:
    trades = load_log(PAPER_LOG)
    if not trades:
        print("[paper] No trades logged yet.")
        return {}

    # ── Per-category breakdown ─────────────────────────────────────────────────
    categories = {}
    for t in trades:
        sport = (t.get("sport") or "UNKNOWN").upper()
        if sport not in categories:
            categories[sport] = {"trades": [], "resolved": [], "wins": [], "pnl": 0.0, "wagered": 0.0}
        categories[sport]["trades"].append(t)
        if t.get("outcome") in ("win", "loss"):
            categories[sport]["resolved"].append(t)
            categories[sport]["pnl"]     += t.get("pnl", 0) or 0
            categories[sport]["wagered"] += t.get("cost_dollars", 0) or 0
            if t["outcome"] == "win":
                categories[sport]["wins"].append(t)

    # ── Overall totals ─────────────────────────────────────────────────────────
    all_resolved = [t for t in trades if t.get("outcome") in ("win", "loss")]
    all_wins     = [t for t in all_resolved if t["outcome"] == "win"]
    total_pnl    = sum(t.get("pnl", 0) or 0 for t in all_resolved)
    total_wagered= sum(t.get("cost_dollars", 0) for t in all_resolved)
    roi          = total_pnl / total_wagered if total_wagered else 0

    print(f"\n{'='*62}")
    print(f"  PAPER TRADING SUMMARY — BY CATEGORY")
    print(f"{'='*62}")
    print(f"  {'CATEGORY':<18} {'TRADES':>6} {'RESOL':>6} {'WIN%':>6} {'P&L':>8} {'ROI':>7}")
    print(f"  {'-'*58}")

    # Sort by number of resolved trades desc
    for sport, data in sorted(categories.items(),
                               key=lambda x: len(x[1]["resolved"]), reverse=True):
        n_trades   = len(data["trades"])
        n_resolved = len(data["resolved"])
        n_wins     = len(data["wins"])
        pnl        = data["pnl"]
        wagered    = data["wagered"]
        win_rate   = f"{n_wins/n_resolved*100:.0f}%" if n_resolved else "n/a"
        roi_str    = f"{pnl/wagered*100:.0f}%" if wagered else "n/a"
        pnl_str    = f"${pnl:+.2f}"
        print(f"  {sport:<18} {n_trades:>6} {n_resolved:>6} {win_rate:>6} {pnl_str:>8} {roi_str:>7}")

    print(f"  {'-'*58}")
    overall_wr = f"{len(all_wins)/len(all_resolved)*100:.0f}%" if all_resolved else "n/a"
    overall_roi = f"{roi*100:.0f}%" if total_wagered else "n/a"
    print(f"  {'TOTAL':<18} {len(trades):>6} {len(all_resolved):>6} {overall_wr:>6} ${total_pnl:>+.2f} {overall_roi:>7}")
    print(f"{'='*62}")

    # CLV report
    try:
        from clv_tracker import update_clv_for_resolved_trades, print_clv_report
        update_clv_for_resolved_trades()
        trades = load_log(PAPER_LOG)  # reload after CLV update
        print_clv_report(trades)
    except Exception as e:
        pass

    return {"total": len(trades), "resolved": len(all_resolved), "pnl": total_pnl, "roi": roi}

# ── Patch size_trade to handle new MarketSignal fields ────────────────────────
# The consensus_prob field is used for edge calculation when Vegas is available
_original_size_trade = size_trade

def size_trade(signal, balance: float) -> TradeDecision:
    # Use consensus_prob for entry price if available
    if hasattr(signal, 'consensus_prob') and signal.consensus_prob > 0:
        if signal.side == "yes":
            entry_price = min(signal.consensus_prob + 0.01, 0.99)
            model_p     = signal.model_prob
        else:
            entry_price = min((1 - signal.consensus_prob) + 0.01, 0.99)
            model_p     = 1 - signal.model_prob

        from nba_model import normalize_team
        notes     = signal.notes or ""
        home_team = None
        away_team = None
        if " at " in notes:
            parts     = notes.split(" at ")
            away_team = parts[0].strip()
            home_team = parts[1].strip()

        import datetime
        fk       = kelly_fraction(model_p, entry_price)
        hk       = fk * 0.25 if QUARTER_KELLY else (fk / 2 if HALF_KELLY else fk)
        fraction = min(hk, MAX_PCT)
        contracts   = max(MIN_CONTRACTS, int(balance * fraction))
        entry_cents = max(1, min(99, round(entry_price * 100)))
        cost        = contracts * entry_cents / 100

        return TradeDecision(
            ticker=signal.ticker,
            title=signal.title,
            sport=signal.sport,
            side=signal.side,
            market_mid=signal.market_mid,
            model_prob=signal.model_prob,
            edge=signal.edge,
            kelly_fraction=round(fk, 4),
            half_kelly=round(hk, 4),
            contracts=contracts,
            cost_dollars=round(cost, 2),
            entry_price_cents=entry_cents,
            timestamp=datetime.datetime.utcnow().isoformat(),
            mode="paper",
            status="pending",
            game_date=datetime.date.today().isoformat(),
            home_team=home_team,
            away_team=away_team,
        )
    return _original_size_trade(signal, balance)