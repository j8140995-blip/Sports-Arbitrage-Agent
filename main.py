"""
main.py
Sports prediction market bot — main entry point.

Usage:
  python main.py --nba              # NBA only
  python main.py --soccer           # soccer only
  python main.py --ncaam            # NCAAM only
  python main.py --fed              # Fed mentions
  python main.py --nba-mentions     # NBA announcer mentions
  python main.py --trump            # Trump mentions
  python main.py --summary          # P&L summary only
  python main.py --force            # force refresh cache
  python main.py --live             # live dry run
  python main.py --live --no-dry-run  # REAL orders
"""

import sys
import datetime
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Position sizing (quarter Kelly with 5% cap)
from position_sizing import size_bet, NOTIONAL_BANKROLL, MIN_EDGE


def main():
    args = set(sys.argv[1:])

    if "--summary" in args:
        from signal_engine import summarize_paper_trades
        summarize_paper_trades()
        return

    force     = "--force"       in args
    live_mode = "--live"        in args
    dry_run   = "--no-dry-run"  not in args

    if "--fed" in args or "--fed-only" in args:
        _run_fed(force); return

    if "--trump" in args or "--trump-only" in args:
        _run_trump(force); return

    if "--nba-mentions" in args or "--nba-mentions-only" in args:
        _run_nba_mentions(force); return

    if "--ncaam" in args or "--ncaam-only" in args:
        _run_ncaam(force, live_mode, dry_run); return

    if "--soccer" in args or "--soccer-only" in args:
        _run_soccer(force, live_mode, dry_run); return

    if "--mlb" in args or "--mlb-only" in args:
        _run_mlb(force, live_mode, dry_run); return

    if "--nba" in args or "--nba-only" in args:
        _run_nba(force, live_mode, dry_run); return

    print(__doc__)


def _print_header():
    import zoneinfo
    ct = datetime.datetime.now(zoneinfo.ZoneInfo("America/Chicago"))
    print(f"\n{'='*60}")
    print(f"  Sports Prediction Market Bot")
    print(f"  {ct.strftime('%Y-%m-%d %H:%M:%S')} CT  |  Mode: PAPER")
    print(f"  Sizing: Quarter-Kelly, 5% cap, ${NOTIONAL_BANKROLL:.0f} notional bankroll")
    print(f"{'='*60}\n")


def _run_nba(force, live_mode, dry_run):
    _print_header()
    print("[ NBA ] Loading model...")
    from nba_model import NBAModel
    m = NBAModel(); m.load(force_refresh=force)
    print("\n[ NBA ] Scanning Kalshi markets...")
    from kalshi_scanner import generate_signals, print_signals
    from signal_engine  import EDGE_THRESHOLD
    sigs = generate_signals(markets=None, nba_model=m, golf_model=None, edge_threshold=EDGE_THRESHOLD)
    print_signals(sigs)
    _execute_nba(sigs)


def _run_soccer(force, live_mode=False, dry_run=True):
    _print_header()
    print("[ SOCCER ] Loading model...")
    from soccer_model   import SoccerModel
    from soccer_scanner import generate_soccer_signals, print_soccer_signals
    m = SoccerModel(); m.load(force_refresh=force)
    print("\n[ SOCCER ] Scanning Kalshi markets...")
    sigs = generate_soccer_signals(m)
    print_soccer_signals(sigs)
    _execute_soccer(sigs)


def _run_ncaam(force, live_mode, dry_run):
    _print_header()
    print("[ NCAAM ] Loading model...")
    from ncaam_model import NCAAMModel
    m = NCAAMModel(); m.load(force_refresh=force)
    print("\n[ NCAAM ] Scanning Kalshi markets...")
    from ncaam_scanner import generate_ncaam_signals, print_ncaam_signals
    sigs = generate_ncaam_signals(m)
    print_ncaam_signals(sigs)
    _execute_ncaam(sigs)


def _run_fed(force):
    _print_header()
    print("[ FED ] Loading transcripts...")
    from fed_mentions import load_transcripts, scan_fed_mentions, print_fed_signals
    t = load_transcripts(force_refresh=force)
    print("\n[ FED ] Scanning Kalshi markets...")
    print_fed_signals(scan_fed_mentions(t))


def _run_trump(force):
    _print_header()
    print("[ TRUMP ] Scanning Kalshi markets via Claude API...")
    from trump_mentions import scan_trump_mentions, print_trump_signals, log_trump_trade
    signals = scan_trump_mentions(force_refresh=force)
    print_trump_signals(signals)
    logged = sum(1 for s in signals if log_trump_trade(s))
    if logged:
        print(f"\n[trump] Logged {logged} new trades to paper_trades.json")


def _run_nba_mentions(force):
    _print_header()
    print("[ NBA MENTIONS ] Loading data...")
    from nba_model    import NBAModel
    from nba_mentions import load_nba_mention_data, scan_nba_mentions, print_nba_mention_signals, log_mention_trade
    m = NBAModel(); m.load(force_refresh=force)
    d = load_nba_mention_data(force_refresh=force)
    print("\n[ NBA MENTIONS ] Scanning Kalshi markets...")
    signals = scan_nba_mentions(d, m)
    print_nba_mention_signals(signals)
    logged = sum(1 for s in signals if log_mention_trade(s, "NBA_MENTION"))
    if logged:
        print(f"\n[nba_mentions] Logged {logged} new trades to paper_trades.json")


# ─── Trade execution with quarter-Kelly sizing ────────────────────────────────

def _compute_entry_price(market_mid, side):
    """Entry price for YES (at ask) or NO (1 - mid + 1¢)."""
    if side == "yes":
        return min(market_mid + 0.01, 0.99)
    else:
        return min((1 - market_mid) + 0.01, 0.99)


def _log_trade(trades, trade_record):
    """Append a trade record to the trades list."""
    trades.append(trade_record)


def _execute_nba(signals):
    """NBA: use position_sizing and write directly to paper_trades.json."""
    print(f"\n[ TRADING ] Paper trading...")
    import datetime
    from signal_engine import PAPER_LOG, load_log, save_log, auto_mark_outcomes, summarize_paper_trades

    print("  Auto-resolving completed games...")
    auto_mark_outcomes()

    trades = load_log(PAPER_LOG)
    logged = 0

    for sig in signals:
        if any(t["ticker"] == sig.ticker for t in trades):
            print(f"  [skip] Already logged: {sig.ticker}")
            continue

        entry_price = _compute_entry_price(sig.market_mid, sig.side)
        sizing = size_bet(abs(sig.edge), entry_price)

        if sizing["skip"]:
            print(f"  [skip] {sig.ticker} — {sizing['reason']}")
            continue

        entry_cents = max(1, min(99, round(entry_price * 100)))
        trades.append({
            "ticker":            sig.ticker,
            "title":             getattr(sig, "title", ""),
            "sport":             "NBA",
            "side":              sig.side,
            "market_mid":        sig.market_mid,
            "model_prob":        sig.model_prob,
            "edge":              sig.edge,
            "kelly_fraction":    sizing["kelly_raw"],
            "quarter_kelly":     sizing["kelly_q"],
            "contracts":         sizing["contracts"],
            "cost_dollars":      sizing["dollars"],
            "entry_price_cents": entry_cents,
            "pct_bankroll":      sizing["pct_bank"],
            "timestamp":         datetime.datetime.utcnow().isoformat(),
            "mode":              "paper",
            "status":            "open",
            "outcome":           None,
            "pnl":               None,
            "game_date":         datetime.date.today().isoformat(),
            "home_team":         getattr(sig, "home_team", ""),
            "away_team":         getattr(sig, "away_team", ""),
        })
        logged += 1
        cap_flag = " ⚠️ CAPPED" if sizing["capped"] else ""
        print(f"\n  📊 NBA | {sig.ticker}")
        print(f"     Side: {sig.side.upper()}  Market: {sig.market_mid*100:.1f}%  "
              f"Model: {sig.model_prob*100:.1f}%  Edge: {sig.edge*100:+.1f}%")
        print(f"     Sizing: {sizing['contracts']} contracts @ {entry_cents}¢ = ${sizing['dollars']:.2f} "
              f"({sizing['pct_bank']:.1f}% of bankroll){cap_flag}")

    save_log(PAPER_LOG, trades)
    if logged:
        print(f"\n[nba] Logged {logged} NBA trades.")
    print("\n" + "="*60)
    summarize_paper_trades()


def _execute_ncaam(signals):
    """NCAAM: use position_sizing and write directly to paper_trades.json."""
    print(f"\n[ TRADING ] Paper trading...")
    import datetime
    from signal_engine import PAPER_LOG, load_log, save_log, auto_mark_outcomes, summarize_paper_trades

    print("  Auto-resolving completed games...")
    auto_mark_outcomes()

    trades = load_log(PAPER_LOG)
    logged = 0

    for sig in signals:
        if any(t["ticker"] == sig.ticker for t in trades):
            print(f"  [skip] Already logged: {sig.ticker}")
            continue

        entry_price = _compute_entry_price(sig.market_mid, sig.side)
        sizing = size_bet(abs(sig.edge), entry_price)

        if sizing["skip"]:
            print(f"  [skip] {sig.ticker} — {sizing['reason']}")
            continue

        entry_cents = max(1, min(99, round(entry_price * 100)))
        trades.append({
            "ticker":            sig.ticker,
            "title":             getattr(sig, "title", ""),
            "sport":             "NCAAM",
            "side":              sig.side,
            "market_mid":        sig.market_mid,
            "model_prob":        sig.model_prob,
            "edge":              sig.edge,
            "kelly_fraction":    sizing["kelly_raw"],
            "quarter_kelly":     sizing["kelly_q"],
            "contracts":         sizing["contracts"],
            "cost_dollars":      sizing["dollars"],
            "entry_price_cents": entry_cents,
            "pct_bankroll":      sizing["pct_bank"],
            "timestamp":         datetime.datetime.utcnow().isoformat(),
            "mode":              "paper",
            "status":            "open",
            "outcome":           None,
            "pnl":               None,
            "game_date":         datetime.date.today().isoformat(),
        })
        logged += 1
        cap_flag = " ⚠️ CAPPED" if sizing["capped"] else ""
        print(f"\n  📊 NCAAM | {sig.ticker}")
        print(f"     Side: {sig.side.upper()}  Edge: {sig.edge*100:+.1f}%")
        print(f"     Sizing: {sizing['contracts']} @ {entry_cents}¢ = ${sizing['dollars']:.2f}{cap_flag}")

    save_log(PAPER_LOG, trades)
    if logged:
        print(f"\n[ncaam] Logged {logged} NCAAM trades.")
    print("\n" + "="*60)
    summarize_paper_trades()


def _execute_soccer(signals):
    """Soccer: use position_sizing."""
    print(f"\n[ TRADING ] Paper trading...")
    import datetime
    from signal_engine import PAPER_LOG, load_log, save_log, auto_mark_outcomes, summarize_paper_trades

    print("  Auto-resolving completed games...")
    auto_mark_outcomes()

    trades = load_log(PAPER_LOG)
    logged = 0
    for sig in signals:
        if any(t["ticker"] == sig.ticker for t in trades):
            print(f"  [skip] Already logged: {sig.ticker}")
            continue

        entry_price = _compute_entry_price(sig.market_mid, sig.side)
        sizing = size_bet(abs(sig.edge), entry_price)

        if sizing["skip"]:
            print(f"  [skip] {sig.ticker} — {sizing['reason']}")
            continue

        entry_cents = max(1, min(99, round(entry_price * 100)))
        trades.append({
            "ticker":            sig.ticker,
            "title":             sig.title,
            "sport":             "SOCCER",
            "side":              sig.side,
            "market_mid":        sig.market_mid,
            "model_prob":        sig.model_prob,
            "edge":              sig.edge,
            "kelly_fraction":    sizing["kelly_raw"],
            "quarter_kelly":     sizing["kelly_q"],
            "contracts":         sizing["contracts"],
            "cost_dollars":      sizing["dollars"],
            "entry_price_cents": entry_cents,
            "pct_bankroll":      sizing["pct_bank"],
            "timestamp":         datetime.datetime.utcnow().isoformat(),
            "mode":              "paper",
            "status":            "open",
            "outcome":           None,
            "pnl":               None,
            "game_date":         datetime.date.today().isoformat(),
            "home_team":         sig.home_team,
            "away_team":         sig.away_team,
        })
        logged += 1
        cap_flag = " ⚠️ CAPPED" if sizing["capped"] else ""
        print(f"\n  📊 SOCCER | {sig.ticker}")
        print(f"     Side: {sig.side.upper()}  Market: {sig.market_mid*100:.1f}%  "
              f"Model: {sig.model_prob*100:.1f}%  Edge: {sig.edge*100:+.1f}%")
        print(f"     Sizing: {sizing['contracts']} contracts @ {entry_cents}¢ = ${sizing['dollars']:.2f} "
              f"({sizing['pct_bank']:.1f}% of bankroll){cap_flag}")

    save_log(PAPER_LOG, trades)
    if logged:
        print(f"\n[soccer] Logged {logged} soccer trades.")
    print("\n" + "="*60)
    summarize_paper_trades()


def _run_mlb(force, live_mode=False, dry_run=True):
    _print_header()
    print("[ MLB ] Loading model...")
    from mlb_model   import MLBModel
    from mlb_scanner import scan_mlb, print_mlb_signals
    m = MLBModel()
    m.load(force_refresh=force)
    print("\n[ MLB ] Scanning Kalshi markets...")
    signals = scan_mlb(m)
    print_mlb_signals(signals)

    import datetime
    from signal_engine import PAPER_LOG, load_log, save_log, summarize_paper_trades, auto_mark_outcomes
    print("\n[ TRADING ] Paper trading...")
    print("  Auto-resolving completed games...")
    auto_mark_outcomes()

    trades = load_log(PAPER_LOG)
    logged = 0
    for sig in signals:
        if any(t["ticker"] == sig.ticker for t in trades):
            print(f"  [skip] Already logged: {sig.ticker}")
            continue

        tbd_warn = ""
        if not sig.home_sp_known or not sig.away_sp_known:
            tbd_warn = "  ⚠️ TBD starter"

        entry_price = _compute_entry_price(sig.market_mid, sig.side)
        sizing = size_bet(abs(sig.edge), entry_price)

        if sizing["skip"]:
            print(f"  [skip] {sig.ticker} — {sizing['reason']}")
            continue

        entry_cents = max(1, min(99, round(entry_price * 100)))
        trades.append({
            "ticker":            sig.ticker,
            "title":             sig.title,
            "sport":             "MLB",
            "side":              sig.side,
            "market_mid":        sig.market_mid,
            "model_prob":        sig.model_prob,
            "consensus":         sig.consensus,
            "edge":              sig.edge,
            "kelly_fraction":    sizing["kelly_raw"],
            "quarter_kelly":     sizing["kelly_q"],
            "contracts":         sizing["contracts"],
            "cost_dollars":      sizing["dollars"],
            "entry_price_cents": entry_cents,
            "pct_bankroll":      sizing["pct_bank"],
            "timestamp":         datetime.datetime.utcnow().isoformat(),
            "mode":              "paper",
            "status":            "open",
            "outcome":           None,
            "pnl":               None,
            "game_date":         datetime.date.today().isoformat(),
            "home_team":         sig.home,
            "away_team":         sig.away,
            "market_source":     sig.market_source,
        })
        logged += 1

        def _fip_str(fip, raw_fip, ip, known):
            if not known:
                return f"FIP {fip:.2f}?"
            if ip > 0 and ip < 60 and abs(raw_fip - fip) >= 0.15:
                return f"FIP {fip:.2f} [raw {raw_fip:.2f}/{ip:.0f}ip]"
            return f"FIP {fip:.2f}"

        away_raw = getattr(sig, 'away_raw_fip', sig.away_fip)
        home_raw = getattr(sig, 'home_raw_fip', sig.home_fip)
        away_ip_ = getattr(sig, 'away_ip', 0.0)
        home_ip_ = getattr(sig, 'home_ip', 0.0)
        cap_flag = " ⚠️ CAPPED" if sizing["capped"] else ""

        print(f"\n  📊 MLB | {sig.ticker}{tbd_warn}")
        print(f"     {sig.away} @ {sig.home}  "
              f"SP: {sig.away_sp or 'TBD'} ({_fip_str(sig.away_fip, away_raw, away_ip_, sig.away_sp_known)}) "
              f"vs {sig.home_sp or 'TBD'} ({_fip_str(sig.home_fip, home_raw, home_ip_, sig.home_sp_known)})")
        print(f"     Side: {sig.side.upper()}  "
              f"[{sig.market_source}] Consensus: {sig.consensus*100:.1f}%  "
              f"Model: {sig.model_prob*100:.1f}%  Edge: {sig.edge*100:+.1f}%")
        print(f"     Bull ERA: {sig.away_bull_era:.2f}/{sig.home_bull_era:.2f}  "
              f"R/G: {sig.away_rpg:.1f}/{sig.home_rpg:.1f}  Park: {sig.park_factor:.2f}")
        print(f"     Sizing: {sizing['contracts']} contracts @ {entry_cents}¢ = ${sizing['dollars']:.2f} "
              f"({sizing['pct_bank']:.1f}% of bankroll){cap_flag}")

    save_log(PAPER_LOG, trades)
    if logged:
        print(f"\n[mlb] Logged {logged} MLB trades.")
    print("\n" + "="*60)
    summarize_paper_trades()


if __name__ == "__main__":
    main()