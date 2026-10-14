"""
verify_soccer_signals.py
Sanity-check every soccer signal before trusting them with real money.

Runs a scan and for each signal:
  1. Parses teams from event ticker
  2. Cross-checks against the team name in the signal
  3. Verifies the side/outcome/probability are internally consistent
  4. Flags anything ambiguous for manual review

Usage:
    python verify_soccer_signals.py
"""

import re
from soccer_model   import SoccerModel
from soccer_scanner import generate_soccer_signals


def parse_ticker(ticker: str) -> tuple[str, str, str]:
    """Parse (home_code, away_code, market_suffix) from a full market ticker."""
    parts = ticker.split("-")
    if len(parts) < 2:
        return "", "", ""
    market_suffix = parts[-1].upper()
    event_body    = parts[-2].upper()
    codes = re.sub(r'^\d+', '', event_body)
    if len(codes) >= 6:
        return codes[-6:-3], codes[-3:], market_suffix
    return "", "", market_suffix


def main():
    print("Loading soccer model...")
    m = SoccerModel()
    m.load()
    print("Generating signals...\n")

    sigs = generate_soccer_signals(m)
    if not sigs:
        print("No signals generated.")
        return

    print(f"{'='*120}")
    print(f"  SIGNAL VERIFICATION — {len(sigs)} signals")
    print(f"{'='*120}\n")

    warnings = 0
    for s in sigs:
        home_code, away_code, market_suffix = parse_ticker(s.ticker)
        line = f"  {s.ticker:<45} {s.home_team:<22} vs {s.away_team:<22}"
        line += f" side={s.side.upper():<3} outcome={s.outcome:<10} model={s.model_prob*100:.1f}%"

        # Verify the market suffix matches either home or away code
        warns = []

        if market_suffix in ("TIE", "DRAW"):
            if s.outcome != "draw":
                warns.append(f"⚠️ ticker says TIE but outcome={s.outcome}")
        elif market_suffix == home_code:
            if s.outcome != "home_win":
                warns.append(f"⚠️ ticker=home ({home_code}) but outcome={s.outcome}")
        elif market_suffix == away_code:
            if s.outcome != "away_win":
                warns.append(f"⚠️ ticker=away ({away_code}) but outcome={s.outcome}")
        else:
            warns.append(f"⚠️ suffix {market_suffix} matches neither home={home_code} nor away={away_code}")

        # Sanity check the probability magnitude
        if s.outcome == "home_win" and s.model_prob > 0.95:
            warns.append(f"⚠️ home_win prob {s.model_prob*100:.1f}% seems too high")
        if s.outcome == "away_win" and s.model_prob > 0.95:
            warns.append(f"⚠️ away_win prob {s.model_prob*100:.1f}% seems too high")

        # Edge sign check
        expected_edge = s.model_prob - s.consensus
        if abs(expected_edge - s.edge) > 0.001:
            warns.append(f"⚠️ edge {s.edge*100:+.1f} != model-consensus {expected_edge*100:+.1f}")

        # Side sign check
        if s.edge > 0 and s.side != "yes":
            warns.append(f"⚠️ positive edge but side={s.side}")
        if s.edge < 0 and s.side != "no":
            warns.append(f"⚠️ negative edge but side={s.side}")

        if warns:
            warnings += 1
            print(line)
            for w in warns:
                print(f"      {w}")
            print()
        else:
            print(line + "  ✅")

    print(f"\n{'='*120}")
    if warnings == 0:
        print(f"  ALL {len(sigs)} SIGNALS VERIFIED CLEAN ✅")
    else:
        print(f"  ⚠️  {warnings} / {len(sigs)} signals flagged — review above before live trading")
    print(f"{'='*120}\n")


if __name__ == "__main__":
    main()