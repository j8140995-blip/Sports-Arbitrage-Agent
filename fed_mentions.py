"""
fed_mentions.py
Model for Federal Reserve press conference mention markets.

Scrapes all Powell press conference transcripts from federalreserve.gov,
calculates word frequency per conference, then scores current Kalshi markets.

Data source: federalreserve.gov/monetarypolicy/fomccalendars.htm (free, public)
"""

import re
import json
import time
import requests
from pathlib import Path
from collections import defaultdict

CACHE_FILE = Path("fed_transcripts_cache.json")
CACHE_TTL  = 7 * 24 * 3600  # 1 week — transcripts don't change

FOMC_BASE  = "https://www.federalreserve.gov"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,*/*",
}


# ── Transcript fetching ───────────────────────────────────────────────────────

def get_fomc_pdf_urls() -> list[str]:
    """
    Generate PDF URLs for all Powell-era FOMC press conferences.
    PDFs are at: federalreserve.gov/mediacenter/files/FOMCpresconf{YYYYMMDD}.pdf
    Powell became chair Feb 2018. Conferences happen ~8x/year at scheduled FOMC dates.
    """
    # All known FOMC press conference dates (Powell era 2018-2026)
    dates = [
        # 2018
        "20180321","20180613","20180926","20181219",
        # 2019
        "20190130","20190320","20190501","20190619","20190731","20190918","20191030","20191211",
        # 2020
        "20200129","20200303","20200315","20200429","20200610","20200729","20200916","20201105","20201216",
        # 2021
        "20210127","20210317","20210428","20210616","20210728","20210922","20211103","20211215",
        # 2022
        "20220126","20220316","20220504","20220615","20220727","20220921","20221102","20221214",
        # 2023
        "20230201","20230322","20230503","20230614","20230726","20230920","20231101","20231213",
        # 2024
        "20240131","20240320","20240501","20240612","20240731","20240918","20241107","20241218",
        # 2025
        "20250129","20250319","20250507","20250618","20250730","20250917","20251029","20251210",
        # 2026
        "20260129","20260319",
    ]
    urls = []
    for d in dates:
        # Try both capitalization variants
        urls.append(f"{FOMC_BASE}/mediacenter/files/FOMCpresconf{d}.pdf")
        urls.append(f"{FOMC_BASE}/mediacenter/files/fomcpresconf{d}.pdf")
    return urls


def fetch_transcript_text(url: str) -> str:
    """Fetch and extract text from a Fed transcript PDF."""
    try:
        import io
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code != 200:
            return ""
        # Use pypdf to extract text
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(resp.content))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        except Exception:
            pass
        # Fallback: try pdfminer
        try:
            from pdfminer.high_level import extract_text_to_fp
            from pdfminer.layout import LAParams
            import io as _io
            output = _io.StringIO()
            extract_text_to_fp(io.BytesIO(resp.content), output, laparams=LAParams())
            return output.getvalue()
        except Exception:
            pass
        return ""
    except Exception as e:
        return ""


def load_transcripts(force_refresh: bool = False) -> list[str]:
    """Load all Fed transcripts, using cache if available."""
    if not force_refresh and CACHE_FILE.exists():
        data = json.loads(CACHE_FILE.read_text())
        if time.time() - data.get("timestamp", 0) < CACHE_TTL:
            transcripts = data.get("transcripts", [])
            print(f"[fed] Loaded {len(transcripts)} transcripts from cache.")
            return transcripts

    print("[fed] Fetching FOMC transcripts from federalreserve.gov (PDFs)...")
    # Install pypdf if needed
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "pypdf", "--break-system-packages", "-q"])

    urls        = get_fomc_pdf_urls()
    transcripts = []
    seen_dates  = set()

    for i, url in enumerate(urls):
        # Skip duplicate dates (we try both capitalizations)
        date_part = url.split("presconf")[-1].replace(".pdf","")
        if date_part in seen_dates:
            continue

        text = fetch_transcript_text(url)
        if len(text) > 1000:
            transcripts.append(text)
            seen_dates.add(date_part)
        time.sleep(0.2)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(urls)}] fetched {len(transcripts)} valid transcripts")

    CACHE_FILE.write_text(json.dumps({
        "transcripts": transcripts,
        "timestamp":   time.time(),
        "count":       len(transcripts),
    }, indent=2))

    print(f"[fed] Fetched and cached {len(transcripts)} transcripts.")
    return transcripts


# ── Word frequency model ──────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    return text.lower()


def word_mentioned(word_or_phrase: str, text: str) -> bool:
    """
    Check if a word/phrase (or any of its variants) appears in the text.
    Handles multi-word phrases and slash-separated variants like 'Yield Curve'.
    """
    text_lower = normalize_text(text)
    # Split on slash for variants like "QT / Quantitative Tightening"
    variants   = [v.strip().lower() for v in word_or_phrase.split("/")]
    # Also split on comma if present
    all_variants = []
    for v in variants:
        all_variants.extend([x.strip() for x in v.split(",")])

    for variant in all_variants:
        if not variant:
            continue
        # Use word boundary for single words, substring for phrases
        if " " in variant:
            if variant in text_lower:
                return True
        else:
            if re.search(r'\b' + re.escape(variant) + r'\b', text_lower):
                return True
    return False


def calculate_word_frequencies(transcripts: list[str]) -> dict[str, dict]:
    """
    For each transcript, calculate which words were mentioned.
    Returns per-word stats: mention_count, total_conferences, base_rate.
    """
    if not transcripts:
        return {}

    # We'll compute this dynamically per query word
    # but store the transcript texts for fast lookup
    return {"transcripts": transcripts, "count": len(transcripts)}


def estimate_mention_probability(
    word: str,
    transcripts: list[str],
    context_boost: float = 0.0,
) -> dict:
    """
    Estimate probability that Powell mentions `word` in his next press conference.
    Uses FOMC-RoBERTa if available, falls back to frequency analysis.

    FOMC-RoBERTa: gtfintechlab/FOMC-RoBERTa — 88% accuracy per research report
    Classifies text as hawkish/dovish/neutral. Used to boost/penalize words
    based on whether current monetary policy stance matches historical context.
    """
    if not transcripts:
        return {"prob": 0.5, "count": 0, "total": 0, "base_rate": 0.5, "confidence": "low"}

    count     = sum(1 for t in transcripts if word_mentioned(word, t))
    total     = len(transcripts)
    base_rate = count / total

    # Laplace smoothing
    smoothed = (count + 1) / (total + 2)

    # ── FOMC-RoBERTa contextual adjustment ────────────────────────────────────
    # Attempt to load and use the fine-tuned Fed language classifier
    # Falls back gracefully if transformers not installed
    roberta_boost = 0.0
    method        = "frequency"
    try:
        roberta_boost, method = _fomc_roberta_adjustment(word, transcripts[-3:])
    except Exception:
        pass

    prob = max(0.02, min(0.98, smoothed + context_boost + roberta_boost))
    confidence = "high" if total >= 10 else "medium" if total >= 5 else "low"

    return {
        "prob":       round(prob, 4),
        "count":      count,
        "total":      total,
        "base_rate":  round(base_rate, 4),
        "smoothed":   round(smoothed, 4),
        "confidence": confidence,
        "method":     method,
    }


def _fomc_roberta_adjustment(word: str, recent_transcripts: list[str]) -> tuple[float, str]:
    """
    Use FOMC-RoBERTa to classify recent Fed communication stance.
    Returns (adjustment, method_name).

    Hawkish stance → boost inflation/rate/tariff words
    Dovish stance  → boost cut/support/growth words
    """
    try:
        from transformers import pipeline
        import os

        cache_path = "fomc_roberta_cache.json"
        import json, hashlib

        # Check cache first
        text_key = hashlib.md5("".join(recent_transcripts[:500]).encode()).hexdigest()
        if os.path.exists(cache_path):
            cache = json.loads(open(cache_path).read())
            if text_key in cache:
                stance = cache[text_key]
                return _stance_to_adjustment(word, stance), "fomc_roberta"

        # Load model (downloads on first run ~1GB, cached after)
        classifier = pipeline(
            "text-classification",
            model="gtfintechlab/FOMC-RoBERTa",
            device=-1  # CPU
        )

        # Classify recent transcript snippets
        stances = []
        for transcript in recent_transcripts:
            # Use last 512 tokens (model limit)
            snippet = transcript[-1500:]
            result  = classifier(snippet[:512], truncation=True)
            stances.append(result[0]["label"].lower())

        # Majority vote
        from collections import Counter
        stance = Counter(stances).most_common(1)[0][0]

        # Cache result
        cache = json.loads(open(cache_path).read()) if os.path.exists(cache_path) else {}
        cache[text_key] = stance
        open(cache_path, "w").write(json.dumps(cache))

        return _stance_to_adjustment(word, stance), "fomc_roberta"

    except ImportError:
        return 0.0, "frequency"
    except Exception:
        return 0.0, "frequency"


def _stance_to_adjustment(word: str, stance: str) -> float:
    """
    Map RoBERTa stance to word probability adjustment.
    Hawkish = more likely to say inflation/rate/tariff words
    Dovish  = more likely to say cut/support/easing words
    """
    word_lower = word.lower()

    hawkish_words = ["inflation", "rate", "tariff", "tight", "restrict",
                     "above", "uncertainty", "risk", "concern", "elevated"]
    dovish_words  = ["cut", "support", "easing", "accommodation", "below",
                     "growth", "employment", "labor", "recovery", "stable"]

    if stance in ("hawkish", "hawk"):
        if any(w in word_lower for w in hawkish_words):
            return 0.08   # boost hawkish words in hawkish environment
        if any(w in word_lower for w in dovish_words):
            return -0.05  # penalize dovish words in hawkish environment
    elif stance in ("dovish", "dove"):
        if any(w in word_lower for w in dovish_words):
            return 0.08
        if any(w in word_lower for w in hawkish_words):
            return -0.05
    return 0.0


# ── Kalshi scanner ────────────────────────────────────────────────────────────

EDGE_THRESHOLD = 0.10   # minimum edge to signal
MIN_VOLUME     = 500


def scan_fed_mentions(transcripts: list[str]) -> list[dict]:
    """Scan open Fed mention markets and find edges."""
    import kalshi_client as kc

    resp   = kc.get("/events", {"series_ticker": "KXFEDMENTION", "limit": 10, "status": "open"})
    events = resp.get("events", [])
    signals = []

    for event in events:
        event_ticker = event.get("event_ticker", "")
        print(f"\n[fed] Scanning {event_ticker}: {event.get('title', '')}")
        markets = kc.get(f"/events/{event_ticker}").get("markets", [])

        for m in markets:
            if m.get("status") != "active":
                continue

            word     = (m.get("yes_sub_title") or "").strip()
            if not word or "does not qualify" in word.lower():
                continue

            yes_bid  = m.get("yes_bid_dollars")
            yes_ask  = m.get("yes_ask_dollars")
            if yes_bid is None or yes_ask is None:
                continue

            bid, ask = float(yes_bid), float(yes_ask)
            mid      = (bid + ask) / 2
            spread   = ask - bid

            # Get model estimate
            result   = estimate_mention_probability(word, transcripts)
            model_p  = result["prob"]
            edge     = model_p - mid

            # Only signal if edge is meaningful and we have data
            if abs(edge) < EDGE_THRESHOLD:
                continue
            if result["confidence"] == "low":
                continue

            side = "yes" if edge > 0 else "no"
            bet_price = ask if side == "yes" else 1 - bid

            signals.append({
                "ticker":     m.get("ticker", ""),
                "word":       word,
                "model_prob": round(model_p, 4),
                "market_mid": round(mid, 4),
                "edge":       round(edge, 4),
                "side":       side,
                "bid":        bid,
                "ask":        ask,
                "spread":     round(spread, 4),
                "count":      result["count"],
                "total":      result["total"],
                "base_rate":  result["base_rate"],
                "confidence": result["confidence"],
            })

    signals.sort(key=lambda s: abs(s["edge"]), reverse=True)
    return signals


def print_fed_signals(signals: list[dict]):
    if not signals:
        print("[fed] No signals above threshold.")
        return

    print(f"\n{'='*90}")
    print(f"  {'WORD':<40} {'MODEL':>7} {'MID':>7} {'EDGE':>7} {'SIDE':<5} {'CONF':<8} {'N'}")
    print(f"{'='*90}")
    for s in signals:
        print(
            f"  {s['word']:<40} "
            f"{s['model_prob']*100:>6.1f}% "
            f"{s['market_mid']*100:>6.1f}% "
            f"{s['edge']*100:>+6.1f}% "
            f"{s['side']:<5} "
            f"{s['confidence']:<8} "
            f"{s['count']}/{s['total']}"
        )
    print(f"{'='*90}")
    print(f"  Total Fed signals: {len(signals)}")


if __name__ == "__main__":
    transcripts = load_transcripts(force_refresh=False)
    if not transcripts:
        print("No transcripts loaded — check federalreserve.gov connectivity.")
    else:
        print(f"\nLoaded {len(transcripts)} Fed transcripts")
        # Quick test
        test_words = ["Uncertainty", "Inflation", "Trump", "Yield Curve", "Tariff"]
        print("\n=== Word frequency test ===")
        for w in test_words:
            r = estimate_mention_probability(w, transcripts)
            print(f"  {w:<25} {r['count']}/{r['total']} = {r['base_rate']*100:.0f}%  model: {r['prob']*100:.0f}%")

        print("\n=== Scanning Kalshi markets ===")
        signals = scan_fed_mentions(transcripts)
        print_fed_signals(signals)