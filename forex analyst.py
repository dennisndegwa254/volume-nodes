"""
forex_analyst.py
----------------
Claude AI analysis for forex pairs.
- Hawkish/Dovish/Neutral sentiment per currency
- Volume Profile context (POC, VAH, VAL, signal)
- Central Bank Drift rule (disables counter-trend signals)
- Correlation Lead/Lag awareness
- Returns structured FX trade signal
"""

import anthropic
import json
import re
import time
from dataclasses import dataclass
from typing import Optional
from forex_volume_profile import VolumeProfile

@dataclass
class FXSignal:
    pair: str
    action: str            # BUY | SELL | HOLD
    confidence: str        # HIGH | MEDIUM | LOW
    entry: float
    stop_loss: float
    take_profit: float
    stop_pips: float
    target_pips: float
    lot_size: float
    cb_sentiment: str      # HAWKISH | DOVISH | NEUTRAL (base currency)
    reasoning: str
    news_summary: str
    signal_type: str       # MEAN_REVERSION | BREAKOUT | HOLD


_client = anthropic.Anthropic()
_cache: dict = {}
CACHE_TTL = 300  # 5 minutes


def _cache_valid(pair: str, signal: str) -> bool:
    if pair not in _cache: return False
    e = _cache[pair]
    return (time.time() - e["ts"]) < CACHE_TTL and e["signal"] == signal


def analyse_pair(vp: VolumeProfile, lot_size: float = 0.02,
                 account_balance: float = 100.0) -> Optional[FXSignal]:

    if _cache_valid(vp.pair, vp.signal):
        print(f"[forex_analyst] Cache hit {vp.pair}")
        return _cache[vp.pair]["result"]

    # Only call Claude on actionable signals — skip NEUTRAL and INSIDE_VA
    if "NEUTRAL" in vp.signal or "INSIDE" in vp.signal or "Watch" in vp.signal:
        return None

    base_ccy  = vp.pair[:3]
    quote_ccy = vp.pair[3:]
    pip       = 0.01 if 'JPY' in vp.pair else 0.0001

    prompt = f"""You are an expert forex analyst specialising in Volume Profile and institutional 
order flow for major currency pairs. Analyse this setup and provide a structured trade decision.

=== PAIR: {vp.pair} ({base_ccy}/{quote_ccy}) ===
Current Price     : {vp.current_price:.5f}
Day Change        : {vp.day_pct:+.2f}%
RSI (1H)          : {vp.rsi_1h:.1f}

=== VOLUME PROFILE (1H, last 5 days) ===
Point of Control  : {vp.poc:.5f}   ← highest volume / fair value
Value Area High   : {vp.vah:.5f}   ← top of 70% volume zone
Value Area Low    : {vp.val:.5f}   ← bottom of 70% volume zone
Price Location    : {vp.price_location}
Volume Intensity  : {vp.volume_intensity:.2f}  (< 0.7 = exhaustion signal)
Dashboard Signal  : {vp.signal_icon} {vp.signal}

=== TREND FILTERS ===
Daily SMA200      : {vp.sma200_1d:.5f} → Trend: {"BULLISH" if vp.trend_bullish else "BEARISH"}
4H SMA50          : {vp.sma50_4h:.5f}  → 4H Trend: {"BULLISH" if vp.trend_4h_bullish else "BEARISH"}
ATR (1H, 14)      : {vp.atr:.5f} ({vp.atr_pct*100:.3f}% of price)

=== ACCOUNT PARAMETERS ===
Account Balance   : ${account_balance:.2f}
Lot Size          : {lot_size} lots
Pip Value         : {pip}
Typical Spread    : {vp.spread_pips} pips

=== YOUR ANALYSIS TASK ===

STEP 1 — VALIDATE THE SETUP:
For MEAN REVERSION: Confirm volume exhaustion at VA edge + trend alignment
For BREAKOUT: Confirm price acceptance outside VA (not a fakeout) + low ATR entry

STEP 2 — SEARCH for latest news on {base_ccy} and {quote_ccy}:
- Central Bank stance (Fed/ECB/BOE/BOJ/RBA/BOC/RBNZ) — Hawkish or Dovish?
- Any scheduled high-impact events in next 24 hours (FOMC, NFP, CPI, rate decisions)?
- Current market risk sentiment (Risk-On or Risk-Off)?
- Any geopolitical events affecting this pair?

STEP 3 — CENTRAL BANK DRIFT RULE:
If AI Sentiment for {base_ccy} is EXTREME BULLISH/HAWKISH:
  → Disable MEAN REVERSION SELL signals, only take BREAKOUT BUY
If AI Sentiment is EXTREME BEARISH/DOVISH:
  → Disable MEAN REVERSION BUY signals, only take BREAKOUT SELL

STEP 4 — CALCULATE SL/TP:
- For Mean Reversion BUY: SL = just below VAL (or nearest LVN below), TP = POC then VAH
- For Breakout BUY: SL = just below VAH (the breakout level), TP = next HVN above
- For Mean Reversion SELL: SL = just above VAH, TP = POC then VAL
- For Breakout SELL: SL = just above VAL, TP = next HVN below
- Minimum 1.5:1 reward-to-risk. Account for {vp.spread_pips} pip spread.

STEP 5 — FINAL DECISION: BUY, SELL, or HOLD

IMPORTANT: Return ONLY valid JSON — no markdown, no preamble, plain text only in strings:
{{
  "action": "BUY" or "SELL" or "HOLD",
  "confidence": "HIGH" or "MEDIUM" or "LOW",
  "entry": {vp.current_price:.5f},
  "stop_loss": <exact price>,
  "take_profit": <exact price>,
  "stop_pips": <distance in pips as float>,
  "target_pips": <distance in pips as float>,
  "lot_size": {lot_size},
  "cb_sentiment": "HAWKISH" or "DOVISH" or "NEUTRAL",
  "signal_type": "MEAN_REVERSION" or "BREAKOUT" or "HOLD",
  "reasoning": "Plain text. 2-3 sentences on setup and key driver.",
  "news_summary": "Plain text. 1-2 sentences on CB stance and key event risk."
}}"""

    try:
        response = _client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
        )

        text_parts = [
            b.text for b in response.content
            if hasattr(b, "type") and b.type == "text"
        ]
        raw = " ".join(text_parts).strip()
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        raw = re.sub(r"<cite[^>]*>|</cite>|<[^>]+>", "", raw)

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            print(f"[forex_analyst] No JSON for {vp.pair}")
            return None

        d = json.loads(match.group())

        result = FXSignal(
            pair=vp.pair,
            action=d.get("action", "HOLD"),
            confidence=d.get("confidence", "LOW"),
            entry=float(d.get("entry", vp.current_price)),
            stop_loss=float(d.get("stop_loss", 0)),
            take_profit=float(d.get("take_profit", 0)),
            stop_pips=float(d.get("stop_pips", 0)),
            target_pips=float(d.get("target_pips", 0)),
            lot_size=float(d.get("lot_size", lot_size)),
            cb_sentiment=d.get("cb_sentiment", "NEUTRAL"),
            reasoning=d.get("reasoning", ""),
            news_summary=d.get("news_summary", ""),
            signal_type=d.get("signal_type", "HOLD"),
        )

        _cache[vp.pair] = {"result": result, "signal": vp.signal, "ts": time.time()}
        return result

    except Exception as e:
        print(f"[forex_analyst] Error {vp.pair}: {e}")
        return None
