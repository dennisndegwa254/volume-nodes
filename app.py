"""
FX Multi-Factor Engine
=======================
Streamlit dashboard combining technical structure (volume profile, trend,
volatility), correlation lead/lag, and AI sentiment gating into a single
signal surface.

DATA HONESTY NOTE
------------------
Spot FX is an OTC market — there is no centralized order book and no such
thing as real-time "Open Interest" the way there is for listed futures or
options. Any tool that claims to show live FX open interest from spot price
data alone is fabricating it. This file resolves that two ways:

  1. Where a real, public proxy exists, we use it: the CFTC's weekly
     Commitment of Traders (COT) report gives actual net futures positioning
     for the major currencies. It's the closest legitimate substitute a
     real desk would reach for. It is NOT real-time — it lags by several
     days and updates weekly (Fridays).
  2. Where no real data is available (COT fetch fails, or between weekly
     updates), we fall back to a deterministic, volume-derived proxy — never
     random noise — and every downstream signal is tagged with
     `data_quality` ("real_cot" / "synthetic_proxy") so the UI layer can
     show the user exactly what it's looking at. Nothing is presented as
     "confirmed institutional positioning" unless it's backed by real COT
     data.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("fx_engine")

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(page_title="FX Multi-Factor Engine", layout="wide",
                    initial_sidebar_state="expanded")

PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
TIMEFRAMES = ["M5", "M15", "H1", "H4", "D1"]

TD_TF_MAP = {"M5": "5min", "M15": "15min", "H1": "1h", "H4": "4h", "D1": "1day"}
TD_PAIRS = {"EURUSD": "EUR/USD", "USDJPY": "USD/JPY", "GBPUSD": "GBP/USD",
            "USDCHF": "USD/CHF", "AUDUSD": "AUD/USD", "USDCAD": "USD/CAD", "NZDUSD": "NZD/USD"}
YF_TICKERS = {"EURUSD": "EURUSD=X", "USDJPY": "USDJPY=X", "GBPUSD": "GBPUSD=X",
              "USDCHF": "USDCHF=X", "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X", "NZDUSD": "NZDUSD=X"}

# CFTC "legacy futures only" report codes for the CME FX futures contracts.
# Used to pull real net non-commercial (large speculator) positioning.
CFTC_CONTRACT_CODES = {
    "EURUSD": "099741",  # Euro FX
    "USDJPY": "097741",  # Japanese Yen (net position is JPY vs USD; sign-flipped below)
    "GBPUSD": "096742",  # British Pound
    "USDCHF": "092741",  # Swiss Franc (sign-flipped below)
    "AUDUSD": "232741",  # Australian Dollar
    "USDCAD": "090741",  # Canadian Dollar (sign-flipped below)
    "NZDUSD": "112741",  # New Zealand Dollar
}
# Contracts quoted as USD/XXX on CME (yen, franc, loonie) need their net
# position sign flipped to align with our XXXUSD-style pair convention.
CFTC_SIGN_FLIP = {"USDJPY", "USDCHF", "USDCAD"}

SIGNAL_LABELS = {2: "Strong Bull", 1: "Bullish", 0: "Neutral", -1: "Bearish", -2: "Strong Bear"}
SIGNAL_COLORS = {2: "#1a7a4a", 1: "#52b788", 0: "#666666", -1: "#e07b5a", -2: "#b5281c"}

BASE_PRICES = {"EURUSD": 1.0850, "USDJPY": 154.20, "GBPUSD": 1.2650,
               "USDCHF": 0.9020, "AUDUSD": 0.6480, "USDCAD": 1.3640, "NZDUSD": 0.5920}
VP_BINS = 30

CORRELATION_GROUPS = {
    "EURUSD": ["GBPUSD", "AUDUSD", "NZDUSD"], "GBPUSD": ["EURUSD", "AUDUSD"],
    "AUDUSD": ["NZDUSD", "EURUSD"], "NZDUSD": ["AUDUSD"],
    "USDJPY": ["USDCHF", "USDCAD"], "USDCHF": ["USDJPY", "USDCAD"], "USDCAD": ["USDJPY", "USDCHF"],
}

BADGE = {
    "blue_pulse": ("🔵", "#1a4fa8", "Mean Reversion"),
    "green_spark": ("🟢", "#1a7a4a", "Momentum Breakout"),
    "grey_dim": ("⚫", "#555555", "High Risk — Avoid"),
    "flashing_red": ("🔴", "#b5281c", "Intervention Risk"),
}

DATA_QUALITY = {
    "real_cot": "Real — CFTC Commitment of Traders (weekly, lagged)",
    "synthetic_proxy": "Synthetic proxy — derived from bar volume/price, NOT real positioning",
}

# ═══════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════
_DEFAULTS = {"last_refresh": 0, "refresh_count": 0,
             "sentiment_cache": {}, "sentiment_ts": {}, "cache_buster": 0, "last_minute": -1,
             "cot_cache": {}, "cot_ts": {},
             "claude_calls_today": 0, "claude_calls_date": "", "claude_tokens_used": 0}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

_current_minute = int(time.time() // 60)
if st.session_state.last_minute != _current_minute:
    st.session_state.last_minute = _current_minute
    st.session_state.cache_buster = _current_minute

TOKENS_PER_CALL = 230
SESSION_LIMITS = {"HIGH": 999, "MED": 20, "LOW": 5, "NONE": 0}
CACHE_DURATION = {"HIGH": 900, "MED": 1800, "LOW": 3600, "NONE": 86400}
COT_CACHE_TTL_SECONDS = 6 * 3600  # COT only updates weekly; 6h cache is plenty


def _reset_daily_if_needed() -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if st.session_state.claude_calls_date != today:
        st.session_state.claude_calls_date = today
        st.session_state.claude_calls_today = 0
        st.session_state.claude_tokens_used = 0


def _get_session_gate() -> tuple[bool, str, str]:
    """Returns (allowed, priority, human-readable reason)."""
    utc = datetime.now(timezone.utc)
    h, wday = utc.hour, utc.weekday()
    if wday >= 5:
        return False, "NONE", "🔴 Weekend — Claude suspended"
    if 12 <= h < 17:
        return True, "HIGH", "🟢 London/NY Overlap — Peak liquidity"
    if 7 <= h < 12:
        return True, "MED", "🟢 London Open"
    if 17 <= h < 21:
        return True, "MED", "🟡 NY Session"
    return False, "LOW", "🟡 Asian/Off-hours — conserving tokens"


def _should_call_claude(confluence_score: int, smooth: int, session_priority: str) -> tuple[bool, str]:
    """Gate valve — only call the model if the signal is worth the tokens."""
    _reset_daily_if_needed()
    if st.session_state.claude_calls_today >= 50:
        return False, f"Daily budget reached ({st.session_state.claude_calls_today}/50)"
    if SESSION_LIMITS.get(session_priority, 0) == 0:
        return False, "Outside active session"

    abs_conf = abs(confluence_score)
    if session_priority == "HIGH":
        if abs_conf >= 3 or smooth >= 2:
            return True, f"Peak session + conf {confluence_score:+d}/5"
        return False, f"Signal too weak (conf {confluence_score:+d}, smooth {smooth}/4)"
    if session_priority == "MED":
        if abs_conf >= 3 and smooth >= 2:
            return True, "Good session + strong signal"
        return False, f"Need conf>=3 AND smooth>=2 (got {abs_conf}, {smooth})"
    if session_priority == "LOW":
        if abs_conf >= 4:
            return True, "Extreme signal during off-hours"
        return False, "Off-hours: only extreme signals (conf>=4)"
    return False, "No session"


# ═══════════════════════════════════════════════════════════════
# DATA FETCHING — price/candles
# ═══════════════════════════════════════════════════════════════
def fetch_td(pair: str, tf: str, td_key: str, limit: int = 300) -> pd.DataFrame:
    """Twelve Data — real forex OHLCV with broker-aggregated volume."""
    if not td_key:
        return pd.DataFrame()
    params = {"symbol": TD_PAIRS.get(pair, pair), "interval": TD_TF_MAP.get(tf, "1h"),
              "outputsize": min(limit, 5000), "timezone": "UTC", "order": "ASC",
              "format": "JSON", "apikey": td_key}
    try:
        r = requests.get("https://api.twelvedata.com/time_series", params=params,
                          headers={"Cache-Control": "no-cache", "Pragma": "no-cache"}, timeout=20)
        d = r.json()
        if d.get("status") == "error":
            log.warning("TwelveData error for %s %s: %s", pair, tf, d.get("message"))
            return pd.DataFrame()
        values = d.get("values", [])
        if not values:
            return pd.DataFrame()
        df = pd.DataFrame(values).rename(columns={"datetime": "time"})
        df["time"] = pd.to_datetime(df["time"])
        for c in ["open", "high", "low", "close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
            if df["volume"].sum() == 0:
                df["volume"] = (df["high"] - df["low"]) * 1e6
                df.attrs["volume_is_proxy"] = True
        else:
            df["volume"] = (df["high"] - df["low"]) * 1e6
            df.attrs["volume_is_proxy"] = True
        df = df[["time", "open", "high", "low", "close", "volume"]].dropna()
        return df.sort_values("time").tail(limit).reset_index(drop=True)
    except Exception as e:
        log.warning("[TD] %s %s: %s", pair, tf, e)
        return pd.DataFrame()


def fetch_yf(pair: str, tf: str, limit: int = 300) -> pd.DataFrame:
    """Yahoo Finance via HTTP — free fallback. Volume is a range-based proxy;
    Yahoo's FX 'volume' field is not reliable tick volume."""
    ticker = YF_TICKERS.get(pair, f"{pair}=X")
    yf_tf = {"M5": "5m", "M15": "15m", "H1": "1h", "H4": "1h", "D1": "1d"}.get(tf, "1h")
    lookback_days = {"5m": 5, "15m": 50, "1h": 59, "1d": 700}.get(yf_tf, 59)
    now = int(datetime.now(timezone.utc).timestamp())
    period1, period2 = now - lookback_days * 86400, now

    for host in ["query2.finance.yahoo.com", "query1.finance.yahoo.com"]:
        url = (f"https://{host}/v8/finance/chart/{ticker}"
               f"?interval={yf_tf}&period1={period1}&period2={period2}"
               f"&includePrePost=false&corsDomain=finance.yahoo.com")
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0",
                                            "Accept": "application/json",
                                            "Cache-Control": "no-cache, no-store",
                                            "Pragma": "no-cache"}, timeout=15)
            d = r.json()
            res = d.get("chart", {}).get("result", [])
            if not res:
                continue
            data_raw = res[0]
            times = data_raw.get("timestamp", [])
            q = data_raw["indicators"]["quote"][0]
            if not times:
                continue

            df = pd.DataFrame({
                "time": pd.to_datetime(times, unit="s"),
                "open": q.get("open", [None] * len(times)),
                "high": q.get("high", [None] * len(times)),
                "low": q.get("low", [None] * len(times)),
                "close": q.get("close", [None] * len(times)),
            }).dropna(subset=["open", "high", "low", "close"])

            base = BASE_PRICES[pair]
            df = df[(df["close"] > base * 0.5) & (df["close"] < base * 2.0)]
            if df.empty:
                continue

            df["volume"] = (df["high"] - df["low"]) * 1e6
            df.attrs["volume_is_proxy"] = True
            df = df.sort_values("time").tail(limit).reset_index(drop=True)

            last_time = df["time"].iloc[-1].to_pydatetime()
            hours_old = (datetime.now(timezone.utc).replace(tzinfo=None) - last_time).total_seconds() / 3600
            if hours_old > 48:
                continue
            return df
        except Exception as e:
            log.debug("[YF] %s via %s failed: %s", pair, host, e)
            continue
    return pd.DataFrame()


def simulate(pair: str, n: int = 300) -> pd.DataFrame:
    """Last-resort synthetic candles when no live source responds. Clearly
    a fallback for UI continuity, not a market signal source."""
    rng = np.random.default_rng(abs(hash(pair)) % 9999)
    base = BASE_PRICES[pair]
    closes = base * np.cumprod(1 + rng.normal(0, 0.0006, n))
    noise = rng.uniform(0.0002, 0.0012, n)
    opens = np.roll(closes, 1)
    opens[0] = base
    times = [datetime.now(timezone.utc) - timedelta(hours=n - i) for i in range(n)]
    df = pd.DataFrame({"time": times, "open": opens, "high": closes * (1 + noise),
                        "low": closes * (1 - noise), "close": closes, "volume": (closes * noise) * 1e6})
    df.attrs["is_simulated"] = True
    return df


def get_candles(pair: str, tf: str, td_key: str, limit: int = 300) -> pd.DataFrame:
    df = fetch_td(pair, tf, td_key, limit) if td_key else pd.DataFrame()
    if df.empty:
        df = fetch_yf(pair, tf, limit)
    if df.empty:
        df = simulate(pair, limit)
    return df


def get_live_price(pair: str, td_key: str) -> float:
    base = BASE_PRICES[pair]
    if td_key:
        try:
            r = requests.get("https://api.twelvedata.com/price",
                              params={"symbol": TD_PAIRS.get(pair, pair), "apikey": td_key}, timeout=8)
            d = r.json()
            if "price" in d:
                p = float(d["price"])
                if base * 0.5 < p < base * 2.0:
                    return round(p, 5)
        except Exception as e:
            log.debug("get_live_price %s: %s", pair, e)
    return base


def get_news(pair: str) -> list[str]:
    """Placeholder headline feed. Swap for a real news API (e.g. NewsAPI,
    Benzinga, TradTheNews) before relying on sentiment output in production."""
    simulated = {
        "EURUSD": ["ECB holds rates steady", "EUR rallies on strong PMI", "Eurozone CPI in focus"],
        "USDJPY": ["BOJ signals rate hike", "JPY weakens on risk-on", "Fed vs BOJ divergence"],
        "GBPUSD": ["UK CPI beats expectations", "BOE cautious on cuts", "GBP supported by data"],
        "USDCHF": ["SNB intervenes to weaken CHF", "Swiss trade surplus widens"],
        "AUDUSD": ["RBA holds rates", "Australia jobs data beats", "China PMI supports AUD"],
        "USDCAD": ["Oil rally boosts CAD", "BOC signals data dependency"],
        "NZDUSD": ["RBNZ cuts 25bps", "NZD pressured by China slowdown"],
    }
    return simulated.get(pair, [f"{pair} markets stable"])


# ═══════════════════════════════════════════════════════════════
# REAL POSITIONING DATA — CFTC Commitment of Traders
# ═══════════════════════════════════════════════════════════════
def fetch_cftc_cot(pair: str) -> Optional[dict]:
    """
    Pull the most recent weekly CFTC Legacy Futures-Only COT report for the
    currency future underlying `pair`, via the CFTC's public Socrata API.

    Returns real net non-commercial (large speculator) positioning:
        {"net_position": int, "report_date": "YYYY-MM-DD", "prior_net": int}
    or None if the fetch fails (no internet, endpoint change, rate limit,
    unsupported pair). Callers MUST fall back to the synthetic proxy on None
    — never fabricate a "real" result.

    NOTE: This is weekly, lagged data (typically 3 business days stale) —
    appropriate for a positioning *backdrop*, not a live trigger.
    """
    code = CFTC_CONTRACT_CODES.get(pair)
    if not code:
        return None

    cache_key = pair
    age = time.time() - st.session_state.cot_ts.get(cache_key, 0)
    if age < COT_CACHE_TTL_SECONDS and cache_key in st.session_state.cot_cache:
        return st.session_state.cot_cache[cache_key]

    try:
        url = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
        params = {
            "$where": f"cftc_contract_market_code='{code}'",
            "$order": "report_date_as_yyyy_mm_dd DESC",
            "$limit": 2,
        }
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            return None

        latest = rows[0]
        net_now = int(latest["noncomm_positions_long_all"]) - int(latest["noncomm_positions_short_all"])
        net_prior = None
        if len(rows) > 1:
            prior = rows[1]
            net_prior = int(prior["noncomm_positions_long_all"]) - int(prior["noncomm_positions_short_all"])

        if pair in CFTC_SIGN_FLIP:
            net_now = -net_now
            net_prior = -net_prior if net_prior is not None else None

        result = {
            "net_position": net_now,
            "prior_net": net_prior if net_prior is not None else net_now,
            "report_date": latest.get("report_date_as_yyyy_mm_dd", "")[:10],
        }
        st.session_state.cot_cache[cache_key] = result
        st.session_state.cot_ts[cache_key] = time.time()
        return result
    except Exception as e:
        log.warning("CFTC COT fetch failed for %s: %s", pair, e)
        return None


# ═══════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS
# ═══════════════════════════════════════════════════════════════
def calc_atr(df: pd.DataFrame, p: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(span=p, adjust=False).mean()


def calc_ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def calc_sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def calc_rsi(s: pd.Series, p: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).ewm(span=p, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(span=p, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def vol_intensity(df: pd.DataFrame) -> pd.Series:
    return (df["volume"] / calc_atr(df).replace(0, np.nan)).fillna(0)


def get_signal(df: pd.DataFrame) -> int:
    if df.empty or len(df) < 55:
        return 0
    c = df["close"]
    e20, e50, last = calc_ema(c, 20).iloc[-1], calc_ema(c, 50).iloc[-1], c.iloc[-1]
    bull = int(e20 > e50) + int(last > e20)
    bear = int(e20 < e50) + int(last < e20)
    if bull == 2: return 2
    if bull == 1: return 1
    if bear == 2: return -2
    if bear == 1: return -1
    return 0


def get_iv(df: pd.DataFrame) -> float:
    if df.empty or len(df) < 10:
        return 8.0
    ret = np.log(df["close"] / df["close"].shift(1)).dropna()
    return round(float(ret.std() * np.sqrt(252 * 24) * 100), 2)


# ═══════════════════════════════════════════════════════════════
# ORDER-FLOW PROXY ENGINE (honest version)
# ═══════════════════════════════════════════════════════════════
def append_order_flow(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """
    Adds two columns to `df`:
      - 'cvd': cumulative volume delta estimated from each bar's close
               location within its high/low range. This is a standard,
               widely-used proxy when true tick-by-tick buy/sell data isn't
               available — it is NOT literal executed buy/sell volume, and
               is labeled as such downstream.
      - 'oi':  a positioning series. If a real CFTC COT read is available
               for this pair, the weekly net position is forward-filled
               across bars as a real positioning backdrop. Otherwise we
               derive a deterministic (non-random) proxy from cumulative
               volume-weighted price momentum — no RNG, fully reproducible,
               and never described as "confirmed" institutional data.

    df.attrs['oi_source'] is set to 'real_cot' or 'synthetic_proxy' so
    every downstream consumer can label output honestly.
    """
    if df.empty:
        df["cvd"], df["oi"] = 0.0, 0.0
        df.attrs["oi_source"] = "synthetic_proxy"
        return df

    # --- CVD: bar-range close-location proxy (standard technique) ---
    range_series = (df["high"] - df["low"]).replace(0, np.nan)
    close_loc = (((df["close"] - df["low"]) - (df["high"] - df["close"])) / range_series).fillna(0)
    delta_volume = df["volume"] * close_loc
    df["cvd"] = delta_volume.cumsum()

    # --- Positioning: try real COT first ---
    cot = fetch_cftc_cot(pair)
    if cot is not None:
        # Real weekly net position, held flat across intraday bars until the
        # next report — genuinely real data, just low-frequency.
        df["oi"] = cot["net_position"]
        df.attrs["oi_source"] = "real_cot"
        df.attrs["oi_report_date"] = cot["report_date"]
        df.attrs["oi_prior"] = cot["prior_net"]
        return df

    # --- Fallback: deterministic volume-momentum proxy, no randomness ---
    price_change = df["close"].diff().fillna(0)
    directional_vol = np.sign(price_change) * df["volume"] * np.sign(delta_volume).replace(0, 1)
    df["oi"] = 50_000.0 + directional_vol.cumsum() * 0.10
    df["oi"] = df["oi"].clip(lower=5_000)
    df.attrs["oi_source"] = "synthetic_proxy"
    return df


# ═══════════════════════════════════════════════════════════════
# VOLUME PROFILE ENGINE
# ═══════════════════════════════════════════════════════════════
def build_vp(df: pd.DataFrame, bins: int = VP_BINS) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["price_mid", "volume", "pct"])
    pmin, pmax = df["low"].min(), df["high"].max()
    if pmin >= pmax:
        return pd.DataFrame(columns=["price_mid", "volume", "pct"])
    edges = np.linspace(pmin, pmax, bins + 1)
    bv = np.zeros(bins)
    for _, r in df.iterrows():
        bl, bh, vol, rng = r["low"], r["high"], r["volume"], r["high"] - r["low"]
        for b in range(bins):
            ol, oh = max(edges[b], bl), min(edges[b + 1], bh)
            if oh > ol:
                bv[b] += vol * ((oh - ol) / rng if rng > 0 else 1.0)
    mids = (edges[:-1] + edges[1:]) / 2
    total = bv.sum() or 1
    out = pd.DataFrame({"price_mid": np.round(mids, 5), "volume": np.round(bv, 2),
                         "pct": np.round(bv / total * 100, 1)})
    return out.dropna().replace([np.inf, -np.inf], 0).sort_values("price_mid", ascending=False).reset_index(drop=True)


def get_poc(vp: pd.DataFrame) -> float:
    if vp is None or vp.empty:
        return 0.0
    vp = vp.dropna(subset=["volume", "price_mid"])
    return 0.0 if vp.empty else float(vp.loc[vp["volume"].idxmax(), "price_mid"])


def get_va(vp: pd.DataFrame, pct: float = 0.70) -> tuple[float, float]:
    if vp is None or vp.empty:
        return 0.0, 0.0
    vs = vp.dropna().sort_values("price_mid").reset_index(drop=True)
    if vs.empty:
        return 0.0, 0.0
    target = vs["volume"].sum() * pct
    pi = vs["volume"].idxmax()
    lo = hi = pi
    acc = vs.loc[pi, "volume"]
    while acc < target:
        can_lo, can_hi = lo > 0, hi < len(vs) - 1
        if not can_lo and not can_hi:
            break
        alo = vs.loc[lo - 1, "volume"] if can_lo else 0
        ahi = vs.loc[hi + 1, "volume"] if can_hi else 0
        if ahi >= alo and can_hi:
            hi += 1; acc += ahi
        elif can_lo:
            lo -= 1; acc += alo
        else:
            hi += 1; acc += ahi
    return float(vs.loc[hi, "price_mid"]), float(vs.loc[lo, "price_mid"])


# ═══════════════════════════════════════════════════════════════
# GOLDEN ENTRY CHECKLIST (Smoothness Score 0-4)
# ═══════════════════════════════════════════════════════════════
def smoothness_score(df: pd.DataFrame, poc: float, vah: float, val_p: float,
                      sent_score: int = 0) -> tuple[int, list[tuple[str, str, str]]]:
    if df.empty or len(df) < 201:
        return 0, []
    score, checks = 0, []
    last = float(df["close"].iloc[-1])
    s50 = float(calc_sma(df["close"], 50).iloc[-1])
    s200 = float(calc_sma(df["close"], 200).iloc[-1])
    atr_n = float(calc_atr(df).iloc[-1])
    atr_m = float(calc_atr(df).rolling(20).mean().iloc[-1])

    if s50 > s200:
        score += 1; checks.append(("✅", "Trend", "SMA50 > SMA200 — Bullish trend"))
    elif s50 < s200:
        score += 1; checks.append(("✅", "Trend", "SMA50 < SMA200 — Bearish trend"))
    else:
        checks.append(("❌", "Trend", "No clear trend — SMA50 ≈ SMA200"))

    tol = atr_n * 0.4
    at_poc, at_vah, at_val = abs(last - poc) < tol, abs(last - vah) < tol, abs(last - val_p) < tol
    if at_poc or at_vah or at_val:
        lbl = "POC" if at_poc else ("VAH" if at_vah else "VAL")
        score += 1; checks.append(("✅", "Location", f"Price at {lbl} — smooth entry zone"))
    else:
        checks.append(("❌", "Location", "Price inside Value Area — no-man's land, avoid"))

    if atr_n < atr_m * 0.85:
        score += 1; checks.append(("✅", "Volatility", f"ATR consolidating ({atr_n:.5f} < mean {atr_m:.5f})"))
    else:
        checks.append(("❌", "Volatility", f"ATR elevated ({atr_n:.5f}) — momentum already running"))

    trend_bull = s50 > s200
    if sent_score != 0 and ((trend_bull and sent_score > 0) or (not trend_bull and sent_score < 0)):
        score += 1; checks.append(("✅", "Sentiment", f"AI score {sent_score:+d} aligns with trend"))
    elif sent_score == 0:
        checks.append(("⚪", "Sentiment", "AI neutral — no confirmation boost"))
    else:
        checks.append(("❌", "Sentiment", f"AI score {sent_score:+d} conflicts with trend"))

    return score, checks


def exhaustion_filter(df: pd.DataFrame, window: int = 5) -> bool:
    if df.empty or len(df) < window + 2:
        return False
    vi = vol_intensity(df)
    return float(vi.iloc[-1]) < float(vi.iloc[-window:].mean()) * 0.75


def tpo_status(df: pd.DataFrame, vah: float, val_p: float, mins_per_bar: int = 60) -> tuple[str, int]:
    if df.empty or len(df) < 3:
        return "Unknown", 0
    last = float(df["close"].iloc[-1])
    if val_p <= last <= vah:
        return "Inside VA", 0
    above = last > vah
    bars = 0
    for i in range(len(df) - 1, max(0, len(df) - 10), -1):
        p = float(df["close"].iloc[i])
        if (above and p > vah) or (not above and p < val_p):
            bars += 1
        else:
            break
    mins = bars * mins_per_bar
    if mins >= 30:
        return "Validated Breakout", mins
    if mins > 0:
        return "Fakeout Risk", mins
    return "Just Broke", 0


def lead_lag(pair: str, all_candles: dict[str, pd.DataFrame]) -> tuple[Optional[str], Optional[str]]:
    peers = CORRELATION_GROUPS.get(pair, [])
    df_m = all_candles.get(pair, pd.DataFrame())
    if df_m.empty or len(df_m) < 5:
        return None, None
    main_ret = float(df_m["close"].pct_change(5).iloc[-1])
    leader, lr = None, 0
    for peer in peers:
        df_p = all_candles.get(peer, pd.DataFrame())
        if df_p.empty or len(df_p) < 5:
            continue
        pr = float(df_p["close"].pct_change(5).iloc[-1])
        same_direction = (main_ret > 0 and pr > 0) or (main_ret < 0 and pr < 0)
        if same_direction and abs(pr) > abs(main_ret) and abs(pr) > abs(lr):
            leader, lr = peer, pr
    if leader:
        direction = "bullish" if lr > 0 else "bearish"
        return leader, f"{leader} moved {lr * 100:+.2f}% — consistent with a {direction} read on {pair}"
    return None, None


# ═══════════════════════════════════════════════════════════════
# AI SENTIMENT (Claude) — gated by session + token budget
# ═══════════════════════════════════════════════════════════════
def ai_sentiment(pair: str, headlines: list[str], claude_key: str,
                  confluence_score: int = 0, smooth: int = 0) -> dict:
    _reset_daily_if_needed()
    default = {"tone": "Neutral", "score": 0, "intervention_risk": False,
               "reasoning": "", "pair": pair, "gated": True, "from_cache": False,
               "session": "", "calls_today": st.session_state.claude_calls_today}

    if not claude_key or not ANTHROPIC_AVAILABLE:
        default["reasoning"] = "anthropic package not installed" if not ANTHROPIC_AVAILABLE else "No Claude key configured"
        return default

    session_ok, session_priority, session_msg = _get_session_gate()
    default["session"] = session_msg

    cache_ttl = CACHE_DURATION.get(session_priority, 1800)
    cache_age = time.time() - st.session_state.sentiment_ts.get(pair, 0)
    if cache_age < cache_ttl and pair in st.session_state.sentiment_cache:
        cached = dict(st.session_state.sentiment_cache[pair])
        cached.update(from_cache=True, calls_today=st.session_state.claude_calls_today, session=session_msg)
        return cached

    if not session_ok:
        default["reasoning"] = session_msg
        return default

    should_call, gate_reason = _should_call_claude(confluence_score, smooth, session_priority)
    if not should_call:
        default["reasoning"] = f"Token gate: {gate_reason}"
        return default

    base_ccy, quote_ccy = pair[:3], pair[3:]
    headline_str = "; ".join(headlines[:3])
    prompt = (f"FX analyst. {base_ccy}/{quote_ccy} news: {headline_str}\n"
              f'JSON only: {{"tone":"Hawkish|Dovish|Neutral",'
              f'"score":-2to+2,"intervention_risk":true/false,'
              f'"reasoning":"<10 words"}}')
    try:
        client = anthropic.Anthropic(api_key=claude_key)
        msg = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=100,
                                      messages=[{"role": "user", "content": prompt}])
        raw = re.sub(r"```json|```", "", msg.content[0].text.strip()).strip()
        result = json.loads(raw)
        result.update(pair=pair, gated=False, from_cache=False, session=session_msg,
                      calls_today=st.session_state.claude_calls_today)
        st.session_state.claude_calls_today += 1
        st.session_state.claude_tokens_used += TOKENS_PER_CALL
        st.session_state.sentiment_cache[pair] = result
        st.session_state.sentiment_ts[pair] = time.time()
        return result
    except Exception as e:
        default["reasoning"] = f"API error: {str(e)[:40]}"
        return default


# ═══════════════════════════════════════════════════════════════
# SIGNAL GENERATION — order-flow + breakout logic, honestly labeled
# ═══════════════════════════════════════════════════════════════
def generate_signals(df: pd.DataFrame, vp: pd.DataFrame, poc: float, vah: float, val: float,
                      pair: str, smooth: int = 0, sent: Optional[dict] = None,
                      tpo: str = "Unknown") -> list[dict]:
    """
    Evaluates CVD (bar-range volume-delta proxy) against price action, and
    positioning (real CFTC COT where available, otherwise a labeled
    synthetic proxy) to build a small set of setups. Every returned signal
    carries `data_quality` so the UI can render an honest badge — real COT
    backing gets stronger language, the synthetic fallback gets explicitly
    hedged language ("proxy suggests", never "confirms").
    """
    if df.empty or len(df) < 20 or vp.empty:
        return []

    df = append_order_flow(df, pair)
    oi_source = df.attrs.get("oi_source", "synthetic_proxy")
    is_real_oi = oi_source == "real_cot"

    sigs = []
    last = float(df["close"].iloc[-1])
    atr_val = float(calc_atr(df).iloc[-1])
    rsi_val = float(calc_rsi(df["close"]).iloc[-1])

    cvd_now = df["cvd"].iloc[-1]
    cvd_prev = df["cvd"].iloc[-5] if len(df) >= 5 else df["cvd"].iloc[0]
    oi_now = df["oi"].iloc[-1]
    oi_prev = df["oi"].iloc[-5] if len(df) >= 5 else df["oi"].iloc[0]

    intervention = sent.get("intervention_risk", False) if sent else False

    def near(p: float, lvl: float, t: float = 0.3) -> bool:
        return abs(p - lvl) < atr_val * t

    def sig(stype: str, cat: str, entry: float, sl: float, tp1: float, tp2: float,
            reason: str, conf: str, badge: str = "blue_pulse") -> dict:
        risk = abs(entry - sl)
        reward = abs(tp2 - entry)
        rr = round(reward / risk, 1) if risk > 0 else 0
        final_conf = conf
        if smooth >= 3 and conf == "Moderate":
            final_conf = "High"
        if intervention and "Short" in stype:
            final_conf, badge = "Low", "flashing_red"
        # Synthetic positioning data can never itself push confidence to High.
        if not is_real_oi and final_conf == "High" and cat in ("MO", "OrderFlow"):
            final_conf = "Moderate"
        return {"type": stype, "cat": cat, "entry": round(entry, 5), "sl": round(sl, 5),
                "tp1": round(tp1, 5), "tp2": round(tp2, 5), "atr": round(atr_val, 5),
                "rsi": round(rsi_val, 1), "reason": reason, "confidence": final_conf, "rr": rr,
                "smooth": smooth, "tpo": tpo, "badge": badge,
                "cvd": round(float(cvd_now), 1), "oi": int(oi_now),
                "data_quality": oi_source, "data_quality_label": DATA_QUALITY[oi_source]}

    positioning_verb = "confirms" if is_real_oi else "is directionally consistent with (unconfirmed proxy)"

    # 1. Volume-delta divergence at value-area extremes
    if near(last, val) and cvd_now > cvd_prev and df["close"].iloc[-1] <= df["close"].iloc[-5]:
        sigs.append(sig(
            "CVD Bullish Divergence", "OrderFlow", last, last - 1.5 * atr_val, poc, vah,
            f"Price flat/lower into VAL ({val:.5f}) while the bar-range volume-delta proxy is rising — "
            f"{'real COT data shows' if is_real_oi else 'the unconfirmed proxy suggests'} building demand.",
            "High" if (oi_now > oi_prev and is_real_oi) else "Moderate", "blue_pulse"))

    if near(last, vah) and cvd_now < cvd_prev and df["close"].iloc[-1] >= df["close"].iloc[-5]:
        sigs.append(sig(
            "CVD Bearish Divergence", "OrderFlow", last, last + 1.5 * atr_val, poc, val,
            f"Price pushing into VAH ({vah:.5f}) while the volume-delta proxy is falling — "
            f"{'real COT data shows' if is_real_oi else 'the unconfirmed proxy suggests'} fading demand.",
            "High" if (oi_now < oi_prev and is_real_oi) else "Moderate", "blue_pulse"))

    # 2. Breakout + positioning alignment
    tpo_ok = tpo in ["Validated Breakout", "Just Broke"]
    if last > vah + 0.1 * atr_val and tpo_ok and oi_now > oi_prev * 1.02 and cvd_now > cvd_prev:
        pct_chg = ((oi_now / oi_prev) - 1) * 100 if oi_prev else 0
        sigs.append(sig(
            "Breakout + Positioning Aligned (Long)", "MO", last, vah - 1.2 * atr_val,
            last + 1.5 * atr_val, last + 3.0 * atr_val,
            f"Breakout past VAH; net positioning {positioning_verb} the move "
            f"({pct_chg:+.1f}% vs. {oi_source.replace('_', ' ')} baseline).",
            "High" if is_real_oi else "Moderate", "green_spark"))

    if last < val - 0.1 * atr_val and tpo_ok and oi_now > oi_prev * 1.02 and cvd_now < cvd_prev:
        sigs.append(sig(
            "Breakout + Positioning Aligned (Short)", "MO", last, val + 1.2 * atr_val,
            last - 1.5 * atr_val, last - 3.0 * atr_val,
            f"Breakdown below VAL; net positioning {positioning_verb} the move "
            f"(proxy: {oi_source.replace('_', ' ')}).",
            "High" if is_real_oi else "Moderate", "green_spark"))

    # 3. Positioning-unwind exhaustion (real COT drop / proxy decline + extreme RSI)
    if rsi_val < 30 and oi_now < oi_prev * 0.96:
        sigs.append(sig(
            "Positioning Unwind Exhaustion (Long side)", "MR", last, last - 1.0 * atr_val, poc, vah,
            f"Oversold RSI plus a decline in {oi_source.replace('_', ' ')} positioning — "
            f"reads as capitulation rather than fresh conviction selling.",
            "Moderate", "blue_pulse"))

    if rsi_val > 70 and oi_now < oi_prev * 0.96:
        sigs.append(sig(
            "Positioning Unwind Exhaustion (Short side)", "MR", last, last + 1.0 * atr_val, poc, val,
            f"Overbought RSI plus a decline in {oi_source.replace('_', ' ')} positioning — "
            f"move looks stretched rather than freshly backed.",
            "Moderate", "blue_pulse"))

    if tpo == "Fakeout Risk":
        for s in sigs:
            if s["cat"] == "MO":
                s["confidence"], s["badge"] = "Low", "grey_dim"
                s["reason"] += " ⚠️ Fakeout risk — less than 30 min outside the value area."

    if intervention:
        for s in sigs:
            s["badge"] = "flashing_red"
            s["reason"] += " 🚨 Central bank intervention risk flagged by sentiment model."

    return sigs


# ═══════════════════════════════════════════════════════════════
# MAIN DATA LOADER
# ═══════════════════════════════════════════════════════════════
@st.cache_data(ttl=0, show_spinner=False)
def load_data(td_key: str, claude_key: str, fr_bars: int, vp_bins_val: int,
              va_pct: float, vp_mode: str, _bust: int = 0):
    candles, prices, iv_data = {}, {}, {}
    all_sigs = {}

    for pair in PAIRS:
        df = get_candles(pair, "H1", td_key, 300)
        df = append_order_flow(df, pair)
        candles[pair] = df
        prices[pair] = (get_live_price(pair, td_key) if td_key
                         else (round(float(df["close"].iloc[-1]), 5) if not df.empty else BASE_PRICES[pair]))
        iv_data[pair] = get_iv(df)

    for pair in PAIRS:
        per_tf = {tf: get_signal(get_candles(pair, tf, td_key, 200)) for tf in TIMEFRAMES}
        all_sigs[pair] = per_tf
    signals_df = pd.DataFrame(all_sigs).T.reindex(columns=TIMEFRAMES)

    for pair in PAIRS:
        df_raw = candles[pair]
        df_vp = df_raw.tail(int(fr_bars)) if "Fixed" in vp_mode else df_raw
        vp = build_vp(df_vp, bins=vp_bins_val)
        poc = get_poc(vp)
        vah, val_p = get_va(vp, pct=va_pct)

        headlines = get_news(pair)
        smooth_pre, _ = smoothness_score(df_raw, poc, vah, val_p, sent_score=0)
        pair_conf = sum(get_signal(get_candles(pair, tf, td_key, 200)) for tf in TIMEFRAMES)
        ai_sentiment(pair, headlines, claude_key, confluence_score=pair_conf, smooth=smooth_pre)

    return candles, prices, signals_df, iv_data
