"""
forex_volume_profile.py
-----------------------
Calculates Volume Profile metrics from 1H OHLCV data:
  - POC  : Point of Control (highest volume price level)
  - VAH  : Value Area High  (top of 70% volume zone)
  - VAL  : Value Area Low   (bottom of 70% volume zone)
  - HVNs : High Volume Nodes (significant support/resistance)
  - LVNs : Low Volume Nodes  (liquidity voids — fast move zones)
  - ATR  : Average True Range (14-period)
  - Signal classification: Mean Reversion / Breakout / High Risk / Intervention Risk
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class VolumeProfile:
    pair: str
    poc: float              # Point of Control price
    vah: float              # Value Area High
    val: float              # Value Area Low
    current_price: float
    prev_price: float
    atr: float              # 14-period ATR
    atr_pct: float          # ATR as % of price
    sma50_4h: float         # SMA 50 on 4H
    sma200_1d: float        # SMA 200 on daily
    hvns: List[float]       # High Volume Node price levels
    lvns: List[float]       # Low Volume Node price levels
    volume_intensity: float # current volume / ATR (exhaustion indicator)
    price_location: str     # "AT_VAH" | "AT_VAL" | "AT_POC" | "ABOVE_VAH" | "BELOW_VAL" | "INSIDE_VA"
    signal: str             # "MEAN_REVERSION" | "BREAKOUT" | "HIGH_RISK" | "NEUTRAL"
    signal_color: str
    signal_icon: str
    trend_bullish: bool     # price > SMA200 daily
    trend_4h_bullish: bool  # price > SMA50 4H
    rsi_1h: float
    day_pct: float          # % change today
    spread_pips: float      # estimated spread


def _calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    try:
        high, low, close = df['High'], df['Low'], df['Close']
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])
    except Exception:
        return 0.0


def _calc_rsi(close: pd.Series, period: int = 14) -> float:
    try:
        delta = close.diff()
        gain  = delta.where(delta > 0, 0).rolling(period).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs    = gain / loss
        return float(100 - (100 / (1 + rs)).iloc[-1])
    except Exception:
        return 50.0


def _build_volume_profile(df: pd.DataFrame, bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a volume profile histogram from OHLCV data.
    Returns (price_levels, volume_at_each_level).
    """
    price_min = df['Low'].min()
    price_max = df['High'].max()
    price_bins = np.linspace(price_min, price_max, bins + 1)
    volume_profile = np.zeros(bins)

    for _, row in df.iterrows():
        # Distribute candle volume across price range (VWAP approximation)
        candle_low  = row['Low']
        candle_high = row['High']
        candle_vol  = row['Volume'] if row['Volume'] > 0 else 1.0

        # Find which bins this candle spans
        low_idx  = np.searchsorted(price_bins, candle_low,  side='left')
        high_idx = np.searchsorted(price_bins, candle_high, side='right')
        low_idx  = max(0, min(low_idx,  bins - 1))
        high_idx = max(0, min(high_idx, bins))

        span = high_idx - low_idx
        if span <= 0:
            span = 1
            high_idx = low_idx + 1

        vol_per_bin = candle_vol / span
        volume_profile[low_idx:high_idx] += vol_per_bin

    price_levels = (price_bins[:-1] + price_bins[1:]) / 2
    return price_levels, volume_profile


def _calc_value_area(price_levels: np.ndarray, volume_profile: np.ndarray,
                     va_pct: float = 0.70) -> Tuple[float, float, float]:
    """
    Calculate POC, VAH, VAL from volume profile.
    VA covers va_pct (70%) of total volume.
    """
    total_vol  = volume_profile.sum()
    target_vol = total_vol * va_pct
    poc_idx    = np.argmax(volume_profile)
    poc        = float(price_levels[poc_idx])

    # Expand outward from POC until 70% of volume is covered
    va_vol  = volume_profile[poc_idx]
    low_idx = poc_idx
    high_idx= poc_idx

    while va_vol < target_vol:
        can_go_up   = high_idx < len(volume_profile) - 1
        can_go_down = low_idx > 0

        vol_up   = volume_profile[high_idx + 1] if can_go_up   else 0
        vol_down = volume_profile[low_idx - 1]  if can_go_down else 0

        if not can_go_up and not can_go_down:
            break

        if vol_up >= vol_down and can_go_up:
            high_idx += 1
            va_vol   += vol_up
        elif can_go_down:
            low_idx -= 1
            va_vol  += vol_down
        else:
            high_idx += 1
            va_vol   += vol_up

    vah = float(price_levels[high_idx])
    val = float(price_levels[low_idx])
    return poc, vah, val


def _find_hvn_lvn(price_levels: np.ndarray, volume_profile: np.ndarray,
                  n_nodes: int = 3) -> Tuple[List[float], List[float]]:
    """Find top HVN and LVN price levels."""
    mean_vol = volume_profile.mean()
    std_vol  = volume_profile.std()

    hvn_mask = volume_profile > (mean_vol + 0.5 * std_vol)
    lvn_mask = volume_profile < (mean_vol - 0.5 * std_vol)

    hvn_prices = price_levels[hvn_mask].tolist()
    lvn_prices = price_levels[lvn_mask].tolist()

    # Return top N by volume magnitude
    hvn_sorted = sorted(
        [(price_levels[i], volume_profile[i]) for i in range(len(price_levels)) if hvn_mask[i]],
        key=lambda x: -x[1]
    )[:n_nodes]
    lvn_sorted = sorted(
        [(price_levels[i], volume_profile[i]) for i in range(len(price_levels)) if lvn_mask[i]],
        key=lambda x: x[1]
    )[:n_nodes]

    return [p for p, _ in hvn_sorted], [p for p, _ in lvn_sorted]


def _classify_location(price: float, vah: float, val: float, poc: float, atr: float) -> str:
    tolerance = atr * 0.3
    if abs(price - vah) <= tolerance: return "AT_VAH"
    if abs(price - val) <= tolerance: return "AT_VAL"
    if abs(price - poc) <= tolerance: return "AT_POC"
    if price > vah:                   return "ABOVE_VAH"
    if price < val:                   return "BELOW_VAL"
    return "INSIDE_VA"


def _classify_signal(location: str, trend_bullish: bool, trend_4h: bool,
                     rsi: float, atr_pct: float, vol_intensity: float) -> Tuple[str, str, str]:
    """
    Returns (signal, color, icon) based on Volume Profile location + filters.
    Logic from your strategy screenshots.
    """
    high_atr = atr_pct > 0.008   # ATR > 0.8% = high volatility
    low_atr  = atr_pct < 0.004   # ATR < 0.4% = consolidation (ideal entry)
    exhaustion = vol_intensity < 0.7   # volume intensity dropping = exhaustion

    # Intervention Risk — extreme RSI or very high ATR
    if atr_pct > 0.015 or (rsi > 80) or (rsi < 20):
        return "INTERVENTION RISK", "#ff2222", "🚨"

    # High Risk — inside value area with high ATR (no man's land)
    if location == "INSIDE_VA" and high_atr:
        return "HIGH RISK — Avoid", "#888888", "⚫"

    # Mean Reversion — price at VA edge with exhaustion signals
    if location in ("AT_VAL", "AT_VAH") and exhaustion:
        if location == "AT_VAL" and trend_bullish:
            return "MEAN REVERSION BUY", "#3399ff", "🔵"
        if location == "AT_VAH" and not trend_bullish:
            return "MEAN REVERSION SELL", "#3399ff", "🔵"
        return "MEAN REVERSION", "#3399ff", "🔵"

    # Breakout Momentum — price clearing levels with trend alignment
    if location == "ABOVE_VAH" and trend_bullish and trend_4h and low_atr:
        return "BREAKOUT BUY", "#00cc44", "🟢"
    if location == "BELOW_VAL" and not trend_bullish and not trend_4h and low_atr:
        return "BREAKOUT SELL", "#ff4444", "🔴"

    # POC bounce — mean reversion to fair value
    if location == "AT_POC":
        return "POC LEVEL — Watch", "#ffaa00", "🟡"

    # Inside VA — neutral / avoid
    if location == "INSIDE_VA":
        return "NEUTRAL — Inside VA", "#666666", "⚪"

    # Above/below VA but conditions not fully met
    if location == "ABOVE_VAH":
        return "ABOVE VAH — Wait Pullback", "#88cc88", "🟢"
    if location == "BELOW_VAL":
        return "BELOW VAL — Wait Rally", "#cc8888", "🔴"

    return "NEUTRAL", "#666666", "⚪"


def compute_volume_profile(pair: str, df_1h: pd.DataFrame,
                            df_4h: Optional[pd.DataFrame],
                            df_1d: Optional[pd.DataFrame]) -> Optional[VolumeProfile]:
    """Main entry point. Returns VolumeProfile or None on failure."""
    try:
        if df_1h is None or len(df_1h) < 24:
            return None

        df_1h = df_1h.copy()
        df_1h.columns = [c.capitalize() for c in df_1h.columns]

        # Use last 5 days of 1H data for intraday volume profile
        df_vp = df_1h.tail(5 * 24)

        current_price = float(df_1h['Close'].iloc[-1])
        prev_price    = float(df_1h['Close'].iloc[-2])
        open_price    = float(df_1h['Close'].iloc[-24]) if len(df_1h) >= 24 else current_price
        day_pct       = (current_price - open_price) / open_price * 100

        # Volume profile
        price_levels, volume_profile = _build_volume_profile(df_vp)
        poc, vah, val = _calc_value_area(price_levels, volume_profile)
        hvns, lvns    = _find_hvn_lvn(price_levels, volume_profile)

        # ATR
        atr     = _calc_atr(df_1h)
        atr_pct = atr / current_price if current_price > 0 else 0

        # RSI
        rsi_1h = _calc_rsi(df_1h['Close'])

        # Volume intensity (volume / ATR — exhaustion signal)
        recent_vol = df_1h['Volume'].tail(5).mean()
        avg_vol    = df_1h['Volume'].tail(50).mean()
        vol_intensity = recent_vol / avg_vol if avg_vol > 0 else 1.0

        # SMA 50 on 4H
        sma50_4h = 0.0
        trend_4h = True
        if df_4h is not None and len(df_4h) >= 50:
            df4 = df_4h.copy()
            df4.columns = [c.capitalize() for c in df4.columns]
            sma50_4h = float(df4['Close'].rolling(50).mean().iloc[-1])
            trend_4h = current_price > sma50_4h

        # SMA 200 on daily
        sma200_1d = 0.0
        trend_1d  = True
        if df_1d is not None and len(df_1d) >= 200:
            d1 = df_1d.copy()
            d1.columns = [c.capitalize() for c in d1.columns]
            sma200_1d = float(d1['Close'].rolling(200).mean().iloc[-1])
            trend_1d  = current_price > sma200_1d

        # Location + signal
        location = _classify_location(current_price, vah, val, poc, atr)
        signal, color, icon = _classify_signal(
            location, trend_1d, trend_4h, rsi_1h, atr_pct, vol_intensity
        )

        # Estimated spread (typical for majors)
        spread_map = {'EURUSD':1.0,'GBPUSD':1.2,'USDJPY':1.0,
                      'USDCHF':1.5,'AUDUSD':1.2,'USDCAD':1.5,'NZDUSD':1.8}
        spread_pips = spread_map.get(pair, 1.5)

        return VolumeProfile(
            pair=pair, poc=poc, vah=vah, val=val,
            current_price=current_price, prev_price=prev_price,
            atr=atr, atr_pct=atr_pct, sma50_4h=sma50_4h, sma200_1d=sma200_1d,
            hvns=hvns, lvns=lvns, volume_intensity=vol_intensity,
            price_location=location, signal=signal, signal_color=color, signal_icon=icon,
            trend_bullish=trend_1d, trend_4h_bullish=trend_4h,
            rsi_1h=rsi_1h, day_pct=day_pct, spread_pips=spread_pips,
        )

    except Exception as e:
        print(f"[volume_profile] Error for {pair}: {e}")
        return None
