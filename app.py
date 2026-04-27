import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone, timedelta
from streamlit_autorefresh import st_autorefresh

from forex_data_fetcher import fetch_all_pairs, get_1h, get_4h, get_1d, get_pip, PAIRS
from forex_volume_profile import compute_volume_profile, VolumeProfile
from forex_analyst import analyse_pair, FXSignal

# ── SESSION GATE ─────────────────────────────────────────────────────────────

def _claude_allowed_forex() -> tuple[bool, str]:
    """
    Gate Claude calls to London Open and London/NY Overlap only.
    Returns (allowed: bool, reason: str)
    """
    from datetime import datetime, timezone
    utc  = datetime.now(timezone.utc)
    h    = utc.hour
    wday = utc.weekday()

    if wday >= 5:
        return False, "🔴 Forex weekend — Claude resumes Monday 07:00 UTC"

    # London Open: 07:00–12:00 UTC
    if 7 <= h < 12:
        return True, "🟢 London Open — Claude ACTIVE (prime session)"

    # London/NY Overlap: 12:00–17:00 UTC (peak liquidity)
    if 12 <= h < 17:
        return True, "🟢 London/NY Overlap — Claude ACTIVE (peak liquidity)"

    # NY only: 17:00–21:00 UTC — USD pairs only
    if 17 <= h < 21:
        return True, "🟡 NY Session — Claude active (USD pairs priority)"

    # Outside active sessions
    if h < 7:
        mins_to_london = (7 - h) * 60 - utc.minute
        return False, f"🕐 Asian Session — Claude activates at London Open in {mins_to_london}min (07:00 UTC)"

    return False, "🔴 After-hours — Claude resumes at London Open (07:00 UTC)"


def _is_usd_pair(pair: str) -> bool:
    return 'USD' in pair


# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="FX Major Pairs Monitor", layout="wide", page_icon="💱")
st_autorefresh(interval=60000, key="fx_refresh")

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ FX Risk Controls")

if "fx_kill"        not in st.session_state: st.session_state.fx_kill        = False
if "fx_daily_pnl"   not in st.session_state: st.session_state.fx_daily_pnl   = 0.0
if "fx_trades_today"not in st.session_state: st.session_state.fx_trades_today= 0
if "fx_open_pair"   not in st.session_state: st.session_state.fx_open_pair   = None
if "fx_trade_log"   not in st.session_state: st.session_state.fx_trade_log   = []

if st.session_state.fx_kill:
    st.sidebar.error("🔴 KILL SWITCH: ON")
    if st.sidebar.button("🟢 Resume Trading", type="primary"):
        st.session_state.fx_kill = False; st.rerun()
else:
    open_pair = st.session_state.fx_open_pair
    if open_pair:
        st.sidebar.warning(f"📊 Open: **{open_pair}**")
    else:
        st.sidebar.success("🟢 Bot ACTIVE — No open position")
    if st.sidebar.button("🔴 HALT Trading", type="secondary"):
        st.session_state.fx_kill = True; st.rerun()

st.sidebar.divider()
pnl = st.session_state.fx_daily_pnl
pnl_color = "green" if pnl >= 0 else "red"
st.sidebar.markdown(f"**Daily P&L:** :{pnl_color}[${pnl:+.2f}]")
st.sidebar.markdown(f"**Trades today:** {st.session_state.fx_trades_today} / 3")

st.sidebar.divider()
account_size = st.sidebar.number_input("Account size (USD)", 10.0, 10000.0, 100.0, 10.0)
lot_size     = st.sidebar.select_slider("Lot size", [0.01, 0.02, 0.03, 0.05], 0.02)
daily_limit  = st.sidebar.slider("Daily loss limit (%)", 1.0, 10.0, 5.0, 0.5)

if st.session_state.fx_trade_log:
    st.sidebar.divider()
    st.sidebar.markdown("**Today's trades**")
    df_log = pd.DataFrame(st.session_state.fx_trade_log)
    st.sidebar.dataframe(df_log, hide_index=True, use_container_width=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("💱 FX Major Pairs — Volume Profile + Claude AI")
st.caption(
    f"Last Update: {pd.Timestamp.now().strftime('%H:%M:%S')} | "
    "Volume Profile (1H) · POC/VAH/VAL · Mean Reversion & Breakout · Claude Sentiment"
)

# ── FOREX SESSION STATUS ──────────────────────────────────────────────────────
now_utc  = datetime.now(timezone.utc)
now_hour = now_utc.hour
now_wday = now_utc.weekday()

if now_wday >= 5:
    session_msg = "🔴 FOREX CLOSED (Weekend)"
    session_col = "error"
elif 22 <= now_hour or now_hour < 7:
    session_msg = f"🟡 Asian Session · {now_utc.strftime('%H:%M')} UTC"
    session_col = "warning"
elif 7 <= now_hour < 12:
    session_msg = f"🟢 London Session OPEN · {now_utc.strftime('%H:%M')} UTC"
    session_col = "success"
elif 12 <= now_hour < 17:
    session_msg = f"🟢 London + New York Overlap (PEAK LIQUIDITY) · {now_utc.strftime('%H:%M')} UTC"
    session_col = "success"
elif 17 <= now_hour < 22:
    session_msg = f"🟡 New York Session · {now_utc.strftime('%H:%M')} UTC"
    session_col = "warning"
else:
    session_msg = f"🟡 Market Open · {now_utc.strftime('%H:%M')} UTC"
    session_col = "warning"

if session_col == "success":   st.success(session_msg)
elif session_col == "warning": st.warning(session_msg)
else:                          st.error(session_msg)

# ── FETCH DATA ────────────────────────────────────────────────────────────────
with st.spinner("Fetching forex data..."):
    data_ok = fetch_all_pairs()

if not data_ok:
    st.error("⚠️ Forex data unavailable. Retrying on next refresh.")
    st.stop()

# ── CORRELATION MATRIX ────────────────────────────────────────────────────────
def render_correlation_matrix():
    st.subheader("🔗 Correlation Matrix — Lead/Lag Analysis")
    st.caption("Values near +1.0 = highly correlated · Near -1.0 = inverse · Find the 'Leader' pair")

    closes = {}
    for pair in PAIRS:
        df = get_1h(pair)
        if df is not None and not df.empty:
            df.columns = [c.capitalize() for c in df.columns]
            closes[pair] = df['Close'].tail(48)   # last 48 hours

    if len(closes) < 2:
        st.warning("Insufficient data for correlation matrix.")
        return

    df_closes = pd.DataFrame(closes).dropna()
    corr = df_closes.pct_change().dropna().corr().round(2)

    def color_corr(val):
        try:
            v = float(val)
            if v == 1.0:   return "background-color:#333;color:#333"
            elif v > 0.7:  return "background-color:#1a5c1a;color:white"
            elif v > 0.3:  return "background-color:#2d7a2d;color:white"
            elif v > -0.3: return "background-color:#555;color:white"
            elif v > -0.7: return "background-color:#7a2d2d;color:white"
            else:           return "background-color:#5c1a1a;color:white"
        except: return ""

    styled = corr.style.map(color_corr).format("{:.2f}")
    st.dataframe(styled, use_container_width=True)
    st.caption("🟢 Strong positive correlation (pairs move together) · 🔴 Strong negative (inverse pairs)")
    st.divider()


render_correlation_matrix()

# ── VOLUME PROFILE SUMMARY TABLE ──────────────────────────────────────────────
st.subheader("📊 Volume Profile Overview")
st.caption("POC = Fair Value · VAH/VAL = Value Area edges · Signal colour = trade type")

vp_data = {}
summary_rows = []

for pair in PAIRS:
    vp = compute_volume_profile(pair, get_1h(pair), get_4h(pair), get_1d(pair))
    if vp:
        vp_data[pair] = vp
        summary_rows.append({
            "Pair":     pair,
            "Price":    f"{vp.current_price:.5f}",
            "Day %":    f"{vp.day_pct:+.2f}%",
            "POC":      f"{vp.poc:.5f}",
            "VAH":      f"{vp.vah:.5f}",
            "VAL":      f"{vp.val:.5f}",
            "Location": vp.price_location.replace("_", " "),
            "RSI":      f"{vp.rsi_1h:.0f}",
            "ATR%":     f"{vp.atr_pct*100:.3f}%",
            "Signal":   f"{vp.signal_icon} {vp.signal}",
        })

if summary_rows:
    df_summary = pd.DataFrame(summary_rows).set_index("Pair")

    def color_signal(val):
        v = str(val)
        if "MEAN REVERSION" in v:   return "background-color:#1a3a5c;color:#88bbff"
        if "BREAKOUT BUY"   in v:   return "background-color:#1a4a1a;color:#88ff88"
        if "BREAKOUT SELL"  in v:   return "background-color:#4a1a1a;color:#ff8888"
        if "HIGH RISK"      in v:   return "background-color:#333;color:#888"
        if "INTERVENTION"   in v:   return "background-color:#4a0000;color:#ff4444"
        if "POC LEVEL"      in v:   return "background-color:#3a2a00;color:#ffcc44"
        return ""

    def color_day(val):
        try:
            v = float(str(val).replace('%',''))
            if v > 0.3:   return "color:#44ff44;font-weight:bold"
            elif v < -0.3:return "color:#ff4444;font-weight:bold"
            return ""
        except: return ""

    styled = (
        df_summary.style
        .map(color_signal, subset=["Signal"])
        .map(color_day,    subset=["Day %"])
    )
    st.dataframe(styled, use_container_width=True, height=300)

st.divider()

# ── PAIR CARDS ────────────────────────────────────────────────────────────────
st.subheader("🔍 Pair Detail + Claude AI Analysis")

cols = st.columns(3)

for idx, pair in enumerate(PAIRS):
    vp = vp_data.get(pair)
    with cols[idx % 3]:
        if vp is None:
            st.warning(f"{pair}: Data unavailable")
            st.divider()
            continue

        # Price header
        delta_pips = (vp.current_price - vp.prev_price) / get_pip(pair)
        st.metric(
            label=f"💱 {pair}",
            value=f"{vp.current_price:.5f}",
            delta=f"{delta_pips:+.1f} pips",
        )

        # Trend tags
        trend_tag  = "🟢 BULLISH" if vp.trend_bullish else "🔴 BEARISH"
        trend4h_tag= "🟢 Bull" if vp.trend_4h_bullish else "🔴 Bear"
        st.markdown(f"**Trend:** {trend_tag} (Daily) · {trend4h_tag} (4H)")

        # Signal badge
        sig_color_map = {
            "MEAN REVERSION BUY":   "blue",
            "MEAN REVERSION SELL":  "blue",
            "MEAN REVERSION":       "blue",
            "BREAKOUT BUY":         "green",
            "BREAKOUT SELL":        "red",
            "HIGH RISK — Avoid":    "gray",
            "INTERVENTION RISK":    "red",
            "POC LEVEL — Watch":    "orange",
        }
        sc = "gray"
        for k, v in sig_color_map.items():
            if k in vp.signal: sc = v; break
        st.markdown(f"**Signal:** :{sc}[{vp.signal_icon} {vp.signal}]")

        # Volume Profile levels
        with st.expander("📊 Volume Profile Details"):
            c1, c2, c3 = st.columns(3)
            c1.metric("POC", f"{vp.poc:.5f}")
            c2.metric("VAH", f"{vp.vah:.5f}")
            c3.metric("VAL", f"{vp.val:.5f}")

            st.write(f"**Location:** {vp.price_location.replace('_', ' ')}")
            st.write(f"**RSI (1H):** {vp.rsi_1h:.1f}")
            st.write(f"**ATR:** {vp.atr:.5f} ({vp.atr_pct*100:.3f}%)")
            st.write(f"**Vol Intensity:** {vp.volume_intensity:.2f} {'⚠️ Exhaustion' if vp.volume_intensity < 0.7 else '✅ Normal'}")
            st.write(f"**SMA 50 (4H):** {vp.sma50_4h:.5f}")
            st.write(f"**SMA 200 (D):** {vp.sma200_1d:.5f}")

            if vp.hvns:
                st.write(f"**HVN levels:** {', '.join([f'{p:.5f}' for p in vp.hvns[:3]])}")
            if vp.lvns:
                st.write(f"**LVN levels:** {', '.join([f'{p:.5f}' for p in vp.lvns[:3]])}")

        # Claude AI Analysis — only on actionable signals
        skip_signals = ["NEUTRAL", "INSIDE", "Watch"]
        should_analyse = not any(s in vp.signal for s in skip_signals)

        if should_analyse:
            with st.expander("🤖 Claude AI Analysis", expanded=True):

                # Session gate
                session_ok, session_msg = _claude_allowed_forex()

                # During NY-only session, skip non-USD pairs to save tokens
                from datetime import datetime, timezone
                utc_now = datetime.now(timezone.utc)
                if session_ok and 17 <= utc_now.hour < 21 and not _is_usd_pair(pair):
                    session_ok  = False
                    session_msg = f"🟡 NY Session — skipping {pair} (not a USD pair)"

                if not session_ok:
                    st.caption(session_msg)
                    ai = None
                else:
                    # Risk checks
                    trade_allowed = True
                    block_reason  = ""
                    if st.session_state.fx_kill:
                        trade_allowed = False; block_reason = "🔴 Kill switch ON"
                    elif st.session_state.fx_open_pair and st.session_state.fx_open_pair != pair:
                        trade_allowed = False; block_reason = f"⚠️ Position open on {st.session_state.fx_open_pair}"
                    elif st.session_state.fx_trades_today >= 3:
                        trade_allowed = False; block_reason = "⚠️ Max 3 trades/day reached"
                    elif abs(min(st.session_state.fx_daily_pnl, 0)) / account_size * 100 >= daily_limit:
                        trade_allowed = False; block_reason = "🔴 Daily loss limit reached"

                    ai: FXSignal = analyse_pair(vp, lot_size, account_size)

                if ai is None and session_ok:
                    st.info("ℹ️ Signal not yet actionable — Claude monitoring.")
                elif ai is not None:
                    # CB sentiment badge
                    cb_color = {"HAWKISH":"green","DOVISH":"red","NEUTRAL":"gray"}.get(ai.cb_sentiment,"gray")
                    st.markdown(
                        f"**CB Sentiment ({pair[:3]}):** :{cb_color}[{ai.cb_sentiment}] | "
                        f"**Type:** {ai.signal_type.replace('_',' ')}"
                    )

                    a_color = {"BUY":"green","SELL":"red","HOLD":"gray"}.get(ai.action,"gray")
                    st.markdown(
                        f"**Decision:** :{a_color}[{ai.action}] | **Confidence:** {ai.confidence}"
                    )
                    st.caption(f"📊 {ai.reasoning}")
                    if ai.news_summary:
                        st.info(f"🗞️ {ai.news_summary}")

                    if ai.action != "HOLD":
                        e1, e2, e3 = st.columns(3)
                        e1.metric("Entry",      f"{ai.entry:.5f}")
                        e2.metric("Stop Loss",  f"{ai.stop_loss:.5f}",
                                  delta=f"{ai.stop_pips:.1f} pips", delta_color="off")
                        e3.metric("Take Profit",f"{ai.take_profit:.5f}",
                                  delta=f"{ai.target_pips:.1f} pips", delta_color="off")

                        rr = ai.target_pips / ai.stop_pips if ai.stop_pips > 0 else 0
                        risk_usd = ai.stop_pips * lot_size * 10
                        st.caption(
                            f"R:R = {rr:.1f}:1 | Risk ~${risk_usd:.2f} | "
                            f"Spread: {vp.spread_pips} pips"
                        )

                        if trade_allowed:
                            if st.button(
                                f"⚡ Execute {ai.action} {pair}",
                                key=f"fx_exec_{pair}",
                                type="primary",
                            ):
                                # Broker execution
                                try:
                                    from broker import place_order
                                    result = place_order(
                                        ticker=pair,
                                        action=ai.action,
                                        stop_loss=ai.stop_loss,
                                        take_profit=ai.take_profit,
                                        lot_size=lot_size,
                                    )
                                    if result.success:
                                        st.session_state.fx_open_pair    = pair
                                        st.session_state.fx_trades_today += 1
                                        st.session_state.fx_trade_log.append({
                                            "time":   pd.Timestamp.now().strftime("%H:%M"),
                                            "pair":   pair,
                                            "action": ai.action,
                                            "entry":  ai.entry,
                                            "sl":     ai.stop_loss,
                                            "tp":     ai.take_profit,
                                        })
                                        st.success(result.message)
                                    else:
                                        st.error(result.message)
                                except ImportError:
                                    st.success(
                                        f"✅ [DEMO] {ai.action} {pair} @ {ai.entry:.5f} | "
                                        f"SL {ai.stop_loss:.5f} | TP {ai.take_profit:.5f}"
                                    )
                        else:
                            st.error(block_reason)
        else:
            st.caption(f"📍 Price inside Value Area — no trade setup. Monitor for VA edge approach.")

        st.divider()
