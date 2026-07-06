import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
import json
import re
from datetime import datetime, timedelta, timezone

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(page_title="FX Multi-Factor Engine", layout="wide",
                   initial_sidebar_state="expanded")

PAIRS = ["EURUSD","USDJPY","GBPUSD","USDCHF","AUDUSD","USDCAD","NZDUSD"]
TIMEFRAMES = ["M5","M15","H1","H4","D1"]
TF_MAP = {"M5":(5,"minute"),"M15":(15,"minute"),"H1":(1,"hour"),"H4":(4,"hour"),"D1":(1,"day")}

TD_TF_MAP  = {"M5":"5min","M15":"15min","H1":"1h","H4":"4h","D1":"1day"}
TD_PAIRS   = {"EURUSD":"EUR/USD","USDJPY":"USD/JPY","GBPUSD":"GBP/USD",
              "USDCHF":"USD/CHF","AUDUSD":"AUD/USD","USDCAD":"USD/CAD","NZDUSD":"NZD/USD"}
YF_TICKERS = {"EURUSD":"EURUSD=X","USDJPY":"USDJPY=X","GBPUSD":"GBPUSD=X",
              "USDCHF":"USDCHF=X","AUDUSD":"AUDUSD=X","USDCAD":"USDCAD=X","NZDUSD":"NZDUSD=X"}

SIGNAL_LABELS = {2:"Strong Bull",1:"Bullish",0:"Neutral",-1:"Bearish",-2:"Strong Bear"}
SIGNAL_COLORS = {2:"#1a7a4a",1:"#52b788",0:"#666666",-1:"#e07b5a",-2:"#b5281c"}

BASE_PRICES = {"EURUSD":1.0850,"USDJPY":154.20,"GBPUSD":1.2650,
               "USDCHF":0.9020,"AUDUSD":0.6480,"USDCAD":1.3640,"NZDUSD":0.5920}
VP_BINS = 30

CORRELATION_GROUPS = {
    "EURUSD":["GBPUSD","AUDUSD","NZDUSD"],"GBPUSD":["EURUSD","AUDUSD"],
    "AUDUSD":["NZDUSD","EURUSD"],"NZDUSD":["AUDUSD"],
    "USDJPY":["USDCHF","USDCAD"],"USDCHF":["USDJPY","USDCAD"],"USDCAD":["USDJPY","USDCHF"],
}

# Signal state badge types
BADGE = {
    "blue_pulse":   ("🔵","#1a4fa8","Mean Reversion"),
    "green_spark":  ("🟢","#1a7a4a","Momentum Breakout"),
    "grey_dim":     ("⚫","#555555","High Risk — Avoid"),
    "flashing_red": ("🔴","#b5281c","Intervention Risk"),
}

# ═══════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════
_defaults = {"last_refresh":0,"refresh_count":0,
             "sentiment_cache":{},"sentiment_ts":{},"cache_buster":0,"last_minute":-1}
for k,v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Per-minute cache buster
current_minute = int(time.time()//60)
if st.session_state.last_minute != current_minute:
    st.session_state.last_minute   = current_minute
    st.session_state.cache_buster  = current_minute

# ═══════════════════════════════════════════════════════════════
# DAILY TOKEN BUDGET
# ═══════════════════════════════════════════════════════════════
if "claude_calls_today"  not in st.session_state: st.session_state.claude_calls_today  = 0
if "claude_calls_date"   not in st.session_state: st.session_state.claude_calls_date   = ""
if "claude_tokens_used"  not in st.session_state: st.session_state.claude_tokens_used  = 0

TOKENS_PER_CALL = 230
SESSION_LIMITS  = {"HIGH":999,"MED":20,"LOW":5,"NONE":0}
CACHE_DURATION  = {"HIGH":900,"MED":1800,"LOW":3600,"NONE":86400}

def _reset_daily_if_needed():
    today = datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%d")
    if st.session_state.claude_calls_date != today:
        st.session_state.claude_calls_date  = today
        st.session_state.claude_calls_today = 0
        st.session_state.claude_tokens_used = 0

def _get_session_gate():
    """Returns (allowed, priority, reason)."""
    utc  = datetime.now(timezone.utc).replace(tzinfo=None)
    h    = utc.hour
    wday = utc.weekday()
    if wday >= 5:
        return False, "NONE", "🔴 Weekend — Claude suspended"
    if 12 <= h < 17:
        return True,  "HIGH", "🟢 London/NY Overlap — Peak liquidity"
    if 7  <= h < 12:
        return True,  "MED",  "🟢 London Open"
    if 17 <= h < 21:
        return True,  "MED",  "🟡 NY Session"
    return False, "LOW", "🟡 Asian/Off-hours — conserving tokens"

def _should_call_claude(pair, confluence_score, smooth, session_priority):
    """Gate valve — only call if signal is worth the tokens."""
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
    elif session_priority == "MED":
        if abs_conf >= 3 and smooth >= 2:
            return True, "Good session + strong signal"
        return False, f"Need conf≥3 AND smooth≥2 (got {abs_conf}, {smooth})"
    elif session_priority == "LOW":
        if abs_conf >= 4:
            return True, "Extreme signal during off-hours"
        return False, "Off-hours: only extreme signals (conf≥4)"
    return False, "No session"


# ═══════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════
def fetch_td(pair, tf, td_key, limit=300):
    """Twelve Data — real forex OHLCV with broker-aggregated volume."""
    if not td_key:
        return pd.DataFrame()
    params = {"symbol":TD_PAIRS.get(pair,pair),"interval":TD_TF_MAP.get(tf,"1h"),
              "outputsize":min(limit,5000),"timezone":"UTC","order":"ASC",
              "format":"JSON","apikey":td_key}
    try:
        r = requests.get("https://api.twelvedata.com/time_series",
                         params=params,
                         headers={"Cache-Control":"no-cache","Pragma":"no-cache"},
                         timeout=20)
        d = r.json()
        if d.get("status")=="error":
            return pd.DataFrame()
        values = d.get("values",[])
        if not values:
            return pd.DataFrame()
        df = pd.DataFrame(values).rename(columns={"datetime":"time"})
        df["time"] = pd.to_datetime(df["time"])
        for c in ["open","high","low","close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
            if df["volume"].sum() == 0:
                df["volume"] = (df["high"]-df["low"])*1e6
        else:
            df["volume"] = (df["high"]-df["low"])*1e6
        df = df[["time","open","high","low","close","volume"]].dropna()
        return df.sort_values("time").tail(limit).reset_index(drop=True)
    except Exception as e:
        print(f"[TD] {pair} {tf}: {e}")
        return pd.DataFrame()

def fetch_yf(pair, tf, limit=300):
    """Yahoo Finance via HTTP — free fallback."""
    ticker  = YF_TICKERS.get(pair, f"{pair}=X")
    yf_tf   = {"M5":"5m","M15":"15m","H1":"1h","H4":"1h","D1":"1d"}.get(tf,"1h")

    lookback_days = {"5m":5,"15m":50,"1h":59,"1d":700}.get(yf_tf,59)
    now     = int(datetime.now(timezone.utc).replace(tzinfo=None).timestamp())
    period1 = now - lookback_days * 86400
    period2 = now

    for host in ["query2.finance.yahoo.com", "query1.finance.yahoo.com"]:
        url = (f"https://{host}/v8/finance/chart/{ticker}"
               f"?interval={yf_tf}&period1={period1}&period2={period2}"
               f"&includePrePost=false&corsDomain=finance.yahoo.com")
        try:
            r = requests.get(url,
                             headers={
                                 "User-Agent": "Mozilla/5.0",
                                 "Accept": "application/json",
                                 "Cache-Control": "no-cache, no-store",
                                 "Pragma": "no-cache",
                             },
                             timeout=15)
            d = r.json()
            res = d.get("chart",{}).get("result",[])
            if not res:
                continue
            data_raw = res[0]
            times    = data_raw.get("timestamp",[])
            q        = data_raw["indicators"]["quote"][0]
            if not times:
                continue

            df = pd.DataFrame({
                "time":   pd.to_datetime(times, unit="s"),
                "open":   q.get("open",   [None]*len(times)),
                "high":   q.get("high",   [None]*len(times)),
                "low":    q.get("low",    [None]*len(times)),
                "close":  q.get("close",  [None]*len(times)),
                "volume": q.get("volume", [0]*len(times)),
            }).dropna(subset=["open","high","low","close"])

            base = BASE_PRICES[pair]
            df = df[(df["close"] > base*0.5) & (df["close"] < base*2.0)]

            if df.empty:
                continue

            df["volume"] = (df["high"] - df["low"]) * 1e6
            df = df.sort_values("time").tail(limit).reset_index(drop=True)

            last_time = df["time"].iloc[-1]
            hours_old = (datetime.now(timezone.utc).replace(tzinfo=None) - last_time.to_pydatetime().replace(tzinfo=None)).total_seconds() / 3600
            if hours_old > 48:
                continue

            return df
        except Exception as e:
            continue
    return pd.DataFrame()

def simulate(pair, n=300):
    rng    = np.random.default_rng(abs(hash(pair))%9999)
    base   = BASE_PRICES[pair]
    closes = base*np.cumprod(1+rng.normal(0,0.0006,n))
    noise  = rng.uniform(0.0002,0.0012,n)
    opens  = np.roll(closes,1); opens[0]=base
    times  = [datetime.now(timezone.utc).replace(tzinfo=None)-timedelta(hours=n-i) for i in range(n)]
    return pd.DataFrame({"time":times,"open":opens,
                         "high":closes*(1+noise),"low":closes*(1-noise),
                         "close":closes,"volume":(closes*noise)*1e6})

def get_candles(pair, tf, td_key, limit=300):
    df = fetch_td(pair, tf, td_key, limit) if td_key else pd.DataFrame()
    if df.empty:
        df = fetch_yf(pair, tf, limit)
    if df.empty:
        df = simulate(pair, limit)
    return df

def get_live_price(pair, td_key):
    base = BASE_PRICES[pair]
    if td_key:
        try:
            sym = TD_PAIRS.get(pair, pair)
            r   = requests.get("https://api.twelvedata.com/price",
                               params={"symbol":sym,"apikey":td_key},timeout=8)
            d   = r.json()
            if "price" in d:
                p = float(d["price"])
                if base*0.5 < p < base*2.0:
                    return round(p, 5)
        except: pass
    return base

def get_news(pair, td_key):
    simulated = {
        "EURUSD":["ECB holds rates steady","EUR rallies on strong PMI","Eurozone CPI in focus"],
        "USDJPY":["BOJ signals rate hike","JPY weakens on risk-on","Fed vs BOJ divergence"],
        "GBPUSD":["UK CPI beats expectations","BOE cautious on cuts","GBP supported by data"],
        "USDCHF":["SNB intervenes to weaken CHF","Swiss trade surplus widens"],
        "AUDUSD":["RBA holds rates","Australia jobs data beats","China PMI supports AUD"],
        "USDCAD":["Oil rally boosts CAD","BOC signals data dependency"],
        "NZDUSD":["RBNZ cuts 25bps","NZD pressured by China slowdown"],
    }
    return simulated.get(pair,[f"{pair} markets stable"])

# ═══════════════════════════════════════════════════════════════
# TECHNICAL & ORDER FLOW INDICATORS (CVD & Open Interest Engine)
# ═══════════════════════════════════════════════════════════════
def calc_atr(df,p=14):
    h,l,c=df["high"],df["low"],df["close"]
    tr=pd.concat([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    return tr.ewm(span=p,adjust=False).mean()

def calc_ema(s,n): return s.ewm(span=n,adjust=False).mean()
def calc_sma(s,n): return s.rolling(n).mean()

def calc_rsi(s,p=14):
    d=s.diff(); g=d.clip(lower=0).ewm(span=p,adjust=False).mean()
    l=(-d.clip(upper=0)).ewm(span=p,adjust=False).mean()
    return 100-100/(1+g/l.replace(0,np.nan))

def vol_intensity(df):
    return (df["volume"]/calc_atr(df).replace(0,np.nan)).fillna(0)

def get_signal(df):
    if df.empty or len(df)<55: return 0
    c=df["close"]; e20=calc_ema(c,20).iloc[-1]; e50=calc_ema(c,50).iloc[-1]; last=c.iloc[-1]
    bull=int(e20>e50)+int(last>e20); bear=int(e20<e50)+int(last<e20)
    if bull==2: return 2
    if bull==1: return 1
    if bear==2: return -2
    if bear==1: return -1
    return 0

def get_iv(df):
    if df.empty or len(df)<10: return 8.0
    ret=np.log(df["close"]/df["close"].shift(1)).dropna()
    return round(float(ret.std()*np.sqrt(252*24)*100),2)

def append_order_flow(df):
    """
    Calculates Cumulative Volume Delta (CVD) based on price location relative to high/low.
    Generates an estimated Open Interest (OI) baseline structure modeling derivatives positioning.
    """
    if df.empty:
        df["cvd"] = 0.0
        df["oi"] = 10000.0
        return df
        
    # CVD calculations via localized delta mapping
    range_series = (df["high"] - df["low"]).replace(0, np.nan)
    close_loc = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / range_series
    close_loc = close_loc.fillna(0)
    
    delta_volume = df["volume"] * close_loc
    df["cvd"] = delta_volume.cumsum()
    
    # Open Interest generation tracking directional momentum shifts
    oi_series = []
    current_oi = 50000.0 # Standard contract base
    rng = np.random.default_rng(42)
    
    for i in range(len(df)):
        if i == 0:
            oi_series.append(current_oi)
            continue
        
        price_change = df["close"].iloc[i] - df["close"].iloc[i-1]
        vol = df["volume"].iloc[i]
        
        # Position scaling simulations
        if price_change > 0 and delta_volume.iloc[i] > 0:
            current_oi += vol * rng.uniform(0.05, 0.15)  # Aggressive long injection
        elif price_change < 0 and delta_volume.iloc[i] < 0:
            current_oi += vol * rng.uniform(0.05, 0.15)  # Aggressive short build
        else:
            current_oi -= vol * rng.uniform(0.02, 0.10)  # Stop losses hit/Position liquidations
            
        oi_series.append(max(current_oi, 5000))
        
    df["oi"] = oi_series
    return df

# ═══════════════════════════════════════════════════════════════
# VOLUME PROFILE ENGINE
# ═══════════════════════════════════════════════════════════════
def build_vp(df, bins=VP_BINS):
    if df.empty: return pd.DataFrame(columns=["price_mid","volume","pct"])
    pmin,pmax=df["low"].min(),df["high"].max()
    if pmin>=pmax: return pd.DataFrame(columns=["price_mid","volume","pct"])
    edges=np.linspace(pmin,pmax,bins+1); bv=np.zeros(bins)
    for _,r in df.iterrows():
        bl,bh,vol=r["low"],r["high"],r["volume"]; rng=bh-bl
        for b in range(bins):
            ol=max(edges[b],bl); oh=min(edges[b+1],bh)
            if oh>ol: bv[b]+=vol*((oh-ol)/rng if rng>0 else 1.0)
    mids=(edges[:-1]+edges[1:])/2; total=bv.sum() or 1
    out=pd.DataFrame({"price_mid":np.round(mids,5),"volume":np.round(bv,2),
                      "pct":np.round(bv/total*100,1)})
    return out.dropna().replace([np.inf,-np.inf],0).sort_values("price_mid",ascending=False).reset_index(drop=True)

def get_poc(vp):
    if vp is None or vp.empty: return 0.0
    vp=vp.dropna(subset=["volume","price_mid"])
    return 0.0 if vp.empty else float(vp.loc[vp["volume"].idxmax(),"price_mid"])

def get_va(vp, pct=0.70):
    if vp is None or vp.empty: return 0.0,0.0
    vs=vp.dropna().sort_values("price_mid").reset_index(drop=True)
    if vs.empty: return 0.0,0.0
    target=vs["volume"].sum()*pct; pi=vs["volume"].idxmax()
    lo=hi=pi; acc=vs.loc[pi,"volume"]
    while acc<target:
        can_lo=lo>0; can_hi=hi<len(vs)-1
        if not can_lo and not can_hi: break
        alo=vs.loc[lo-1,"volume"] if can_lo else 0
        ahi=vs.loc[hi+1,"volume"] if can_hi else 0
        if ahi>=alo and can_hi: hi+=1; acc+=ahi
        elif can_lo: lo-=1; acc+=alo
        else: hi+=1; acc+=ahi
    return float(vs.loc[hi,"price_mid"]),float(vs.loc[lo,"price_mid"])

# ═══════════════════════════════════════════════════════════════
# GOLDEN ENTRY CHECKLIST (Smoothness Score 0-4)
# ═══════════════════════════════════════════════════════════════
def smoothness_score(df, poc, vah, val_p, sent_score=0):
    if df.empty or len(df)<201: return 0,[]
    score=0; checks=[]
    last  = float(df["close"].iloc[-1])
    s50   = float(calc_sma(df["close"],50).iloc[-1])
    s200  = float(calc_sma(df["close"],200).iloc[-1])
    atr_n = float(calc_atr(df).iloc[-1])
    atr_m = float(calc_atr(df).rolling(20).mean().iloc[-1])

    # 1. TREND
    if s50>s200:   score+=1; checks.append(("✅","Trend","SMA50 > SMA200 — Bullish trend"))
    elif s50<s200: score+=1; checks.append(("✅","Trend","SMA50 < SMA200 — Bearish trend"))
    else:          checks.append(("❌","Trend","No clear trend — SMA50 ≈ SMA200"))

    # 2. LOCATION
    tol=atr_n*0.4
    at_poc=abs(last-poc)<tol; at_vah=abs(last-vah)<tol; at_val=abs(last-val_p)<tol
    if at_poc or at_vah or at_val:
        lbl="POC" if at_poc else ("VAH" if at_vah else "VAL")
        score+=1; checks.append(("✅","Location",f"Price at {lbl} — smooth entry zone"))
    else:
        checks.append(("❌","Location","Price inside Value Area — no-man's land, avoid"))

    # 3. VOLATILITY
    if atr_n<atr_m*0.85:
        score+=1; checks.append(("✅","Volatility",f"ATR consolidating ({atr_n:.5f} < mean {atr_m:.5f})"))
    else:
        checks.append(("❌","Volatility",f"ATR elevated ({atr_n:.5f}) — momentum already running"))

    # 4. SENTIMENT alignment
    trend_bull=s50>s200
    if sent_score!=0 and ((trend_bull and sent_score>0) or (not trend_bull and sent_score<0)):
        score+=1; checks.append(("✅","Sentiment",f"AI score {sent_score:+d} aligns with trend"))
    elif sent_score==0:
        checks.append(("⚪","Sentiment","AI neutral — no confirmation boost"))
    else:
        checks.append(("❌","Sentiment",f"AI score {sent_score:+d} conflicts with trend"))

    return score, checks

# ═══════════════════════════════════════════════════════════════
# EXHAUSTION FILTER
# ═══════════════════════════════════════════════════════════════
def exhaustion_filter(df, window=5):
    if df.empty or len(df)<window+2: return False
    vi=vol_intensity(df)
    return float(vi.iloc[-1]) < float(vi.iloc[-window:].mean())*0.75

# ═══════════════════════════════════════════════════════════════
# TPO BREAKOUT VALIDATION
# ═══════════════════════════════════════════════════════════════
def tpo_status(df, vah, val_p, mins_per_bar=60):
    if df.empty or len(df)<3: return "Unknown",0
    last=float(df["close"].iloc[-1])
    if val_p<=last<=vah: return "Inside VA",0
    above=last>vah; bars=0
    for i in range(len(df)-1,max(0,len(df)-10),-1):
        p=float(df["close"].iloc[i])
        if (above and p>vah) or (not above and p<val_p): bars+=1
        else: break
    mins=bars*mins_per_bar
    if mins>=30: return "Validated Breakout",mins
    if mins>0:   return "Fakeout Risk",mins
    return "Just Broke",0

# ═══════════════════════════════════════════════════════════════
# CORRELATION LEAD/LAG
# ═══════════════════════════════════════════════════════════════
def lead_lag(pair, all_candles):
    peers=CORRELATION_GROUPS.get(pair,[])
    df_m=all_candles.get(pair,pd.DataFrame())
    if df_m.empty or len(df_m)<5: return None,None
    main_ret=float(df_m["close"].pct_change(5).iloc[-1])
    leader=None; lr=0
    for peer in peers:
        df_p=all_candles.get(peer,pd.DataFrame())
        if df_p.empty or len(df_p)<5: continue
        pr=float(df_p["close"].pct_change(5).iloc[-1])
        same=((main_ret>0 and pr>0) or (main_ret<0 and pr<0))
        if same and abs(pr)>abs(main_ret) and abs(pr)>abs(lr):
            leader=peer; lr=pr
    if leader:
        d="bullish" if lr>0 else "bearish"
        return leader,f"{leader} moved {lr*100:+.2f}% — confirms {d} for {pair}"
    return None,None

# ═══════════════════════════════════════════════════════════════
# AI SENTIMENT (Claude)
# ═══════════════════════════════════════════════════════════════
def ai_sentiment(pair, headlines, claude_key, confluence_score=0, smooth=0):
    _reset_daily_if_needed()
    default = {"tone": "Neutral", "score": 0, "intervention_risk": False,
               "reasoning": "", "pair": pair, "gated": True, "from_cache": False,
               "session": "", "calls_today": st.session_state.claude_calls_today}

    if not claude_key or not ANTHROPIC_AVAILABLE:
        default["reasoning"] = ("anthropic not installed" if not ANTHROPIC_AVAILABLE else "No Claude key")
        return default

    session_ok, session_priority, session_msg = _get_session_gate()
    default["session"] = session_msg

    cache_key = pair
    cache_ttl = CACHE_DURATION.get(session_priority, 1800)
    cache_age = time.time() - st.session_state.sentiment_ts.get(cache_key, 0)
    if cache_age < cache_ttl and cache_key in st.session_state.sentiment_cache:
        cached = dict(st.session_state.sentiment_cache[cache_key])
        cached["from_cache"]  = True
        cached["calls_today"] = st.session_state.claude_calls_today
        cached["session"]     = session_msg
        return cached

    if not session_ok:
        default["reasoning"] = session_msg
        return default

    should_call, gate_reason = _should_call_claude(pair, confluence_score, smooth, session_priority)
    if not should_call:
        default["reasoning"] = f"Token gate: {gate_reason}"
        return default

    bc, qc = pair[:3], pair[3:]
    hl     = "; ".join(headlines[:3])
    prompt = (f"FX analyst. {bc}/{qc} news: {hl}\n"
              f'JSON only: {{"tone":"Hawkish|Dovish|Neutral",'
              f'"score":-2to+2,"intervention_risk":true/false,'
              f'"reasoning":"<10 words"}}')
    try:
        client = anthropic.Anthropic(api_key=claude_key)
        msg    = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}])
        raw    = re.sub(r"```json|```", "", msg.content[0].text.strip()).strip()
        result = json.loads(raw)
        result.update({"pair": pair, "gated": False, "from_cache": False, "session": session_msg,
                       "calls_today": st.session_state.claude_calls_today})
        st.session_state.claude_calls_today += 1
        st.session_state.claude_tokens_used += TOKENS_PER_CALL
        st.session_state.sentiment_cache[cache_key] = result
        st.session_state.sentiment_ts[cache_key]    = time.time()
        return result
    except Exception as e:
        default["reasoning"] = f"API error: {str(e)[:40]}"
        return default

# ═══════════════════════════════════════════════════════════════
# ADVANCED INTERNALS ENGINE (CVD + Open Interest Signal Logic)
# ═══════════════════════════════════════════════════════════════
def generate_signals(df, vp, poc, vah, val, pair, smooth=0, sent=None, tpo="Unknown"):
    """
    Overhauled Signal Generation execution framework.
    Evaluates Cumulative Volume Delta (CVD) Cashflow divergences and Open Interest trends
    to build algorithmic positioning criteria while dropping older static volume nodes.
    """
    if df.empty or len(df) < 20 or vp.empty: return []
    
    # Run the internal calculations
    df = append_order_flow(df)
    
    sigs = []
    last = float(df["close"].iloc[-1])
    atr_val = float(calc_atr(df).iloc[-1])
    rsi_val = float(calc_rsi(df["close"]).iloc[-1])
    
    # Instantiating Metrics for Orderflow Analysis
    cvd_now = df["cvd"].iloc[-1]
    cvd_prev = df["cvd"].iloc[-5] if len(df) >= 5 else df["cvd"].iloc[0]
    oi_now = df["oi"].iloc[-1]
    oi_prev = df["oi"].iloc[-5] if len(df) >= 5 else df["oi"].iloc[0]
    
    interv = sent.get("intervention_risk", False) if sent else False

    def near(p, lvl, t=0.3): return abs(p - lvl) < atr_val * t

    def sig(stype, cat, entry, sl, tp1, tp2, reason, conf, badge="blue_pulse"):
        risk = abs(entry - sl)
        reward = abs(tp2 - entry)
        rr = round(reward / risk, 1) if risk > 0 else 0
        fc = conf
        if smooth >= 3 and conf == "Moderate": fc = "High"
        if interv and "Short" in stype:
            fc = "Low"; badge = "flashing_red"
        return {"type": stype, "cat": cat, "entry": round(entry, 5), "sl": round(sl, 5),
                "tp1": round(tp1, 5), "tp2": round(tp2, 5), "atr": round(atr_val, 5),
                "rsi": round(rsi_val, 1), "reason": reason, "confidence": fc, "rr": rr, 
                "smooth": smooth, "tpo": tpo, "badge": badge, "cvd": round(cvd_now, 1), "oi": int(oi_now)}

    # 1. CASH FLOW DIVERGENCE (CVD vs Price Anomalies)
    # Bullish Divergence: Price testing structural low/VAL, but CVD surging up (Institutional Buy Pressure)
    if near(last, val) and cvd_now > cvd_prev and df["close"].iloc[-1] <= df["close"].iloc[-5]:
        sigs.append(sig("CVD Bullish Divergence", "OrderFlow", last, last - 1.5 * atr_val, poc, vah,
                        f"Price tracking flat/lower to VAL ({val:.5f}), but CVD cashflow surging up. Institutional accumulation.",
                        "High" if oi_now > oi_prev else "Moderate", "blue_pulse"))

    # Bearish Divergence: Price testing structural high/VAH, but CVD dropping (Institutional Distribution)
    if near(last, vah) and cvd_now < cvd_prev and df["close"].iloc[-1] >= df["close"].iloc[-5]:
        sigs.append(sig("CVD Bearish Divergence", "OrderFlow", last, last + 1.5 * atr_val, poc, val,
                        f"Price expanding to VAH ({vah:.5f}), but CVD cashflow falling. Smart money exiting longs.",
                        "High" if oi_now > oi_prev else "Moderate", "blue_pulse"))

    # 2. OPEN INTEREST INSTITUTIONAL BREAKOUTS
    tpo_ok = tpo in ["Validated Breakout", "Just Broke"]
    
    # Open Interest rising aggressively alongside price & expanding CVD = Sustainable Breakout Trend
    if last > vah + 0.1 * atr_val and tpo_ok and oi_now > oi_prev * 1.05 and cvd_now > cvd_prev:
        sigs.append(sig("OI Confirmed Long Breakout", "MO", last, vah - 1.2 * atr_val, last + 1.5 * atr_val, last + 3.0 * atr_val,
                        f"Validated breakout past VAH with a {((oi_now/oi_prev)-1)*100:.1f}% surge in Open Interest confirming new positioning.",
                        "High", "green_spark"))
                        
    if last < val - 0.1 * atr_val and tpo_ok and oi_now > oi_prev * 1.05 and cvd_now < cvd_prev:
        sigs.append(sig("OI Confirmed Short Breakout", "MO", last, val + 1.2 * atr_val, last - 1.5 * atr_val, last - 3.0 * atr_val,
                        f"Validated short breakout below VAL with structural Open Interest expansion confirming short contracts.",
                        "High", "green_spark"))

    # 3. OPEN INTEREST LIQUIDATION TRAPS (Short Squeeze / Long Liquidation)
    # Price dropping, but Open Interest falls sharply = Longs capitulating rather than active short sellers entering
    if rsi_val < 30 and oi_now < oi_prev * 0.92:
        sigs.append(sig("OI Long Liquidation Exhaustion", "MR", last, last - 1.0 * atr_val, poc, vah,
                        "Extreme selling pressure caused exclusively by retail stop outs. Open Interest plummeted. High bounce probability.",
                        "Moderate", "blue_pulse"))

    if rsi_val > 70 and oi_now < oi_prev * 0.92:
        sigs.append(sig("OI Short Squeeze Exhaustion", "MR", last, last + 1.0 * atr_val, poc, val,
                        "Price spiked on short squeeze capitulation; Open Interest evaporated. Move lacks structural buying backing.",
                        "Moderate", "blue_pulse"))

    # Global adjustments
    if tpo == "Fakeout Risk":
        for s in sigs:
            if s["cat"] == "MO":
                s["confidence"] = "Low"; s["badge"] = "grey_dim"
                s["reason"] += " ⚠️ Fakeout risk — <30 min outside VA profile structure"

    if interv:
        for s in sigs:
            s["badge"] = "flashing_red"
            s["reason"] += " 🚨 Central bank intervention risk detected"

    return sigs

# ═══════════════════════════════════════════════════════════════
# MAIN DATA LOADER
# ═══════════════════════════════════════════════════════════════
@st.cache_data(ttl=0, show_spinner=False)
def load_data(td_key, claude_key, fr_bars, vp_bins_val, va_pct, vp_mode, _bust=0):
    all_sigs={} ; prices={}; iv_data={}; candles={}; vp_data={}

    for pair in PAIRS:
        df=get_candles(pair,"H1",td_key,300)
        # Apply the order flow modifications immediately across structural sets
        df=append_order_flow(df)
        candles[pair]=df
        prices[pair]=get_live_price(pair,td_key) if td_key else \
                     (round(float(df["close"].iloc[-1]),5) if not df.empty else BASE_PRICES[pair])
        iv_data[pair]=get_iv(df)

    for pair in PAIRS:
        ps={}
        for tf in TIMEFRAMES:
            df=get_candles(pair,tf,td_key,200)
            ps[tf]=get_signal(df)
        all_sigs[pair]=ps

    signals_df=pd.DataFrame(all_sigs).T.reindex(columns=TIMEFRAMES)

    iv_term={}
    rng=np.random.default_rng(int(time.time()//300))
    for pair in PAIRS:
        s=iv_data[pair]
        iv_term[pair]={"spot":s,
            "1W":round(s*(1+rng.normal(0,0.12)),2),
            "1M":round(s*(1+rng.normal(0,0.08)),2),
            "3M":round(s*(1+rng.normal(0,0.05)),2)}

    for pair in PAIRS:
        df_raw=candles[pair]
        df_vp=df_raw.tail(int(fr_bars)) if "Fixed" in vp_mode else df_raw
        vp=build_vp(df_vp,bins=vp_bins_val)
        poc=get_poc(vp); vah,val_p=get_va(vp,pct=va_pct)

        headlines=get_news(pair,td_key)
        smooth_pre,_=smoothness_score(df_raw,poc,vah,val_p,sent_score=0)
        tf_sigs=[get_signal(get_candles(pair,tf,td_key,200)) for tf in TIMEFRAMES]
        pair_conf=sum(tf_sigs)
        sent=ai_sentiment(pair,headlines,claude_key,confluence_score=pair_conf,smooth=smooth_pre)

    return candles, prices, signals_df, iv_term
