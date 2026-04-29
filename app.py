import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
import json
import re
from datetime import datetime, timedelta

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

# Signal state badge types (from your screenshots)
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
# DAILY TOKEN BUDGET — defined early so sidebar can use it
# ═══════════════════════════════════════════════════════════════
if "claude_calls_today"  not in st.session_state: st.session_state.claude_calls_today  = 0
if "claude_calls_date"   not in st.session_state: st.session_state.claude_calls_date   = ""
if "claude_tokens_used"  not in st.session_state: st.session_state.claude_tokens_used  = 0

TOKENS_PER_CALL = 230
SESSION_LIMITS  = {"HIGH":999,"MED":20,"LOW":5,"NONE":0}
CACHE_DURATION  = {"HIGH":900,"MED":1800,"LOW":3600,"NONE":86400}

def _reset_daily_if_needed():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if st.session_state.claude_calls_date != today:
        st.session_state.claude_calls_date  = today
        st.session_state.claude_calls_today = 0
        st.session_state.claude_tokens_used = 0

def _get_session_gate():
    """Returns (allowed, priority, reason)."""
    utc  = datetime.utcnow()
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
    """
    Yahoo Finance via HTTP — free fallback.
    Uses short period (recent=True) to avoid stale cached data.
    """
    ticker  = YF_TICKERS.get(pair, f"{pair}=X")
    yf_tf   = {"M5":"5m","M15":"15m","H1":"1h","H4":"1h","D1":"1d"}.get(tf,"1h")

    # Use SHORT lookback to force fresh recent data — avoids Yahoo cache
    lookback_days = {"5m":5,"15m":50,"1h":59,"1d":700}.get(yf_tf,59)
    now     = int(datetime.utcnow().timestamp())
    period1 = now - lookback_days * 86400
    period2 = now

    # Try query2 first (less cached), then query1
    for host in ["query2.finance.yahoo.com", "query1.finance.yahoo.com"]:
        url = (f"https://{host}/v8/finance/chart/{ticker}"
               f"?interval={yf_tf}&period1={period1}&period2={period2}"
               f"&includePrePost=false&corsDomain=finance.yahoo.com")
        try:
            r = requests.get(url,
                             headers={
                                 "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
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

            # Validate prices are in realistic range for the pair
            base = BASE_PRICES[pair]
            df = df[(df["close"] > base*0.5) & (df["close"] < base*2.0)]

            if df.empty:
                continue

            df["volume"] = (df["high"] - df["low"]) * 1e6
            df = df.sort_values("time").tail(limit).reset_index(drop=True)

            # Verify last bar is recent (within 48 hours for H1)
            last_time = df["time"].iloc[-1]
            hours_old = (datetime.utcnow() - last_time.to_pydatetime().replace(tzinfo=None)).total_seconds() / 3600
            if hours_old > 48:
                print(f"[YF] {pair} data is {hours_old:.0f}h old — stale, skipping")
                continue

            print(f"[YF] {pair} {tf}: {len(df)} bars, last={last_time}, price={df['close'].iloc[-1]:.5f}")
            return df

        except Exception as e:
            print(f"[YF] {pair} {tf} from {host}: {e}")
            continue

    return pd.DataFrame()

def simulate(pair, n=300):
    rng    = np.random.default_rng(abs(hash(pair))%9999)
    base   = BASE_PRICES[pair]
    closes = base*np.cumprod(1+rng.normal(0,0.0006,n))
    noise  = rng.uniform(0.0002,0.0012,n)
    opens  = np.roll(closes,1); opens[0]=base
    times  = [datetime.utcnow()-timedelta(hours=n-i) for i in range(n)]
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

    # Priority 1: Twelve Data real-time price
    if td_key:
        try:
            sym = TD_PAIRS.get(pair, pair)
            r   = requests.get("https://api.twelvedata.com/price",
                               params={"symbol":sym,"apikey":td_key},timeout=8)
            d   = r.json()
            if "price" in d:
                p = float(d["price"])
                # Validate realistic range
                if base*0.5 < p < base*2.0:
                    return round(p, 5)
        except: pass

    # Priority 2: Yahoo Finance real-time (1-min bar)
    try:
        ticker = YF_TICKERS.get(pair, f"{pair}=X")
        now    = int(datetime.utcnow().timestamp())
        for host in ["query2.finance.yahoo.com","query1.finance.yahoo.com"]:
            r = requests.get(
                f"https://{host}/v8/finance/chart/{ticker}"
                f"?interval=1m&period1={now-600}&period2={now}",
                headers={"User-Agent":"Mozilla/5.0","Cache-Control":"no-cache"},
                timeout=10)
            d = r.json()
            res = d.get("chart",{}).get("result",[])
            if res:
                closes = [c for c in res[0]["indicators"]["quote"][0].get("close",[]) if c]
                if closes:
                    p = float(closes[-1])
                    if base*0.5 < p < base*2.0:
                        return round(p, 5)
    except: pass

    # Fallback: use last close from candles
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
# TECHNICAL INDICATORS
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

# ═══════════════════════════════════════════════════════════════
# VOLUME PROFILE
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

def get_nodes(vp, top=0.25, bot=0.25):
    if vp is None or vp.empty: return [],[]
    return (vp[vp["volume"]>=vp["volume"].quantile(1-top)]["price_mid"].tolist(),
            vp[vp["volume"]<=vp["volume"].quantile(bot)]["price_mid"].tolist())

def next_hvn(price, hvn, direction):
    if not hvn: return None
    cands=[h for h in hvn if h>price] if direction=="Long" else [h for h in hvn if h<price]
    return (min(cands) if direction=="Long" else max(cands)) if cands else None

# ═══════════════════════════════════════════════════════════════
# GOLDEN ENTRY CHECKLIST (Smoothness Score 0-4)
# ═══════════════════════════════════════════════════════════════
def smoothness_score(df, poc, vah, val_p, sent_score=0):
    """
    From your screenshots:
    1. Trend: SMA50 > SMA200 (Daily) and SMA50 (4H)
    2. Location: Price at Fixed Range POC or Visible Range VAH
    3. Volatility: ATR is low (entering before spike)
    4. Sentiment: AI call confirms Risk-On or neutral
    """
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

    # 2. LOCATION (price at key VP level)
    tol=atr_n*0.4
    at_poc=abs(last-poc)<tol; at_vah=abs(last-vah)<tol; at_val=abs(last-val_p)<tol
    if at_poc or at_vah or at_val:
        lbl="POC" if at_poc else ("VAH" if at_vah else "VAL")
        score+=1; checks.append(("✅","Location",f"Price at {lbl} — smooth entry zone"))
    else:
        checks.append(("❌","Location","Price inside Value Area — no-man's land, avoid"))

    # 3. VOLATILITY (low ATR = entering before spike)
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
# EXHAUSTION FILTER (Volume/Delta Divergence)
# ═══════════════════════════════════════════════════════════════
def exhaustion_filter(df, window=5):
    """From screenshots: Volume Intensity drops at VAL/VAH = sellers/buyers exhausted."""
    if df.empty or len(df)<window+2: return False
    vi=vol_intensity(df)
    return float(vi.iloc[-1]) < float(vi.iloc[-window:].mean())*0.75

# ═══════════════════════════════════════════════════════════════
# TPO BREAKOUT VALIDATION
# ═══════════════════════════════════════════════════════════════
def tpo_status(df, vah, val_p, mins_per_bar=60):
    """
    From screenshots:
    >30 mins outside VA = Validated Breakout
    <10 mins = Fakeout (Look Above and Fail)
    """
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
def ai_sentiment(pair, headlines, claude_key):
    """
    From screenshots: Hawkish/Dovish/Neutral + intervention_risk
    Only fires Entry Alert if AI sentiment aligns with technical trend.
    """
    cache_key=pair
    if (time.time()-st.session_state.sentiment_ts.get(cache_key,0))<1800:
        if cache_key in st.session_state.sentiment_cache:
            return st.session_state.sentiment_cache[cache_key]

    default={"tone":"Neutral","score":0,"intervention_risk":False,
             "reasoning":"No API key — using neutral default.","pair":pair}

    if not claude_key or not ANTHROPIC_AVAILABLE:
        if not ANTHROPIC_AVAILABLE:
            default["reasoning"]="anthropic package not installed"
        return default

    bc,qc=pair[:3],pair[3:]
    hl="\n".join(f"- {h}" for h in headlines)
    prompt=(f"You are a professional FX analyst. Analyze these headlines for {bc}/{qc}:\n\n{hl}\n\n"
            f"Respond ONLY with JSON — no preamble:\n"
            f'{{"tone":"Hawkish"|"Dovish"|"Neutral","score":-2 to +2,"intervention_risk":true|false,'
            f'"reasoning":"one sentence max"}}')
    try:
        client=anthropic.Anthropic(api_key=claude_key)
        msg=client.messages.create(model="claude-sonnet-4-20250514",max_tokens=200,
                                   messages=[{"role":"user","content":prompt}])
        raw=re.sub(r"```json|```","",msg.content[0].text.strip()).strip()
        result=json.loads(raw); result["pair"]=pair
        st.session_state.sentiment_cache[cache_key]=result
        st.session_state.sentiment_ts[cache_key]=time.time()
        return result
    except Exception as e:
        default["reasoning"]=f"API error: {str(e)[:60]}"
        return default

# ═══════════════════════════════════════════════════════════════
# SIGNAL ENGINE — ATR + Smoothness + TPO + Exhaustion + Liquidity Void
# ═══════════════════════════════════════════════════════════════
def generate_signals(df, vp, poc, vah, val, hvn, lvn, pair,
                     smooth=0, sent=None, tpo="Unknown"):
    if df.empty or len(df)<20 or vp.empty: return []
    sigs=[]
    last    =float(df["close"].iloc[-1])
    atr_val =float(calc_atr(df).iloc[-1])
    rsi_val =float(calc_rsi(df["close"]).iloc[-1])
    vol_sma =df["volume"].rolling(20).mean().iloc[-1]
    last_vol=df["volume"].iloc[-1]
    vol_surge=last_vol>vol_sma*1.4
    vi_now  =float(vol_intensity(df).iloc[-1])
    vi_mean =float(vol_intensity(df).rolling(20).mean().iloc[-1])
    exhausted=vi_now<vi_mean*0.75

    sent_score=sent.get("score",0) if sent else 0
    interv=sent.get("intervention_risk",False) if sent else False

    def near(p,lvl,t=0.3): return abs(p-lvl)<atr_val*t

    def sig(stype,cat,entry,sl,tp1,tp2r,reason,conf,badge="blue_pulse"):
        d="Long" if "Long" in stype else "Short"
        hvn_tp=next_hvn(last,hvn,d)
        tp2=hvn_tp if hvn_tp else tp2r
        risk=abs(entry-sl); reward=abs(tp2-entry)
        rr=round(reward/risk,1) if risk>0 else 0
        fc=conf
        if smooth>=3 and conf=="Moderate": fc="High"
        if interv and "Reversion" in stype and "Short" in stype:
            fc="Low"; badge="flashing_red"
        return {"type":stype,"cat":cat,"entry":round(entry,5),"sl":round(sl,5),
                "tp1":round(tp1,5),"tp2":round(tp2,5),"atr":round(atr_val,5),
                "rsi":round(rsi_val,1),"vi":round(vi_now,1),"exhausted":exhausted,
                "reason":reason,"confidence":fc,"rr":rr,"smooth":smooth,
                "tpo":tpo,"badge":badge}

    # ── MEAN REVERSION (Blue Pulse) ─────────────────────────
    # VAL bounce — price at VAL, RSI oversold, VI exhaustion
    if near(last,val) and rsi_val<40 and exhausted:
        sigs.append(sig("Mean Reversion Long","MR",last,last-1.5*atr_val,poc,vah,
            f"Price at VAL ({val:.5f}), RSI {rsi_val:.0f}, VI exhaustion — sellers done",
            "High" if rsi_val<30 else "Moderate","blue_pulse"))

    # VAH rejection — price at VAH, RSI overbought, VI exhaustion
    if near(last,vah) and rsi_val>60 and exhausted:
        sigs.append(sig("Mean Reversion Short","MR",last,last+1.5*atr_val,poc,val,
            f"Price at VAH ({vah:.5f}), RSI {rsi_val:.0f}, VI exhaustion — buyers done",
            "High" if rsi_val>70 else "Moderate","blue_pulse"))

    # LVN snap — price in thin liquidity zone
    for lp in lvn:
        if near(last,lp,0.2):
            d="Long" if last>poc else "Short"; sd=-1 if d=="Long" else 1
            sigs.append(sig(f"LVN Snap {d}","MR",last,last+sd*atr_val,poc,
                poc+(1 if d=="Long" else -1)*atr_val,
                f"LVN {lp:.5f} — liquidity void, price snaps to next HVN",
                "Moderate","blue_pulse"))

    # POC rejection
    if near(last,poc,0.25):
        e20=float(calc_ema(df["close"],20).iloc[-1])
        if last>e20 and rsi_val<55:
            sigs.append(sig("POC Rejection Long","MR",last,poc-1.2*atr_val,
                poc+1.5*atr_val,vah,f"Held POC ({poc:.5f}), above EMA20","Moderate","blue_pulse"))
        elif last<e20 and rsi_val>45:
            sigs.append(sig("POC Rejection Short","MR",last,poc+1.2*atr_val,
                poc-1.5*atr_val,val,f"Rejected POC ({poc:.5f}), below EMA20","Moderate","blue_pulse"))

    # HVN support/resistance
    for hp in hvn[:3]:
        if near(last,hp,0.2):
            e20=float(calc_ema(df["close"],20).iloc[-1])
            d="Long" if last>=e20 else "Short"; sd=-1 if d=="Long" else 1
            tp_t=poc if d=="Long" else val
            sigs.append(sig(f"HVN {d}","MR",last,last+sd*atr_val,tp_t,
                tp_t+(1 if d=="Long" else -1)*atr_val,
                f"HVN {hp:.5f} — institutional level, wait for wick rejection",
                "Moderate","blue_pulse"))

    # ── MOMENTUM BREAKOUT (Green Spark) ─────────────────────
    # TPO must confirm (>30 mins outside VA)
    tpo_ok=tpo in ["Validated Breakout","Just Broke"]

    if last>vah+0.1*atr_val and vol_surge and rsi_val>50 and tpo_ok:
        conf="High" if rsi_val>60 and tpo=="Validated Breakout" else "Moderate"
        sigs.append(sig("Breakout Long","MO",vah+0.05*atr_val,vah-atr_val,
            last+1.5*atr_val,last+3.0*atr_val,
            f"Broke VAH ({vah:.5f}) with vol surge, TPO: {tpo}",conf,"green_spark"))

    if last<val-0.1*atr_val and vol_surge and rsi_val<50 and tpo_ok:
        conf="High" if rsi_val<40 and tpo=="Validated Breakout" else "Moderate"
        sigs.append(sig("Breakout Short","MO",val-0.05*atr_val,val+atr_val,
            last-1.5*atr_val,last-3.0*atr_val,
            f"Broke VAL ({val:.5f}) with vol surge, TPO: {tpo}",conf,"green_spark"))

    # Fakeout downgrade
    if tpo=="Fakeout Risk":
        for s in sigs:
            if s["cat"]=="MO":
                s["confidence"]="Low"; s["badge"]="grey_dim"
                s["reason"]+=" ⚠️ Fakeout — <30 min outside VA (Look Above and Fail)"

    # Intervention override
    if interv:
        for s in sigs:
            s["badge"]="flashing_red"
            s["reason"]+=" 🚨 Intervention risk detected"

    return sigs

# ═══════════════════════════════════════════════════════════════
# MAIN DATA LOADER — no cache, fresh every call
# ═══════════════════════════════════════════════════════════════
@st.cache_data(ttl=0, show_spinner=False)
def load_data(td_key, claude_key, fr_bars, vp_bins_val, va_pct, vp_mode, _bust=0):
    all_sigs={} ; prices={}; iv_data={}; candles={}; vp_data={}

    # Fetch H1 candles for all pairs (used for VP + signals)
    for pair in PAIRS:
        df=get_candles(pair,"H1",td_key,300)
        candles[pair]=df
        prices[pair]=get_live_price(pair,td_key) if td_key else \
                     (round(float(df["close"].iloc[-1]),5) if not df.empty else BASE_PRICES[pair])
        iv_data[pair]=get_iv(df)

    # Per-pair signals for heatmap (5 TFs)
    for pair in PAIRS:
        ps={}
        for tf in TIMEFRAMES:
            df=get_candles(pair,tf,td_key,200)
            ps[tf]=get_signal(df)
        all_sigs[pair]=ps

    signals_df=pd.DataFrame(all_sigs).T.reindex(columns=TIMEFRAMES)

    # IV term structure
    iv_term={}
    rng=np.random.default_rng(int(time.time()//300))
    for pair in PAIRS:
        s=iv_data[pair]
        iv_term[pair]={"spot":s,
            "1W":round(s*(1+rng.normal(0,0.12)),2),
            "1M":round(s*(1+rng.normal(0,0.08)),2),
            "3M":round(s*(1+rng.normal(0,0.05)),2)}

    # Volume profile + advanced analysis per pair
    for pair in PAIRS:
        df_raw=candles[pair]
        df_vp=df_raw.tail(int(fr_bars)) if "Fixed" in vp_mode else df_raw
        vp=build_vp(df_vp,bins=vp_bins_val)
        poc=get_poc(vp); vah,val_p=get_va(vp,pct=va_pct)
        hvn,lvn=get_nodes(vp)

        headlines=get_news(pair,td_key)
        # Compute confluence before calling AI so gate can check signal strength
        tf_sigs=[get_signal(get_candles(pair,tf,td_key,200)) for tf in TIMEFRAMES]
        pair_conf=sum(tf_sigs)
        sent=ai_sentiment(pair,headlines,claude_key,
                          confluence_score=pair_conf,smooth=smooth)
        smooth,checks=smoothness_score(df_raw,poc,vah,val_p,sent.get("score",0))
        tpo,tpo_mins=tpo_status(df_raw,vah,val_p)
        ldr,lag_msg=lead_lag(pair,candles)
        exhausted=exhaustion_filter(df_raw)
        vi_now=float(vol_intensity(df_raw).iloc[-1]) if not df_raw.empty else 0
        sigs=generate_signals(df_raw,vp,poc,vah,val_p,hvn,lvn,pair,smooth,sent,tpo)

        vp_data[pair]={
            "vp":vp,"poc":poc,"vah":vah,"val":val_p,
            "hvn":hvn,"lvn":lvn,"sigs":sigs,
            "atr":round(float(calc_atr(df_raw).iloc[-1]),5) if not df_raw.empty else 0,
            "smooth":smooth,"checks":checks,"sent":sent,"headlines":headlines,
            "tpo":tpo,"tpo_mins":tpo_mins,"leader":ldr,"lag":lag_msg,
            "exhausted":exhausted,"vi":vi_now,
        }

    # Zones (simulated — swap with order flow API)
    rng2=np.random.default_rng(int(time.time()//600))
    def make_zones(base):
        zones=[]
        for _ in range(rng2.integers(2,5)):
            zt=rng2.choice(["Order Block","Fair Value Gap","Point of Interest"])
            di=rng2.choice(["Bullish","Bearish"])
            off=rng2.uniform(-0.008,0.008)
            lv=round(base*(1+off),5); w=round(base*rng2.uniform(0.001,0.004),5)
            zones.append({"type":zt,"direction":di,"level":lv,
                          "zone_low":round(lv-w/2,5),"zone_high":round(lv+w/2,5),
                          "timeframe":rng2.choice(TIMEFRAMES),"strength":int(rng2.integers(1,4))})
        return zones

    last_ts={p:(str(candles[p]["time"].iloc[-1])[:16] if not candles[p].empty else "N/A")
             for p in PAIRS}

    return {"signals":signals_df,"prices":prices,"iv":iv_data,"iv_term":iv_term,
            "vp_data":vp_data,"zones":{p:make_zones(BASE_PRICES[p]) for p in PAIRS},
            "candles":candles,"last_ts":last_ts,
            "use_live":bool(td_key)}

# ═══════════════════════════════════════════════════════════════
# RENDER HELPERS
# ═══════════════════════════════════════════════════════════════
def conf_score(row): return int(row.sum())

def conf_bar(score):
    pct=int((score+5)/10*100)
    c="#1a7a4a" if score>0 else ("#b5281c" if score<0 else "#666")
    return (f'<div style="background:#2a2a3e;border-radius:4px;height:10px;width:110px;">'
            f'<div style="width:{pct}%;background:{c};height:10px;border-radius:4px;"></div>'
            f'</div><small style="color:{c};font-weight:600;">{score:+d}/5</small>')

def iv_col(v):
    if v<7: return "#52b788"
    if v<12: return "#e09a2a"
    return "#b5281c"

def dots(s): return "●"*s+"○"*(3-s)

def smooth_bar(score):
    colors=["#666","#e07b5a","#e09a2a","#52b788","#1a7a4a"]
    labels=["No Setup","Weak","Moderate","Strong","⭐ Golden Entry"]
    c=colors[min(score,4)]; lbl=labels[min(score,4)]; pct=score/4*100
    return (f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0;">'
            f'<div style="background:#2a2a3e;border-radius:4px;height:12px;width:100px;">'
            f'<div style="width:{pct}%;background:{c};height:12px;border-radius:4px;"></div></div>'
            f'<span style="color:{c};font-weight:700;font-size:12px;">{score}/4 {lbl}</span></div>')

def render_vp(vp, poc, vah, val_p, hvn, lvn, cp, title):
    if vp is None or vp.empty:
        st.caption("Volume data loading..."); return
    vp=vp.dropna(subset=["volume","price_mid"]).reset_index(drop=True)
    if vp.empty: st.caption("No VP data."); return
    max_vol=vp["volume"].max()
    if max_vol==0 or pd.isna(max_vol): st.caption("Zero volume."); return
    if not cp: cp=float(vp["price_mid"].mean())
    ci=(vp["price_mid"]-cp).abs().idxmin()
    rows=""
    for i,r in vp.iterrows():
        bw=max(0,int(r["volume"]/max_vol*155)); pm=r["price_mid"]
        is_hvn=any(abs(pm-h)<1e-4 for h in hvn)
        is_lvn=any(abs(pm-l)<1e-4 for l in lvn)
        is_poc=abs(pm-poc)<(vp["price_mid"].max()-vp["price_mid"].min())/(len(vp)*2)
        bc=("#e09a2a" if is_poc else "#1a4fa8" if is_hvn else "#444" if is_lvn
            else "#52b788" if val_p<=pm<=vah else "#2a3a4a")
        cp_m=" ◀" if i==ci else ""
        tag=(" POC" if is_poc else " HVN" if is_hvn else " LVN" if is_lvn else "")
        tag_c=("#e09a2a" if is_poc else "#1a4fa8" if is_hvn else "#666" if is_lvn else "")
        rows+=(f'<tr><td style="text-align:right;padding:1px 3px;font-size:9px;'
               f'font-family:monospace;color:#999;white-space:nowrap;">{pm:.5f}{cp_m}</td>'
               f'<td style="padding:1px 2px;"><div style="width:{bw}px;height:7px;'
               f'background:{bc};border-radius:2px;"></div></td>'
               f'<td style="padding:1px 3px;font-size:9px;color:{tag_c};">{r["pct"]:.0f}%{tag}</td></tr>')
    leg=(f'<div style="font-size:10px;margin:3px 0;">'
         f'<span style="color:#e09a2a;">■ POC {poc:.5f}</span> &nbsp;'
         f'<span style="color:#1a4fa8;">■ HVN</span> &nbsp;'
         f'<span style="color:#444;">■ LVN</span> &nbsp;'
         f'<span style="color:#52b788;">■ VA {val_p:.5f}–{vah:.5f}</span></div>')
    st.markdown(f"**{title}**")
    st.markdown(f'<div style="overflow-y:auto;max-height:400px;">{leg}'
                f'<table style="border-collapse:collapse;">{rows}</table></div>',
                unsafe_allow_html=True)

def render_sent(sent, headlines):
    tone   = sent.get("tone","Neutral")
    score  = sent.get("score",0)
    interv = sent.get("intervention_risk",False)
    reason = sent.get("reasoning","")
    gated  = sent.get("gated",True)
    session= sent.get("session","")
    calls  = sent.get("calls_today",0)
    cached = sent.get("from_cache",False)

    tc = {"Hawkish":"#1a7a4a","Dovish":"#b5281c","Neutral":"#666"}.get(tone,"#666")
    sc = "#1a7a4a" if score>0 else ("#b5281c" if score<0 else "#666")
    iv_html = ('<span style="background:#b5281c;color:#fff;padding:1px 8px;border-radius:3px;'
               'font-size:10px;font-weight:700;margin-left:8px;">🚨 INTERVENTION RISK</span>'
               if interv else "")

    # Token status badge
    if gated:
        status_html = (f'<span style="background:#333;color:#aaa;padding:1px 7px;'
                       f'border-radius:3px;font-size:9px;">⏸ Gated</span>')
    elif cached:
        status_html = (f'<span style="background:#1a3a1a;color:#52b788;padding:1px 7px;'
                       f'border-radius:3px;font-size:9px;">♻ Cached</span>')
    else:
        status_html = (f'<span style="background:#1a4fa8;color:#fff;padding:1px 7px;'
                       f'border-radius:3px;font-size:9px;">⚡ Live</span>')

    st.markdown(
        f'<div style="background:#12122a;border-radius:8px;padding:10px 12px;margin-bottom:6px;">'
        f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;flex-wrap:wrap;">'
        f'<span style="background:{tc};color:#fff;padding:2px 10px;border-radius:4px;font-size:12px;font-weight:700;">{tone}</span>'
        f'<span style="color:{sc};font-size:14px;font-weight:700;">Score {score:+d}</span>'
        f'{status_html}{iv_html}</div>'
        f'<div style="font-size:10px;color:#bbb;font-style:italic;margin-bottom:4px;">{reason}</div>'
        f'<div style="font-size:9px;color:#555;">{session} · Calls today: {calls}/50</div>'
        f'</div>',unsafe_allow_html=True)
    with st.expander(f"Headlines ({len(headlines)})"):
        for h in headlines: st.caption(f"• {h}")

def render_sig_card(s):
    bk=s.get("badge","blue_pulse")
    icon,bc,cat=BADGE.get(bk,("🔵","#1a4fa8","Signal"))
    cc={"High":"#1a7a4a","Moderate":"#e09a2a","Low":"#555"}.get(s["confidence"],"#555")
    sc=["#666","#e07b5a","#e09a2a","#52b788","#1a7a4a"][min(s.get("smooth",0),4)]
    tc=("#1a7a4a" if "Validated" in s.get("tpo","") else
        "#e09a2a" if "Fakeout" in s.get("tpo","") else "#666")
    ex="⚡ Exhaustion" if s.get("exhausted") else ""
    st.markdown(
        f'<div style="border-left:4px solid {bc};background:#12122a;padding:10px 12px;'
        f'border-radius:6px;margin-bottom:8px;">'
        f'<div style="display:flex;justify-content:space-between;">'
        f'<span style="color:{bc};font-size:11px;font-weight:700;">{icon} {cat}</span>'
        f'<span style="color:#888;font-size:10px;">{ex}</span></div>'
        f'<div style="color:#fff;font-size:13px;font-weight:700;margin:2px 0;">{s["type"]}</div>'
        f'<div style="color:#bbb;font-size:10px;margin-bottom:6px;">{s["reason"]}</div>'
        f'<table style="font-size:11px;color:#ddd;width:100%;border-collapse:collapse;">'
        f'<tr><td style="color:#aaa;padding:1px 6px 1px 0;">Entry</td>'
        f'<td style="font-family:monospace;font-weight:700;color:#fff;">{s["entry"]}</td>'
        f'<td style="color:#aaa;padding:1px 0 1px 8px;">ATR</td>'
        f'<td style="font-family:monospace;">{s["atr"]}</td></tr>'
        f'<tr><td style="color:#f87171;">SL</td>'
        f'<td style="font-family:monospace;color:#f87171;font-weight:700;">{s["sl"]}</td>'
        f'<td style="color:#aaa;padding:1px 0 1px 8px;">RSI</td>'
        f'<td style="font-family:monospace;">{s["rsi"]}</td></tr>'
        f'<tr><td style="color:#4ade80;">TP1</td>'
        f'<td style="font-family:monospace;color:#4ade80;">{s["tp1"]}</td>'
        f'<td style="color:#4ade80;padding:1px 0 1px 8px;">TP2</td>'
        f'<td style="font-family:monospace;color:#4ade80;">{s["tp2"]}</td></tr>'
        f'</table>'
        f'<div style="display:flex;gap:6px;margin-top:6px;flex-wrap:wrap;">'
        f'<span style="background:{cc};color:#fff;padding:1px 7px;border-radius:3px;font-size:10px;font-weight:600;">{s["confidence"]}</span>'
        f'<span style="color:#aaa;font-size:10px;">R:R 1:{s.get("rr",0)}</span>'
        f'<span style="color:{sc};font-size:10px;">Smooth {s.get("smooth",0)}/4</span>'
        f'<span style="color:{tc};font-size:10px;">TPO:{s.get("tpo","—")}</span>'
        f'</div></div>',unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# COUNTDOWN FRAGMENT
# ═══════════════════════════════════════════════════════════════
@st.fragment(run_every=1)
def countdown():
    elapsed=time.time()-st.session_state.last_refresh
    remaining=max(0,int(60-elapsed%60))  # show seconds to next minute refresh
    st.caption(f"Data refreshes every minute · Next refresh in {remaining}s")
    st.progress(1.0-remaining/60)

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ Settings")

    td_key     = st.text_input("Twelve Data API Key", type="password",
                                placeholder="Free at twelvedata.com")
    claude_key = st.text_input("Claude API Key",type="password",
                                placeholder="Anthropic — for AI sentiment")
    st.caption("Twelve Data = real forex volume. Claude = AI sentiment filter.")

    st.divider()
    st.markdown("**Volume Profile**")
    vp_bins_val = st.slider("Price bins",15,60,VP_BINS,5)
    va_pct_val  = st.slider("Value Area %",50,90,70,5)/100
    vp_mode     = st.radio("Mode",["Visible Range","Fixed Range"])
    fr_bars     = st.number_input("Fixed bars",10,300,50) if "Fixed" in vp_mode else 200

    st.divider()
    st.markdown("**Signal Filters**")
    show_mr  = st.checkbox("Mean Reversion (🔵)",value=True)
    show_mo  = st.checkbox("Momentum Breakout (🟢)",value=True)
    min_conf = st.selectbox("Min confidence",["Moderate","High"],index=0)

    st.divider()
    if td_key:     st.success("🟢 Twelve Data — Live volume")
    else:          st.info("📡 Yahoo Finance — synthetic volume")
    if claude_key: st.success("🤖 Claude AI — Active")
    else:          st.warning("🤖 Claude AI — No key")

    # Token budget display
    st.divider()
    st.markdown("**🤖 Claude Token Budget**")
    _reset_daily_if_needed()
    calls = st.session_state.claude_calls_today
    tokens = st.session_state.claude_tokens_used
    budget_pct = calls / 50 * 100
    budget_c = "#1a7a4a" if calls < 20 else ("#e09a2a" if calls < 40 else "#b5281c")
    st.markdown(
        f'<div style="background:#12122a;padding:8px;border-radius:6px;font-size:11px;">'
        f'<div style="display:flex;justify-content:space-between;">'
        f'<span>Calls today</span><span style="color:{budget_c};font-weight:700;">{calls}/50</span></div>'
        f'<div style="background:#2a2a3e;border-radius:3px;height:6px;margin:4px 0;">'
        f'<div style="width:{min(budget_pct,100):.0f}%;background:{budget_c};height:6px;border-radius:3px;"></div></div>'
        f'<div style="color:#666;font-size:9px;">~{tokens} tokens used · Shared budget</div>'
        f'</div>', unsafe_allow_html=True)

    # Session gate status
    _, s_priority, s_msg = _get_session_gate()
    gate_c = "#1a7a4a" if s_priority=="HIGH" else ("#e09a2a" if s_priority=="MED" else "#666")
    st.markdown(f'<div style="font-size:10px;color:{gate_c};margin-top:4px;">{s_msg}</div>',
                unsafe_allow_html=True)
    st.caption("Gate: HIGH(≥3conf/≥2smooth) · MED(≥3+≥2) · LOW(≥4 extreme only)")

    st.divider()
    # Debug connection test
    if td_key and st.button("🔍 Test TD Connection"):
        try:
            r=requests.get("https://api.twelvedata.com/time_series",
                           params={"symbol":"EUR/USD","interval":"1h","outputsize":2,
                                   "order":"ASC","format":"JSON","apikey":td_key},timeout=15)
            d=r.json()
            if d.get("status")=="error":
                st.error(f"API Error: {d.get('message')}")
            else:
                v=d.get("values",[])
                if v: st.success(f"✅ Latest bar: {v[-1].get('datetime','?')}")
                else: st.warning(f"Connected — no values: {d}")
        except Exception as e:
            st.error(f"Failed: {e}")

# ═══════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════
st.title("💱 FX Multi-Factor Engine")

tc1,tc2=st.columns([5,1])
with tc2:
    if st.button("🔄 Refresh",use_container_width=True):
        st.cache_data.clear()
        st.session_state.cache_buster=int(time.time())
        st.rerun()
countdown()

# Load data
with st.spinner("Fetching live data..."):
    data=load_data(td_key,claude_key,fr_bars,vp_bins_val,va_pct_val,vp_mode,
                   _bust=st.session_state.cache_buster)

signals=data["signals"]

# Show data freshness
ts_str=" · ".join(f"{p}:{data['last_ts'][p]}" for p in PAIRS[:4])
st.caption(f"📡 Last candle timestamps (UTC): {ts_str}")

# ── Metric cards ──────────────────────────────────────────────
st.markdown("### 📊 Market Overview")
mc=st.columns(len(PAIRS))
for i,pair in enumerate(PAIRS):
    sc=conf_score(signals.loc[pair])
    d="▲" if sc>0 else ("▼" if sc<0 else "—")
    vd=data["vp_data"][pair]; price=data["prices"][pair]
    sm=vd["smooth"]; star="⭐" if sm==4 else ""
    rel="↑POC" if price>vd["poc"] else "↓POC"
    with mc[i]:
        st.metric(label=f"{pair}{star}",value=f"{price:.5f}",
                  delta=f"{d}{abs(sc)}/5 · {rel}")

st.divider()

# ── Signal Heatmap + Alerts ───────────────────────────────────
st.markdown("### 🗺 Signal Heatmap — All Timeframes")
col_hm,col_al=st.columns([3,1])

with col_hm:
    hdr=('<div style="overflow-x:auto;"><table style="width:100%;min-width:800px;'
         'border-collapse:collapse;font-size:12px;table-layout:fixed;"><thead><tr>'
         '<th style="text-align:left;padding:5px 6px;border-bottom:1px solid #333;width:65px;">Pair</th>'
         '<th style="text-align:center;padding:5px 6px;border-bottom:1px solid #333;width:85px;">Price</th>')
    for tf in TIMEFRAMES:
        hdr+=f'<th style="text-align:center;padding:5px 4px;border-bottom:1px solid #333;width:88px;">{tf}</th>'
    hdr+=('<th style="text-align:center;padding:5px 6px;border-bottom:1px solid #333;width:115px;">Confluence</th>'
          '<th style="text-align:center;padding:5px 5px;border-bottom:1px solid #333;width:65px;">IV</th>'
          '<th style="text-align:center;padding:5px 5px;border-bottom:1px solid #333;width:75px;">Sent.</th>'
          '<th style="text-align:center;padding:5px 5px;border-bottom:1px solid #333;width:85px;">Smooth</th>'
          '</tr></thead><tbody>')
    rows_h=""
    for idx,pair in enumerate(PAIRS):
        row=signals.loc[pair]; sc=conf_score(row)
        price=data["prices"][pair]; iv_v=data["iv"][pair]
        ivc=iv_col(iv_v); vd=data["vp_data"][pair]
        sent=vd["sent"]; tone=sent.get("tone","Neutral")
        tc={"Hawkish":"#1a7a4a","Dovish":"#b5281c","Neutral":"#666"}.get(tone,"#666")
        sm=vd["smooth"]
        smc=["#666","#e07b5a","#e09a2a","#52b788","#1a7a4a"][min(sm,4)]
        bg="#1a1a2e" if idx%2==0 else "#12122a"
        rows_h+=(f'<tr style="background:{bg};">'
                 f'<td style="padding:5px 6px;font-weight:700;color:#fff;">{pair}</td>'
                 f'<td style="text-align:center;padding:5px 6px;font-family:monospace;font-size:11px;color:#fff;">{price:.5f}</td>')
        for tf in TIMEFRAMES:
            v=int(row[tf])
            rows_h+=(f'<td style="text-align:center;padding:3px 3px;">'
                     f'<span style="background:{SIGNAL_COLORS[v]};color:#fff;padding:2px 6px;'
                     f'border-radius:4px;font-size:10px;white-space:nowrap;">{SIGNAL_LABELS[v]}</span></td>')
        rows_h+=(f'<td style="text-align:center;padding:5px 6px;">{conf_bar(sc)}</td>'
                 f'<td style="text-align:center;padding:5px 5px;">'
                 f'<span style="background:{ivc};color:#fff;padding:2px 7px;border-radius:4px;'
                 f'font-size:11px;font-weight:700;">{iv_v}%</span></td>'
                 f'<td style="text-align:center;padding:5px 5px;">'
                 f'<span style="background:{tc};color:#fff;padding:2px 7px;border-radius:4px;'
                 f'font-size:10px;">{tone}</span></td>'
                 f'<td style="text-align:center;padding:5px 5px;">'
                 f'<span style="color:{smc};font-weight:700;">{"⭐" if sm==4 else sm}/4</span></td>'
                 f'</tr>')
    st.markdown(hdr+rows_h+'</tbody></table></div>',unsafe_allow_html=True)

with col_al:
    st.markdown("### 🔔 Alerts")
    fired=False
    for pair in PAIRS:
        sc=conf_score(signals.loc[pair]); iv_v=data["iv"][pair]
        vd=data["vp_data"][pair]; nsigs=len(vd["sigs"])
        sent=vd["sent"]
        if sent.get("intervention_risk"):
            st.error(f"🚨 **{pair}** Intervention risk"); fired=True
        if abs(sc)>=4:
            st.error(f"**{pair}** {'BULL' if sc>0 else 'BEAR'} ({sc:+d}/5)"); fired=True
        elif abs(sc)==3:
            st.warning(f"**{pair}** Moderate {'▲' if sc>0 else '▼'} ({sc:+d}/5)"); fired=True
        if iv_v>14: st.error(f"**{pair}** Extreme IV {iv_v}%"); fired=True
        elif iv_v>10: st.info(f"**{pair}** Elevated IV {iv_v}%"); fired=True
        if vd["smooth"]==4: st.success(f"⭐ **{pair}** Golden Entry!"); fired=True
        elif nsigs>0: st.success(f"**{pair}** {nsigs} signal(s)"); fired=True
    if not fired: st.write("No significant signals.")

st.divider()

# ── Per-pair detailed analysis ────────────────────────────────
st.markdown("### 📈 Volume Profile · AI Sentiment · Entry Signals")
st.caption("🟡 POC · 🔵 HVN · ⬛ LVN · 🟢 Value Area · ◀ Price · "
           "🔵 Mean Reversion · 🟢 Breakout · ⚫ Avoid · 🔴 Intervention")

tabs=st.tabs(PAIRS)
co={"High":2,"Moderate":1,"Low":0}

for i,pair in enumerate(PAIRS):
    with tabs[i]:
        vd=data["vp_data"][pair]; cp=data["prices"][pair]
        atr_v=vd["atr"]

        # ── Row 1: VP | Key Levels + Smoothness | AI Sentiment ──
        c1,c2,c3=st.columns([1,1,1])

        with c1:
            render_vp(vd["vp"],vd["poc"],vd["vah"],vd["val"],
                      vd["hvn"],vd["lvn"],cp,
                      f"{pair} VP [{'FR' if 'Fixed' in vp_mode else 'VR'}]")

        with c2:
            st.markdown("**📐 Key Levels**")
            poc_d=round((cp-vd["poc"])*10000,1)
            vah_d=round((vd["vah"]-cp)*10000,1)
            val_d=round((cp-vd["val"])*10000,1)
            st.markdown(
                f'<div style="background:#12122a;padding:10px;border-radius:6px;'
                f'font-size:11px;line-height:2;">'
                f'<b>Price:</b> {cp:.5f}<br>'
                f'<b>ATR(14):</b> {atr_v:.5f}<br>'
                f'<b>POC:</b> {vd["poc"]:.5f} <span style="color:#e09a2a;">({poc_d:+.1f} pips)</span><br>'
                f'<b>VAH:</b> {vd["vah"]:.5f} <span style="color:#52b788;">(↑{vah_d:.1f} pips)</span><br>'
                f'<b>VAL:</b> {vd["val"]:.5f} <span style="color:#52b788;">(↓{val_d:.1f} pips)</span><br>'
                f'<b>HVNs:</b> {len(vd["hvn"])} nodes &nbsp; <b>LVNs:</b> {len(vd["lvn"])} nodes<br>'
                f'<b>VI:</b> {vd["vi"]:.1f} {"⚡ Exhaustion" if vd["exhausted"] else ""}<br>'
                f'<b>TPO:</b> <span style="color:{"#1a7a4a" if "Validated" in vd["tpo"] else "#e09a2a"};">'
                f'{vd["tpo"]} ({vd["tpo_mins"]}m)</span>'
                f'</div>',unsafe_allow_html=True)

            st.markdown("**🏆 Golden Entry Checklist**")
            st.markdown(smooth_bar(vd["smooth"]),unsafe_allow_html=True)
            for icon,label,desc in vd["checks"]:
                col=("#1a7a4a" if icon=="✅" else "#888" if icon=="⚪" else "#b5281c")
                st.markdown(f'<div style="font-size:10px;color:{col};margin:1px 0;">'
                            f'{icon} <b>{label}:</b> {desc}</div>',unsafe_allow_html=True)

            if vd["leader"]:
                st.markdown(f'<div style="margin-top:6px;background:#12122a;padding:7px;'
                            f'border-radius:5px;font-size:11px;color:#e09a2a;">'
                            f'🔗 <b>Lead/Lag:</b> {vd["lag"]}</div>',unsafe_allow_html=True)

        with c3:
            st.markdown("**🤖 AI Sentiment — Claude**")
            render_sent(vd["sent"],vd["headlines"])

            # Summary state badge
            nsigs=len(vd["sigs"])
            if vd["sent"].get("intervention_risk"):
                st.markdown('🔴 **Flashing Red** — Intervention risk')
            elif any(s["badge"]=="green_spark" for s in vd["sigs"]):
                st.markdown('🟢 **Green Spark** — Breakout momentum')
            elif any(s["badge"]=="blue_pulse" for s in vd["sigs"]):
                st.markdown('🔵 **Blue Pulse** — Mean reversion')
            elif not vd["sigs"]:
                st.markdown('⚫ **No signal** — Wait for setup')

        st.markdown("---")

        # ── Row 2: Entry Signals ──────────────────────────────
        st.markdown(f"**🎯 Entry Signals — {pair}**")
        st.caption("ATR-backed SL/TP · Smoothness filter · TPO validated · Exhaustion confirmed")

        all_s=vd["sigs"]
        filtered=[s for s in all_s if
                  ((show_mr and s["cat"]=="MR") or (show_mo and s["cat"]=="MO"))
                  and co.get(s["confidence"],0)>=co.get(min_conf,0)]

        if not filtered:
            st.info("No signals match filters. Conditions: price near VP level + RSI + VI exhaustion + TPO confirmation.")
        else:
            sc2=st.columns(min(len(filtered),3))
            for j,s in enumerate(filtered):
                with sc2[j%3]: render_sig_card(s)

st.divider()

# ── IV Term Structure ──────────────────────────────────────────
st.markdown("### 📉 Implied Volatility — Term Structure")
iv_rows=[{"Pair":p,"Spot IV%":data["iv_term"][p]["spot"],
          "1W IV%":data["iv_term"][p]["1W"],"1M IV%":data["iv_term"][p]["1M"],
          "3M IV%":data["iv_term"][p]["3M"],
          "Skew":round(data["iv_term"][p]["1W"]-data["iv_term"][p]["3M"],2),
          "Structure":"Backwardation" if data["iv_term"][p]["1W"]>data["iv_term"][p]["3M"] else "Contango"}
         for p in PAIRS]
iv_df=pd.DataFrame(iv_rows)
def civ(v): return f"color:{iv_col(v)};font-weight:600;" if isinstance(v,(int,float)) else ""
def csk(v): return f"color:{'#b5281c' if v>0 else '#1a7a4a'};font-weight:600;" if isinstance(v,(int,float)) else ""
st.dataframe(iv_df.style.map(civ,subset=["Spot IV%","1W IV%","1M IV%","3M IV%"])
             .map(csk,subset=["Skew"]),use_container_width=True,hide_index=True)
st.caption("Backwardation (1W>3M): near-term event risk. Contango: calm near-term.")
st.divider()

# ── Key Level Zones ────────────────────────────────────────────
st.markdown("### 🏛 Key Level Zones — Order Blocks · FVGs · Points of Interest")
sel=st.selectbox("Select pair:",PAIRS,key="zone_sel")
cp=data["prices"][sel]; atr_z=data["vp_data"][sel]["atr"]
st.markdown(f"**Price:** `{cp:.5f}` &nbsp;&nbsp; **ATR:** `{atr_z:.5f}`")

zr=[]
for idx,z in enumerate(sorted(data["zones"][sel],key=lambda z:abs(z["level"]-cp))):
    dp=round(abs(z["level"]-cp)*10000,1)
    zr.append({"Zone Type":f"{'⭐ ' if idx==0 else ''}{z['type']}",
               "Direction":z["direction"],"TF":z["timeframe"],
               "Zone Low":z["zone_low"],"Mid Level":z["level"],"Zone High":z["zone_high"],
               "Distance":f"{'↑' if z['level']>cp else '↓'} {dp} pips",
               "Strength":dots(z["strength"])})
zdf=pd.DataFrame(zr)
def sz(row):
    c="#1a7a4a" if row.get("Direction")=="Bullish" else "#b5281c"
    return [f"background:{c};color:#fff;"]*len(row)
def st2(v):
    if "Order Block" in str(v): return "color:#fff;background:#1a4fa8;font-weight:700;"
    if "Fair Value Gap" in str(v): return "color:#fff;background:#7c3aed;font-weight:700;"
    if "Point of Interest" in str(v): return "color:#fff;background:#b45309;font-weight:700;"
    return "color:#fff;font-weight:700;"
st.dataframe(zdf.style.apply(sz,axis=1).map(st2,subset=["Zone Type"])
             .map(lambda v:"color:#fff;",
                  subset=["Direction","TF","Zone Low","Mid Level","Zone High","Distance","Strength"]),
             use_container_width=True,hide_index=True)

st.divider()
st.info("⭐ Golden Entry = 4/4 checklist · 🔵 Blue Pulse = Mean Reversion at VAL/VAH/POC · "
        "🟢 Green Spark = Momentum Breakout (TPO validated >30min) · "
        "⚫ Grey/Dim = Inside VA no-man's land · 🔴 Flashing Red = AI intervention risk detected · "
        "VI Exhaustion = Volume Intensity drop at extreme = fuel running out")
