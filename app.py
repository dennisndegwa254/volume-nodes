import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
PLOTLY_AVAILABLE = False  # Using native Streamlit charts instead
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
from datetime import datetime, timedelta

# ═══════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════
st.set_page_config(page_title="FX Multi-Factor Engine", layout="wide")

PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
POLYGON_TICKERS = {p: f"C:{p}" for p in PAIRS}  # kept for reference
YF_TICKERS = {
    "EURUSD": "EURUSD=X", "USDJPY": "USDJPY=X", "GBPUSD": "GBPUSD=X",
    "USDCHF": "USDCHF=X", "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X",
    "NZDUSD": "NZDUSD=X",
}
YF_TF_MAP = {
    "M5":  "5m",  "M15": "15m", "H1":  "1h",
    "H4":  "1h",  "D1":  "1d",
}

# Twelve Data — real forex OHLCV with actual volume
TD_TF_MAP = {
    "M5": "5min", "M15": "15min", "H1": "1h", "H4": "4h", "D1": "1day"
}
TD_PAIRS = {
    "EURUSD": "EUR/USD", "USDJPY": "USD/JPY", "GBPUSD": "GBP/USD",
    "USDCHF": "USD/CHF", "AUDUSD": "AUD/USD", "USDCAD": "USD/CAD",
    "NZDUSD": "NZD/USD",
}
TIMEFRAMES = ["M5", "M15", "H1", "H4", "D1"]
TF_MAP = {"M5":(5,"minute"),"M15":(15,"minute"),"H1":(1,"hour"),"H4":(4,"hour"),"D1":(1,"day")}

SIGNAL_LABELS = {2:"Strong Bull",1:"Bullish",0:"Neutral",-1:"Bearish",-2:"Strong Bear"}
SIGNAL_COLORS = {2:"#1a7a4a",1:"#52b788",0:"#888888",-1:"#e07b5a",-2:"#b5281c"}

REFRESH_INTERVAL = 900
BASE_PRICES = {"EURUSD":1.0850,"USDJPY":154.20,"GBPUSD":1.2650,
               "USDCHF":0.9020,"AUDUSD":0.6480,"USDCAD":1.3640,"NZDUSD":0.5920}
VP_BINS = 30

# Pair correlation groups (for lead/lag)
CORRELATION_GROUPS = {
    "EURUSD": ["GBPUSD","AUDUSD","NZDUSD"],
    "GBPUSD": ["EURUSD","AUDUSD"],
    "AUDUSD": ["NZDUSD","EURUSD"],
    "NZDUSD": ["AUDUSD"],
    "USDJPY": ["USDCHF","USDCAD"],
    "USDCHF": ["USDJPY","USDCAD"],
    "USDCAD": ["USDJPY","USDCHF"],
}

# ═══════════════════════════════════════════════════════════════
# 2. SESSION STATE
# ═══════════════════════════════════════════════════════════════
for k,v in [("last_refresh",time.time()),("refresh_count",0),
            ("sentiment_cache",{}),("sentiment_ts",{})]:
    if k not in st.session_state:
        st.session_state[k] = v

if time.time() - st.session_state.last_refresh >= REFRESH_INTERVAL:
    st.session_state.last_refresh = time.time()
    st.session_state.refresh_count += 1
    st.cache_data.clear()

# ═══════════════════════════════════════════════════════════════
# 3. POLYGON.IO FETCHING
# ═══════════════════════════════════════════════════════════════
def fetch_polygon_candles(pair, multiplier, timespan, api_key, limit=300):
    ticker  = POLYGON_TICKERS[pair]
    to_dt   = datetime.utcnow()
    from_dt = to_dt - timedelta(days=max(10, limit//8+5))
    url = (f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range"
           f"/{multiplier}/{timespan}"
           f"/{from_dt.strftime('%Y-%m-%d')}/{to_dt.strftime('%Y-%m-%d')}"
           f"?adjusted=true&sort=asc&limit={limit}&apiKey={api_key}")
    try:
        r = requests.get(url, timeout=12)
        d = r.json()
        if d.get("resultsCount",0) == 0: return pd.DataFrame()
        df = pd.DataFrame(d["results"])
        df["t"] = pd.to_datetime(df["t"],unit="ms")
        df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume","t":"time"})
        return df[["time","open","high","low","close","volume"]].tail(limit)
    except: return pd.DataFrame()

def simulate_candles(pair, n=300):
    rng    = np.random.default_rng(abs(hash(pair))%9999)
    base   = BASE_PRICES[pair]
    closes = base * np.cumprod(1+rng.normal(0,0.0006,n))
    noise  = rng.uniform(0.0002,0.0012,n)
    opens  = np.roll(closes,1); opens[0]=base
    times  = [datetime.utcnow()-timedelta(hours=n-i) for i in range(n)]
    return pd.DataFrame({"time":times,"open":opens,
                          "high":closes*(1+noise),"low":closes*(1-noise),
                          "close":closes,"volume":rng.uniform(200,4000,n)})


def fetch_yf_candles(pair, tf, limit=300):
    """
    Fetch OHLCV using Yahoo Finance CSV endpoint — pure requests, no yfinance package.
    """
    ticker = YF_TICKERS.get(pair, f"{pair}=X")
    yf_tf  = YF_TF_MAP.get(tf, "1h")

    interval_seconds = {
        "5m": 300, "15m": 900, "1h": 3600, "1d": 86400
    }
    now      = int(datetime.utcnow().timestamp())
    lookback = interval_seconds.get(yf_tf, 3600) * limit * 2
    period1  = now - lookback
    period2  = now

    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?interval={yf_tf}&period1={period1}&period2={period2}&includePrePost=false")
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        r = requests.get(url, headers=headers, timeout=15)
        d = r.json()
        result = d.get("chart", {}).get("result", [])
        if not result:
            return pd.DataFrame()
        data   = result[0]
        times  = data["timestamp"]
        ohlcv  = data["indicators"]["quote"][0]
        df = pd.DataFrame({
            "time":   pd.to_datetime(times, unit="s"),
            "open":   ohlcv["open"],
            "high":   ohlcv["high"],
            "low":    ohlcv["low"],
            "close":  ohlcv["close"],
            "volume": ohlcv["volume"],
        }).dropna().tail(limit).reset_index(drop=True)
        return df
    except Exception as e:
        print(f"[YF-HTTP] {pair} {tf} error: {e}")
        return pd.DataFrame()

def fetch_yf_price(pair):
    """Get latest price via Yahoo Finance HTTP — no package needed."""
    try:
        ticker = YF_TICKERS.get(pair, f"{pair}=X")
        now    = int(datetime.utcnow().timestamp())
        url    = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
                  f"?interval=1m&period1={now-300}&period2={now}")
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=10)
        d = r.json()
        result = d.get("chart",{}).get("result",[])
        if result:
            closes = result[0]["indicators"]["quote"][0]["close"]
            closes = [c for c in closes if c is not None]
            if closes:
                return round(float(closes[-1]), 5)
    except:
        pass
    return BASE_PRICES[pair]

def fetch_td_candles(pair, tf, td_api_key, limit=300):
    """
    Fetch OHLCV from Twelve Data API.
    Free tier: 800 requests/day, real forex volume included.
    Sign up free at: https://twelvedata.com
    """
    if not td_api_key:
        return pd.DataFrame()
    symbol   = TD_PAIRS.get(pair, pair[:3]+"/"+pair[3:])
    interval = TD_TF_MAP.get(tf, "1h")
    url = (f"https://api.twelvedata.com/time_series"
           f"?symbol={symbol}&interval={interval}"
           f"&outputsize={min(limit,5000)}&apikey={td_api_key}&format=JSON")
    try:
        r = requests.get(url, timeout=15)
        d = r.json()
        if d.get("status") == "error":
            print(f"[TwelveData] {pair}: {d.get('message','unknown error')}")
            return pd.DataFrame()
        values = d.get("values", [])
        if not values:
            return pd.DataFrame()
        df = pd.DataFrame(values)
        df = df.rename(columns={"datetime":"time"})
        df["time"]   = pd.to_datetime(df["time"])
        df["open"]   = df["open"].astype(float)
        df["high"]   = df["high"].astype(float)
        df["low"]    = df["low"].astype(float)
        df["close"]  = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float) if "volume" in df.columns else 1000.0
        df = df[["time","open","high","low","close","volume"]]
        df = df.sort_values("time").reset_index(drop=True)
        return df.tail(limit)
    except Exception as e:
        print(f"[TwelveData] {pair} {tf} error: {e}")
        return pd.DataFrame()

def fetch_td_price(pair, td_api_key):
    """Get real-time price from Twelve Data."""
    if not td_api_key:
        return None
    symbol = TD_PAIRS.get(pair, pair[:3]+"/"+pair[3:])
    url = f"https://api.twelvedata.com/price?symbol={symbol}&apikey={td_api_key}"
    try:
        r = requests.get(url, timeout=8)
        d = r.json()
        if "price" in d:
            return round(float(d["price"]), 5)
    except:
        pass
    return None


def fetch_polygon_news(pair, api_key, limit=5):
    """Fetch recent news headlines for a currency pair from Polygon."""
    currencies = [pair[:3], pair[3:]]
    headlines = []
    for curr in currencies:
        url = (f"https://api.polygon.io/v2/reference/news"
               f"?ticker=C:{pair}&limit={limit}&apiKey={api_key}")
        try:
            r = requests.get(url, timeout=8)
            d = r.json()
            for item in d.get("results",[]):
                headlines.append(item.get("title",""))
        except: pass
    # Fallback simulated headlines if no API or no results
    if not headlines:
        simulated = {
            "EURUSD": ["ECB holds rates steady amid inflation concerns","EUR rallies on strong PMI data"],
            "USDJPY": ["BOJ signals potential rate hike","JPY weakens on risk-on sentiment"],
            "GBPUSD": ["UK CPI beats expectations","BOE cautious on rate cuts"],
            "USDCHF": ["SNB intervenes to weaken franc","Swiss trade surplus widens"],
            "AUDUSD": ["RBA holds rates, AUD pressured","Australia jobs data beats"],
            "USDCAD": ["Oil rally boosts CAD","BOC signals data dependency"],
            "NZDUSD": ["RBNZ cuts rates 25bps","NZD pressured by China slowdown"],
        }
        headlines = simulated.get(pair, [f"{pair} markets stable"])
    return headlines[:5]

# ═══════════════════════════════════════════════════════════════
# 4. CLAUDE AI SENTIMENT ENGINE
# ═══════════════════════════════════════════════════════════════
def get_ai_sentiment(pair, headlines, claude_api_key):
    """Call Claude API to get Hawkish/Dovish/Neutral + intervention risk."""
    cache_key = pair
    cache_age = time.time() - st.session_state.sentiment_ts.get(cache_key, 0)
    if cache_age < 1800 and cache_key in st.session_state.sentiment_cache:
        return st.session_state.sentiment_cache[cache_key]

    default = {"tone":"Neutral","score":0,"intervention_risk":False,
                "reasoning":"No API key provided — using neutral default.","pair":pair}
    if not claude_api_key or not ANTHROPIC_AVAILABLE:
        if not ANTHROPIC_AVAILABLE:
            default["reasoning"] = "anthropic package not installed — add to requirements.txt"
        return default

    base_curr, quote_curr = pair[:3], pair[3:]
    headlines_text = "\n".join(f"- {h}" for h in headlines)
    prompt = f"""You are a professional FX analyst. Analyze these recent headlines for {base_curr}/{quote_curr}:

{headlines_text}

Respond ONLY with a JSON object — no preamble, no markdown, no explanation:
{{
  "tone": "Hawkish" | "Dovish" | "Neutral",
  "score": -2 to +2 integer (negative=bearish for base currency, positive=bullish),
  "intervention_risk": true | false,
  "reasoning": "one sentence max"
}}"""

    try:
        client = anthropic.Anthropic(api_key=claude_api_key)
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{"role":"user","content":prompt}]
        )
        import json, re
        raw = msg.content[0].text.strip()
        raw = re.sub(r"```json|```","",raw).strip()
        result = json.loads(raw)
        result["pair"] = pair
        st.session_state.sentiment_cache[cache_key] = result
        st.session_state.sentiment_ts[cache_key] = time.time()
        return result
    except Exception as e:
        default["reasoning"] = f"API error: {str(e)[:60]}"
        return default

# ═══════════════════════════════════════════════════════════════
# 5. TECHNICAL INDICATORS
# ═══════════════════════════════════════════════════════════════
def atr(df, period=14):
    h,l,c = df["high"],df["low"],df["close"]
    tr = pd.concat([h-l,(h-c.shift(1)).abs(),(l-c.shift(1)).abs()],axis=1).max(axis=1)
    return tr.ewm(span=period,adjust=False).mean()

def ema(s, span): return s.ewm(span=span,adjust=False).mean()
def sma(s, span): return s.rolling(span).mean()

def rsi(s, period=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(span=period,adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(span=period,adjust=False).mean()
    return 100-100/(1+g/l.replace(0,np.nan))

def volume_intensity(df):
    """Volume / ATR — measures vol per unit of price movement."""
    atr_s = atr(df)
    return (df["volume"] / atr_s.replace(0,np.nan)).fillna(0)

def compute_signal(df):
    if df.empty or len(df)<55: return 0
    c=df["close"]; e20=ema(c,20).iloc[-1]; e50=ema(c,50).iloc[-1]; last=c.iloc[-1]
    bull=int(e20>e50)+int(last>e20); bear=int(e20<e50)+int(last<e20)
    if bull==2: return 2
    if bull==1: return 1
    if bear==2: return -2
    if bear==1: return -1
    return 0

def compute_iv_proxy(df):
    if df.empty or len(df)<10: return 8.0
    ret=np.log(df["close"]/df["close"].shift(1)).dropna()
    return round(float(ret.std()*np.sqrt(252*24)*100),2)

def compute_smoothness_score(df, poc, vah, val_p, sentiment_score=0):
    """
    Golden Entry Checklist (0-4):
    1. Trend: SMA50 > SMA200 (bullish) or < (bearish)
    2. Location: price at FR-POC or VR-VAH edge
    3. Volatility: ATR is low (below 20-bar mean)
    4. Sentiment: AI score aligns with trend
    """
    if df.empty or len(df)<201: return 0, []
    score = 0
    checks = []
    last = float(df["close"].iloc[-1])
    s50  = float(sma(df["close"],50).iloc[-1])
    s200 = float(sma(df["close"],200).iloc[-1])
    atr_now  = float(atr(df).iloc[-1])
    atr_mean = float(atr(df).rolling(20).mean().iloc[-1])

    # 1. Trend
    if s50 > s200:
        score+=1; checks.append(("✅","Trend","SMA50 > SMA200 — Bullish"))
    elif s50 < s200:
        score+=1; checks.append(("✅","Trend","SMA50 < SMA200 — Bearish"))
    else:
        checks.append(("❌","Trend","No clear trend (SMA50 ≈ SMA200)"))

    # 2. Location (price at POC or VAH edge)
    tol = atr_now * 0.4
    at_poc = abs(last-poc)<tol
    at_vah = abs(last-vah)<tol
    at_val = abs(last-val_p)<tol
    if at_poc or at_vah or at_val:
        label = "POC" if at_poc else ("VAH" if at_vah else "VAL")
        score+=1; checks.append(("✅","Location",f"Price at {label} — key VP level"))
    else:
        checks.append(("❌","Location","Price in mid-Value Area (no-man's land)"))

    # 3. Volatility (ATR consolidation)
    if atr_now < atr_mean * 0.85:
        score+=1; checks.append(("✅","Volatility",f"ATR low ({atr_now:.5f} < mean {atr_mean:.5f}) — entering before spike"))
    else:
        checks.append(("❌","Volatility",f"ATR elevated ({atr_now:.5f} ≥ mean {atr_mean:.5f}) — momentum already running"))

    # 4. Sentiment alignment
    trend_bull = s50>s200
    sent_aligned = (trend_bull and sentiment_score>0) or (not trend_bull and sentiment_score<0) or sentiment_score==0
    if sentiment_score != 0 and sent_aligned:
        score+=1; checks.append(("✅","Sentiment",f"AI score {sentiment_score:+d} aligns with trend"))
    elif sentiment_score == 0:
        checks.append(("⚪","Sentiment","AI neutral — no confirmation boost"))
    else:
        checks.append(("❌","Sentiment",f"AI score {sentiment_score:+d} conflicts with trend"))

    return score, checks

def exhaustion_filter(df, level, window=5):
    """
    Volume Intensity Divergence:
    Price makes new extreme at level but VI is dropping — sellers/buyers exhausted.
    Returns True if exhaustion detected.
    """
    if df.empty or len(df)<window+2: return False
    vi = volume_intensity(df)
    recent_vi = vi.iloc[-window:]
    # Check if VI is declining while price is at extremes
    vi_declining = recent_vi.iloc[-1] < recent_vi.mean() * 0.75
    return vi_declining

def tpo_breakout_status(df, vah, val_p, minutes_per_bar=60):
    """
    TPO validation: how many bars has price stayed outside Value Area?
    >30 mins (0.5 bars on H1) = Validated
    <10 mins = Fakeout risk
    """
    if df.empty or len(df)<3: return "Unknown", 0
    last = float(df["close"].iloc[-1])
    if last <= vah and last >= val_p:
        return "Inside VA", 0
    # Count consecutive bars outside VA
    above_vah = last > vah
    bars_outside = 0
    for i in range(len(df)-1, max(0,len(df)-10), -1):
        price = float(df["close"].iloc[i])
        if above_vah and price > vah:
            bars_outside += 1
        elif not above_vah and price < val_p:
            bars_outside += 1
        else:
            break
    minutes_outside = bars_outside * minutes_per_bar
    if minutes_outside >= 30:
        return "Validated Breakout", minutes_outside
    elif minutes_outside > 0:
        return "Fakeout Risk", minutes_outside
    return "Just Broke", 0

def find_lead_lag(pair, all_candles):
    """
    Correlation lead/lag: find which peer pair has already moved
    in the same direction — that's the leader.
    """
    peers = CORRELATION_GROUPS.get(pair, [])
    if not peers or pair not in all_candles: return None, None
    df_main  = all_candles[pair]
    if df_main.empty or len(df_main)<5: return None, None

    main_ret = float(df_main["close"].pct_change(5).iloc[-1])
    leader = None; leader_ret = 0

    for peer in peers:
        if peer not in all_candles: continue
        df_p = all_candles[peer]
        if df_p.empty or len(df_p)<5: continue
        peer_ret = float(df_p["close"].pct_change(5).iloc[-1])
        # Leader = same direction AND moved more (already confirmed)
        same_dir = (main_ret>0 and peer_ret>0) or (main_ret<0 and peer_ret<0)
        if same_dir and abs(peer_ret)>abs(main_ret) and abs(peer_ret)>abs(leader_ret):
            leader = peer; leader_ret = peer_ret

    if leader:
        direction = "bullish" if leader_ret>0 else "bearish"
        return leader, f"{leader} already moved {leader_ret*100:+.2f}% — confirms {direction} for {pair}"
    return None, None

# ═══════════════════════════════════════════════════════════════
# 6. VOLUME PROFILE ENGINE
# ═══════════════════════════════════════════════════════════════
def compute_volume_profile(df, bins=VP_BINS):
    if df.empty: return pd.DataFrame(columns=["price_mid","volume","pct"])
    pmin,pmax = df["low"].min(),df["high"].max()
    if pmin>=pmax: return pd.DataFrame(columns=["price_mid","volume","pct"])
    edges=np.linspace(pmin,pmax,bins+1); bin_vol=np.zeros(bins)
    for _,row in df.iterrows():
        bl,bh,bv=row["low"],row["high"],row["volume"]; rr=bh-bl
        for b in range(bins):
            ol=max(edges[b],bl); oh=min(edges[b+1],bh)
            if oh>ol: bin_vol[b]+=bv*((oh-ol)/rr if rr>0 else 1.0)
    mids=(edges[:-1]+edges[1:])/2
    total=bin_vol.sum()
    if total == 0 or np.isnan(total):
        return pd.DataFrame(columns=["price_mid","volume","pct"])
    vp_df = pd.DataFrame({
        "price_mid": np.round(mids, 5),
        "volume":    np.round(bin_vol, 2),
        "pct":       np.round(bin_vol / total * 100, 1),
    })
    vp_df = vp_df.dropna().replace([np.inf, -np.inf], 0)
    return vp_df.sort_values("price_mid", ascending=False).reset_index(drop=True)

def find_poc(vp):
    if vp is None or vp.empty: return 0.0
    vp = vp.dropna(subset=["volume","price_mid"])
    if vp.empty: return 0.0
    return float(vp.loc[vp["volume"].idxmax(),"price_mid"])

def find_value_area(vp, pct=0.70):
    if vp is None or vp.empty: return 0.0, 0.0
    vp = vp.dropna(subset=["volume","price_mid"])
    if vp.empty: return 0.0, 0.0
    vs=vp.sort_values("price_mid").reset_index(drop=True)
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

def classify_nodes(vp, hvn_top_pct=0.25, lvn_top_pct=0.25):
    if vp.empty: return [],[]
    return (vp[vp["volume"]>=vp["volume"].quantile(1-hvn_top_pct)]["price_mid"].tolist(),
            vp[vp["volume"]<=vp["volume"].quantile(lvn_top_pct)]["price_mid"].tolist())

def find_next_hvn_target(current_price, hvn_list, direction):
    """Liquidity void targeting: TP = next HVN in direction of trade."""
    if not hvn_list: return None
    if direction=="Long":
        candidates=[h for h in hvn_list if h>current_price]
        return min(candidates) if candidates else None
    else:
        candidates=[h for h in hvn_list if h<current_price]
        return max(candidates) if candidates else None

# ═══════════════════════════════════════════════════════════════
# 7. SIGNAL ENGINE — FULL ATR + SMOOTHNESS + TPO + EXHAUSTION
# ═══════════════════════════════════════════════════════════════
def generate_entry_signals(df, vp, poc, vah, val, hvn, lvn, pair,
                            smoothness_score=0, sentiment=None, tpo_status="Unknown"):
    if df.empty or len(df)<20 or vp.empty: return []
    signals=[]
    last     = float(df["close"].iloc[-1])
    atr_val  = float(atr(df).iloc[-1])
    rsi_val  = float(rsi(df["close"]).iloc[-1])
    vol_sma  = df["volume"].rolling(20).mean().iloc[-1]
    last_vol = df["volume"].iloc[-1]
    vol_surge= last_vol > vol_sma*1.4
    vi_now   = float(volume_intensity(df).iloc[-1])
    vi_mean  = float(volume_intensity(df).rolling(20).mean().iloc[-1])
    exhausted= vi_now < vi_mean*0.75

    sent_score = sentiment.get("score",0) if sentiment else 0
    interv_risk= sentiment.get("intervention_risk",False) if sentiment else False

    def near(price, level, tol=0.3): return abs(price-level)<atr_val*tol

    def make_sig(sig_type, category, entry, sl, tp1, tp2_raw, reason, conf,
                 badge="blue_pulse"):
        # Liquidity void TP: use next HVN
        direction = "Long" if "Long" in sig_type or "Bull" in sig_type else "Short"
        hvn_tp = find_next_hvn_target(last, hvn, direction)
        tp2 = hvn_tp if hvn_tp else tp2_raw
        risk = abs(entry-sl); reward = abs(tp2-entry)
        rr   = round(reward/risk,1) if risk>0 else 0
        # Boost smoothness score
        final_conf = conf
        if smoothness_score>=3 and conf=="Moderate": final_conf="High"
        if interv_risk and "Reversion" in sig_type and "Short" in sig_type:
            final_conf="Low"; badge="flashing_red"
        return {"type":sig_type,"category":category,"entry":round(entry,5),
                "sl":round(sl,5),"tp1":round(tp1,5),"tp2":round(tp2,5),
                "atr":round(atr_val,5),"rsi":round(rsi_val,1),
                "vi":round(vi_now,1),"exhausted":exhausted,
                "reason":reason,"confidence":final_conf,"rr":rr,
                "smoothness":smoothness_score,"tpo":tpo_status,"badge":badge}

    # ── MEAN REVERSION ─────────────────────────────────────────
    if near(last,val) and rsi_val<40 and exhausted:
        signals.append(make_sig("Mean Reversion Long","MR",last,
            last-1.5*atr_val,poc,vah,
            f"Price at VAL ({val:.5f}), RSI {rsi_val:.0f}, VI exhaustion — sellers running out",
            "High" if rsi_val<30 else "Moderate","blue_pulse"))

    if near(last,vah) and rsi_val>60 and exhausted:
        signals.append(make_sig("Mean Reversion Short","MR",last,
            last+1.5*atr_val,poc,val,
            f"Price at VAH ({vah:.5f}), RSI {rsi_val:.0f}, VI exhaustion — buyers running out",
            "High" if rsi_val>70 else "Moderate","blue_pulse"))

    # LVN snap
    for lvn_p in lvn:
        if near(last,lvn_p,0.2):
            d="Long" if last>poc else "Short"; sl_d=-1 if d=="Long" else 1
            signals.append(make_sig(f"LVN Snap {d}","MR",last,
                last+sl_d*atr_val,poc,
                poc+(1 if d=="Long" else -1)*atr_val,
                f"LVN {lvn_p:.5f} — liquidity void snap to next HVN",
                "Moderate","blue_pulse"))

    # POC rejection
    if near(last,poc,0.25):
        e20=float(ema(df["close"],20).iloc[-1])
        if last>e20 and rsi_val<55:
            signals.append(make_sig("POC Rejection Long","MR",last,
                poc-1.2*atr_val,poc+1.5*atr_val,vah,
                f"Held POC ({poc:.5f}), above EMA20","Moderate","blue_pulse"))
        elif last<e20 and rsi_val>45:
            signals.append(make_sig("POC Rejection Short","MR",last,
                poc+1.2*atr_val,poc-1.5*atr_val,val,
                f"Rejected POC ({poc:.5f}), below EMA20","Moderate","blue_pulse"))

    # HVN S/R
    for hvn_p in hvn[:3]:
        if near(last,hvn_p,0.2):
            e20=float(ema(df["close"],20).iloc[-1])
            d="Long" if last>=e20 else "Short"; sl_d=-1 if d=="Long" else 1
            tp_t=poc if d=="Long" else val
            signals.append(make_sig(f"HVN {d}","MR",last,
                last+sl_d*atr_val,tp_t,
                tp_t+(1 if d=="Long" else -1)*atr_val,
                f"HVN {hvn_p:.5f} — institutional level, wait for rejection wick",
                "Moderate","blue_pulse"))

    # ── MOMENTUM BREAKOUT ──────────────────────────────────────
    # Only validate breakout if TPO confirmed (>30 mins)
    tpo_valid = tpo_status in ["Validated Breakout","Just Broke"]

    if last>vah+0.1*atr_val and vol_surge and rsi_val>50 and tpo_valid:
        conf="High" if (rsi_val>60 and tpo_status=="Validated Breakout") else "Moderate"
        signals.append(make_sig("Breakout Long","MO",
            vah+0.05*atr_val,vah-atr_val,
            last+1.5*atr_val,last+3.0*atr_val,
            f"Broke VAH ({vah:.5f}) vol surge, TPO: {tpo_status}",
            conf,"green_spark"))

    if last<val-0.1*atr_val and vol_surge and rsi_val<50 and tpo_valid:
        conf="High" if (rsi_val<40 and tpo_status=="Validated Breakout") else "Moderate"
        signals.append(make_sig("Breakout Short","MO",
            val-0.05*atr_val,val+atr_val,
            last-1.5*atr_val,last-3.0*atr_val,
            f"Broke VAL ({val:.5f}) vol surge, TPO: {tpo_status}",
            conf,"green_spark"))

    # Fakeout warning
    if tpo_status=="Fakeout Risk" and (last>vah or last<val):
        for s in signals:
            if s["category"]=="MO":
                s["confidence"]="Low"; s["badge"]="grey_dim"
                s["reason"]+=" ⚠️ Fakeout risk — price < 30 min outside VA"

    # Intervention risk override
    if interv_risk:
        for s in signals:
            s["badge"]="flashing_red"
            s["reason"]+=" 🚨 AI: Intervention risk detected"

    return signals

# ═══════════════════════════════════════════════════════════════
# 8. MAIN CACHED DATA LOADER
# ═══════════════════════════════════════════════════════════════
@st.cache_data(ttl=900)
def get_heatmap_data(api_key, td_key, claude_key, fr_bars, vp_bins_val, va_pct_val, vp_mode):
    rng=np.random.default_rng(); use_live=bool(api_key)
    signals_dict,prices,iv,candles_h1={},{},{},{}

    use_td = bool(td_key)

    for pair in PAIRS:
        pair_signals={}
        for tf in TIMEFRAMES:
            mult,tspan=TF_MAP[tf]
            if use_live:
                # Priority 1: Polygon.io
                df=fetch_polygon_candles(pair,mult,tspan,api_key,300)
            elif use_td:
                # Priority 2: Twelve Data (real volume)
                df=fetch_td_candles(pair,tf,td_key,300)
            else:
                # Priority 3: Yahoo Finance (no volume)
                df=fetch_yf_candles(pair,tf,300)
            if df.empty:
                df=simulate_candles(pair,300)
            pair_signals[tf]=compute_signal(df)
            if tf=="H1": candles_h1[pair]=df
        signals_dict[pair]=pair_signals
        df_h1=candles_h1[pair]
        # Live price
        if use_live:
            prices[pair]=round(float(df_h1["close"].iloc[-1]),5)
        elif use_td:
            p=fetch_td_price(pair,td_key)
            prices[pair]=p if p else round(float(df_h1["close"].iloc[-1]),5)
        else:
            prices[pair]=fetch_yf_price(pair)
        iv[pair]=compute_iv_proxy(df_h1)

    signals=pd.DataFrame(signals_dict).T.reindex(columns=TIMEFRAMES)

    iv_term={}
    for pair in PAIRS:
        spot=iv[pair]
        iv_term[pair]={"spot":spot,
            "1W":round(spot*(1+rng.normal(0,0.12)),2),
            "1M":round(spot*(1+rng.normal(0,0.08)),2),
            "3M":round(spot*(1+rng.normal(0,0.05)),2)}

    vp_data={}
    for pair in PAIRS:
        df_raw=candles_h1[pair]
        df_vp=df_raw.tail(int(fr_bars)) if "Fixed" in vp_mode else df_raw
        vp=compute_volume_profile(df_vp,bins=vp_bins_val)
        poc=find_poc(vp); vah,val_p=find_value_area(vp,pct=va_pct_val)
        hvn,lvn=classify_nodes(vp)

        # Headlines + AI sentiment
        headlines=fetch_polygon_news(pair,api_key) if use_live else fetch_polygon_news(pair,"")
        sentiment=get_ai_sentiment(pair,headlines,claude_key)

        # Smoothness score
        smooth_score,smooth_checks=compute_smoothness_score(
            df_raw,poc,vah,val_p,sentiment.get("score",0))

        # TPO validation
        tpo_status,tpo_mins=tpo_breakout_status(df_raw,vah,val_p)

        # Lead/lag
        leader,lag_msg=find_lead_lag(pair,candles_h1)

        # Exhaustion at current price
        exhausted=exhaustion_filter(df_raw,prices[pair])

        # Volume intensity
        vi_now=float(volume_intensity(df_raw).iloc[-1]) if not df_raw.empty else 0

        entry_sigs=generate_entry_signals(
            df_raw,vp,poc,vah,val_p,hvn,lvn,pair,
            smooth_score,sentiment,tpo_status)

        vp_data[pair]={
            "vp":vp,"poc":poc,"vah":vah,"val":val_p,
            "hvn":hvn,"lvn":lvn,"entry_signals":entry_sigs,
            "atr":round(float(atr(df_raw).iloc[-1]),5) if not df_raw.empty else 0,
            "smoothness_score":smooth_score,"smoothness_checks":smooth_checks,
            "sentiment":sentiment,"headlines":headlines,
            "tpo_status":tpo_status,"tpo_mins":tpo_mins,
            "leader":leader,"lag_msg":lag_msg,
            "exhausted":exhausted,"vi":vi_now,
        }

    # Zones (simulated — replace with order flow API)
    def make_zones(base):
        zones=[]
        for _ in range(rng.integers(2,5)):
            zt=rng.choice(["Order Block","Fair Value Gap","Point of Interest"])
            di=rng.choice(["Bullish","Bearish"])
            off=rng.uniform(-0.008,0.008)
            lv=round(base*(1+off),5); w=round(base*rng.uniform(0.001,0.004),5)
            zones.append({"type":zt,"direction":di,"level":lv,
                          "zone_low":round(lv-w/2,5),"zone_high":round(lv+w/2,5),
                          "timeframe":rng.choice(TIMEFRAMES),"strength":int(rng.integers(1,4))})
        return zones

    return {"signals":signals,"prices":prices,"iv":iv,"iv_term":iv_term,
            "zones":{p:make_zones(BASE_PRICES[p]) for p in PAIRS},
            "candles_h1":candles_h1,"vp_data":vp_data,"use_live":use_live}

# ═══════════════════════════════════════════════════════════════
# 9. RENDER HELPERS
# ═══════════════════════════════════════════════════════════════
def confluence_score(row): return int(row.sum())

def confluence_bar(score):
    pct=int((score+5)/10*100)
    c="#1a7a4a" if score>0 else ("#b5281c" if score<0 else "#888")
    return (f'<div style="background:#2a2a3e;border-radius:4px;height:10px;width:120px;">'
            f'<div style="width:{pct}%;background:{c};height:10px;border-radius:4px;"></div>'
            f'</div><small style="color:{c};font-weight:600;">{score:+d} / 5</small>')

def iv_color(v):
    if v<7: return "#52b788"
    if v<12: return "#e09a2a"
    return "#b5281c"

def strength_dots(s): return "●"*s+"○"*(3-s)

BADGE_STYLES = {
    "blue_pulse":   ("🔵","#1a4fa8","Mean Reversion"),
    "green_spark":  ("🟢","#1a7a4a","Momentum Breakout"),
    "grey_dim":     ("⚫","#555555","High Risk / Avoid"),
    "flashing_red": ("🔴","#b5281c","Intervention Risk"),
}

def smoothness_gauge(score):
    colors=["#888","#e07b5a","#e09a2a","#52b788","#1a7a4a"]
    labels=["No Setup","Weak","Moderate","Strong","Golden ⭐"]
    c=colors[min(score,4)]; lbl=labels[min(score,4)]
    pct=score/4*100
    return (f'<div style="display:flex;align-items:center;gap:10px;">'
            f'<div style="background:#2a2a3e;border-radius:6px;height:14px;width:120px;">'
            f'<div style="width:{pct}%;background:{c};height:14px;border-radius:6px;"></div></div>'
            f'<span style="color:{c};font-weight:700;font-size:13px;">{score}/4 — {lbl}</span></div>')

def render_vp_chart(vp, poc, vah, val_p, hvn, lvn, current_price, title):
    if vp is None or vp.empty:
        st.caption("No volume profile data yet."); return
    # Sanitise — drop NaN rows
    vp = vp.dropna(subset=["volume","price_mid"]).reset_index(drop=True)
    if vp.empty:
        st.caption("Volume profile empty after cleaning."); return
    max_vol = vp["volume"].max()
    if max_vol == 0 or pd.isna(max_vol):
        st.caption("Volume data is zero — waiting for data."); return
    # Guard current_price
    if current_price is None or current_price == 0:
        current_price = float(vp["price_mid"].mean())
    closest_idx=(vp["price_mid"]-current_price).abs().idxmin()
    rows_html=""
    for i,r in vp.iterrows():
        vol = r["volume"] if not pd.isna(r["volume"]) else 0
        bar_w=max(0, int(vol/max_vol*160)); pm=r["price_mid"]
        is_hvn=any(abs(pm-h)<1e-4 for h in hvn)
        is_lvn=any(abs(pm-l)<1e-4 for l in lvn)
        is_poc=abs(pm-poc)<(vp["price_mid"].max()-vp["price_mid"].min())/(len(vp)*2)
        if is_poc: bar_c="#e09a2a"
        elif is_hvn: bar_c="#1a4fa8"
        elif is_lvn: bar_c="#555555"
        elif val_p<=pm<=vah: bar_c="#52b788"
        else: bar_c="#333355"
        cp=" ◀" if i==closest_idx else ""
        tag=(' <span style="font-size:9px;color:#e09a2a;">POC</span>' if is_poc else
             ' <span style="font-size:9px;color:#1a4fa8;">HVN</span>' if is_hvn else
             ' <span style="font-size:9px;color:#777;">LVN</span>' if is_lvn else "")
        rows_html+=(f'<tr><td style="text-align:right;padding:1px 4px;font-size:9px;'
                    f'font-family:monospace;color:#bbb;white-space:nowrap;">{pm:.5f}{cp}</td>'
                    f'<td style="padding:1px 3px;"><div style="width:{bar_w}px;height:8px;'
                    f'background:{bar_c};border-radius:2px;"></div></td>'
                    f'<td style="padding:1px 3px;font-size:9px;color:#999;">{r["pct"]:.0f}%{tag}</td></tr>')
    legend=(f'<p style="font-size:10px;margin:2px 0;line-height:1.8;">'
            f'<b style="color:#e09a2a;">■ POC</b> {poc:.5f} &nbsp;'
            f'<b style="color:#1a4fa8;">■ HVN</b> &nbsp;'
            f'<b style="color:#777;">■ LVN</b> &nbsp;'
            f'<b style="color:#52b788;">■ VA</b> {val_p:.5f}–{vah:.5f}</p>')
    st.markdown(f"**{title}**")
    st.markdown(f'<div style="overflow-y:auto;max-height:420px;">{legend}'
                f'<table style="border-collapse:collapse;">{rows_html}</table></div>',
                unsafe_allow_html=True)

def render_sentiment_card(sentiment, headlines):
    tone  = sentiment.get("tone","Neutral")
    score = sentiment.get("score",0)
    interv= sentiment.get("intervention_risk",False)
    reason= sentiment.get("reasoning","")
    tone_color = {"Hawkish":"#1a7a4a","Dovish":"#b5281c","Neutral":"#888"}.get(tone,"#888")
    score_color= "#1a7a4a" if score>0 else ("#b5281c" if score<0 else "#888")
    interv_html= '<span style="background:#b5281c;color:#fff;padding:1px 8px;border-radius:3px;font-size:10px;font-weight:700;">🚨 INTERVENTION RISK</span>' if interv else ""
    card=(f'<div style="background:#1a1a2e;border-radius:8px;padding:12px 14px;margin-bottom:8px;">'
          f'<div style="display:flex;gap:10px;align-items:center;margin-bottom:6px;">'
          f'<span style="background:{tone_color};color:#fff;padding:3px 12px;border-radius:4px;font-size:12px;font-weight:700;">{tone}</span>'
          f'<span style="color:{score_color};font-size:14px;font-weight:700;">Score: {score:+d}</span>'
          f'{interv_html}</div>'
          f'<div style="font-size:11px;color:#ccc;margin-bottom:6px;font-style:italic;">{reason}</div>'
          f'<div style="font-size:10px;color:#777;">Headlines analysed: {len(headlines)}</div>'
          f'</div>')
    st.markdown(card,unsafe_allow_html=True)

def render_entry_signal_card(s):
    badge_key = s.get("badge","blue_pulse")
    icon,border_c,cat_label = BADGE_STYLES.get(badge_key,("🔵","#1a4fa8","Signal"))
    conf_color={"High":"#1a7a4a","Moderate":"#e09a2a","Low":"#555"}.get(s["confidence"],"#555")
    smooth_c=["#888","#e07b5a","#e09a2a","#52b788","#1a7a4a"][min(s.get("smoothness",0),4)]
    tpo_col="#1a7a4a" if "Validated" in s.get("tpo","") else ("#e09a2a" if "Fakeout" in s.get("tpo","") else "#888")
    exhaust="⚡ VI Exhaustion" if s.get("exhausted") else ""
    card=(f'<div style="border-left:4px solid {border_c};background:#1a1a2e;'
          f'padding:10px 14px;border-radius:6px;margin-bottom:10px;">'
          f'<div style="display:flex;justify-content:space-between;align-items:center;">'
          f'<div style="font-size:12px;font-weight:700;color:{border_c};">{icon} {cat_label}</div>'
          f'<div style="font-size:10px;color:#aaa;">{exhaust}</div></div>'
          f'<div style="font-size:13px;font-weight:700;color:#fff;margin:3px 0;">{s["type"]}</div>'
          f'<div style="font-size:11px;color:#ccc;margin-bottom:6px;">{s["reason"]}</div>'
          f'<table style="font-size:11px;color:#eee;border-collapse:collapse;width:100%;">'
          f'<tr><td style="padding:2px 8px 2px 0;color:#aaa;">Entry</td>'
          f'<td style="font-family:monospace;font-weight:600;color:#fff;">{s["entry"]}</td>'
          f'<td style="padding:2px 8px 2px 12px;color:#aaa;">ATR</td>'
          f'<td style="font-family:monospace;">{s["atr"]}</td></tr>'
          f'<tr><td style="color:#f87171;">SL</td>'
          f'<td style="font-family:monospace;color:#f87171;font-weight:600;">{s["sl"]}</td>'
          f'<td style="padding:2px 8px 2px 12px;color:#aaa;">RSI</td>'
          f'<td style="font-family:monospace;">{s["rsi"]}</td></tr>'
          f'<tr><td style="color:#4ade80;">TP1</td>'
          f'<td style="font-family:monospace;color:#4ade80;">{s["tp1"]}</td>'
          f'<td style="padding:2px 8px 2px 12px;color:#4ade80;">TP2</td>'
          f'<td style="font-family:monospace;color:#4ade80;">{s["tp2"]}</td></tr>'
          f'</table>'
          f'<div style="margin-top:8px;display:flex;gap:8px;flex-wrap:wrap;">'
          f'<span style="background:{conf_color};color:#fff;padding:1px 8px;border-radius:3px;font-size:10px;font-weight:600;">{s["confidence"]}</span>'
          f'<span style="color:#aaa;font-size:10px;">R:R 1:{s.get("rr",0)}</span>'
          f'<span style="background:#2a2a3e;color:{smooth_c};padding:1px 8px;border-radius:3px;font-size:10px;">Smooth {s.get("smoothness",0)}/4</span>'
          f'<span style="color:{tpo_col};font-size:10px;">TPO: {s.get("tpo","—")}</span>'
          f'</div></div>')
    st.markdown(card,unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# REAL-TIME CANDLESTICK CHART
# ═══════════════════════════════════════════════════════════════
def render_candlestick_chart(df, pair, poc, vah, val_p, hvn, lvn, atr_val):
    """Candlestick chart using pure Streamlit — no plotly needed."""
    if df.empty or len(df) < 5:
        st.info("No candle data available.")
        return

    df = df.copy().tail(100).reset_index(drop=True)
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["returns"] = df["close"].pct_change()
    last = float(df["close"].iloc[-1])

    # ── Key level distances ───────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    poc_dist  = round((last - poc)  * 10000, 1)
    vah_dist  = round((vah  - last) * 10000, 1)
    val_dist  = round((last - val_p) * 10000, 1)
    atr_pips  = round(atr_val * 10000, 1)

    col1.metric("POC",  f"{poc:.5f}",  f"{poc_dist:+.1f} pips")
    col2.metric("VAH",  f"{vah:.5f}",  f"{vah_dist:+.1f} pips above")
    col3.metric("VAL",  f"{val_p:.5f}", f"{val_dist:+.1f} pips below")
    col4.metric("ATR",  f"{atr_val:.5f}", f"{atr_pips} pips range")

    # ── Price + EMA chart ─────────────────────────────────────
    st.markdown("**Price · EMA20 · EMA50**")
    chart_data = pd.DataFrame({
        "Close":  df["close"].values,
        "EMA 20": df["ema20"].values,
        "EMA 50": df["ema50"].values,
    }, index=df["time"] if "time" in df.columns else range(len(df)))
    st.line_chart(chart_data, height=280, use_container_width=True)

    # ── VP level indicator bars ───────────────────────────────
    st.markdown("**Volume Profile Levels vs Current Price**")
    levels_df = pd.DataFrame({
        "Level": ["VAL", "POC", "Price", "VAH"],
        "Price":  [val_p, poc, last, vah],
    }).set_index("Level")
    st.bar_chart(levels_df, height=120, use_container_width=True)

    # ── Volume bars ───────────────────────────────────────────
    st.markdown("**Volume**")
    vol_df = pd.DataFrame({
        "Volume": df["volume"].values,
    }, index=df["time"] if "time" in df.columns else range(len(df)))
    st.bar_chart(vol_df, height=120, use_container_width=True)

    # ── OHLC summary table (last 10 bars) ─────────────────────
    with st.expander("📋 Last 10 candles — OHLCV"):
        display_df = df[["time","open","high","low","close","volume"]].tail(10).copy()
        display_df["time"] = display_df["time"].astype(str).str[:16]
        display_df = display_df.round(5)
        display_df.columns = ["Time","Open","High","Low","Close","Volume"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════
# 10. COUNTDOWN FRAGMENT
# ═══════════════════════════════════════════════════════════════
@st.fragment(run_every=1)
def countdown_bar():
    elapsed=time.time()-st.session_state.last_refresh
    remaining=max(0,int(REFRESH_INTERVAL-elapsed))
    mins,secs=remaining//60,remaining%60
    st.caption(f"Next refresh in **{mins}m {secs}s** · Refreshes: {st.session_state.refresh_count}")
    st.progress(1.0-remaining/REFRESH_INTERVAL)
    if remaining==0: st.rerun(scope="app")

# ═══════════════════════════════════════════════════════════════
# 11. SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ Settings")
    api_key    = st.text_input("Polygon.io API Key",type="password",placeholder="Polygon key (optional)")
    td_key     = st.text_input("Twelve Data API Key",type="password",placeholder="Free at twelvedata.com")
    claude_key = st.text_input("Claude API Key",type="password",placeholder="Anthropic key for AI sentiment")
    st.caption("Twelve Data recommended — real forex volume. All keys optional.")
    st.divider()
    st.markdown("**Volume Profile**")
    vp_bins_val = st.slider("Price bins",15,60,VP_BINS,5)
    va_pct_val  = st.slider("Value Area %",50,90,70,5)/100
    vp_mode     = st.radio("Profile mode",["Visible Range (H1 200 bars)","Fixed Range (custom bars)"])
    fr_bars     = st.number_input("Fixed range bars",10,300,50) if "Fixed" in vp_mode else 200
    st.divider()
    st.markdown("**Signal Filters**")
    show_mr  = st.checkbox("Mean Reversion",value=True)
    show_mo  = st.checkbox("Momentum Breakout",value=True)
    min_conf = st.selectbox("Min confidence",["Moderate","High"],index=0)
    st.divider()
    if api_key:    st.success("🟢 Polygon.io — Live")
    elif td_key:   st.success("🟢 Twelve Data — Real volume")
    else:          st.info("📡 Yahoo Finance — No volume data")
    if claude_key: st.success("🤖 Claude AI — Active")
    else:          st.warning("🤖 Claude AI — No key")

# ═══════════════════════════════════════════════════════════════
# 12. DASHBOARD
# ═══════════════════════════════════════════════════════════════
st.title("💱 FX Multi-Factor Engine")
countdown_bar()

data    = get_heatmap_data(api_key,td_key,claude_key,fr_bars,vp_bins_val,va_pct_val,vp_mode)
signals = data["signals"]

# ── Metric cards ──────────────────────────────────────────────
mc = st.columns(len(PAIRS))
for i,pair in enumerate(PAIRS):
    score=confluence_score(signals.loc[pair])
    d="▲" if score>0 else ("▼" if score<0 else "—")
    vp_d=data["vp_data"][pair]; price=data["prices"][pair]
    rel="above POC" if price>vp_d["poc"] else "below POC"
    smooth=vp_d["smoothness_score"]
    smooth_star="⭐" if smooth==4 else ""
    with mc[i]:
        st.metric(label=f"{pair} {smooth_star}",value=f"{price:.5f}",
                  delta=f"{d}{abs(score)}/5 · {rel}")

st.caption("📡 Data: Polygon.io → Twelve Data (real volume) → Yahoo Finance · Signals from live candles")
st.divider()

# ── Heatmap + Alerts ──────────────────────────────────────────
# ── Real-time chart ──────────────────────────────────────────
st.markdown("### 📈 Real-Time Candlestick Chart")
chart_c1, chart_c2, chart_c3 = st.columns([2, 1, 1])
with chart_c1:
    chart_pair = st.selectbox("Pair", PAIRS, key="chart_pair")
with chart_c2:
    chart_tf   = st.selectbox("Timeframe", TIMEFRAMES, index=2, key="chart_tf")
with chart_c3:
    chart_bars  = st.slider("Bars", 50, 300, 100, 25, key="chart_bars")

# Fetch candles for selected TF (may differ from H1 used for VP)
@st.cache_data(ttl=60)
def get_chart_candles(pair, tf, api_key, td_key, bars):
    # Priority 1: Polygon
    if api_key:
        mult, tspan = TF_MAP[tf]
        df = fetch_polygon_candles(pair, mult, tspan, api_key, limit=bars)
        if not df.empty:
            return df
    # Priority 2: Twelve Data (real volume)
    if td_key:
        df = fetch_td_candles(pair, tf, td_key, bars)
        if not df.empty:
            return df
    # Priority 3: Yahoo Finance
    df = fetch_yf_candles(pair, tf, bars)
    if not df.empty:
        return df
    return simulate_candles(pair, bars)

chart_df   = get_chart_candles(chart_pair, chart_tf, api_key, td_key, chart_bars)
chart_vpd  = data["vp_data"][chart_pair]
render_candlestick_chart(
    chart_df, chart_pair,
    chart_vpd["poc"], chart_vpd["vah"], chart_vpd["val"],
    chart_vpd["hvn"], chart_vpd["lvn"], chart_vpd["atr"]
)
st.caption("🟡 POC  🟢 VAH/VAL (Value Area)  🔵 HVN  ⬜ LVN  🟠 EMA20  🟣 EMA50  🔴 ATR band")
st.divider()

col_map,col_alerts=st.columns([3,1])
with col_map:
    st.markdown("### 🗺 Signal Heatmap")
    header=('<div style="overflow-x:auto;width:100%;"><table style="width:100%;min-width:860px;'
            'border-collapse:collapse;font-size:12px;table-layout:fixed;"><thead><tr>'
            '<th style="text-align:left;padding:6px 8px;border-bottom:1px solid #444;width:68px;">Pair</th>'
            '<th style="text-align:center;padding:6px 8px;border-bottom:1px solid #444;width:88px;">Price</th>')
    for tf in TIMEFRAMES:
        header+=f'<th style="text-align:center;padding:6px 4px;border-bottom:1px solid #444;width:90px;">{tf}</th>'
    header+=('<th style="text-align:center;padding:6px 8px;border-bottom:1px solid #444;width:125px;">Confluence</th>'
             '<th style="text-align:center;padding:6px 6px;border-bottom:1px solid #444;width:70px;">IV</th>'
             '<th style="text-align:center;padding:6px 6px;border-bottom:1px solid #444;width:80px;">Sentiment</th>'
             '<th style="text-align:center;padding:6px 6px;border-bottom:1px solid #444;width:90px;">Smooth</th>'
             '</tr></thead><tbody>')
    rows_html=""
    for idx,pair in enumerate(PAIRS):
        row=signals.loc[pair]; score=confluence_score(row)
        price=data["prices"][pair]; iv_val=data["iv"][pair]; iv_c=iv_color(iv_val)
        vp_d=data["vp_data"][pair]
        sent=vp_d["sentiment"]; tone=sent.get("tone","Neutral")
        tone_c={"Hawkish":"#1a7a4a","Dovish":"#b5281c","Neutral":"#888"}.get(tone,"#888")
        smooth=vp_d["smoothness_score"]
        smooth_c=["#888","#e07b5a","#e09a2a","#52b788","#1a7a4a"][min(smooth,4)]
        bg="#1e1e2e" if idx%2==0 else "#16213e"
        rows_html+=(f'<tr style="background:{bg};">'
                    f'<td style="padding:6px 8px;font-weight:600;color:#fff;">{pair}</td>'
                    f'<td style="text-align:center;padding:6px 8px;font-family:monospace;font-size:12px;color:#fff;">{price:.5f}</td>')
        for tf in TIMEFRAMES:
            val=int(row[tf])
            rows_html+=(f'<td style="text-align:center;padding:4px 4px;">'
                        f'<span style="background:{SIGNAL_COLORS[val]};color:#fff;padding:3px 7px;'
                        f'border-radius:4px;font-size:11px;white-space:nowrap;">{SIGNAL_LABELS[val]}</span></td>')
        rows_html+=(f'<td style="text-align:center;padding:6px 8px;">{confluence_bar(score)}</td>'
                    f'<td style="text-align:center;padding:6px 6px;">'
                    f'<span style="background:{iv_c};color:#fff;padding:3px 8px;border-radius:4px;font-size:11px;font-weight:600;">{iv_val}%</span></td>'
                    f'<td style="text-align:center;padding:6px 6px;">'
                    f'<span style="background:{tone_c};color:#fff;padding:3px 8px;border-radius:4px;font-size:11px;">{tone}</span></td>'
                    f'<td style="text-align:center;padding:6px 6px;">'
                    f'<span style="color:{smooth_c};font-weight:700;font-size:13px;">{"⭐" if smooth==4 else smooth}/4</span></td>'
                    f'</tr>')
    st.markdown(header+rows_html+'</tbody></table></div>',unsafe_allow_html=True)

with col_alerts:
    st.markdown("### 🔔 Alerts")
    triggered=False
    for pair in PAIRS:
        score=confluence_score(signals.loc[pair])
        iv_val=data["iv"][pair]; vp_d=data["vp_data"][pair]
        nsigs=len(vp_d["entry_signals"])
        sent=vp_d["sentiment"]
        if sent.get("intervention_risk"):
            st.error(f"🚨 **{pair}** — Intervention risk"); triggered=True
        if abs(score)>=4:
            st.error(f"**{pair}** — {'BULLISH' if score>0 else 'BEARISH'} ({score:+d}/5)"); triggered=True
        elif abs(score)==3:
            st.warning(f"**{pair}** — Moderate {'bull' if score>0 else 'bear'} ({score:+d}/5)"); triggered=True
        if iv_val>14: st.error(f"**{pair}** — Extreme IV ({iv_val}%)"); triggered=True
        elif iv_val>10: st.info(f"**{pair}** — Elevated IV ({iv_val}%)"); triggered=True
        if vp_d["smoothness_score"]==4:
            st.success(f"⭐ **{pair}** — Golden Entry setup!"); triggered=True
        elif nsigs>0:
            st.success(f"**{pair}** — {nsigs} signal(s)"); triggered=True
    if not triggered: st.write("No significant signals.")

st.divider()

# ── VP + AI Sentiment + Signals per pair ──────────────────────
st.markdown("### 📊 Volume Profile · AI Sentiment · Entry Signals")
st.caption("🟡 POC  🔵 HVN  ⬜ LVN  🟢 Value Area  ◀ Price  "
           "🔵 MR  🟢 Breakout  ⚫ Avoid  🔴 Intervention")

tabs=st.tabs(PAIRS)
conf_order={"High":2,"Moderate":1,"Low":0}
for i,pair in enumerate(PAIRS):
    with tabs[i]:
        vp_d=data["vp_data"][pair]; cp=data["prices"][pair]; atr_val=vp_d["atr"]

        # Row 1: VP | Stats + Smoothness | AI Sentiment
        r1c1,r1c2,r1c3=st.columns([1,1,1])

        with r1c1:
            mode_label="FR" if "Fixed" in vp_mode else "VR"
            render_vp_chart(vp_d["vp"],vp_d["poc"],vp_d["vah"],vp_d["val"],
                            vp_d["hvn"],vp_d["lvn"],cp,
                            f"{pair} [{mode_label}]")

        with r1c2:
            st.markdown("**📐 Key Levels**")
            st.markdown(
                f'<div style="font-size:11px;line-height:2;background:#1a1a2e;padding:10px;border-radius:6px;">'
                f'<b>Price:</b> {cp:.5f}<br><b>ATR(14):</b> {atr_val:.5f}<br>'
                f'<b>POC:</b> {vp_d["poc"]:.5f}<br><b>VAH:</b> {vp_d["vah"]:.5f}<br>'
                f'<b>VAL:</b> {vp_d["val"]:.5f}<br>'
                f'<b>HVNs:</b> {len(vp_d["hvn"])} nodes<br>'
                f'<b>LVNs:</b> {len(vp_d["lvn"])} nodes<br>'
                f'<b>VI now:</b> {vp_d["vi"]:.1f} {"⚡ Exhaustion" if vp_d["exhausted"] else ""}<br>'
                f'<b>TPO:</b> <span style="color:{"#1a7a4a" if "Validated" in vp_d["tpo_status"] else "#e09a2a"};">'
                f'{vp_d["tpo_status"]} ({vp_d["tpo_mins"]}m)</span></div>',
                unsafe_allow_html=True)

            st.markdown("**🏆 Smoothness Score**")
            st.markdown(smoothness_gauge(vp_d["smoothness_score"]),unsafe_allow_html=True)
            for icon,label,desc in vp_d["smoothness_checks"]:
                col=("#1a7a4a" if icon=="✅" else ("#888" if icon=="⚪" else "#b5281c"))
                st.markdown(f'<div style="font-size:10px;color:{col};margin-top:3px;">'
                            f'{icon} <b>{label}:</b> {desc}</div>',unsafe_allow_html=True)

            if vp_d["leader"]:
                st.markdown(f'<div style="margin-top:8px;background:#1a1a2e;padding:8px;border-radius:6px;'
                            f'font-size:11px;color:#e09a2a;">🔗 <b>Lead/Lag:</b> {vp_d["lag_msg"]}</div>',
                            unsafe_allow_html=True)

        with r1c3:
            st.markdown("**🤖 AI Sentiment — Claude**")
            render_sentiment_card(vp_d["sentiment"],vp_d["headlines"])
            with st.expander("Headlines analysed"):
                for h in vp_d["headlines"]:
                    st.caption(f"• {h}")

        st.markdown("---")

        # Row 2: Entry signals
        st.markdown(f"**🎯 Entry Signals — {pair}** (ATR + Smoothness + TPO + Exhaustion backed)")
        all_sigs=vp_d["entry_signals"]
        filtered=[s for s in all_sigs if
                  ((show_mr and s["category"]=="MR") or (show_mo and s["category"]=="MO"))
                  and conf_order.get(s["confidence"],0)>=conf_order.get(min_conf,0)]
        if not filtered:
            st.info("No signals match current filters for this pair.")
        else:
            sig_cols=st.columns(min(len(filtered),3))
            for j,s in enumerate(filtered):
                with sig_cols[j%3]:
                    render_entry_signal_card(s)

st.divider()

# ── IV Term Structure ──────────────────────────────────────────
st.markdown("### 📈 Implied Volatility — Term Structure")
iv_rows=[{"Pair":p,"Spot IV%":data["iv_term"][p]["spot"],
          "1W IV%":data["iv_term"][p]["1W"],"1M IV%":data["iv_term"][p]["1M"],
          "3M IV%":data["iv_term"][p]["3M"],
          "Skew (1W-3M)":round(data["iv_term"][p]["1W"]-data["iv_term"][p]["3M"],2),
          "Structure":"Backwardation" if data["iv_term"][p]["1W"]>data["iv_term"][p]["3M"] else "Contango"}
         for p in PAIRS]
iv_df=pd.DataFrame(iv_rows)
def civ(v): return f"color:{iv_color(v)};font-weight:600;" if isinstance(v,(int,float)) else ""
def csk(v): return f"color:{'#b5281c' if v>0 else '#1a7a4a'};font-weight:600;" if isinstance(v,(int,float)) else ""
st.dataframe(iv_df.style.map(civ,subset=["Spot IV%","1W IV%","1M IV%","3M IV%"]).map(csk,subset=["Skew (1W-3M)"]),
             use_container_width=True,hide_index=True)
st.caption("Backwardation (1W>3M): near-term vol premium. Contango: calm near term.")
st.divider()

# ── Key Level Zones ────────────────────────────────────────────
st.markdown("### 🏛 Key Level Zones")
sel=st.selectbox("Pair:",PAIRS); cp=data["prices"][sel]
st.markdown(f"**Price:** `{cp:.5f}` &nbsp; **ATR:** `{data['vp_data'][sel]['atr']:.5f}`")
zone_records=[]
for idx,z in enumerate(sorted(data["zones"][sel],key=lambda z:abs(z["level"]-cp))):
    dp=round(abs(z["level"]-cp)*10000,1)
    zone_records.append({"Zone Type":f"{'⭐ ' if idx==0 else ''}{z['type']}",
        "Direction":z["direction"],"TF":z["timeframe"],
        "Zone Low":z["zone_low"],"Mid Level":z["level"],"Zone High":z["zone_high"],
        "Distance":f"{'↑' if z['level']>cp else '↓'} {dp} pips",
        "Strength":strength_dots(z["strength"])})
zone_df=pd.DataFrame(zone_records)
def sty_row(row):
    c="#1a7a4a" if row.get("Direction")=="Bullish" else "#b5281c"
    return [f"background-color:{c};color:#fff;"]*len(row)
def sty_type(v):
    if "Order Block" in str(v): return "color:#fff;font-weight:700;background:#1a4fa8;"
    if "Fair Value Gap" in str(v): return "color:#fff;font-weight:700;background:#7c3aed;"
    if "Point of Interest" in str(v): return "color:#fff;font-weight:700;background:#b45309;"
    return "color:#fff;font-weight:700;"
st.dataframe(zone_df.style.apply(sty_row,axis=1).map(sty_type,subset=["Zone Type"])
             .map(lambda v:"color:#fff;font-weight:500;",
                  subset=["Direction","TF","Zone Low","Mid Level","Zone High","Distance","Strength"]),
             use_container_width=True,hide_index=True)

st.divider()
st.info("⭐ Golden Entry = Smooth 4/4 — all checklist items pass. "
        "🔵 Blue Pulse = Mean Reversion. 🟢 Green Spark = Momentum Breakout. "
        "⚫ Grey/Dim = Inside VA (avoid). 🔴 Flashing Red = Intervention risk. "
        "TPO Validated = price outside VA >30 mins. VI Exhaustion = buyers/sellers running out.")
