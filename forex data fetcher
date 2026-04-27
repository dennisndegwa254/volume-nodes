"""
forex_data_fetcher.py
---------------------
Fetches forex OHLCV data via yfinance with parallel threading + session cache.
yfinance forex symbols: EURUSD=X, GBPUSD=X, USDJPY=X etc.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import time
import streamlit as st
import threading
from typing import Optional, Tuple

PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD']

YF_MAP = {
    'EURUSD': 'EURUSD=X', 'GBPUSD': 'GBPUSD=X', 'USDJPY': 'USDJPY=X',
    'USDCHF': 'USDCHF=X', 'AUDUSD': 'AUDUSD=X', 'USDCAD': 'USDCAD=X',
    'NZDUSD': 'NZDUSD=X',
}

# Pip values (used for SL/TP display)
PIP_MAP = {
    'EURUSD': 0.0001, 'GBPUSD': 0.0001, 'USDJPY': 0.01,
    'USDCHF': 0.0001, 'AUDUSD': 0.0001, 'USDCAD': 0.0001, 'NZDUSD': 0.0001,
}

CACHE_TTL   = 65
FETCH_TIMEOUT = 10


def _fetch_with_timeout(func, timeout=FETCH_TIMEOUT):
    result = [None]
    def target():
        try: result[0] = func()
        except Exception as e: print(f"[forex_fetcher] {e}")
    t = threading.Thread(target=target, daemon=True)
    t.start(); t.join(timeout=timeout)
    return result[0]


def _cache_valid():
    return (time.time() - st.session_state.get("fx_fetch_ts", 0)) < CACHE_TTL


def _store(key, data):
    st.session_state[key] = data

def _load(key):
    return st.session_state.get(key)


def _fetch_pair(pair: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    sym = YF_MAP[pair]

    def get_1h():
        df = yf.Ticker(sym).history(period="30d", interval="1h").ffill().bfill()
        return df if not df.empty else None

    def get_4h():
        df = yf.Ticker(sym).history(period="60d", interval="1h").ffill().bfill()
        if df is None or df.empty: return None
        # Resample 1h → 4h
        df.index = pd.to_datetime(df.index)
        df4 = df.resample('4h').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
        return df4

    def get_1d():
        df = yf.Ticker(sym).history(period="365d", interval="1d").ffill().bfill()
        return df if not df.empty else None

    df_1h = _fetch_with_timeout(get_1h, FETCH_TIMEOUT)
    df_4h = _fetch_with_timeout(get_4h, FETCH_TIMEOUT)
    df_1d = _fetch_with_timeout(get_1d, FETCH_TIMEOUT)
    return df_1h, df_4h, df_1d


def fetch_all_pairs() -> bool:
    if _cache_valid():
        return True
    print(f"[forex_fetcher] Fetching {len(PAIRS)} pairs in parallel...")
    start = time.time()
    results = {}
    threads = []

    def fetch_and_store(pair):
        results[pair] = _fetch_pair(pair)

    for pair in PAIRS:
        t = threading.Thread(target=fetch_and_store, args=(pair,), daemon=True)
        threads.append(t); t.start()

    for t in threads:
        t.join(timeout=FETCH_TIMEOUT + 2)

    any_ok = False
    for pair, (df_1h, df_4h, df_1d) in results.items():
        if df_1h is not None: any_ok = True
        _store(f"fx_1h_{pair}", df_1h)
        _store(f"fx_4h_{pair}", df_4h)
        _store(f"fx_1d_{pair}", df_1d)

    if any_ok:
        st.session_state["fx_fetch_ts"] = time.time()
        print(f"[forex_fetcher] Done in {time.time()-start:.1f}s")
    return any_ok


def get_1h(pair): return _load(f"fx_1h_{pair}")
def get_4h(pair): return _load(f"fx_4h_{pair}")
def get_1d(pair): return _load(f"fx_1d_{pair}")
def get_pip(pair): return PIP_MAP.get(pair, 0.0001)
