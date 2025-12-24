import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================
# CONFIG
# =========================
FINNHUB_API_KEY = st.secrets.get("FINNHUB_API_KEY", "")
MAX_WORKERS = 8

st.set_page_config(page_title="Earnings Calendar", layout="wide")

# =========================
# HELPERS
# =========================
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def pct(a, b):
    if a is None or b in (None, 0):
        return None
    return (a - b) / b * 100


# =========================
# FINNHUB
# =========================
def finnhub_next_earnings(ticker, lookahead_days=90):
    try:
        url = f"https://finnhub.io/api/v1/calendar/earnings?symbol={ticker}&token={FINNHUB_API_KEY}"
        r = requests.get(url, timeout=10).json()
        for e in r.get("earningsCalendar", []):
            d = pd.to_datetime(e["date"])
            if d >= pd.Timestamp.today() and d <= pd.Timestamp.today() + pd.Timedelta(days=lookahead_days):
                return d.date()
    except Exception:
        pass
    return None


def finnhub_past_earnings(ticker, limit=4):
    try:
        url = f"https://finnhub.io/api/v1/stock/earnings?symbol={ticker}&token={FINNHUB_API_KEY}"
        r = requests.get(url, timeout=10).json()
        df = pd.DataFrame(r[:limit])
        if not df.empty:
            df["date"] = pd.to_datetime(df["period"])
            return df
    except Exception:
        pass
    return pd.DataFrame()


# =========================
# YFINANCE (CACHED)
# =========================
@st.cache_data(ttl=3600)
def yf_prices(tickers, period):
    return yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        group_by="ticker",
        threads=True,
        progress=False,
    )


def market_cap(ticker):
    try:
        fi = yf.Ticker(ticker).fast_info
        return safe_float(fi.get("market_cap") or fi.get("marketCap"))
    except Exception:
        return None


# =========================
# REACTIONS
# =========================
def reaction(price_df, date, days):
    try:
        d = pd.to_datetime(date).normalize()
        pre = price_df.loc[:d].iloc[-1]["Close"]
        post = price_df.loc[d + timedelta(days=days):].iloc[0]["Close"]
        return pct(post, pre)
    except Exception:
        return None


# =========================
# MAIN FETCH
# =========================
def fetch_all(tickers, lookahead_days, progress):
    rows = []

    prices_1y = yf_prices(tickers, "1y")
    prices_2y = yf_prices(tickers, "2y")

    # market caps in parallel
    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        mcaps = dict(zip(tickers, ex.map(market_cap, tickers)))

    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        futures = {ex.submit(finnhub_next_earnings, t, lookahead_days): t for t in tickers}
        next_earn = {futures[f]: f.result() for f in as_completed(futures)}

    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        futures = {ex.submit(finnhub_past_earnings, t): t for t in tickers}
        past_earn = {futures[f]: f.result() for f in as_completed(futures)}

    for i, t in enumerate(tickers):
        try:
            p1 = prices_1y[t] if isinstance(prices_1y.columns, pd.MultiIndex) else prices_1y
            p2 = prices_2y[t] if isinstance(prices_2y.columns, pd.MultiIndex) else prices_2y

            current = safe_float(p1["Close"].iloc[-1])
            high52 = safe_float(p1["High"].max())
            low52 = safe_float(p1["Low"].min())

            earn_rows = []
            df = past_earn.get(t, pd.DataFrame())
            for _, r in df.iterrows():
                earn_rows.append({
                    "Date": r["date"].date(),
                    "EPS Actual": r.get("actual"),
                    "EPS Est.": r.get("estimate"),
                    "Surprise": r.get("surprise"),
                    "1D Reaction %": reaction(p2, r["date"], 1),
                    "3D Reaction %": reaction(p2, r["date"], 3),
                })

            rows.append({
                "Ticker": t,
                "Market Cap": mcaps.get(t),
                "Current Price": current,
                "52W High": high52,
                "52W Low": low52,
                "Δ vs 52W High %": pct(current, high52),
                "Δ vs 52W Low %": pct(current, low52),
                "Next Earnings": next_earn.get(t),
                "Earnings": earn_rows,
            })

        except Exception:
            rows.append({"Ticker": t})

        progress.progress((i + 1) / len(tickers))

    return rows


# =========================
# UI
# =========================
st.title("📅 Earnings Calendar Tracker")

tickers_input = st.text_area(
    "Enter tickers (comma or newline separated)",
    "AAPL\nMSFT\nNVDA\nGOOGL"
)

lookahead_days = st.slider("Lookahead days", 30, 180, 90)

if st.button("Fetch Earnings"):
    tickers = sorted({t.strip().upper() for t in tickers_input.replace(",", "\n").split() if t.strip()})
    progress = st.progress(0.0)

    data = fetch_all(tickers, lookahead_days, progress)

    # flatten earnings
    rows = []
    for r in data:
        if not r.get("Earnings"):
            rows.append(r)
        else:
            for e in r["Earnings"]:
                row = r.copy()
                row.update(e)
                row.pop("Earnings", None)
                rows.append(row)

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)
