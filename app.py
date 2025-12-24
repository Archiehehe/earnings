import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================
# CONFIG
# =========================
FINNHUB_API_KEY = st.secrets.get("FINNHUB_API_KEY", "")
MAX_WORKERS = 8

st.set_page_config(page_title="Earnings Radar", layout="wide")

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
# NEXT EARNINGS (YFINANCE)
# =========================
def yf_next_earnings(ticker):
    """Fetch next earnings date using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        
        # Try calendar first
        if hasattr(stock, 'calendar') and stock.calendar is not None:
            cal = stock.calendar
            if not cal.empty and 'Earnings Date' in cal.index:
                earnings_date = cal.loc['Earnings Date']
                if isinstance(earnings_date, pd.Series):
                    earnings_date = earnings_date.iloc[0]
                if pd.notna(earnings_date):
                    return pd.to_datetime(earnings_date).date()
        
        # Try earnings_dates as fallback
        earnings_dates = stock.earnings_dates
        if earnings_dates is not None and not earnings_dates.empty:
            # Filter for future dates
            future_dates = earnings_dates[earnings_dates.index > pd.Timestamp.now()]
            if not future_dates.empty:
                return future_dates.index[0].date()
                
    except Exception:
        pass
    
    return None


# =========================
# FINNHUB (PAST EARNINGS)
# =========================
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
def fetch_all(tickers, progress):
    rows = []

    prices_1y = yf_prices(tickers, "1y")
    prices_2y = yf_prices(tickers, "2y")

    # Market caps (parallel)
    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        mcaps = dict(zip(tickers, ex.map(market_cap, tickers)))

    # Next earnings (parallel) - NOW USING YFINANCE
    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        futures = {ex.submit(yf_next_earnings, t): t for t in tickers}
        next_earn = {futures[f]: f.result() for f in as_completed(futures)}

    # Past earnings (parallel) - STILL USING FINNHUB
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
st.title("📊 Earnings Radar")

# ---- UPLOAD FILES ----
uploaded_files = st.file_uploader(
    "Upload CSV or Excel files (Ticker column required)",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True,
)

tickers_text = st.text_area(
    "Enter tickers (comma or newline separated)",
    "AAPL\nMSFT\nNVDA\nGOOGL",
)

# ---- PARSE TICKERS ----
tickers = set()

if uploaded_files:
    for f in uploaded_files:
        try:
            if f.name.lower().endswith(".csv"):
                df = pd.read_csv(f)
            else:
                df = pd.read_excel(f)

            for col in ["Ticker", "Symbol", "ticker", "symbol"]:
                if col in df.columns:
                    tickers.update(
                        df[col].dropna().astype(str).str.upper()
                    )
                    break
        except Exception:
            st.warning(f"Could not read {f.name}")

tickers.update(
    t.strip().upper()
    for t in tickers_text.replace(",", "\n").split()
    if t.strip()
)

tickers = sorted(tickers)

# ---- RUN ----
if st.button("Fetch Earnings"):
    if not tickers:
        st.warning("No tickers provided")
    else:
        progress = st.progress(0.0)
        data = fetch_all(tickers, progress)

        # Flatten earnings rows
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
