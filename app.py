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

def is_future(date_obj):
    if date_obj is None:
        return False
    return date_obj >= datetime.now().date()

def format_market_cap(val):
    if val is None or not isinstance(val, (int, float)):
        return "N/A"
    if val >= 1e12:
        return f"{val / 1e12:.2f}T"
    elif val >= 1e9:
        return f"{val / 1e9:.2f}B"
    elif val >= 1e6:
        return f"{val / 1e6:.2f}M"
    return f"{val:.2f}"

# =========================
# NEXT EARNINGS
# =========================
def get_next_earnings(ticker):
    try:
        stock = yf.Ticker(ticker)
        # Strategy 1: Calendar
        cal = stock.calendar
        if cal is not None and not cal.empty and 'Earnings Date' in cal.index:
            dates = cal.loc['Earnings Date']
            if isinstance(dates, (list, pd.Series, pd.Index)):
                for d in dates:
                    dt = pd.to_datetime(d).date()
                    if is_future(dt): return dt
            else:
                dt = pd.to_datetime(dates).date()
                if is_future(dt): return dt
        
        # Strategy 2: Info fallback
        info = stock.info
        for field in ['earningsDate', 'nextEarningsDate']:
            val = info.get(field)
            if val:
                dt = pd.to_datetime(val, unit='s' if isinstance(val, int) else None).date()
                if is_future(dt): return dt
    except: pass
    return "TBD"

# =========================
# STURDY REACTION LOGIC
# =========================
def reaction(price_df, earnings_date, trading_days):
    """
    Calculates reaction. If the specific target day (1D/3D) hasn't happened yet,
    it falls back to the latest available trading day's price.
    """
    try:
        # Ensure index is datetime and normalized
        price_df.index = pd.to_datetime(price_df.index).normalize()
        e_date = pd.to_datetime(earnings_date).normalize()
        
        # 1. Price BEFORE earnings (last close on or before reporting date)
        pre_data = price_df.loc[:e_date]
        if pre_data.empty: return None
        pre_price = pre_data.iloc[-1]["Close"]
        
        # 2. Prices AFTER earnings
        post_data = price_df.loc[e_date + timedelta(days=1):]
        if post_data.empty: 
            # If earnings were literally today/yesterday and no post-data exists yet
            return None
            
        # 3. Find target index
        # If we want 3-day but only 2 days have passed, we take the 2nd day (index 1)
        target_idx = min(trading_days - 1, len(post_data) - 1)
        post_price = post_data.iloc[target_idx]["Close"]
        
        return pct(post_price, pre_price)
    except:
        return None

# =========================
# DATA FETCHING
# =========================
def finnhub_past_earnings(ticker):
    try:
        url = f"https://finnhub.io/api/v1/stock/earnings?symbol={ticker}&token={FINNHUB_API_KEY}"
        r = requests.get(url, timeout=10).json()
        df = pd.DataFrame(r[:4])
        if not df.empty:
            df["date"] = pd.to_datetime(df["period"])
            return df
    except: pass
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_prices(tickers):
    return yf.download(tickers=list(tickers), period="2y", group_by="ticker", progress=False)

def fetch_ticker_data(t, all_prices, next_earn_val, past_earn_df):
    try:
        # Handle single vs multi-ticker DF
        p_data = all_prices[t] if len(next_earn_val) > 1 else all_prices
        
        current = safe_float(p_data["Close"].iloc[-1])
        # Use last 252 trading days for 52W High/Low
        high52 = safe_float(p_data["High"].iloc[-252:].max())
        low52 = safe_float(p_data["Low"].iloc[-252:].min())
        
        mcap_raw = yf.Ticker(t).fast_info.get("market_cap")

        rows = []
        if not past_earn_df.empty:
            for _, r in past_earn_df.iterrows():
                rows.append({
                    "Ticker": t,
                    "Market Cap": format_market_cap(mcap_raw),
                    "Current Price": current,
                    "52W High": high52,
                    "52W Low": low52,
                    "Δ vs 52W High %": pct(current, high52),
                    "Δ vs 52W Low %": pct(current, low52),
                    "Next Earnings": next_earn_val[t],
                    "Date": r["date"].date(),
                    "EPS Actual": r.get("actual"),
                    "EPS Est.": r.get("estimate"),
                    "Surprise": r.get("surprise"),
                    "1D Reaction %": reaction(p_data, r["date"], 1),
                    "3D Reaction %": reaction(p_data, r["date"], 3),
                })
        else:
            rows.append({"Ticker": t, "Next Earnings": next_earn_val[t]})
        return rows
    except:
        return [{"Ticker": t}]

# =========================
# UI & FLOW
# =========================
st.title("📊 Earnings Radar")

uploaded_files = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"], accept_multiple_files=True)
tickers_text = st.text_area("Or enter tickers", "AAPL\nMSFT\nNVDA")

tickers = set()
if uploaded_files:
    for f in uploaded_files:
        df_u = pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)
        for c in ["Ticker", "Symbol", "ticker"]:
            if c in df_u.columns:
                tickers.update(df_u[c].dropna().astype(str).str.upper())
                break
tickers.update(t.strip().upper() for t in tickers_text.replace(",", "\n").split() if t.strip())
ticker_list = sorted(list(tickers))

if st.button("Fetch Earnings"):
    if not ticker_list:
        st.error("No tickers found.")
    else:
        progress = st.progress(0.0)
        all_prices = fetch_prices(ticker_list)
        
        # Parallel fetch for next/past earnings meta-data
        with ThreadPoolExecutor(MAX_WORKERS) as ex:
            next_earn_map = {t: ex.submit(get_next_earnings, t) for t in ticker_list}
            past_earn_map = {t: ex.submit(finnhub_past_earnings, t) for t in ticker_list}
            
            next_earn_res = {t: f.result() for t, f in next_earn_map.items()}
            past_earn_res = {t: f.result() for t, f in past_earn_map.items()}

        final_rows = []
        for i, t in enumerate(ticker_list):
            res = fetch_ticker_data(t, all_prices, next_earn_res, past_earn_res[t])
            final_rows.extend(res)
            progress.progress((i + 1) / len(ticker_list))
        
        df_final = pd.DataFrame(final_rows)
        pct_cols = ["Δ vs 52W High %", "Δ vs 52W Low %", "1D Reaction %", "3D Reaction %"]
        
        st.dataframe(
            df_final,
            use_container_width=True,
            column_config={c: st.column_config.NumberColumn(format="%.2f%%") for c in pct_cols}
        )
