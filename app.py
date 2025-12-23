# app.py
import os
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(
    page_title="Earnings Radar (Portfolio)",
    page_icon="📅",
    layout="wide",
)

# -----------------------------
# Constants
# -----------------------------
SP500_UNIVERSE_CSV = "sp500_universe.csv"  # keep in repo root (optional)
FINNHUB_BASE = "https://finnhub.io/api/v1"

# -----------------------------
# Helpers
# -----------------------------
def _clean_ticker(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip().upper()
    s = s.replace("$", "").replace("'", "").replace('"', "")
    s = s.replace(" ", "")
    # Heuristic: strip common suffixes like AAPL.US / AAPL:US
    for sep in (":", ".", "/"):
        if sep in s and len(s.split(sep)[0]) >= 1 and len(s.split(sep)[-1]) <= 3:
            s = s.split(sep)[0]
            break
    return s


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float, np.floating)) and (isinstance(x, float) and np.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None


def _to_date(x) -> Optional[date]:
    """Best-effort parse to python date."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, pd.Timestamp):
        return x.date()
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d", "%Y%m%d"):
            try:
                return datetime.strptime(s, fmt).date()
            except Exception:
                pass
        try:
            return pd.to_datetime(s).date()
        except Exception:
            return None
    return None


def _fmt_money(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    v = float(x)
    abs_v = abs(v)
    if abs_v >= 1e12:
        return f"{v/1e12:.2f}T"
    if abs_v >= 1e9:
        return f"{v/1e9:.2f}B"
    if abs_v >= 1e6:
        return f"{v/1e6:.2f}M"
    return f"{v:.0f}"


def _read_holdings_file(uploaded) -> pd.DataFrame:
    """Read CSV / Excel and standardize to a DataFrame with at least a Ticker column."""
    if uploaded is None:
        return pd.DataFrame()

    name = uploaded.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # 1-column sheet = ticker list
    if df.shape[1] == 1:
        df.columns = ["Ticker"]
        df["Ticker"] = df["Ticker"].astype(str).map(_clean_ticker)
        return df.dropna().drop_duplicates()

    # Try to locate a ticker column
    rename = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in ("symbol", "ticker", "tickers"):
            rename[c] = "Ticker"
        elif cl in ("company", "company name", "name"):
            rename[c] = "Company"
        elif cl == "sector":
            rename[c] = "Sector"
        elif cl == "industry":
            rename[c] = "Industry"

    df = df.rename(columns=rename)
    if "Ticker" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Ticker"})

    df["Ticker"] = df["Ticker"].astype(str).map(_clean_ticker)
    keep = [c for c in ["Ticker", "Company", "Sector", "Industry"] if c in df.columns]
    df = df[keep].drop_duplicates(subset=["Ticker"])
    return df


@st.cache_data(ttl=24 * 3600)
def load_sp500_universe(path: str = SP500_UNIVERSE_CSV) -> Dict[str, Dict]:
    """Optional metadata table (Ticker->Company/Sector/Industry)."""
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
        if df is None or df.empty:
            return {}
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]

        ticker_col = None
        for c in df.columns:
            if str(c).strip().lower() in ("ticker", "symbol"):
                ticker_col = c
                break
        if ticker_col is None:
            ticker_col = df.columns[0]

        df[ticker_col] = df[ticker_col].astype(str).map(_clean_ticker)

        out: Dict[str, Dict] = {}
        for _, r in df.iterrows():
            t = r.get(ticker_col)
            if not t:
                continue
            out[t] = {
                "Company": r.get("Company") or r.get("Name") or r.get("Security") or None,
                "Sector": r.get("Sector") or None,
                "Industry": r.get("Industry") or None,
            }
        return out
    except Exception:
        return {}

# -----------------------------
# Finnhub
# -----------------------------
def finnhub_get(endpoint: str, params: Dict, api_key: str) -> Tuple[int, object]:
    if not api_key:
        return 401, {}
    try:
        p = dict(params or {})
        p["token"] = api_key
        r = requests.get(f"{FINNHUB_BASE}{endpoint}", params=p, timeout=20)
        try:
            js = r.json()
        except Exception:
            js = {}
        return r.status_code, js
    except Exception:
        return 0, {}


@st.cache_data(ttl=6 * 3600)
def finnhub_next_earnings_date(symbol: str, lookahead_days: int, api_key: str) -> Optional[date]:
    """Next earnings announcement date from Finnhub calendar."""
    sym = _clean_ticker(symbol)
    today = date.today()
    to_d = today + timedelta(days=int(lookahead_days))

    status, js = finnhub_get(
        "/calendar/earnings",
        {"from": today.isoformat(), "to": to_d.isoformat(), "symbol": sym},
        api_key=api_key,
    )
    if status != 200 or not isinstance(js, dict):
        return None

    data = js.get("earningsCalendar", [])
    if not isinstance(data, list) or not data:
        return None

    dates: List[date] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if _clean_ticker(item.get("symbol", "")) != sym:
            continue
        d = _to_date(item.get("date"))
        if d is not None and d >= today:
            dates.append(d)

    return min(dates) if dates else None


@st.cache_data(ttl=12 * 3600)
def finnhub_past_earnings(symbol: str, limit: int, api_key: str) -> pd.DataFrame:
    """Past earnings *announcement* events (Finnhub /calendar/earnings).

    NOTE: /stock/earnings returns fiscal period-end ("period"), not announcement date.
    For price reaction we need the announcement date/time -> use /calendar/earnings.
    """
    sym = _clean_ticker(symbol)
    today = date.today()
    from_d = today - timedelta(days=730)

    status, js = finnhub_get(
        "/calendar/earnings",
        {"from": from_d.isoformat(), "to": today.isoformat(), "symbol": sym},
        api_key=api_key,
    )
    if status != 200 or not isinstance(js, dict):
        return pd.DataFrame()

    data = js.get("earningsCalendar", [])
    if not isinstance(data, list) or not data:
        return pd.DataFrame()

    rows: List[Dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if _clean_ticker(item.get("symbol", "")) != sym:
            continue

        d = _to_date(item.get("date"))
        if d is None or d > today:
            continue

        rows.append(
            {
                "Earnings Date": d,
                "Hour": (item.get("hour") or "").strip().lower() or None,  # bmo/amc/dmh (sometimes empty)
                "EPS Actual": _safe_float(item.get("epsActual")),
                "EPS Est.": _safe_float(item.get("epsEstimate")),
                "Surprise": _safe_float(item.get("epsSurprise")),
                "Surprise %": _safe_float(item.get("epsSurprisePercent")),
                "Year": item.get("year"),
                "Quarter": item.get("quarter"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    return df.sort_values("Earnings Date", ascending=False).head(int(limit)).reset_index(drop=True)

# -----------------------------
# yfinance
# -----------------------------
@st.cache_data(ttl=24 * 3600)
def yf_fast_info(ticker: str) -> Dict:
    """Fast info per ticker (market cap etc). Cached heavily."""
    try:
        t = yf.Ticker(ticker)
        fi = getattr(t, "fast_info", None)
        return dict(fi) if fi is not None else {}
    except Exception:
        return {}


@st.cache_data(ttl=6 * 3600)
def yf_history_1y(tickers: Tuple[str, ...]) -> pd.DataFrame:
    """Batch download 1Y daily prices for many tickers."""
    if not tickers:
        return pd.DataFrame()
    try:
        return yf.download(
            list(tickers),
            period="1y",
            interval="1d",
            auto_adjust=False,
            group_by="ticker",
            threads=True,
            progress=False,
        )
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=6 * 3600)
def yf_history_3y_single(ticker: str) -> pd.DataFrame:
    """3Y daily history for one ticker (for reactions around last 4 earnings)."""
    try:
        df = yf.download(
            ticker,
            period="3y",
            interval="1d",
            auto_adjust=False,
            group_by="column",
            progress=False,
            threads=False,
        )
        if isinstance(df, pd.DataFrame) and not df.empty:
            df = df.copy()
            df.index = pd.to_datetime(df.index).date
        return df
    except Exception:
        return pd.DataFrame()


def compute_52w_stats_from_batch(history_1y: pd.DataFrame, ticker: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Returns (current_close, high_52w, low_52w) from yfinance batch download.
    Handles both common MultiIndex layouts:
      - [ticker][field]  (group_by="ticker")
      - [field][ticker]  (some yfinance behaviors)
    """
    if history_1y is None or history_1y.empty:
        return None, None, None

    # Single ticker: columns are single-index
    if isinstance(history_1y.columns, pd.Index):
        close = history_1y.get("Close")
        if close is None or close.dropna().empty:
            return None, None, None
        series = close.dropna()
        return float(series.iloc[-1]), float(series.max()), float(series.min())

    # MultiIndex: try [ticker][field]
    try:
        if ticker in history_1y.columns.get_level_values(0):
            close = history_1y[ticker]["Close"].dropna()
            if not close.empty:
                return float(close.iloc[-1]), float(close.max()), float(close.min())
    except Exception:
        pass

    # MultiIndex: try [field][ticker]
    try:
        if "Close" in history_1y.columns.get_level_values(0) and ticker in history_1y.columns.get_level_values(1):
            close = history_1y["Close"][ticker].dropna()
            if not close.empty:
                return float(close.iloc[-1]), float(close.max()), float(close.min())
    except Exception:
        pass

    return None, None, None


def compute_reaction_from_history(hist: pd.DataFrame, event_date: date, hour: Optional[str] = None) -> Dict:
    """Compute close-to-close reaction around an earnings announcement.

    Uses daily closes from yfinance. Interprets Finnhub `hour`:
      - bmo: before market open -> pre = previous close, post(1D) = same-day close
      - amc: after market close -> pre = same-day close, post(1D) = next close
      - dmh/unknown -> generic: pre = last close strictly before date, post(1D) = first close strictly after date

    3D reaction uses the close ~3 sessions after the announcement:
      post(3D) = 2 trading days after the post(1D) day.
    """
    out = {
        "Pre Close Date": None,
        "Pre Close": None,
        "Post Close (1D) Date": None,
        "Post Close (1D)": None,
        "1D Reaction %": None,
        "Post Close (3D) Date": None,
        "Post Close (3D)": None,
        "3D Reaction %": None,
    }

    if hist is None or hist.empty or "Close" not in hist.columns:
        return out

    trading_days = sorted(set(hist.index))
    if not trading_days:
        return out

    close_map = hist["Close"].to_dict()

    def prev_td(d: date) -> Optional[date]:
        ds = [x for x in trading_days if x < d]
        return ds[-1] if ds else None

    def next_td(d: date) -> Optional[date]:
        ds = [x for x in trading_days if x > d]
        return ds[0] if ds else None

    def same_or_prev_td(d: date) -> Optional[date]:
        if d in trading_days:
            return d
        ds = [x for x in trading_days if x < d]
        return ds[-1] if ds else None

    def same_or_next_td(d: date) -> Optional[date]:
        if d in trading_days:
            return d
        ds = [x for x in trading_days if x > d]
        return ds[0] if ds else None

    h = (hour or "").strip().lower()
    if h in ("am", "morning"):
        h = "bmo"
    if h in ("pm", "after"):
        h = "amc"

    pre_d: Optional[date] = None
    post1_d: Optional[date] = None

    if h == "bmo":
        pre_d = prev_td(event_date)
        post1_d = same_or_next_td(event_date)
    elif h == "amc":
        pre_d = same_or_prev_td(event_date)
        post1_d = next_td(pre_d) if pre_d else None
    else:
        pre_d = prev_td(event_date)
        post1_d = next_td(event_date)

    # fallback if hour-based logic couldn't compute
    if pre_d is None or post1_d is None:
        pre_d = prev_td(event_date)
        post1_d = next_td(event_date)

    if pre_d is None or post1_d is None:
        return out

    out["Pre Close Date"] = pre_d
    out["Pre Close"] = float(close_map.get(pre_d)) if pre_d in close_map else None
    out["Post Close (1D) Date"] = post1_d
    out["Post Close (1D)"] = float(close_map.get(post1_d)) if post1_d in close_map else None

    if out["Pre Close"] and out["Post Close (1D)"]:
        out["1D Reaction %"] = (out["Post Close (1D)"] / out["Pre Close"] - 1.0) * 100.0

    after_post1 = [d for d in trading_days if d > post1_d]
    if len(after_post1) >= 2:
        post3_d = after_post1[1]
        out["Post Close (3D) Date"] = post3_d
        out["Post Close (3D)"] = float(close_map.get(post3_d)) if post3_d in close_map else None
        if out["Pre Close"] and out["Post Close (3D)"]:
            out["3D Reaction %"] = (out["Post Close (3D)"] / out["Pre Close"] - 1.0) * 100.0

    return out

# -----------------------------
# UI
# -----------------------------
st.title("📅 Earnings Radar (Portfolio)")
st.caption(
    "Upload a portfolio (or paste tickers) to see upcoming earnings + context and the stock's recent reaction "
    "around the last 4 earnings. Earnings dates & EPS fields from Finnhub (calendar). Prices from Yahoo Finance (yfinance)."
)

sp500_lookup = load_sp500_universe()

with st.sidebar:
    st.header("Settings")

    # Prefer secrets if available, but allow manual input
    default_key = ""
    try:
        default_key = st.secrets.get("FINNHUB_API_KEY", "")
    except Exception:
        default_key = ""

    api_key = st.text_input("Finnhub API key", value=default_key, type="password")

    lookahead_days = st.slider("Next earnings lookahead (days)", 7, 365, 180)
    max_tickers = st.slider("Max tickers to process", 1, 200, 60)

    only_with_upcoming = st.checkbox("Show only holdings with an upcoming earnings date", value=True)
    within_days = st.slider("Upcoming earnings within (days)", 1, 180, 60)

    st.caption("Tip: fewer tickers = faster. Caching helps a lot after the first run.")

col_u1, col_u2 = st.columns([2, 3], vertical_alignment="top")
with col_u1:
    uploaded = st.file_uploader(
        "Upload holdings file (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        help="Must contain a ticker column (Ticker/Symbol) or have tickers in the first column.",
    )
with col_u2:
    pasted = st.text_input("No file? Paste tickers (comma-separated)", value="")

run = st.button("🚀 Run Earnings Radar", type="primary")

if not run:
    st.stop()

if not api_key:
    st.error("Please provide a Finnhub API key (sidebar).")
    st.stop()

# Collect tickers
tickers: List[str] = []
if uploaded is not None:
    df_holdings = _read_holdings_file(uploaded)
    if not df_holdings.empty:
        tickers.extend([t for t in df_holdings["Ticker"].tolist() if t])

if pasted.strip():
    tickers.extend([_clean_ticker(x) for x in pasted.split(",") if _clean_ticker(x)])

# De-dupe preserving order
seen = set()
tickers = [t for t in tickers if t and (t not in seen and not seen.add(t))]
tickers = tickers[: int(max_tickers)]

if not tickers:
    st.info("Upload a file or paste tickers to begin.")
    st.stop()

st.success(f"Loaded {len(tickers)} unique tickers.")

# -----------------------------
# 2) Portfolio overview
# -----------------------------
st.subheader("2) Portfolio Overview")

rows: List[Dict] = []
progress = st.progress(0, text="Fetching earnings + price stats...")
total = len(tickers)

# Batch 1Y prices for 52W stats
hist_1y = yf_history_1y(tuple(tickers))

for i, t in enumerate(tickers):
    meta = sp500_lookup.get(t, {})
    company = meta.get("Company")
    sector = meta.get("Sector")
    industry = meta.get("Industry")

    # Next earnings
    next_e = finnhub_next_earnings_date(t, lookahead_days=lookahead_days, api_key=api_key)
    days_to = (next_e - date.today()).days if next_e else None

    # 52W stats
    current, high52, low52 = compute_52w_stats_from_batch(hist_1y, t)

    # Market cap
    fi = yf_fast_info(t)
    mcap = _safe_float(fi.get("market_cap")) or _safe_float(fi.get("marketCap"))

    # Fallback company name (slower)
    if not company:
        try:
            info = yf.Ticker(t).get_info()
            company = info.get("shortName") or info.get("longName")
        except Exception:
            pass

    d_vs_high = (current - high52) / high52 * 100.0 if current is not None and high52 not in (None, 0) else None
    d_vs_low = (current - low52) / low52 * 100.0 if current is not None and low52 not in (None, 0) else None
    r52 = (high52 - low52) / low52 * 100.0 if high52 is not None and low52 not in (None, 0) else None

    rows.append(
        {
            "Ticker": t,
            "Company": company,
            "Sector": sector,
            "Industry": industry,
            "Market Cap": mcap,
            "Next Earnings": next_e,
            "Days to Earnings": days_to,
            "Current": current,
            "52W High": high52,
            "52W Low": low52,
            "Δ vs 52W High (%)": d_vs_high,
            "Δ vs 52W Low (%)": d_vs_low,
            "52W Range (%)": r52,
        }
    )

    progress.progress((i + 1) / total, text=f"Fetched {i+1}/{total}: {t}")

progress.empty()

overview = pd.DataFrame(rows)

if only_with_upcoming:
    overview = overview[overview["Next Earnings"].notna()].copy()

if not overview.empty:
    overview = overview[overview["Days to Earnings"].notna()].copy()
    overview = overview[(overview["Days to Earnings"] >= 0) & (overview["Days to Earnings"] <= within_days)].copy()

if not overview.empty:
    overview = overview.sort_values(["Days to Earnings", "Ticker"], ascending=[True, True])

if overview.empty:
    st.warning("No tickers matched the current filters (try increasing 'Upcoming earnings within').")
else:
    st.dataframe(
        overview,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Next Earnings": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "Current": st.column_config.NumberColumn(format="%.2f"),
            "52W High": st.column_config.NumberColumn(format="%.2f"),
            "52W Low": st.column_config.NumberColumn(format="%.2f"),
            "Δ vs 52W High (%)": st.column_config.NumberColumn(format="%.2f"),
            "Δ vs 52W Low (%)": st.column_config.NumberColumn(format="%.2f"),
            "52W Range (%)": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    view = overview.copy()
    if "Market Cap" in view.columns:
        view["Market Cap"] = view["Market Cap"].apply(_fmt_money)
    csv_bytes = view.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Portfolio Overview (CSV)",
        data=csv_bytes,
        file_name="earnings_radar_portfolio_overview.csv",
        mime="text/csv",
    )

# -----------------------------
# 3) Past 4 earnings + price reaction
# -----------------------------
st.subheader("3) Past 4 Earnings + Price Reaction (per holding)")
st.caption(
    "Earnings events from Finnhub calendar (includes timing: bmo/amc when available). "
    "Price reaction uses daily closes from Yahoo Finance (yfinance)."
)

for t in tickers:
    company = None
    try:
        company = next((r.get("Company") for r in rows if r.get("Ticker") == t), None)
    except Exception:
        company = None

    title = f"{t}" + (f" — {company}" if company else "")
    with st.expander(title, expanded=False):
        earn_df = finnhub_past_earnings(t, limit=4, api_key=api_key)
        if earn_df is None or earn_df.empty:
            st.info("No recent earnings events returned for this ticker (common for ETFs/funds or incomplete listings).")
            continue

        hist3y = yf_history_3y_single(t)
        if hist3y is None or hist3y.empty:
            st.warning("Could not fetch 3Y daily price history from Yahoo Finance for reactions.")
            st.dataframe(earn_df, use_container_width=True, hide_index=True)
            continue

        enriched: List[Dict] = []
        for _, row in earn_df.iterrows():
            ed = _to_date(row.get("Earnings Date"))
            base = dict(row)
            if ed is None:
                enriched.append(base)
                continue

            reaction = compute_reaction_from_history(hist3y, ed, hour=row.get("Hour"))
            base.update(reaction)
            enriched.append(base)

        out_df = pd.DataFrame(enriched)

        keep_cols = [
            "Earnings Date",
            "Hour",
            "Year",
            "Quarter",
            "EPS Actual",
            "EPS Est.",
            "Surprise",
            "Surprise %",
            "Pre Close Date",
            "Pre Close",
            "Post Close (1D) Date",
            "Post Close (1D)",
            "1D Reaction %",
            "Post Close (3D) Date",
            "Post Close (3D)",
            "3D Reaction %",
        ]
        keep_cols = [c for c in keep_cols if c in out_df.columns]
        out_df = out_df[keep_cols].copy()

        st.dataframe(
            out_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Earnings Date": st.column_config.DateColumn(format="YYYY-MM-DD"),
                "Pre Close Date": st.column_config.DateColumn(format="YYYY-MM-DD"),
                "Post Close (1D) Date": st.column_config.DateColumn(format="YYYY-MM-DD"),
                "Post Close (3D) Date": st.column_config.DateColumn(format="YYYY-MM-DD"),
                "Pre Close": st.column_config.NumberColumn(format="%.2f"),
                "Post Close (1D)": st.column_config.NumberColumn(format="%.2f"),
                "Post Close (3D)": st.column_config.NumberColumn(format="%.2f"),
                "1D Reaction %": st.column_config.NumberColumn(format="%.2f"),
                "3D Reaction %": st.column_config.NumberColumn(format="%.2f"),
                "EPS Actual": st.column_config.NumberColumn(format="%.3f"),
                "EPS Est.": st.column_config.NumberColumn(format="%.3f"),
                "Surprise": st.column_config.NumberColumn(format="%.3f"),
                "Surprise %": st.column_config.NumberColumn(format="%.2f"),
            },
        )
