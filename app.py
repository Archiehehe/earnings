import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import altair as alt

# IMPORTANT: curl_cffi improves Yahoo endpoints reliability (esp earnings dates)
from curl_cffi import requests as c_requests

# -----------------------------------------------------------------------------
# Page
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Earnings Radar", page_icon="📡", layout="wide")
alt.data_transformers.disable_max_rows()

st.title("📡 Earnings Radar")
st.caption(
    "Portfolio-first earnings monitoring. Upload holdings to get next earnings + price context, "
    "and review last 4 earnings reactions. (ETFs usually have no earnings.)"
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def _fmt_num(x: Any, d: int = 2) -> str:
    v = _safe_float(x)
    if v is None:
        return "—"
    return f"{v:,.{d}f}"


def _fmt_pct(x: Any, d: int = 1) -> str:
    v = _safe_float(x)
    if v is None:
        return "—"
    return f"{v:+.{d}f}%"


def chunk_list(xs: List[str], n: int) -> List[List[str]]:
    return [xs[i: i + n] for i in range(0, len(xs), n)]


def today_utc_date() -> datetime:
    return datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


def _normalize_ticker(t: str) -> str:
    return str(t).upper().strip().replace(".", "-")


def read_portfolio_file(uploaded) -> pd.DataFrame:
    name = (uploaded.name or "").lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded)  # requires openpyxl
    raise ValueError("Unsupported file type. Please upload a .csv, .xlsx, or .xls file.")


def find_ticker_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["Ticker", "Symbol", "Ticker Symbol", "Trading Symbol", "Security", "Instrument"]
    cols = list(df.columns)

    for c in candidates:
        if c in cols:
            return c

    lower_map = {str(c).strip().lower(): c for c in cols}
    for c in candidates:
        key = str(c).strip().lower()
        if key in lower_map:
            return lower_map[key]

    return None


# -----------------------------------------------------------------------------
# Sessions (curl_cffi)
# -----------------------------------------------------------------------------
@st.cache_resource
def get_impersonated_session() -> c_requests.Session:
    # Chrome impersonation helps avoid Yahoo blocking / partial responses
    return c_requests.Session(impersonate="chrome")


def get_ticker_obj(ticker: str) -> yf.Ticker:
    # yfinance supports passing a requests-like session
    return yf.Ticker(ticker, session=get_impersonated_session())


# -----------------------------------------------------------------------------
# Universe (optional tagging)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=24 * 3600)
def load_universe_csv(path: str = "sp500_universe.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Ticker", "Company", "Sector", "Industry"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Universe CSV missing columns: {missing}. Required: {sorted(required)}")

    df = df.copy()
    df["Ticker"] = df["Ticker"].astype(str).apply(_normalize_ticker)
    df["Company"] = df["Company"].fillna("").astype(str).str.strip()
    df["Sector"] = df["Sector"].fillna("").astype(str).str.strip()
    df["Industry"] = df["Industry"].fillna("").astype(str).str.strip()
    df = df.dropna(subset=["Ticker"]).drop_duplicates(subset=["Ticker"]).reset_index(drop=True)
    return df


try:
    universe = load_universe_csv("sp500_universe.csv")
    meta = universe[["Ticker", "Company", "Sector", "Industry"]].drop_duplicates(subset=["Ticker"]).set_index("Ticker")
except Exception:
    # Universe is optional; app still works without it
    meta = pd.DataFrame(columns=["Company", "Sector", "Industry"]).set_index(pd.Index([], name="Ticker"))

# -----------------------------------------------------------------------------
# Price metrics (batched) via yf.download (fast + reliable)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=6 * 3600)
def fetch_1y_price_metrics(tickers: List[str]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    results = []
    for batch in chunk_list(tickers, 120):
        try:
            data = yf.download(
                tickers=batch,
                period="1y",
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
            )
        except Exception:
            data = pd.DataFrame()

        if data is None or data.empty:
            for tk in batch:
                results.append({"Ticker": tk})
            continue

        if isinstance(data.columns, pd.MultiIndex):
            for tk in batch:
                if tk not in data.columns.get_level_values(0):
                    results.append({"Ticker": tk})
                    continue
                closes = data[(tk, "Close")].dropna()
                highs = data[(tk, "High")].dropna()
                lows = data[(tk, "Low")].dropna()

                cur = _safe_float(closes.iloc[-1]) if not closes.empty else None
                hi = _safe_float(highs.max()) if not highs.empty else None
                lo = _safe_float(lows.min()) if not lows.empty else None

                d_hi = (cur - hi) / hi * 100.0 if (cur is not None and hi not in (None, 0)) else None
                d_lo = (cur - lo) / lo * 100.0 if (cur is not None and lo not in (None, 0)) else None
                rng = (hi - lo) / lo * 100.0 if (hi is not None and lo not in (None, 0)) else None

                results.append(
                    {
                        "Ticker": tk,
                        "Current": cur,
                        "52W High": hi,
                        "52W Low": lo,
                        "Δ vs 52W High (%)": d_hi,
                        "Δ vs 52W Low (%)": d_lo,
                        "52W Range (%)": rng,
                    }
                )
        else:
            tk = batch[0]
            closes = data["Close"].dropna()
            highs = data["High"].dropna()
            lows = data["Low"].dropna()

            cur = _safe_float(closes.iloc[-1]) if not closes.empty else None
            hi = _safe_float(highs.max()) if not highs.empty else None
            lo = _safe_float(lows.min()) if not lows.empty else None

            d_hi = (cur - hi) / hi * 100.0 if (cur is not None and hi not in (None, 0)) else None
            d_lo = (cur - lo) / lo * 100.0 if (cur is not None and lo not in (None, 0)) else None
            rng = (hi - lo) / lo * 100.0 if (hi is not None and lo not in (None, 0)) else None

            results.append(
                {
                    "Ticker": tk,
                    "Current": cur,
                    "52W High": hi,
                    "52W Low": lo,
                    "Δ vs 52W High (%)": d_hi,
                    "Δ vs 52W Low (%)": d_lo,
                    "52W Range (%)": rng,
                }
            )

    return pd.DataFrame(results).drop_duplicates(subset=["Ticker"])


# -----------------------------------------------------------------------------
# Earnings: robust next earnings + history
# -----------------------------------------------------------------------------
def _is_etf_or_fund(info: Dict[str, Any]) -> bool:
    qt = str(info.get("quoteType", "")).upper()
    return qt in {"ETF", "MUTUALFUND", "FUND", "INDEX"}


def _pick_next_from_info(info: Dict[str, Any]) -> Optional[pd.Timestamp]:
    """
    Yahoo sometimes provides:
      earningsTimestampStart / earningsTimestampEnd / earningsTimestamp
    We try these if get_earnings_dates doesn't include future events.
    """
    candidates = []
    for k in ["earningsTimestampStart", "earningsTimestamp", "earningsTimestampEnd"]:
        v = info.get(k)
        if v is None:
            continue
        try:
            ts = pd.to_datetime(int(v), unit="s", utc=True, errors="coerce")
            if pd.notna(ts):
                candidates.append(ts)
        except Exception:
            pass

    candidates = [c for c in candidates if c >= pd.Timestamp(today_utc_date())]
    if candidates:
        return pd.Timestamp(min(candidates))
    return None


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def fetch_earnings_dates_df(ticker: str, limit: int = 16) -> pd.DataFrame:
    """
    Returns earnings dates table from yfinance (may be empty depending on Yahoo).
    Uses curl_cffi session for better reliability.
    """
    try:
        t = get_ticker_obj(ticker)
        df = t.get_earnings_dates(limit=limit)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df[~df.index.isna()].sort_index()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def fetch_info_minimal(ticker: str) -> Dict[str, Any]:
    try:
        t = get_ticker_obj(ticker)
        info = t.info or {}
        return info
    except Exception:
        return {}


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def fetch_next_earnings_date(ticker: str) -> Optional[pd.Timestamp]:
    # Step 1: try earnings dates table (future + past)
    df = fetch_earnings_dates_df(ticker, limit=20)
    now = pd.Timestamp(today_utc_date())

    if df is not None and not df.empty:
        idx = pd.to_datetime(df.index, utc=True, errors="coerce").dropna()
        future = idx[idx >= now]
        if len(future) > 0:
            return pd.Timestamp(future.min())

    # Step 2: try info timestamps (future earnings)
    info = fetch_info_minimal(ticker)
    if info and _is_etf_or_fund(info):
        return None

    ts = _pick_next_from_info(info) if info else None
    if ts is not None:
        return ts

    # Step 3: older-style info key 'earningsDate' (sometimes list)
    if info:
        ed = info.get("earningsDate")
        try:
            if isinstance(ed, list) and len(ed) > 0:
                parsed = [pd.to_datetime(x, utc=True, errors="coerce") for x in ed]
                parsed = [p for p in parsed if pd.notna(p)]
                parsed = [p for p in parsed if p >= now]
                if parsed:
                    return pd.Timestamp(min(parsed))
            else:
                p = pd.to_datetime(ed, utc=True, errors="coerce")
                if pd.notna(p) and p >= now:
                    return pd.Timestamp(p)
        except Exception:
            pass

    return None


def fetch_earnings_for_list(tickers: List[str]) -> pd.DataFrame:
    bar = st.progress(0.0)
    rows = []
    n = len(tickers)
    for i, tk in enumerate(tickers, start=1):
        rows.append({"Ticker": tk, "Next Earnings (UTC)": fetch_next_earnings_date(tk)})
        bar.progress(i / max(1, n))
    bar.empty()
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Past 4 earnings + price context
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=12 * 3600)
def fetch_price_history_for_events(ticker: str, period: str = "3y") -> pd.DataFrame:
    try:
        df = yf.download(
            tickers=ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df[~df.index.isna()]
        return df
    except Exception:
        return pd.DataFrame()


def _event_context_from_history(hist: pd.DataFrame, event_ts: pd.Timestamp) -> Optional[Dict[str, Any]]:
    if hist is None or hist.empty or "Close" not in hist.columns:
        return None

    close = hist["Close"].dropna()
    if close.empty:
        return None

    event_day = pd.to_datetime(event_ts, utc=True, errors="coerce")
    if pd.isna(event_day):
        return None
    event_day = event_day.normalize()

    idx_norm = close.index.normalize()
    pos = np.where(idx_norm <= event_day)[0]
    if len(pos) == 0:
        return None
    p0 = int(pos[-1])

    prev_close = _safe_float(close.iloc[p0])
    next1 = _safe_float(close.iloc[p0 + 1]) if p0 + 1 < len(close) else None
    next5 = _safe_float(close.iloc[p0 + 5]) if p0 + 5 < len(close) else None

    move_1d = (next1 / prev_close - 1.0) * 100.0 if (prev_close not in (None, 0) and next1 is not None) else None
    move_5d = (next5 / prev_close - 1.0) * 100.0 if (prev_close not in (None, 0) and next5 is not None) else None

    roll_hi = close.rolling(window=252, min_periods=20).max()
    roll_lo = close.rolling(window=252, min_periods=20).min()

    hi = _safe_float(roll_hi.iloc[p0])
    lo = _safe_float(roll_lo.iloc[p0])

    dist_hi = (prev_close - hi) / hi * 100.0 if (prev_close is not None and hi not in (None, 0)) else None
    dist_lo = (prev_close - lo) / lo * 100.0 if (prev_close is not None and lo not in (None, 0)) else None

    return {
        "Event Close (prev)": prev_close,
        "1D Move (%)": move_1d,
        "5D Move (%)": move_5d,
        "Dist vs 52W High (%)": dist_hi,
        "Dist vs 52W Low (%)": dist_lo,
    }


def last4_earnings_with_context(ticker: str) -> pd.DataFrame:
    df = fetch_earnings_dates_df(ticker, limit=20)
    if df is None or df.empty:
        return pd.DataFrame()

    now = pd.Timestamp(today_utc_date())
    past = df[df.index < now].sort_index()
    if past.empty:
        return pd.DataFrame()

    last4 = past.tail(4)
    hist = fetch_price_history_for_events(ticker, period="3y")

    rows = []
    for ts, row in last4.iterrows():
        ctx = _event_context_from_history(hist, ts) if hist is not None and not hist.empty else None
        base = {"Ticker": ticker, "Earnings Date (UTC)": ts.strftime("%Y-%m-%d")}

        for col in ["EPS Estimate", "Reported EPS", "Surprise(%)"]:
            base[col] = _safe_float(row.get(col)) if col in last4.columns else None

        if ctx:
            base.update(ctx)

        rows.append(base)

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# UI — Portfolio-first
# -----------------------------------------------------------------------------
st.subheader("📤 Upload Portfolio (Primary)")
st.markdown("Upload a **CSV or Excel** file containing a ticker column (e.g. `Ticker` or `Symbol`).")

portfolio_file = st.file_uploader("Portfolio file", type=["csv", "xlsx", "xls"])

colA, colB, colC = st.columns([1, 1, 2])
with colA:
    pf_max_names = st.slider("Max tickers to process (speed)", 5, 150, 40, 5)
with colB:
    pf_only_next = st.checkbox("Only tickers with next earnings", value=False)
with colC:
    debug_mode = st.checkbox("Debug mode (show raw earnings fetch for one ticker)", value=False)

pf_run = st.button("Run Portfolio Earnings Radar", type="primary")

with st.popover("ℹ️ File format help"):
    st.markdown(
        """
Accepted files:
- `.csv`
- `.xlsx` / `.xls`

Ticker column examples:
`Ticker`, `Symbol`, `Ticker Symbol`

Notes:
- ETFs (QQQ/VOO/SOXX etc.) usually have no earnings.
- Yahoo can temporarily block earnings endpoints; we use a browser-impersonated session to improve reliability.
        """.strip()
    )

if pf_run:
    if portfolio_file is None:
        st.error("Upload a portfolio file first.")
        st.stop()

    try:
        pf = read_portfolio_file(portfolio_file)
    except Exception as e:
        st.error(f"Could not read portfolio file: {e}")
        st.stop()

    ticker_col = find_ticker_column(pf)
    if ticker_col is None:
        st.error("Could not find a ticker column. Rename your column to `Ticker` or `Symbol` and try again.")
        st.stop()

    pf_tickers = pf[ticker_col].astype(str).apply(_normalize_ticker).dropna().tolist()
    pf_tickers = [t for t in pf_tickers if t]
    pf_tickers = list(dict.fromkeys(pf_tickers))

    if not pf_tickers:
        st.error("No valid tickers found in your file.")
        st.stop()

    if len(pf_tickers) > pf_max_names:
        st.warning(f"Portfolio has {len(pf_tickers)} tickers — processing first {pf_max_names} for speed.")
        pf_tickers = pf_tickers[:pf_max_names]

    st.divider()
    st.subheader("📦 Portfolio Overview")

    with st.spinner("Fetching 1Y price metrics..."):
        pm = fetch_1y_price_metrics(pf_tickers)

    with st.spinner("Fetching next earnings dates (cached)..."):
        ed = fetch_earnings_for_list(pf_tickers)

    ov = pm.merge(ed, on="Ticker", how="left")
    ov["Next Earnings (UTC)"] = pd.to_datetime(ov["Next Earnings (UTC)"], utc=True, errors="coerce")

    def _meta_get(t: str, col: str) -> str:
        try:
            return str(meta.loc[t, col]) if t in meta.index else ""
        except Exception:
            return ""

    ov["Company"] = ov["Ticker"].apply(lambda t: _meta_get(t, "Company"))
    ov["Sector"] = ov["Ticker"].apply(lambda t: _meta_get(t, "Sector"))
    ov["Industry"] = ov["Ticker"].apply(lambda t: _meta_get(t, "Industry"))

    if pf_only_next:
        ov = ov[ov["Next Earnings (UTC)"].notna()].reset_index(drop=True)

    disp = ov.copy()
    disp["Next Earnings (UTC)"] = disp["Next Earnings (UTC)"].dt.strftime("%Y-%m-%d")
    disp["Current"] = disp["Current"].apply(lambda x: _fmt_num(x, 2))
    disp["52W High"] = disp["52W High"].apply(lambda x: _fmt_num(x, 2))
    disp["52W Low"] = disp["52W Low"].apply(lambda x: _fmt_num(x, 2))
    for c in ["Δ vs 52W High (%)", "Δ vs 52W Low (%)", "52W Range (%)"]:
        disp[c] = disp[c].apply(lambda x: _fmt_pct(x, 1))

    disp = disp.sort_values("Next Earnings (UTC)", ascending=True, na_position="last")

    st.dataframe(
        disp[
            [
                "Ticker",
                "Company",
                "Sector",
                "Industry",
                "Next Earnings (UTC)",
                "Current",
                "Δ vs 52W High (%)",
                "Δ vs 52W Low (%)",
                "52W Range (%)",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇️ Download Portfolio Overview (CSV)",
        ov.to_csv(index=False),
        file_name="portfolio_earnings_overview.csv",
        mime="text/csv",
    )

    # Optional debug for one ticker
    if debug_mode and pf_tickers:
        st.divider()
        st.subheader("🛠 Debug: Earnings fetch (first ticker)")
        tk0 = pf_tickers[0]
        st.write("Ticker:", tk0)
        st.write("Next earnings:", fetch_next_earnings_date(tk0))
        raw = fetch_earnings_dates_df(tk0, limit=20)
        st.write("Raw earnings_dates table (if any):")
        st.dataframe(raw.reset_index().rename(columns={"index": "Earnings Date"}), use_container_width=True)

    st.divider()
    st.subheader("🧾 Past 4 Earnings + Price Reaction (per holding)")

    for tk in pf_tickers:
        company = _meta_get(tk, "Company")
        title = f"{tk} — {company}" if company else tk

        with st.expander(title, expanded=False):
            # ETFs: show a friendly message
            info = fetch_info_minimal(tk)
            if info and _is_etf_or_fund(info):
                st.info("This looks like an ETF/fund (no earnings events).")
                continue

            out = last4_earnings_with_context(tk)
            if out.empty:
                st.info("No past earnings events returned by Yahoo for this ticker (endpoint can be flaky).")
                continue

            disp2 = out.copy()

            for col in ["EPS Estimate", "Reported EPS", "Event Close (prev)"]:
                if col in disp2.columns:
                    disp2[col] = disp2[col].apply(lambda x: _fmt_num(x, 2))

            for col in ["Surprise(%)", "1D Move (%)", "5D Move (%)", "Dist vs 52W High (%)", "Dist vs 52W Low (%)"]:
                if col in disp2.columns:
                    disp2[col] = disp2[col].apply(lambda x: _fmt_pct(x, 1))

            order = [
                "Earnings Date (UTC)",
                "EPS Estimate",
                "Reported EPS",
                "Surprise(%)",
                "Event Close (prev)",
                "1D Move (%)",
                "5D Move (%)",
                "Dist vs 52W High (%)",
                "Dist vs 52W Low (%)",
            ]
            order = [c for c in order if c in disp2.columns]
            st.dataframe(disp2[order], use_container_width=True, hide_index=True)
