import os
import io
import time
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

import yfinance as yf
import finnhub


# -----------------------------
# Page
# -----------------------------
st.set_page_config(
    page_title="Earnings Radar (Portfolio)",
    page_icon="📅",
    layout="wide",
)

st.title("📅 Earnings Radar (Portfolio)")
st.caption(
    "Upload your portfolio, see upcoming earnings dates, and review the past 4 earnings with simple 1D / 3D price reaction."
)


# -----------------------------
# Helpers
# -----------------------------
def _safe_upper(x: str) -> str:
    if x is None:
        return ""
    return str(x).strip().upper()


def _fmt_date(d):
    if d is None or (isinstance(d, float) and np.isnan(d)):
        return None
    if isinstance(d, (datetime, pd.Timestamp)):
        return d.date().isoformat()
    if isinstance(d, date):
        return d.isoformat()
    try:
        return date.fromisoformat(str(d)[:10]).isoformat()
    except Exception:
        return str(d)


def _to_date(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, (datetime, pd.Timestamp)):
        return x.date()
    s = str(x)[:10]
    try:
        return date.fromisoformat(s)
    except Exception:
        return None


def _human_money(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    try:
        x = float(x)
    except Exception:
        return None
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e12:
        return f"{sign}{x/1e12:.2f}T"
    if x >= 1e9:
        return f"{sign}{x/1e9:.2f}B"
    if x >= 1e6:
        return f"{sign}{x/1e6:.2f}M"
    if x >= 1e3:
        return f"{sign}{x/1e3:.2f}K"
    return f"{sign}{x:.0f}"


def _pick_ticker_column(df: pd.DataFrame) -> str | None:
    # Try common names first
    candidates = [
        "ticker", "tickers", "symbol", "symbols", "stock", "security", "instrument", "code"
    ]
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in cols:
            return cols[c]
    # Fallback: if any column has lots of uppercase-ish short strings
    best = None
    best_score = 0
    for c in df.columns:
        s = df[c].astype(str).str.strip()
        # score: how many look like tickers
        score = ((s.str.len() >= 1) & (s.str.len() <= 8) & (s.str.match(r"^[A-Za-z\.\-]+$"))).sum()
        if score > best_score:
            best_score = score
            best = c
    return best


@st.cache_data(show_spinner=False)
def load_sp500_universe_map(path="sp500_universe.csv"):
    """
    Optional: if you keep sp500_universe.csv in the repo, we can map Sector/Industry/Company quickly.
    Expected columns often include:
    - Symbol
    - Security
    - GICS Sector
    - GICS Sub-Industry
    """
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    # normalize common column names
    col_symbol = None
    for c in df.columns:
        if c.lower() in ["symbol", "ticker"]:
            col_symbol = c
            break
    if col_symbol is None:
        return None

    df[col_symbol] = df[col_symbol].astype(str).str.upper().str.strip()
    col_company = None
    col_sector = None
    col_industry = None

    for c in df.columns:
        cl = c.lower()
        if col_company is None and cl in ["security", "company", "company name", "name"]:
            col_company = c
        if col_sector is None and ("sector" in cl):
            # prefer "gics sector" if present
            if "gics" in cl:
                col_sector = c
            elif col_sector is None:
                col_sector = c
        if col_industry is None and ("industry" in cl or "sub-industry" in cl):
            if "sub" in cl or "gics" in cl:
                col_industry = c
            elif col_industry is None:
                col_industry = c

    out = df[[col_symbol] + [c for c in [col_company, col_sector, col_industry] if c is not None]].copy()
    out = out.drop_duplicates(subset=[col_symbol])
    out = out.set_index(col_symbol)
    return out


@st.cache_data(show_spinner=False)
def read_portfolio_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    # fallback
    return pd.read_csv(uploaded_file)


@st.cache_data(show_spinner=False)
def yf_download_daily(tickers: tuple[str, ...], days: int = 730) -> dict[str, pd.DataFrame]:
    """
    Batch daily candles for all tickers (2 years default).
    Returns dict ticker -> OHLCV dataframe with DatetimeIndex.
    """
    if len(tickers) == 0:
        return {}

    period = f"{days}d"
    # yf.download returns:
    # - single ticker: columns = [Open, High, Low, Close, Adj Close, Volume]
    # - multi ticker: columns MultiIndex: (field, ticker) or (ticker, field) depending on group_by
    raw = yf.download(
        list(tickers),
        period=period,
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    out = {}
    if isinstance(raw.columns, pd.MultiIndex):
        # group_by="ticker" => top level should be ticker
        # but sometimes yfinance flips; handle both
        top = raw.columns.get_level_values(0)
        if any(t in top for t in tickers):
            # (ticker, field)
            for t in tickers:
                if t in raw.columns.get_level_values(0):
                    df_t = raw[t].dropna(how="all")
                    out[t] = df_t
        else:
            # (field, ticker)
            for t in tickers:
                if t in raw.columns.get_level_values(1):
                    df_t = raw.xs(t, level=1, axis=1).dropna(how="all")
                    out[t] = df_t
    else:
        # single ticker
        t = tickers[0]
        out[t] = raw.dropna(how="all")

    return out


def compute_current_52w(df: pd.DataFrame):
    """
    From OHLCV daily df, compute current close, 52W high, 52W low (last ~252 trading days).
    """
    if df is None or df.empty:
        return None, None, None

    df2 = df.dropna(subset=["Close"], how="any").copy()
    if df2.empty:
        return None, None, None

    tail = df2.tail(252) if len(df2) >= 10 else df2

    current = float(df2["Close"].iloc[-1]) if "Close" in df2.columns else None

    high52 = None
    low52 = None
    if "High" in tail.columns:
        try:
            high52 = float(tail["High"].max())
        except Exception:
            high52 = None
    if "Low" in tail.columns:
        try:
            low52 = float(tail["Low"].min())
        except Exception:
            low52 = None

    return current, high52, low52


def compute_reaction_from_prices(df: pd.DataFrame, event_dt: date):
    """
    Compute:
      pre_close: last close strictly before event_dt
      post_close_1d: first close strictly after event_dt
      post_close_3d: third trading close strictly after event_dt
    """
    if df is None or df.empty or event_dt is None:
        return {}

    dfi = df.dropna(subset=["Close"]).copy()
    if dfi.empty:
        return {}

    # Ensure date index
    idx_dates = pd.to_datetime(dfi.index).date

    # pre = last trading day strictly before event date
    mask_pre = np.array([d < event_dt for d in idx_dates])
    if not mask_pre.any():
        return {}
    pre_i = np.where(mask_pre)[0][-1]
    pre_date = idx_dates[pre_i]
    pre_close = float(dfi["Close"].iloc[pre_i])

    # post candidates strictly after event date
    post_idx = np.where(np.array([d > event_dt for d in idx_dates]))[0]
    if len(post_idx) == 0:
        return {
            "Pre Close Date": pre_date,
            "Pre Close": pre_close,
        }

    post1_i = post_idx[0]
    post1_date = idx_dates[post1_i]
    post1_close = float(dfi["Close"].iloc[post1_i])

    res = {
        "Pre Close Date": pre_date,
        "Pre Close": pre_close,
        "Post Close (1D) Date": post1_date,
        "Post Close (1D)": post1_close,
        "1D Reaction %": (post1_close / pre_close - 1.0) * 100.0 if pre_close else None,
    }

    if len(post_idx) >= 3:
        post3_i = post_idx[2]
        post3_date = idx_dates[post3_i]
        post3_close = float(dfi["Close"].iloc[post3_i])
        res.update(
            {
                "Post Close (3D) Date": post3_date,
                "Post Close (3D)": post3_close,
                "3D Reaction %": (post3_close / pre_close - 1.0) * 100.0 if pre_close else None,
            }
        )

    return res


@st.cache_data(show_spinner=False)
def yf_fast_profile(ticker: str):
    """
    Lightweight profile/labels + market cap via yfinance fast_info when possible.
    """
    t = yf.Ticker(ticker)
    info = {}
    # fast_info is quicker but not always complete
    try:
        fi = getattr(t, "fast_info", None)
        if fi:
            info["marketCap"] = fi.get("market_cap")
            info["lastPrice"] = fi.get("last_price")
    except Exception:
        pass

    # sector/industry/name are in .info (heavier). Only do if missing.
    try:
        ii = t.info or {}
        info["shortName"] = ii.get("shortName") or ii.get("longName")
        info["sector"] = ii.get("sector")
        info["industry"] = ii.get("industry")
    except Exception:
        pass

    return info


@st.cache_data(show_spinner=False)
def finnhub_calendar_for_symbol(symbol: str, frm: str, to: str, api_key: str):
    client = finnhub.Client(api_key=api_key)
    try:
        return client.earnings_calendar(_from=frm, to=to, symbol=symbol)
    except Exception:
        return {"earningsCalendar": []}


def extract_next_and_past_from_calendar(cal_list: list[dict], today: date, past_n: int = 4):
    """
    cal_list entries typically include fields like:
    date, symbol, epsActual, epsEstimate, surprise, surprisePercent, quarter, year, hour, revenueActual, revenueEstimate...
    """
    events = []
    for e in cal_list or []:
        d = _to_date(e.get("date"))
        if d is None:
            continue
        events.append({**e, "_date": d})

    events.sort(key=lambda x: x["_date"])

    next_dt = None
    for e in events:
        if e["_date"] >= today:
            next_dt = e["_date"]
            break

    past = [e for e in events if e["_date"] < today]
    past.sort(key=lambda x: x["_date"], reverse=True)
    past = past[:past_n]

    return next_dt, past


# -----------------------------
# Secrets / API Key
# -----------------------------
API_KEY = st.secrets.get("FINNHUB_API_KEY", None)

if API_KEY is None or str(API_KEY).strip() == "":
    st.error("Missing FINNHUB_API_KEY in Streamlit Secrets.")
    st.stop()


# -----------------------------
# Sidebar settings
# -----------------------------
with st.sidebar:
    st.header("Settings")

    lookahead_days = st.slider("Next earnings lookahead (days)", 30, 365, 180, 10)
    max_tickers = st.slider("Max tickers to process", 10, 200, 60, 5)

    st.markdown("---")
    only_upcoming = st.checkbox("Show only holdings with an upcoming earnings date", value=True)
    upcoming_within = st.slider("Upcoming earnings within (days)", 7, 180, 60, 1)

    st.markdown("---")
    st.caption("Tip: fewer tickers = faster. Caching helps a lot after the first run.")


# -----------------------------
# Upload section (VERY visible)
# -----------------------------
st.subheader("1) Upload your portfolio (CSV / Excel)")

colA, colB = st.columns([2, 2])
with colA:
    uploaded = st.file_uploader(
        "Upload a portfolio file (.csv, .xlsx). We’ll try to find a ticker/symbol column automatically.",
        type=["csv", "xlsx", "xls"],
        label_visibility="visible",
    )

with colB:
    manual = st.text_input("No file? Paste tickers (comma-separated):", value="", placeholder="AAPL, MSFT, NVDA")


def get_tickers_from_inputs():
    tickers = []

    if uploaded is not None:
        df = read_portfolio_file(uploaded)
        col = _pick_ticker_column(df)
        if col is None:
            st.error("Could not detect a ticker/symbol column in the uploaded file.")
            return []
        raw = df[col].dropna().astype(str).tolist()
        tickers += raw

    if manual.strip():
        tickers += [t.strip() for t in manual.split(",")]

    tickers = [_safe_upper(t) for t in tickers if _safe_upper(t)]
    # remove obvious junk
    tickers = [t for t in tickers if t not in {"NAN", "NONE"}]
    # unique preserve order
    seen = set()
    uniq = []
    for t in tickers:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq


tickers_all = get_tickers_from_inputs()
if uploaded is not None or manual.strip():
    st.success(f"Loaded {len(tickers_all)} unique tickers.")
else:
    st.info("Upload a file or paste tickers to begin.")


run = st.button("🚀 Run Earnings Radar", type="primary", use_container_width=False)
if not run:
    st.stop()

if len(tickers_all) == 0:
    st.stop()

tickers = tickers_all[:max_tickers]
today = date.today()
frm = (today - timedelta(days=730)).isoformat()
to = (today + timedelta(days=lookahead_days)).isoformat()

# Optional SP500 map (if file exists in repo)
sp500_map = load_sp500_universe_map("sp500_universe.csv")

# -----------------------------
# 2) Calendar + prices (batch)
# -----------------------------
with st.spinner("Fetching earnings calendars (Finnhub) ..."):
    next_map = {}
    past_map = {}

    # Basic throttling to be nice (Finnhub can rate-limit)
    for i, tkr in enumerate(tickers):
        cal = finnhub_calendar_for_symbol(tkr, frm=frm, to=to, api_key=API_KEY)
        cal_list = cal.get("earningsCalendar", []) if isinstance(cal, dict) else []
        nxt, past = extract_next_and_past_from_calendar(cal_list, today=today, past_n=4)
        next_map[tkr] = nxt
        past_map[tkr] = past

        if i % 15 == 0 and i > 0:
            time.sleep(0.25)

with st.spinner("Fetching price history (Yahoo / yfinance) ..."):
    price_map = yf_download_daily(tuple(tickers), days=730)

# -----------------------------
# Build overview rows
# -----------------------------
rows = []
with st.spinner("Building portfolio overview ..."):
    for tkr in tickers:
        # Labels / profile
        company = None
        sector = None
        industry = None

        if sp500_map is not None and tkr in sp500_map.index:
            r = sp500_map.loc[tkr]
            company = r.get("Security", None) if "Security" in sp500_map.columns else None
            # try likely column names
            for c in sp500_map.columns:
                cl = c.lower()
                if sector is None and "sector" in cl:
                    sector = r.get(c)
                if industry is None and ("sub" in cl or "industry" in cl):
                    industry = r.get(c)

        # fallback to yfinance
        prof = yf_fast_profile(tkr)
        company = company or prof.get("shortName")
        sector = sector or prof.get("sector")
        industry = industry or prof.get("industry")

        # market cap from yfinance fast_info (fallback)
        mcap = prof.get("marketCap")

        # price stats from batch candles
        dfp = price_map.get(tkr)
        current, high52, low52 = compute_current_52w(dfp)

        d_vs_high = None
        d_vs_low = None
        range_pct = None
        if current is not None and high52 is not None and high52 != 0:
            d_vs_high = (current / high52 - 1.0) * 100.0
        if current is not None and low52 is not None and low52 != 0:
            d_vs_low = (current / low52 - 1.0) * 100.0
        if high52 is not None and low52 is not None and low52 != 0:
            range_pct = (high52 / low52 - 1.0) * 100.0

        nxt = next_map.get(tkr)
        days_to = (nxt - today).days if isinstance(nxt, date) else None

        rows.append(
            {
                "Ticker": tkr,
                "Company": company,
                "Sector": sector,
                "Industry": industry,
                "Market Cap": mcap,
                "Next Earnings": nxt,
                "Days to Earnings": days_to,
                "Current": current,
                "52W High": high52,
                "52W Low": low52,
                "Δ vs 52W High (%)": d_vs_high,
                "Δ vs 52W Low (%)": d_vs_low,
                "52W Range (%)": range_pct,
            }
        )

overview = pd.DataFrame(rows)

# Clean types/formatting
overview["Market Cap"] = overview["Market Cap"].apply(_human_money)
overview["Next Earnings"] = overview["Next Earnings"].apply(_fmt_date)

# Filter upcoming-only if asked
filtered = overview.copy()
if only_upcoming:
    # Keep only rows that have a next earnings date and are within upcoming_within
    def _within(x):
        d = _to_date(x)
        if d is None:
            return False
        return 0 <= (d - today).days <= upcoming_within

    filtered = filtered[filtered["Next Earnings"].apply(_within)]

# -----------------------------
# 2) Portfolio overview table
# -----------------------------
st.subheader("2) Portfolio Overview")

st.dataframe(
    filtered,
    use_container_width=True,
    hide_index=True,
)

csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Portfolio Overview (CSV)",
    data=csv,
    file_name="earnings_radar_overview.csv",
    mime="text/csv",
)

# -----------------------------
# 3) Past 4 earnings + price reaction (per holding)
# -----------------------------
st.subheader("3) Past 4 Earnings + Price Reaction (per holding)")
st.caption(
    "Earnings events come from Finnhub calendar (reliable dates). Price reaction uses Yahoo daily closes (yfinance). "
    "Pre-close = last close before earnings date; 1D/3D = 1st and 3rd trading closes after earnings date."
)

# Use the same ticker set shown in overview, but limit for speed
tickers_for_history = filtered["Ticker"].tolist()[:max_tickers]

for tkr in tickers_for_history:
    company = overview.loc[overview["Ticker"] == tkr, "Company"].iloc[0] if (overview["Ticker"] == tkr).any() else None
    label = f"{tkr}" + (f" — {company}" if company else "")
    with st.expander(label, expanded=(tkr == tickers_for_history[0] if tickers_for_history else False)):

        past_events = past_map.get(tkr, [])
        if not past_events:
            st.info("No recent earnings events found for this ticker in Finnhub calendar.")
            continue

        dfp = price_map.get(tkr)
        if dfp is None or dfp.empty:
            st.warning("No price history returned by Yahoo for this ticker (may be ETF/fund/illiquid).")
            dfp = None

        out_rows = []
        for ev in sorted(past_events, key=lambda x: x["_date"], reverse=True):
            ev_date = ev.get("_date")

            eps_a = ev.get("epsActual", None)
            eps_e = ev.get("epsEstimate", None)
            surprise = ev.get("surprise", None)
            surprise_pct = ev.get("surprisePercent", None)
            q = ev.get("quarter", None)
            y = ev.get("year", None)

            reaction = compute_reaction_from_prices(dfp, ev_date) if dfp is not None else {}

            out_rows.append(
                {
                    "Earnings Date": ev_date.isoformat() if isinstance(ev_date, date) else None,
                    "Year": y,
                    "Quarter": q,
                    "EPS Actual": eps_a,
                    "EPS Est.": eps_e,
                    "Surprise": surprise,
                    "Surprise %": surprise_pct,
                    "Pre Close Date": _fmt_date(reaction.get("Pre Close Date")),
                    "Pre Close": reaction.get("Pre Close"),
                    "Post Close (1D) Date": _fmt_date(reaction.get("Post Close (1D) Date")),
                    "Post Close (1D)": reaction.get("Post Close (1D)"),
                    "1D Reaction %": reaction.get("1D Reaction %"),
                    "Post Close (3D) Date": _fmt_date(reaction.get("Post Close (3D) Date")),
                    "Post Close (3D)": reaction.get("Post Close (3D)"),
                    "3D Reaction %": reaction.get("3D Reaction %"),
                }
            )

        hist_df = pd.DataFrame(out_rows)

        # nicer numeric formatting in the UI
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        # quick note if reactions couldn't be computed
        if hist_df["Pre Close"].isna().all():
            st.info(
                "Price reaction is blank because we couldn’t find trading days around those earnings dates in Yahoo daily data "
                "(common for some ADRs/funds/odd symbols)."
            )
