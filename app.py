import os
import time
import math
import datetime as dt
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# ----------------------------
# App config
# ----------------------------
st.set_page_config(page_title="Earnings Radar", page_icon="📆", layout="wide")

# ----------------------------
# Helpers
# ----------------------------
def _today_utc_date() -> dt.date:
    return dt.datetime.utcnow().date()

def _safe_upper(x: str) -> str:
    return str(x).strip().upper()

def _fmt_money(n: Optional[float]) -> str:
    if n is None or (isinstance(n, float) and (math.isnan(n) or math.isinf(n))):
        return "—"
    absn = abs(n)
    if absn >= 1e12:
        return f"{n/1e12:.2f}T"
    if absn >= 1e9:
        return f"{n/1e9:.2f}B"
    if absn >= 1e6:
        return f"{n/1e6:.2f}M"
    if absn >= 1e3:
        return f"{n/1e3:.2f}K"
    return f"{n:.0f}"

def _fmt_pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:+.1f}%"

def _coerce_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None

def _pick_ticker_column(df: pd.DataFrame) -> Optional[str]:
    # common portfolio formats
    candidates = [
        "Ticker", "ticker", "Symbol", "symbol", "Security", "security", "Instrument", "instrument"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: first column that looks ticker-ish
    for c in df.columns:
        if df[c].astype(str).str.len().median() <= 6:
            return c
    return None

@st.cache_data(show_spinner=False)
def load_sp500_universe() -> pd.DataFrame:
    """
    Looks for sp500_universe.csv in repo root.
    Must contain at least: Ticker (or ticker)
    Optionally: Company / Sector / Industry
    """
    for p in ["sp500_universe.csv", "./sp500_universe.csv"]:
        if os.path.exists(p):
            df = pd.read_csv(p)
            # normalize
            cols = {c: c.strip() for c in df.columns}
            df = df.rename(columns=cols)
            if "ticker" in df.columns and "Ticker" not in df.columns:
                df = df.rename(columns={"ticker": "Ticker"})
            if "Ticker" not in df.columns:
                return pd.DataFrame()
            df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
            return df
    return pd.DataFrame()

def universe_lookup(universe: pd.DataFrame, ticker: str) -> Dict[str, Any]:
    if universe is None or universe.empty:
        return {}
    hit = universe[universe["Ticker"] == ticker]
    if hit.empty:
        return {}
    row = hit.iloc[0].to_dict()
    return {
        "company": row.get("Company") or row.get("Company Name") or row.get("Name"),
        "sector": row.get("Sector"),
        "industry": row.get("Industry"),
    }

def get_finnhub_key() -> Optional[str]:
    # Streamlit Cloud Secrets: FINNHUB_API_KEY
    k = st.secrets.get("FINNHUB_API_KEY", None)
    if k:
        return str(k).strip()
    # local fallback: env var
    k2 = os.getenv("FINNHUB_API_KEY")
    if k2:
        return str(k2).strip()
    return None

def finnhub_get(url: str, params: Dict[str, Any], token: str, timeout: int = 20) -> Dict[str, Any]:
    params = dict(params or {})
    params["token"] = token
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def finnhub_profile2(symbol: str, token: str) -> Dict[str, Any]:
    return finnhub_get("https://finnhub.io/api/v1/stock/profile2", {"symbol": symbol}, token)

@st.cache_data(show_spinner=False, ttl=60 * 10)
def finnhub_quote(symbol: str, token: str) -> Dict[str, Any]:
    return finnhub_get("https://finnhub.io/api/v1/quote", {"symbol": symbol}, token)

@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def finnhub_basic_financials(symbol: str, token: str) -> Dict[str, Any]:
    # Includes 52W high/low for many tickers in "metric"
    return finnhub_get("https://finnhub.io/api/v1/stock/metric", {"symbol": symbol, "metric": "all"}, token)

@st.cache_data(show_spinner=False, ttl=60 * 30)
def finnhub_earnings_calendar(symbol: str, token: str, date_from: str, date_to: str) -> List[Dict[str, Any]]:
    data = finnhub_get(
        "https://finnhub.io/api/v1/calendar/earnings",
        {"symbol": symbol, "from": date_from, "to": date_to},
        token,
    )
    cal = data.get("earningsCalendar", []) or []
    # normalize a bit
    for e in cal:
        # ensure keys exist
        e.setdefault("symbol", symbol)
    return cal

@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def yf_fast_info(symbol: str) -> Dict[str, Any]:
    t = yf.Ticker(symbol)
    try:
        fi = dict(t.fast_info)
    except Exception:
        fi = {}
    return fi

@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def yf_sector_industry_name(symbol: str) -> Dict[str, Any]:
    t = yf.Ticker(symbol)
    out = {"company": None, "sector": None, "industry": None}
    try:
        info = t.get_info()
        out["company"] = info.get("shortName") or info.get("longName")
        out["sector"] = info.get("sector")
        out["industry"] = info.get("industry")
    except Exception:
        pass
    return out

@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def yf_daily_history(symbol: str, start: str, end: str) -> pd.DataFrame:
    # yfinance end is exclusive-ish sometimes; add a buffer day
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    return df

def nearest_trading_close(hist: pd.DataFrame, target: dt.date, direction: str) -> Optional[Tuple[dt.date, float]]:
    """
    direction: "before" => last row with Date < target
               "on_or_after" => first row with Date >= target
    """
    if hist is None or hist.empty:
        return None
    if direction == "before":
        d = hist[hist["Date"] < target].sort_values("Date")
        if d.empty:
            return None
        row = d.iloc[-1]
        return row["Date"], float(row["Close"])
    if direction == "on_or_after":
        d = hist[hist["Date"] >= target].sort_values("Date")
        if d.empty:
            return None
        row = d.iloc[0]
        return row["Date"], float(row["Close"])
    return None

def nth_trading_close_after(hist: pd.DataFrame, on_or_after_date: dt.date, n: int) -> Optional[Tuple[dt.date, float]]:
    """
    Finds nth trading day close AFTER (or on) on_or_after_date, 1-indexed:
      n=1 -> first close on_or_after_date
      n=3 -> third close on_or_after_date
    """
    if hist is None or hist.empty:
        return None
    d = hist[hist["Date"] >= on_or_after_date].sort_values("Date")
    if len(d) < n:
        return None
    row = d.iloc[n - 1]
    return row["Date"], float(row["Close"])

def compute_52w_from_sources(symbol: str, token: str) -> Tuple[Optional[float], Optional[float]]:
    # Try Finnhub metric first
    try:
        m = finnhub_basic_financials(symbol, token).get("metric", {}) or {}
        hi = _coerce_float(m.get("52WeekHigh"))
        lo = _coerce_float(m.get("52WeekLow"))
        if hi is not None and lo is not None:
            return hi, lo
    except Exception:
        pass
    # Fallback yfinance fast_info
    try:
        fi = yf_fast_info(symbol)
        hi = _coerce_float(fi.get("year_high"))
        lo = _coerce_float(fi.get("year_low"))
        return hi, lo
    except Exception:
        return None, None

def get_market_cap(symbol: str, token: str) -> Optional[float]:
    # Finnhub profile2 returns marketCapitalization in *millions*
    try:
        p = finnhub_profile2(symbol, token) or {}
        mc_m = _coerce_float(p.get("marketCapitalization"))
        if mc_m is not None:
            return mc_m * 1e6
    except Exception:
        pass
    # fallback yfinance
    try:
        fi = yf_fast_info(symbol)
        return _coerce_float(fi.get("market_cap"))
    except Exception:
        return None

def get_current_price(symbol: str, token: str) -> Optional[float]:
    try:
        q = finnhub_quote(symbol, token) or {}
        return _coerce_float(q.get("c"))
    except Exception:
        pass
    try:
        fi = yf_fast_info(symbol)
        return _coerce_float(fi.get("last_price"))
    except Exception:
        return None

def next_earnings_date(symbol: str, token: str, lookahead_days: int) -> Optional[dt.date]:
    today = _today_utc_date()
    to = today + dt.timedelta(days=int(lookahead_days))
    cal = finnhub_earnings_calendar(symbol, token, today.isoformat(), to.isoformat())
    if not cal:
        return None
    # find earliest date >= today
    dates = []
    for e in cal:
        d = e.get("date")
        if not d:
            continue
        try:
            dd = dt.date.fromisoformat(str(d))
            if dd >= today:
                dates.append(dd)
        except Exception:
            continue
    if not dates:
        return None
    return min(dates)

def last_earnings_events(symbol: str, token: str, n: int = 4) -> List[Dict[str, Any]]:
    """
    Pulls a wider window into the past and returns last n by date desc.
    Using Finnhub calendar for historical earnings is surprisingly decent.
    """
    today = _today_utc_date()
    start = today - dt.timedelta(days=900)  # ~2.5y
    cal = finnhub_earnings_calendar(symbol, token, start.isoformat(), today.isoformat())
    if not cal:
        return []
    parsed = []
    for e in cal:
        d = e.get("date")
        if not d:
            continue
        try:
            dd = dt.date.fromisoformat(str(d))
        except Exception:
            continue
        # keep only past dates
        if dd <= today:
            e2 = dict(e)
            e2["_date_obj"] = dd
            parsed.append(e2)
    parsed.sort(key=lambda x: x["_date_obj"], reverse=True)
    return parsed[:n]

def earnings_price_reaction(symbol: str, event_date: dt.date, hour: Optional[str]) -> Dict[str, Any]:
    """
    Uses yfinance daily closes around event_date.
    Finnhub 'hour' often: 'amc' (after market close) or 'bmo' (before market open)
    """
    # Fetch a small window around event date (buffer)
    start = (event_date - dt.timedelta(days=10)).isoformat()
    end = (event_date + dt.timedelta(days=15)).isoformat()
    hist = yf_daily_history(symbol, start=start, end=end)
    if hist.empty:
        return {
            "pre_close_date": None, "pre_close": None,
            "post_1d_date": None, "post_1d": None, "rxn_1d": None,
            "post_3d_date": None, "post_3d": None, "rxn_3d": None,
        }

    hour_norm = (hour or "").strip().lower()

    # Pre close is always last close BEFORE the earnings date
    pre = nearest_trading_close(hist, event_date, "before")
    if not pre:
        return {
            "pre_close_date": None, "pre_close": None,
            "post_1d_date": None, "post_1d": None, "rxn_1d": None,
            "post_3d_date": None, "post_3d": None, "rxn_3d": None,
        }
    pre_date, pre_close = pre

    # If BMO, the market reacts the SAME DAY close
    # If AMC/unknown, reaction is next trading day close
    if hour_norm == "bmo":
        post_anchor = event_date
    else:
        post_anchor = event_date + dt.timedelta(days=1)

    post1 = nth_trading_close_after(hist, post_anchor, 1)
    post3 = nth_trading_close_after(hist, post_anchor, 3)

    out = {
        "pre_close_date": pre_date, "pre_close": pre_close,
        "post_1d_date": None, "post_1d": None, "rxn_1d": None,
        "post_3d_date": None, "post_3d": None, "rxn_3d": None,
    }

    if post1:
        d1, c1 = post1
        out["post_1d_date"] = d1
        out["post_1d"] = c1
        out["rxn_1d"] = ((c1 / pre_close) - 1.0) * 100.0 if pre_close else None

    if post3:
        d3, c3 = post3
        out["post_3d_date"] = d3
        out["post_3d"] = c3
        out["rxn_3d"] = ((c3 / pre_close) - 1.0) * 100.0 if pre_close else None

    return out

def build_company_row(
    symbol: str,
    token: str,
    universe: pd.DataFrame,
    lookahead_days: int,
) -> Dict[str, Any]:
    sym = _safe_upper(symbol)

    # Company metadata: universe first, then yfinance, then finnhub profile
    meta = universe_lookup(universe, sym)
    if not meta.get("company") or not meta.get("sector") or not meta.get("industry"):
        yf_meta = yf_sector_industry_name(sym)
        meta["company"] = meta.get("company") or yf_meta.get("company")
        meta["sector"] = meta.get("sector") or yf_meta.get("sector")
        meta["industry"] = meta.get("industry") or yf_meta.get("industry")

    if not meta.get("company"):
        try:
            p = finnhub_profile2(sym, token) or {}
            meta["company"] = p.get("name") or meta.get("company")
        except Exception:
            pass

    mc = get_market_cap(sym, token)
    nxt = next_earnings_date(sym, token, lookahead_days=lookahead_days)
    cur = get_current_price(sym, token)
    hi52, lo52 = compute_52w_from_sources(sym, token)

    # percent distances
    d_hi = None
    d_lo = None
    rng = None
    if cur is not None and hi52 is not None and hi52 != 0:
        d_hi = ((cur / hi52) - 1.0) * 100.0
    if cur is not None and lo52 is not None and lo52 != 0:
        d_lo = ((cur / lo52) - 1.0) * 100.0
    if hi52 is not None and lo52 is not None and lo52 != 0:
        rng = ((hi52 / lo52) - 1.0) * 100.0

    days_to = (nxt - _today_utc_date()).days if nxt else None

    return {
        "Ticker": sym,
        "Company": meta.get("company"),
        "Sector": meta.get("sector"),
        "Industry": meta.get("industry"),
        "Market Cap ($)": mc,
        "Next Earnings": nxt,
        "Days to Earnings": days_to,
        "Current": cur,
        "52W High": hi52,
        "52W Low": lo52,
        "Δ vs 52W High (%)": d_hi,
        "Δ vs 52W Low (%)": d_lo,
        "52W Range (%)": rng,
    }

# ----------------------------
# UI
# ----------------------------
st.title("📆 Earnings Radar")
st.caption(
    "Upload a holdings file and instantly see upcoming earnings + 52-week context. "
    "Past 4 earnings uses Finnhub; price reaction uses Yahoo daily closes."
)

token = get_finnhub_key()
universe = load_sp500_universe()

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    lookahead_days = st.slider("Next earnings lookahead (days)", 30, 365, 180, 10)
    max_tickers = st.slider("Max tickers to process", 5, 250, 60, 5)
    only_with_upcoming = st.checkbox("Show only holdings with an upcoming earnings date", value=True)
    upcoming_within_days = st.slider("Upcoming earnings within (days)", 1, 365, 60, 5)
    st.divider()
    st.caption("Tip: fewer tickers = faster. Caching helps a lot after the first run.")

# Top: Upload / Paste
colA, colB = st.columns([2, 1], gap="large")
with colA:
    st.subheader("1) Upload Portfolio (CSV / Excel)")
    uploaded = st.file_uploader("Upload holdings file", type=["csv", "xlsx", "xls"], label_visibility="collapsed")

    st.caption("No file? Paste tickers (comma-separated):")
    tickers_text = st.text_input("Tickers", value="AAPL, MSFT, NVDA", label_visibility="collapsed")

with colB:
    if token is None:
        st.warning(
            "Missing `FINNHUB_API_KEY`. Set it in Streamlit Cloud: **Manage app → Settings → Secrets**.\n\n"
            "Example:\n\n"
            "`FINNHUB_API_KEY = \"your_key\"`"
        )
    else:
        st.success("Finnhub key detected ✅")

# Parse tickers
tickers: List[str] = []

if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".csv"):
            df_up = pd.read_csv(uploaded)
        else:
            df_up = pd.read_excel(uploaded)
        col = _pick_ticker_column(df_up)
        if col is None:
            st.error("Couldn't find a ticker/symbol column in the uploaded file.")
        else:
            tickers = [_safe_upper(x) for x in df_up[col].dropna().tolist()]
            tickers = [t for t in tickers if t and t != "NAN"]
            tickers = list(dict.fromkeys(tickers))
            st.success(f"Loaded {len(tickers)} unique tickers from file.")
    except Exception as e:
        st.error(f"Failed to read upload: {e}")

if not tickers:
    # fallback: pasted
    parts = [p.strip() for p in (tickers_text or "").split(",")]
    tickers = [_safe_upper(p) for p in parts if p.strip()]
    tickers = list(dict.fromkeys(tickers))

# Run button
run = st.button("🚀 Run Earnings Radar", type="primary", disabled=(token is None or len(tickers) == 0))

if not run:
    st.stop()

if token is None:
    st.error("Please set FINNHUB_API_KEY and rerun.")
    st.stop()

tickers = tickers[: int(max_tickers)]

# ----------------------------
# Build overview table
# ----------------------------
st.subheader("2) Portfolio Overview")

rows: List[Dict[str, Any]] = []
progress = st.progress(0)
status = st.empty()

for i, t in enumerate(tickers, start=1):
    status.write(f"Fetching: **{t}** ({i}/{len(tickers)})")
    try:
        row = build_company_row(t, token, universe, lookahead_days=lookahead_days)
        rows.append(row)
    except Exception as e:
        rows.append({"Ticker": _safe_upper(t), "Company": None, "Sector": None, "Industry": None, "Error": str(e)})
    progress.progress(i / len(tickers))

status.empty()
progress.empty()

overview = pd.DataFrame(rows)

# Filtering: upcoming earnings only + within days
if "Next Earnings" not in overview.columns:
    overview["Next Earnings"] = pd.NaT

if only_with_upcoming:
    overview = overview[overview["Next Earnings"].notna()]

if not overview.empty and "Days to Earnings" in overview.columns:
    overview = overview[(overview["Days to Earnings"].notna()) & (overview["Days to Earnings"] <= int(upcoming_within_days))]

# Sort by next earnings date
if "Next Earnings" in overview.columns:
    overview = overview.sort_values(["Next Earnings", "Ticker"], ascending=[True, True], na_position="last")

# Display formatting copy
disp = overview.copy()

# Format market cap for display
if "Market Cap ($)" in disp.columns:
    disp["Market Cap"] = disp["Market Cap ($)"].apply(_fmt_money)
    disp = disp.drop(columns=["Market Cap ($)"])

# Format numeric columns
for c in ["Current", "52W High", "52W Low"]:
    if c in disp.columns:
        disp[c] = disp[c].apply(lambda x: "—" if _coerce_float(x) is None else f"{float(x):.2f}")

for c in ["Δ vs 52W High (%)", "Δ vs 52W Low (%)", "52W Range (%)"]:
    if c in disp.columns:
        disp[c] = disp[c].apply(_fmt_pct)

# Next earnings as date string
if "Next Earnings" in disp.columns:
    disp["Next Earnings"] = disp["Next Earnings"].apply(lambda d: d.isoformat() if isinstance(d, dt.date) else "—")

# Keep a clean column order
wanted = [
    "Ticker", "Company", "Sector", "Industry",
    "Market Cap",
    "Next Earnings", "Days to Earnings",
    "Current", "52W High", "52W Low",
    "Δ vs 52W High (%)", "Δ vs 52W Low (%)", "52W Range (%)",
]
cols_present = [c for c in wanted if c in disp.columns]
disp = disp[cols_present + [c for c in disp.columns if c not in cols_present]]

st.dataframe(disp, use_container_width=True, hide_index=True)

# Download
csv_bytes = disp.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Portfolio Overview (CSV)",
    data=csv_bytes,
    file_name="earnings_radar_overview.csv",
    mime="text/csv",
)

# ----------------------------
# Past 4 earnings + price reaction
# ----------------------------
st.subheader("3) Past 4 Earnings + Price Reaction (per holding)")
st.caption(
    "Earnings events come from Finnhub calendar (reliable dates). "
    "Price reaction uses Yahoo daily closes (yfinance): pre-close = last close before earnings date; "
    "1D/3D = 1st and 3rd trading closes after earnings date (BMO uses same-day close)."
)

# Use the original tickers list order (not filtered view) so you can still expand any holding
for sym in tickers:
    sym_u = _safe_upper(sym)

    # Company name for label
    name = None
    # try from overview rows
    try:
        m = overview[overview["Ticker"] == sym_u]
        if not m.empty:
            name = m.iloc[0].get("Company")
    except Exception:
        pass
    label = f"{sym_u} — {name}" if name else sym_u

    with st.expander(label, expanded=False):
        try:
            ev = last_earnings_events(sym_u, token, n=4)
        except Exception as e:
            st.warning(f"Failed to fetch earnings history: {e}")
            continue

        if not ev:
            st.info("No earnings history found for this ticker in Finnhub calendar.")
            continue

        # Build table rows
        out_rows = []
        for e in ev:
            ed = e.get("_date_obj")
            hour = e.get("hour")  # often "amc" / "bmo"
            eps_a = _coerce_float(e.get("epsActual"))
            eps_e = _coerce_float(e.get("epsEstimate"))
            surprise = None
            surprise_pct = None
            if eps_a is not None and eps_e is not None:
                surprise = eps_a - eps_e
                if eps_e != 0:
                    surprise_pct = (surprise / abs(eps_e)) * 100.0

            rx = earnings_price_reaction(sym_u, ed, hour)

            out_rows.append({
                "Earnings Date": ed.isoformat() if isinstance(ed, dt.date) else "—",
                "Hour": (hour or "—").upper(),
                "Period": str(e.get("period") or e.get("quarter") or "—"),
                "EPS Actual": "—" if eps_a is None else f"{eps_a:.3f}",
                "EPS Est.": "—" if eps_e is None else f"{eps_e:.3f}",
                "Surprise": "—" if surprise is None else f"{surprise:+.3f}",
                "Surprise %": "—" if surprise_pct is None else f"{surprise_pct:+.1f}%",
                "Pre Close Date": rx["pre_close_date"].isoformat() if isinstance(rx["pre_close_date"], dt.date) else "—",
                "Pre Close": "—" if rx["pre_close"] is None else f"{rx['pre_close']:.2f}",
                "Post Close (1D) Date": rx["post_1d_date"].isoformat() if isinstance(rx["post_1d_date"], dt.date) else "—",
                "Post Close (1D)": "—" if rx["post_1d"] is None else f"{rx['post_1d']:.2f}",
                "1D Reaction %": _fmt_pct(rx["rxn_1d"]),
                "Post Close (3D) Date": rx["post_3d_date"].isoformat() if isinstance(rx["post_3d_date"], dt.date) else "—",
                "Post Close (3D)": "—" if rx["post_3d"] is None else f"{rx['post_3d']:.2f}",
                "3D Reaction %": _fmt_pct(rx["rxn_3d"]),
            })

        st.dataframe(pd.DataFrame(out_rows), use_container_width=True, hide_index=True)

