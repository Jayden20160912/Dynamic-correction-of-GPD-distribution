"""
Data download + local cache for the dynamic GPD project.

Sources
    Alpaca      daily OHLCV bars (factors 1-7 and Y)
    FRED        VIXCLS daily close (factor 8)
    SEC EDGAR   8-K / 10-K / 10-Q filings (factors 9-10)

Everything is cached under data_cache/ so the APIs are hit only once.
Delete a cache file (or pass refresh=True) to download it again.

API keys are read from API.env (or .env) in the project root -- never hard-coded.
See API.env.example for the expected variable names.
"""

from __future__ import annotations

import gzip
import html
import io
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"
EDGAR_DIR = CACHE_DIR / "edgar"
EDGAR_TEXT_DIR = EDGAR_DIR / "text"

ALPACA_BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
FRED_VIX_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"

MAX_TEXT_CHARS = 3_000_000      # cap on stripped filing text
SEC_MIN_INTERVAL = 0.15         # seconds between SEC requests (< 10 requests / second)


# ==========================================
# 0. Keys (read from API.env / .env, filled in by the user)
# ==========================================

_ENV_ALIASES = {
    "ALPACA_API_KEY_ID": ["ALPACA_API_KEY_ID", "APCA_API_KEY_ID", "ALPACA_API_KEY", "ALPACA_KEY_ID"],
    "ALPACA_API_SECRET_KEY": ["ALPACA_API_SECRET_KEY", "APCA_API_SECRET_KEY", "ALPACA_SECRET_KEY",
                              "ALPACA_API_SECRET"],
    "SEC_USER_AGENT": ["SEC_USER_AGENT"],
}


def load_env():
    for name in ("API.env", ".env"):
        path = ROOT / name
        if path.exists():
            load_dotenv(path, override=False)


def get_key(name):
    load_env()
    for alias in _ENV_ALIASES.get(name, [name]):
        value = (os.environ.get(alias) or "").strip().strip('"').strip("'")
        if value:
            return value
    raise RuntimeError(
        f"Missing {name}. Add a line `{name}=...` to API.env in the project root "
        f"(see API.env.example) and save the file."
    )


# ==========================================
# 1. Alpaca daily bars
# ==========================================

def fetch_alpaca_bars(tickers, start="2016-01-01", end="2024-12-31", refresh=False,
                      asof=None, require_all=True):
    """Adjusted daily bars for all tickers.

    asof: Alpaca symbol mapping date. "2016-01-04" means "the company that used this ticker on
          2016-01-04", followed through later renames (e.g. FB -> META) and NOT switching to a
          different company that reuses the ticker later. "-" = no mapping (plain ticker).
          None = Alpaca default (the company using the ticker today).
    Each asof setting has its own cache file, so caches never mix.

    Returns a DataFrame with columns: ticker, date, open, high, low, close, volume
    (date = New York trading date, tz-naive).
    """
    tag = "" if asof is None else ("_plain" if asof == "-" else f"_asof{asof}")
    cache = CACHE_DIR / f"alpaca_bars{tag}.parquet"
    if cache.exists() and not refresh:
        df = pd.read_parquet(cache)
        if set(tickers) <= set(df["ticker"].unique()):
            return df[df["ticker"].isin(tickers)].reset_index(drop=True)
        print(f"Alpaca cache {cache.name} does not cover all tickers, downloading again")

    headers = {
        "APCA-API-KEY-ID": get_key("ALPACA_API_KEY_ID"),
        "APCA-API-SECRET-KEY": get_key("ALPACA_API_SECRET_KEY"),
        "accept": "application/json",
    }

    rows, page = [], 0
    tickers = list(dict.fromkeys(tickers))
    for i in range(0, len(tickers), 100):              # keep the URL short
        params = {
            "symbols": ",".join(tickers[i:i + 100]),
            "timeframe": "1Day",
            "start": f"{start}T00:00:00Z",
            "end": f"{end}T23:59:59Z",
            "limit": 10000,          # total across all symbols, not per symbol
            "adjustment": "all",     # split + dividend adjusted (no fake crashes on split days)
            "feed": "sip",           # full-market volume / range
            "sort": "asc",
        }
        if asof is not None:
            params["asof"] = asof
        token = None
        while True:
            if token:
                params["page_token"] = token
            resp = _get_with_retry(ALPACA_BARS_URL, headers=headers, params=params)
            payload = resp.json()
            for sym, bars in (payload.get("bars") or {}).items():
                for b in bars:
                    rows.append((sym, b["t"], b["o"], b["h"], b["l"], b["c"], b["v"]))
            page += 1
            print(f"Alpaca page {page}: {len(rows)} bars so far")
            token = payload.get("next_page_token")
            if not token:
                break

    if not rows:
        raise RuntimeError("Alpaca returned no bars -- check the keys and the request parameters")

    df = pd.DataFrame(rows, columns=["ticker", "t", "open", "high", "low", "close", "volume"])
    ts = pd.to_datetime(df.pop("t"), utc=True).dt.tz_convert("America/New_York")
    df.insert(1, "date", ts.dt.tz_localize(None).dt.normalize())
    df = (df.drop_duplicates(["ticker", "date"], keep="last")
            .sort_values(["ticker", "date"]).reset_index(drop=True))

    missing = sorted(set(tickers) - set(df["ticker"].unique()))
    if missing and require_all:
        raise RuntimeError(f"Alpaca returned no data for: {missing}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


# ==========================================
# 2. FRED VIX
# ==========================================

def fetch_vix(refresh=False):
    """VIXCLS daily close as a Series indexed by date (holidays are NaN; caller forward-fills)."""
    cache = CACHE_DIR / "vix_fred.csv"
    if not cache.exists() or refresh:
        resp = _get_with_retry(FRED_VIX_CSV_URL)   # default UA; FRED stalls on a bare "Mozilla/5.0"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(resp.text)

    df = pd.read_csv(io.StringIO(cache.read_text()))
    dates = pd.to_datetime(df.iloc[:, 0])
    values = pd.to_numeric(df.iloc[:, 1], errors="coerce")   # "." on holidays -> NaN
    s = pd.Series(values.values, index=dates, name="vix").sort_index()
    return s[~s.index.duplicated(keep="last")]


# ==========================================
# 3. SEC EDGAR
# ==========================================

_last_sec_call = [0.0]


def _sec_get(url):
    wait = SEC_MIN_INTERVAL - (time.time() - _last_sec_call[0])
    if wait > 0:
        time.sleep(wait)
    headers = {"User-Agent": get_key("SEC_USER_AGENT"), "Accept-Encoding": "gzip, deflate"}
    try:
        return _get_with_retry(url, headers=headers)
    finally:
        _last_sec_call[0] = time.time()


def _get_with_retry(url, headers=None, params=None, tries=6):
    delay = 2.0
    for attempt in range(tries):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=60)
        except requests.RequestException:
            if attempt == tries - 1:
                raise
            time.sleep(delay)
            delay *= 2
            continue
        if resp.status_code == 200:
            return resp
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < tries - 1:
            time.sleep(delay)
            delay *= 2
            continue
        raise RuntimeError(f"HTTP {resp.status_code} for {resp.url.split('?')[0]}: {resp.text[:300]}")
    raise RuntimeError(f"Failed to fetch {url}")


def _cached_json(path, url, refresh=False):
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    data = _sec_get(url).json()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def ticker_to_cik(tickers):
    data = _cached_json(EDGAR_DIR / "company_tickers.json", SEC_TICKERS_URL)
    table = {row["ticker"].upper(): int(row["cik_str"]) for row in data.values()}
    missing = [t for t in tickers if t not in table]
    if missing:
        raise RuntimeError(f"No CIK found for: {missing}")
    return {t: table[t] for t in tickers}


def fetch_filing_index(ticker, cik, start, end, refresh=False):
    """All 8-K / 10-K / 10-Q filings of one company accepted between start and end.

    Returns a DataFrame: form, accepted (naive US/Eastern timestamp), accession, primary_doc.
    """
    cik10 = f"{cik:010d}"
    sub = _cached_json(EDGAR_DIR / "submissions" / f"CIK{cik10}.json",
                       f"{SEC_SUBMISSIONS_URL}CIK{cik10}.json", refresh)
    blocks = [sub["filings"]["recent"]]
    # Older filings live in extra JSON files; read the ones overlapping our window
    for extra in sub["filings"].get("files", []):
        if extra.get("filingTo", "9999") < str(start)[:10] or extra.get("filingFrom", "0000") > str(end)[:10]:
            continue
        blocks.append(_cached_json(EDGAR_DIR / "submissions" / extra["name"],
                                   SEC_SUBMISSIONS_URL + extra["name"], refresh))

    frames = []
    for b in blocks:
        frames.append(pd.DataFrame({
            "form": b["form"],
            "filing_date": b["filingDate"],
            "accepted": b["acceptanceDateTime"],
            "accession": b["accessionNumber"],
            "primary_doc": b["primaryDocument"],
        }))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["form"].isin(["8-K", "10-K", "10-Q"])].copy()

    # acceptanceDateTime is treated as US/Eastern wall-clock time (EDGAR's convention).
    # If a value were really UTC this reading is later than the truth, i.e. still conservative.
    accepted = pd.to_datetime(df["accepted"].astype(str).str.slice(0, 19), errors="coerce")
    fallback = pd.to_datetime(df["filing_date"], errors="coerce") + pd.Timedelta(hours=23)
    df["accepted"] = accepted.fillna(fallback)

    df = df[(df["accepted"] >= pd.Timestamp(start)) &
            (df["accepted"] < pd.Timestamp(end) + pd.Timedelta(days=1))]
    df = df.drop_duplicates("accession").sort_values("accepted").reset_index(drop=True)
    df.insert(0, "ticker", ticker)
    df.insert(1, "cik", cik)
    return df


def available_date(accepted):
    """First date a filing may be used: same day if accepted before 16:00 ET, else the next day.

    The result may fall on a weekend / holiday; it is matched against trading days with
    `available_date <= trading_day`, which is the same as "next trading day".
    """
    accepted = pd.DatetimeIndex(accepted)
    day = accepted.normalize()
    return day + pd.to_timedelta((accepted.hour >= 16).astype(int), unit="D")


_DROP_BLOCK_RE = re.compile(r"(?is)<(script|style|ix:header)\b.*?</\1\s*>")
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


def strip_html(raw):
    text = _DROP_BLOCK_RE.sub(" ", raw)       # scripts, styles, hidden inline-XBRL header
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return _SPACE_RE.sub(" ", text).strip()[:MAX_TEXT_CHARS]


def fetch_filing_text(cik, accession, primary_doc):
    """Main document of a filing with HTML removed. Cached as gzip text."""
    acc_nodash = accession.replace("-", "")
    cache = EDGAR_TEXT_DIR / f"{cik}_{acc_nodash}.txt.gz"
    if cache.exists():
        with gzip.open(cache, "rt", encoding="utf-8") as fh:
            return fh.read()

    doc = primary_doc if isinstance(primary_doc, str) and primary_doc.strip() else f"{accession}.txt"
    url = f"{SEC_ARCHIVES_URL}/{cik}/{acc_nodash}/{doc}"
    try:
        resp = _sec_get(url)
        text = strip_html(resp.content.decode(resp.encoding or "utf-8", errors="ignore"))
    except RuntimeError as err:
        print(f"  warning: could not download {url}: {str(err)[:80]}")
        return ""            # not cached, so it is retried on the next run

    EDGAR_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache, "wt", encoding="utf-8") as fh:
        fh.write(text)
    return text
