"""
Build X (N x 10), Y (N x 1), dates, tickers for the dynamic GPD model.

Row t of X uses only information known at the close of day t (no look-ahead);
Y_t is the realised return of the NEXT trading day.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

import data_fetch

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"

DATA_START = "2016-01-01"
DATA_END = "2024-12-31"
SEC_LOOKBACK_DAYS = 550      # need one prior 10-K before DATA_START for the first similarity

# Y definition:
#   "close_to_close": Y_t = Close(t+1) / Close(t) - 1   (default)
#   "overnight"     : Y_t = Open(t+1)  / Close(t) - 1
TARGET_MODE = "close_to_close"

# Fixed cross-sector large-cap universe that traded through the whole window
# (survivorship bias: must be stated in the report)
TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "INTC", "CSCO", "ORCL", "IBM",
    "JPM", "BAC", "WFC", "GS",
    "XOM", "CVX",
    "JNJ", "PFE", "MRK", "UNH",
    "PG", "KO", "PEP", "WMT",
    "HD", "MCD", "NKE", "DIS",
    "CAT", "BA", "MMM",
    "T", "VZ",
]

FEATURE_NAMES = [
    "vol_20d",          # 1  20-day annualised realised volatility
    "downside_vol_20d", # 2  20-day annualised downside semi-volatility
    "kurtosis_60d",     # 3  60-day excess kurtosis of daily log returns
    "momentum_60d",     # 4  60-day price momentum
    "reversal_5d",      # 5  5-day return (short-term reversal)
    "range_5d",         # 6  5-day mean (High-Low)/Close
    "abn_volume",       # 7  log(5-day avg volume / 60-day avg volume)
    "vix",              # 8  CBOE VIX close (FRED VIXCLS)
    "n_8k_30d",         # 9  number of 8-K filings in the past 30 calendar days (SEC)
    "filing_change",    # 10 1 - cosine similarity of latest 10-K/10-Q vs previous same-type filing
]

_HASHER = HashingVectorizer(
    n_features=2 ** 18,
    alternate_sign=False,
    norm="l2",
    lowercase=True,
    stop_words="english",
    token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
)
MIN_TEXT_CHARS = 1000


# ---------- price factors + target for one stock ----------
def price_factors(px, target_mode=TARGET_MODE):
    c, h, l, o, v = px["close"], px["high"], px["low"], px["open"], px["volume"]
    r = np.log(c).diff()
    f = pd.DataFrame(index=px.index)
    f["vol_20d"] = r.rolling(20).std() * np.sqrt(252)
    f["downside_vol_20d"] = np.sqrt((r.clip(upper=0) ** 2).rolling(20).mean()) * np.sqrt(252)
    f["kurtosis_60d"] = r.rolling(60).kurt()
    f["momentum_60d"] = c / c.shift(60) - 1
    f["reversal_5d"] = c / c.shift(5) - 1
    f["range_5d"] = ((h - l) / c).rolling(5).mean()
    f["abn_volume"] = np.log((v.rolling(5).mean() + 1) / (v.rolling(60).mean() + 1))

    if target_mode == "overnight":
        y = o.shift(-1) / c - 1
    else:
        y = c.shift(-1) / c - 1
    return f, y


# ---------- SEC factors for one stock ----------
def sec_factors(ticker, cik, dates):
    """n_8k_30d and filing_change for each trading date (point-in-time)."""
    start = pd.Timestamp(DATA_START) - pd.Timedelta(days=SEC_LOOKBACK_DAYS)
    filings = data_fetch.fetch_filing_index(ticker, cik, start, DATA_END)
    filings["available"] = data_fetch.available_date(filings["accepted"])

    d = dates.values.astype("datetime64[ns]")
    d30 = (dates - pd.Timedelta(days=30)).values.astype("datetime64[ns]")

    # (a) 8-K event intensity: filings with  d - 30 days < available date <= d
    #     (every 8-K counts once; 8-K/A amendments are not counted)
    k8 = np.sort(filings.loc[filings["form"] == "8-K", "available"].values.astype("datetime64[ns]"))
    n_8k = (np.searchsorted(k8, d, side="right")
            - np.searchsorted(k8, d30, side="right")).astype(float)

    # (b) Lazy-Prices change score: 1 - cos(latest filing, previous same-type filing)
    events = []   # (available_date, change_score)
    for form in ("10-K", "10-Q"):
        vecs = []
        for row in filings[filings["form"] == form].itertuples():
            text = data_fetch.fetch_filing_text(cik, row.accession, row.primary_doc)
            if len(text) < MIN_TEXT_CHARS:
                continue
            vecs.append((row.available, _HASHER.transform([text])))
        for i in range(1, len(vecs)):
            sim = float(vecs[i][1].multiply(vecs[i - 1][1]).sum())
            events.append((vecs[i][0], 1.0 - sim))

    change = np.full(len(dates), np.nan)
    if events:
        events.sort(key=lambda x: x[0])
        ev_dates = np.array([e[0] for e in events], dtype="datetime64[ns]")
        ev_vals = np.array([e[1] for e in events])
        idx = np.searchsorted(ev_dates, d, side="right") - 1
        ok = idx >= 0
        change[ok] = ev_vals[idx[ok]]

    return n_8k, change


# ---------- assemble X, Y ----------
def build_dataset(target_mode=TARGET_MODE, refresh=False, save=True):
    """Returns X (N,10) float32, Y (N,1) float32, dates (N,), tickers (N,), sorted by date."""
    bars = data_fetch.fetch_alpaca_bars(TICKERS, DATA_START, DATA_END, refresh=refresh)
    vix = data_fetch.fetch_vix(refresh=refresh).ffill()
    ciks = data_fetch.ticker_to_cik(TICKERS)

    parts = []
    for t in TICKERS:
        px = bars[bars["ticker"] == t].set_index("date").sort_index()
        f, y = price_factors(px, target_mode)
        f["vix"] = vix.reindex(f.index, method="ffill")

        n_8k, change = sec_factors(t, ciks[t], f.index.to_series())
        f["n_8k_30d"] = n_8k
        f["filing_change"] = change

        f["Y"] = y
        f = f[f.index >= pd.Timestamp(DATA_START)]
        # price/market factors and target must exist; SEC similarity may be NaN early on
        f = f.dropna(subset=FEATURE_NAMES[:8] + ["Y"])
        f.insert(0, "ticker", t)
        parts.append(f)
        print(f"{t}: {len(f)} rows")

    df = pd.concat(parts).rename_axis("date").reset_index()
    df = df.sort_values("date", kind="stable").reset_index(drop=True)   # chronological, stocks interleaved
    df = df[["date", "ticker"] + FEATURE_NAMES + ["Y"]]

    if save:
        ARTIFACT_DIR.mkdir(exist_ok=True)
        out = df.copy()
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        out.to_csv(ARTIFACT_DIR / "dataset.csv", index=False)
        print(f"Saved dataset to {ARTIFACT_DIR / 'dataset.csv'}")

    X = df[FEATURE_NAMES].to_numpy(dtype="float32")
    Y = df["Y"].to_numpy(dtype="float32").reshape(-1, 1)
    dates = df["date"].values.astype("datetime64[ns]")
    tickers = df["ticker"].to_numpy(dtype=object)
    return X, Y, dates, tickers


if __name__ == "__main__":
    X, Y, dates, tickers = build_dataset()
    print(f"X shape {X.shape}, Y shape {Y.shape}")
