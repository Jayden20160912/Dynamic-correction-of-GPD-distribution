"""
Build X (N x 10), Y (N x 1), dates, tickers for the dynamic GPD model.

Row t of X uses only information known at the close of day t (no look-ahead);
Y_t is the realised return of the NEXT trading day.

Stock universe: universe/selected_universe_300.csv -- 300 stocks chosen with information
available in January 2016 only (stratified by 2016 market cap and sector, delisted stocks
included). Each stock is identified by its 2016 ticker + SEC CIK; its price history is
cut at the verified `history_end` (delisting / acquisition date).
"""

from __future__ import annotations

import json
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

# Point-in-time universe (see universe/screening_scripts/ for how it was selected).
# The old 32 large caps and their dataset are kept in backup_dataset_32stocks/.
UNIVERSE_CSV = ROOT / "universe" / "selected_universe_300.csv"
ALPACA_ASOF = "2016-01-04"        # tickers mean "the company using this ticker on this date"
CONTINUATION_GAP_DAYS = 7         # asof history ending this long before history_end -> link break


def load_universe():
    u = pd.read_csv(UNIVERSE_CSV, dtype={"symbol_2016": str})
    u["cik"] = u["cik"].astype(int)
    u["history_end"] = pd.to_datetime(u["history_end"]).clip(upper=pd.Timestamp(DATA_END))
    return u.set_index("symbol_2016")


UNIVERSE = load_universe()
TICKERS = UNIVERSE.index.tolist()

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
    "worst252",         # 11 largest single-day loss (-return) over the past 252 trading days
    "e63",              # 12 max |return| 60-66 trading days before the loss day (~1 quarter ago)
    "echo_q",           # 13 mean of the same max-|return| windows 63/126/189/252 days before the loss day
]
HISTORY_FEATURES = ["worst252", "e63", "echo_q"]   # computed from the Y column itself


def add_history_features(df):
    """Add worst252, e63, echo_q to a dataset with columns date, ticker, Y.

    Row t has Y_t = return realised on the NEXT trading day (the "loss day", t+1).
    The return realised k trading days before the loss day is Y shifted by k rows
    within the same stock, so every value used here is known at the close of day t.
    """
    df = df.sort_values(["ticker", "date"], kind="stable").copy()
    g = df.groupby("ticker", sort=False)["Y"]
    past = lambda k: g.shift(k)                      # return realised k days before the loss day

    # worst252: returns of the 252 days up to and including day t (k = 1..252)
    loss = -past(1)
    df["worst252"] = loss.groupby(df["ticker"]).transform(
        lambda s: s.rolling(252, min_periods=60).max())

    def window_max(center):                          # max |r| over k = center-3 .. center+3
        return pd.concat([past(k).abs() for k in range(center - 3, center + 4)], axis=1).max(axis=1)

    windows = pd.concat([window_max(c) for c in (63, 126, 189, 252)], axis=1)
    df["e63"] = windows.iloc[:, 0]
    df["echo_q"] = windows.mean(axis=1)               # mean of the windows that exist so far
    return df.sort_values("date", kind="stable").reset_index(drop=True)

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
def load_price_history(refresh=False):
    """Clean daily bars for every universe stock, one entity per 2016 ticker.

    1. Bars of the company that used the ticker on ALPACA_ASOF (handles renames and reuse).
    2. Where Alpaca's asof link breaks at a corporate action although the same company keeps
       trading (checked against SEC filings during selection), append the plain-ticker bars
       up to history_end.
    3. Cut at history_end (delisting / acquisition) and drop zero-volume bars: after a
       delisting Alpaca can keep returning frozen, zero-volume prices for years.
    """
    bars = data_fetch.fetch_alpaca_bars(TICKERS, DATA_START, DATA_END, refresh=refresh,
                                        asof=ALPACA_ASOF)
    last = bars.groupby("ticker")["date"].max()
    gap = UNIVERSE["history_end"] - last.reindex(UNIVERSE.index)
    cont = gap[gap > pd.Timedelta(days=CONTINUATION_GAP_DAYS)].index.tolist()
    if cont:
        plain = data_fetch.fetch_alpaca_bars(cont, DATA_START, DATA_END, refresh=refresh,
                                             asof="-", require_all=False)
        plain = plain[plain["date"] > plain["ticker"].map(last)]
        bars = pd.concat([bars, plain], ignore_index=True)
        print(f"Appended plain-ticker continuation for {len(cont)} stocks: {', '.join(cont)}")

    bars = bars[bars["date"] <= bars["ticker"].map(UNIVERSE["history_end"])]
    n_before = len(bars)
    bars = bars[bars["volume"] > 0]
    print(f"Dropped {n_before - len(bars)} zero-volume bars")
    bars = bars.sort_values(["ticker", "date"]).reset_index(drop=True)
    return apply_price_fixes(bars)


PRICE_FIXES_CSV = ROOT / "universe" / "price_fixes.csv"


def apply_price_fixes(bars):
    """Back-adjust spin-offs that Alpaca left unadjusted (listed in universe/price_fixes.csv).

    All prices before the spin-off date are scaled so the close-to-close return on that day
    is 0 instead of a fake crash; the true (small) market return of that day is lost.
    """
    if not PRICE_FIXES_CSV.exists():
        return bars
    fixes = pd.read_csv(PRICE_FIXES_CSV, parse_dates=["date"])
    for fx in fixes[fixes["action"] == "spinoff_backadjust"].itertuples():
        stock = bars["ticker"] == fx.ticker
        on_day = stock & (bars["date"] == fx.date)
        before = stock & (bars["date"] < fx.date)
        if not on_day.any() or not before.any():
            print(f"price fix skipped (no bars): {fx.ticker} {fx.date.date()}")
            continue
        prev_close = bars.loc[before, "close"].iloc[-1]
        factor = bars.loc[on_day, "close"].iloc[0] / prev_close
        bars.loc[before, ["open", "high", "low", "close"]] *= factor
        print(f"price fix: {fx.ticker} {fx.date.date()} spin-off back-adjusted (factor {factor:.3f})")
    return bars


def build_dataset(target_mode=TARGET_MODE, refresh=False, save=True):
    """Returns X (N,10) float32, Y (N,1) float32, dates (N,), tickers (N,), sorted by date."""
    bars = load_price_history(refresh=refresh)
    vix = data_fetch.fetch_vix(refresh=refresh).ffill()

    parts = []
    for i, t in enumerate(TICKERS, 1):
        px = bars[bars["ticker"] == t].set_index("date").sort_index()
        if len(px) < 80:                 # not enough history for the 60-day factors + target
            print(f"{t}: only {len(px)} bars, skipped")
            continue
        f, y = price_factors(px, target_mode)
        f["vix"] = vix.reindex(f.index, method="ffill")

        n_8k, change = sec_factors(t, int(UNIVERSE.at[t, "cik"]), f.index.to_series())
        f["n_8k_30d"] = n_8k
        f["filing_change"] = change

        f["Y"] = y
        f = f[f.index >= pd.Timestamp(DATA_START)]
        # price/market factors and target must exist; SEC similarity may be NaN early on
        f = f.dropna(subset=FEATURE_NAMES[:8] + ["Y"])
        f.insert(0, "ticker", t)
        parts.append(f)
        print(f"[{i}/{len(TICKERS)}] {t}: {len(f)} rows", flush=True)

    df = pd.concat(parts).rename_axis("date").reset_index()
    df = add_history_features(df)   # chronological, stocks interleaved
    df = df[["date", "ticker"] + FEATURE_NAMES + ["Y"]]

    if save:
        ARTIFACT_DIR.mkdir(exist_ok=True)
        out = df.copy()
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        out.to_csv(DATASET_CSV, index=False)
        DATASET_INFO.write_text(json.dumps(_dataset_info(target_mode), indent=2))
        print(f"Saved dataset to {DATASET_CSV}")

    return _to_arrays(df)


# ---------- reuse the saved dataset (no API calls) ----------
DATASET_CSV = ARTIFACT_DIR / "dataset.csv"
DATASET_INFO = ARTIFACT_DIR / "dataset_info.json"


def _dataset_info(target_mode):
    return {"target_mode": target_mode, "data_start": DATA_START, "data_end": DATA_END,
            "universe_file": UNIVERSE_CSV.relative_to(ROOT).as_posix(), "alpaca_asof": ALPACA_ASOF,
            "tickers": TICKERS, "feature_names": FEATURE_NAMES}


def _to_arrays(df):
    X = df[FEATURE_NAMES].to_numpy(dtype="float32")
    Y = df["Y"].to_numpy(dtype="float32").reshape(-1, 1)
    dates = df["date"].values.astype("datetime64[ns]")
    tickers = df["ticker"].to_numpy(dtype=object)
    return X, Y, dates, tickers


def load_saved_dataset(target_mode=TARGET_MODE):
    """Read artifacts/dataset.csv. Returns None if it is missing or was built with other settings."""
    if not DATASET_CSV.exists():
        print(f"No saved dataset at {DATASET_CSV}")
        return None

    if DATASET_INFO.exists():
        info = json.loads(DATASET_INFO.read_text())
        wanted = _dataset_info(target_mode)
        diff = [k for k in wanted if info.get(k) != wanted[k]]
        if diff:
            print(f"Saved dataset was built with different settings ({', '.join(diff)})")
            return None
    else:
        print(f"Note: {DATASET_INFO.name} not found; assuming dataset.csv matches the current "
              f"settings (TARGET_MODE = {target_mode!r}).")

    df = pd.read_csv(DATASET_CSV, parse_dates=["date"])
    missing = [c for c in ["date", "ticker", "Y"] + FEATURE_NAMES if c not in df.columns]
    if missing:
        print(f"Saved dataset is missing columns {missing}")
        return None

    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    print(f"Loaded saved dataset {DATASET_CSV} ({len(df)} rows) -- no API calls")
    return _to_arrays(df)


def get_dataset(target_mode=TARGET_MODE, rebuild=False):
    """Use artifacts/dataset.csv when possible; otherwise build it from data_cache/ or the APIs."""
    if not rebuild:
        arrays = load_saved_dataset(target_mode)
        if arrays is not None:
            return arrays
    print("Building dataset from data_cache/ (downloads anything that is not cached)")
    return build_dataset(target_mode=target_mode)


if __name__ == "__main__":
    X, Y, dates, tickers = build_dataset()
    print(f"X shape {X.shape}, Y shape {Y.shape}")
