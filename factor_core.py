"""
The 13 model factors -- the ONLY implementation, shared by training and QuantConnect.

    features.py            builds artifacts/dataset.csv (training) with these functions
    quantconnect/main.py   computes the same factors live (add this file to the QC project)

Only numpy + pandas: no API calls, no sklearn, no QuantConnect imports.

Timing: row t uses only information known at the close of day t; the model predicts the
loss of the NEXT trading day t+1.

    r_log  = log(close).diff()            log return (vol / downside vol / kurtosis)
    r      = close / close.shift(1) - 1   simple return realised on day t (the training Y, shifted)

    vol_20d          std(r_log, 20) * sqrt(252)
    downside_vol_20d sqrt(mean(min(r_log, 0)^2, 20)) * sqrt(252)
    kurtosis_60d     excess kurtosis of r_log over 60 days
    momentum_60d     close / close.shift(60) - 1
    reversal_5d      close / close.shift(5) - 1
    range_5d         mean((high - low) / close, 5)
    abn_volume       log((mean(volume, 5) + 1) / (mean(volume, 60) + 1))
    vix              CBOE VIX close, last value on or before t
    n_8k_30d         8-K filings public in the last 30 calendar days      } SEC EDGAR: computed in
    filing_change    1 - cosine similarity of the latest 10-K/10-Q vs     } features.sec_factors;
                     the previous same-type filing                        } QC reads them from a CSV
    worst252         max(-r) over the 252 days up to t (at least 60)
    e63              max |r| over the days 60..66 trading days before the loss day t+1
    echo_q           mean of the same 7-day window max at 63, 126, 189, 252 days (windows that exist)
"""
from __future__ import annotations

from io import StringIO

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "vol_20d", "downside_vol_20d", "kurtosis_60d", "momentum_60d", "reversal_5d", "range_5d",
    "abn_volume", "vix", "n_8k_30d", "filing_change", "worst252", "e63", "echo_q",
]
PRICE_FEATURES = FEATURE_NAMES[:7]
SEC_FEATURES = ["n_8k_30d", "filing_change"]
HISTORY_FEATURES = ["worst252", "e63", "echo_q"]
ECHO_LAGS = (63, 126, 189, 252)


# -----------------------------------------------------------------------------
# price factors (one stock)
# -----------------------------------------------------------------------------
def price_factors(bars):
    """Daily bars of ONE stock (columns close, high, low, volume; sorted by time) ->
    DataFrame with the 7 price factors, same index."""
    c, h, l, v = bars["close"], bars["high"], bars["low"], bars["volume"]
    r = np.log(c).diff()
    f = pd.DataFrame(index=bars.index)
    f["vol_20d"] = r.rolling(20).std() * np.sqrt(252)
    f["downside_vol_20d"] = np.sqrt((r.clip(upper=0) ** 2).rolling(20).mean()) * np.sqrt(252)
    f["kurtosis_60d"] = r.rolling(60).kurt()
    f["momentum_60d"] = c / c.shift(60) - 1
    f["reversal_5d"] = c / c.shift(5) - 1
    f["range_5d"] = ((h - l) / c).rolling(5).mean()
    f["abn_volume"] = np.log((v.rolling(5).mean() + 1) / (v.rolling(60).mean() + 1))
    return f


def next_day_return(bars, target_mode="close_to_close"):
    """Training target Y_t: close_to_close = close(t+1)/close(t) - 1, overnight = open(t+1)/close(t) - 1."""
    if target_mode == "overnight":
        return bars["open"].shift(-1) / bars["close"] - 1
    return bars["close"].shift(-1) / bars["close"] - 1


# -----------------------------------------------------------------------------
# history factors (one stock)
# -----------------------------------------------------------------------------
def history_factors(r):
    """worst252, e63, echo_q of ONE stock.

    r: simple return realised on each row's day (row t -> return of day t), consecutive rows.
       Training passes Y shifted by one row (Y_{t-1} = return of day t); QuantConnect passes
       close.pct_change(). The return realised k trading days before the loss day t+1 is
       r.shift(k - 1).
    """
    out = pd.DataFrame(index=r.index)
    out["worst252"] = (-r).rolling(252, min_periods=60).max()        # k = 1..252
    abs_r = r.abs()

    def window_max(center):                                          # k = center-3 .. center+3
        cols = [abs_r.shift(k - 1) for k in range(center - 3, center + 4)]
        return pd.concat(cols, axis=1).max(axis=1)                   # NaN only if all 7 missing

    windows = pd.concat([window_max(c) for c in ECHO_LAGS], axis=1)
    out["e63"] = windows.iloc[:, 0]
    out["echo_q"] = windows.mean(axis=1)                             # windows that exist so far
    return out


def add_history_features(df):
    """Add worst252, e63, echo_q to a long dataset with columns date, ticker, Y (Y_t = return
    realised on day t+1). Returns the frame sorted by date (stocks interleaved)."""
    df = df.sort_values(["ticker", "date"], kind="stable").copy()
    r = df.groupby("ticker", sort=False)["Y"].shift(1)               # return realised on day t
    hist = pd.concat([history_factors(s) for _, s in r.groupby(df["ticker"], sort=False)])
    df[HISTORY_FEATURES] = hist.loc[df.index, HISTORY_FEATURES].to_numpy()
    return df.sort_values("date", kind="stable").reset_index(drop=True)


# -----------------------------------------------------------------------------
# alignment helpers (VIX, SEC table)
# -----------------------------------------------------------------------------
def align_asof(series, index):
    """Value of `series` at the last time <= each time in `index` (no look-ahead); NaN before
    the first value. Missing values in `series` are skipped."""
    if series is None:
        return pd.Series(np.nan, index=index)
    s = series.dropna().sort_index()
    if s.empty:
        return pd.Series(np.nan, index=index)
    pos = s.index.searchsorted(index, side="right") - 1
    vals = np.where(pos >= 0, s.to_numpy()[np.clip(pos, 0, None)], np.nan)
    return pd.Series(vals, index=index)


def load_sec_table(text):
    """CSV text with columns date, ticker, n_8k_30d, filing_change (value on date d = known at
    the close of d) -> {ticker: DataFrame indexed by date}."""
    df = pd.read_csv(StringIO(text))
    missing = [c for c in ["date", "ticker"] + SEC_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"SEC factor file is missing columns {missing}")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    return {t: g.set_index("date")[SEC_FEATURES].sort_index() for t, g in df.groupby("ticker")}


def align_sec(sec, dates):
    """SEC factors of one stock on each trading date: the latest row dated on or before that day.
    Dates before the first or after the last row of the table -> NaN (training median later)."""
    dates = pd.DatetimeIndex(dates).normalize()
    out = pd.DataFrame(np.nan, index=dates, columns=SEC_FEATURES)
    if sec is None or len(sec) == 0:
        return out
    inside = (dates >= sec.index[0]) & (dates <= sec.index[-1])
    for col in SEC_FEATURES:
        out[col] = np.where(inside, align_asof(sec[col], dates).to_numpy(), np.nan)
    return out


def bar_dates(index):
    """Trading date of QuantConnect daily bars, which are stamped with their END time (the
    close, or midnight after it): step back one second before taking the date."""
    return (pd.DatetimeIndex(index) - pd.Timedelta(seconds=1)).normalize()


# -----------------------------------------------------------------------------
# all 13 factors of one stock from its bars (used live; training assembles the same pieces)
# -----------------------------------------------------------------------------
def compute_factor_panel(bars, vix=None, sec=None, end_stamped=False):
    """Daily bars of ONE stock (open, high, low, close, volume) -> all 13 factors per day.

    vix: Series of VIX closes indexed by time (aligned as-of), or None -> NaN.
    sec: this stock's SEC table (from load_sec_table), or None -> NaN.
    end_stamped: True for QuantConnect daily bars (index = bar end time); False when the index
                 already holds trading dates (training data).
    NaN factors are replaced by the training median in the model's preprocessing.
    """
    bars = bars.sort_index()
    f = price_factors(bars)
    f["vix"] = align_asof(vix, f.index).to_numpy() if vix is not None else np.nan
    dates = bar_dates(f.index) if end_stamped else pd.DatetimeIndex(f.index).normalize()
    sec_vals = align_sec(sec, dates)
    for col in SEC_FEATURES:
        f[col] = sec_vals[col].to_numpy()
    hist = history_factors(bars["close"].pct_change())
    for col in HISTORY_FEATURES:
        f[col] = hist[col].to_numpy()
    return f[FEATURE_NAMES]
