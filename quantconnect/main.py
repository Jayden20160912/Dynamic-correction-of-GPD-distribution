"""
GPD-mixture tail-risk allocation strategy for QuantConnect (LEAN, Python).

Each trading day:
  1. take daily bars of the chosen stocks up to yesterday's close (day t)
  2. compute the 13 factors for every stock
  3. run the trained gated GPD-mixture network -> 5 parameters per stock
  4. turn them into a catastrophic-risk score R, relative weights v and total exposure c
  5. rebalance 30 minutes after today's open (day t+1); the rest stays in cash

How to run
  1. Locally:  python export_model_for_qc.py --artifacts ../artifacts
     -> writes gpd_mixture_model.json (weights + preprocessing statistics in one file;
        gpd_model.pt and preprocess.json are not needed on QuantConnect)
  2. Commit and push gpd_mixture_model.json and qc_sec_factors.csv to GitHub (public repository).
  3. New Python project on QuantConnect, paste this file as main.py, edit the CONFIG section,
     run a backtest. The model (MODEL_URL) and the SEC factors (SEC_FACTORS_URL) are downloaded
     from GitHub with self.download().

The network forward pass is written in numpy (no torch needed on QuantConnect);
export_model_for_qc.py checks that it matches the PyTorch model exactly.
"""

from AlgorithmImports import *

import json
from io import StringIO

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================
# The 30 stocks covered by qc_sec_factors.csv
TICKERS = ["AAPL", "JPM", "XOM", "MRK", "PG", "DIS", "UNH", "NKE", "BA", "LULU",
           "ETSY", "CROX", "DAR", "CAKE", "ALB", "MTN", "POOL", "MOS", "ENPH", "BBWI",
           "SMP", "TILE", "HCSG", "ARLO", "PRDO", "MLKN", "LZB", "WGO", "HELE", "OI"]
# The model was trained on 2016-01-01 .. 2024-12-31 (data after 2023-03-30 was used for model
# selection), so 2025-01-01 onwards is genuinely out of sample.
START_DATE = (2023, 1, 1)
END_DATE = (2026, 10, 30)
CASH = 10000000

# Files downloaded from GitHub at start-up ("raw" links of a PUBLIC repository).
# GITHUB_BRANCH is the branch the files are read from; a commit hash pins an exact version.
GITHUB_RAW = "https://raw.githubusercontent.com/Jayden20160912/Dynamic-correction-of-GPD-distribution"
GITHUB_BRANCH = "Neural-Networks,-Loss-Function,-and-Model-Training-Algorithm"
# exported model (written by export_model_for_qc.py)
MODEL_URL = f"{GITHUB_RAW}/{GITHUB_BRANCH}/quantconnect/gpd_mixture_model.json"
# SEC factors per stock and calendar day: columns date, ticker, n_8k_30d, filing_change
# (value on date d = what was known at the close of d). Stocks / dates not in the file fall back
# to the training median.
SEC_FACTORS_URL = f"{GITHUB_RAW}/{GITHUB_BRANCH}/qc_sec_factors.csv"

MODE = "gpd"            # "gpd" = this strategy; "equal" / "inverse_vol" = benchmarks on the same stocks
D_LOSS = 0.07           # disaster loss threshold d
GAMMA = 1.0             # sensitivity of inverse-risk weighting
DELTA = 1e-6            # numerical stability
R_TARGET = None         # fixed risk budget, or None = calibrate from the CALIB_DAYS before START_DATE
CALIB_DAYS = 252        # calibration window (trading days) when R_TARGET is None
TARGET_QUANTILE = 0.5   # R_target = this quantile of the full-investment risk B_t in the calibration window
REBALANCE_MINUTES_AFTER_OPEN = 30
HISTORY_BARS = 320      # daily bars needed per stock (echo_q looks back 255 returns)

# ---- turnover control ----
# Trade only when the relative weights v move enough. Distance between two days' weights:
#     TV(v_today, v_ref) = 0.5 * sum_i |v_today,i - v_ref,i|      (0 = identical, 1 = no overlap)
# Otherwise the current holdings are kept unchanged (no orders that day).
REBALANCE_TV_THRESHOLD = 0.12        # 0 = rebalance every day
REBALANCE_REFERENCE = "previous_day"   # "previous_day":   compare with yesterday's v
                                       # "last_rebalance": compare with v of the last actual rebalance
                                       #                   (catches slow drift that never jumps in one day)


# =============================================================================
# 1. FACTORS  (same definitions as features.py in the training repository)
# =============================================================================
# t = the factor date; every factor only uses bars up to and including the close of day t.
# Prices are split- and dividend-adjusted, like the Alpaca bars used in training
# (adjustment="all"), which matches QuantConnect's default adjusted data.
#
#   price_factors() in features.py      -> log returns for the three return-distribution factors
#   add_history_features() in features.py -> simple close-to-close returns (the training target Y)
#
# n_8k_30d and filing_change come from SEC EDGAR in training (8-K count, 10-K/10-Q text change).
# QuantConnect has no SEC filings, so they are read from qc_sec_factors.csv (SEC_FACTORS_URL).
# Stocks or dates not covered by the file are NaN, which the preprocessing replaces with the
# training median (1.0 and 0.068).
SEC_COLS = ["n_8k_30d", "filing_change"]


def load_sec_factors(csv_text):
    """CSV text (date, ticker, n_8k_30d, filing_change) -> {ticker: DataFrame indexed by date}."""
    df = pd.read_csv(StringIO(csv_text))
    missing = [c for c in ["date", "ticker"] + SEC_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"SEC factor file is missing columns {missing}; found {list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    return {t: g.set_index("date")[SEC_COLS].sort_index() for t, g in df.groupby("ticker")}


def align_sec(sec, index):
    """SEC factors of one stock on each bar date in `index`: the latest row dated on or before that
    day (no look-ahead). Days before the first or after the last row of the file -> NaN."""
    out = pd.DataFrame(np.nan, index=index, columns=SEC_COLS)
    if sec is None or sec.empty:
        return out
    days = pd.DatetimeIndex(index).normalize()
    inside = (days >= sec.index[0]) & (days <= sec.index[-1])
    for col in SEC_COLS:
        vals = align_asof(sec[col], days).to_numpy()
        out[col] = np.where(inside, vals, np.nan)
    return out

def echo_window_max(abs_r, lag):
    """max |r_tau| for tau in [t+1-lag-3, t+1-lag+3], i.e. k = lag-3 .. lag+3 trading days before
    the loss day t+1 (features.py window_max). Like pandas' row-wise max there, missing days are
    skipped: the window is NaN only if all 7 days are missing."""
    return abs_r.rolling(7, min_periods=1).max().shift(lag - 4)


def compute_factor_panel(bars, vix=None, sec=None):
    """Daily bars of ONE stock (index = time; open, high, low, close, volume) -> factors per day.

    vix: optional Series of VIX closes indexed by time (aligned as-of to the stock's dates).
    sec: optional DataFrame of this stock's SEC factors indexed by date (from load_sec_factors).
    Returns a DataFrame with one row per day and one column per factor.
    """
    bars = bars.sort_index()
    close, high, low, volume = bars["close"], bars["high"], bars["low"], bars["volume"]
    log_r = np.log(close).diff()            # features.py price_factors: r = log(c).diff()
    simple_r = close.pct_change()           # features.py history features: Y = c(t+1)/c(t) - 1

    f = pd.DataFrame(index=bars.index)
    # ---- price factors: features.price_factors ----
    f["vol_20d"] = log_r.rolling(20).std() * np.sqrt(252)
    f["downside_vol_20d"] = np.sqrt((log_r.clip(upper=0) ** 2).rolling(20).mean()) * np.sqrt(252)
    f["kurtosis_60d"] = log_r.rolling(60).kurt()                    # excess kurtosis
    f["momentum_60d"] = close / close.shift(60) - 1
    f["reversal_5d"] = close / close.shift(5) - 1
    f["range_5d"] = ((high - low) / close).rolling(5).mean()
    f["abn_volume"] = np.log((volume.rolling(5).mean() + 1) / (volume.rolling(60).mean() + 1))
    # ---- market factor: FRED VIXCLS in training, forward-filled over holidays ----
    f["vix"] = align_asof(vix, f.index) if vix is not None else np.nan
    # ---- SEC factors: from qc_sec_factors.csv (NaN where not covered -> training median) ----
    sec_vals = align_sec(sec, f.index)
    f["n_8k_30d"] = sec_vals["n_8k_30d"].to_numpy()
    f["filing_change"] = sec_vals["filing_change"].to_numpy()
    # ---- history factors: features.add_history_features ----
    f["worst252"] = (-simple_r).rolling(252, min_periods=60).max()
    windows = pd.concat([echo_window_max(simple_r.abs(), L) for L in (63, 126, 189, 252)], axis=1)
    f["e63"] = windows.iloc[:, 0]
    f["echo_q"] = windows.mean(axis=1)      # mean of the windows that exist so far (skips NaN)
    return f


def align_asof(series, index):
    """Value of `series` at the last time <= each time in `index` (no look-ahead)."""
    s = series.dropna().sort_index()
    if s.empty:
        return pd.Series(np.nan, index=index)
    pos = s.index.searchsorted(index, side="right") - 1
    vals = np.where(pos >= 0, s.to_numpy()[np.clip(pos, 0, None)], np.nan)
    return pd.Series(vals, index=index)


# =============================================================================
# 2. MODEL  (numpy forward pass of the gated GPD-mixture network)
# =============================================================================
PARAM_COLS = ["xi_b", "beta_b", "xi_h", "beta_h", "pi"]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softplus(x):
    return np.logaddexp(0.0, x)


class GPDMixtureModel:
    """Loads the JSON written by export_model_for_qc.py and reproduces the PyTorch forward pass:

    trunk  Linear(13,224,no bias)-BN-ReLU-Linear(224,88,no bias)-BN-ReLU -> h
    base   Linear(88,2):  xi_b = 0.01 + 0.34*sigmoid, beta_b = s*softplus + 1e-6
    gate   Linear(88,32,no bias)-BN-ReLU-Linear(32,1): pi = pi_min + (1-pi_min)*sigmoid
    heavy  global scalars: xi_h = 0.35 + 0.45*sigmoid(theta_xi), beta_h = s*softplus(theta_beta) + 1e-6
    BatchNorm uses the trained moving statistics (inference mode).
    """

    def __init__(self, model_json):
        m = model_json["meta"]
        self.feature_names = list(m["feature_names"])
        self.u = float(m["threshold_u_loss"])
        self.pi_min = float(m["pi_min"])
        self.eps = float(m["bn_eps"])
        self.median = np.asarray(m["feature_median_fill"], dtype=float)
        self.mean = np.asarray(m["feature_mean"], dtype=float)
        self.std = np.asarray(m["feature_std"], dtype=float)
        self.clip_z = float(m["clip_z"])
        self.w = {k: np.asarray(v, dtype=float) for k, v in model_json["weights"].items()}
        self.scale = float(np.asarray(self.w["scale"]).reshape(()))

    def zscores(self, X):
        """Raw factors (n, 13) -> standardised factors before clipping (used for diagnostics)."""
        X = np.asarray(X, dtype=float)
        X = np.where(np.isnan(X), self.median, X)
        return (X - self.mean) / self.std

    def preprocess(self, X):
        """Same as training: median fill -> z-score with training stats -> clip to +/- clip_z."""
        return np.clip(self.zscores(X), -self.clip_z, self.clip_z)

    def _bn(self, x, prefix):
        w = self.w
        return ((x - w[prefix + ".running_mean"]) / np.sqrt(w[prefix + ".running_var"] + self.eps)
                * w[prefix + ".weight"] + w[prefix + ".bias"])

    def forward(self, X_raw):
        """Raw factor matrix (n, 13) in feature_names order -> parameters (n, 5)."""
        w = self.w
        x = self.preprocess(X_raw)
        h = np.maximum(self._bn(x @ w["trunk.0.weight"].T, "trunk.1"), 0.0)
        h = np.maximum(self._bn(h @ w["trunk.3.weight"].T, "trunk.4"), 0.0)
        a = h @ w["base_head.weight"].T + w["base_head.bias"]
        xi_b = 0.01 + 0.34 * _sigmoid(a[:, 0])
        beta_b = self.scale * _softplus(a[:, 1]) + 1e-6
        g = np.maximum(self._bn(h @ w["gate.0.weight"].T, "gate.1"), 0.0)
        g = (g @ w["gate.3.weight"].T + w["gate.3.bias"])[:, 0]
        pi = np.clip(self.pi_min + (1.0 - self.pi_min) * _sigmoid(g), 1e-6, 1 - 1e-6)
        ones = np.ones_like(xi_b)
        xi_h = (0.35 + 0.45 * _sigmoid(float(np.asarray(w["theta_xi"]).reshape(())))) * ones
        beta_h = (self.scale * _softplus(float(np.asarray(w["theta_beta"]).reshape(()))) + 1e-6) * ones
        return np.column_stack([xi_b, beta_b, xi_h, beta_h, pi])


# =============================================================================
# 3. STRATEGY  (five parameters -> risk score -> weights and exposure)
# =============================================================================

def gpd_expected_excess(xi, beta, a):
    """E[(Y - a)^+] of one GPD = P(Y > a) * E[Y - a | Y > a]; requires 0 < xi < 1."""
    survival = np.exp(-np.log1p(xi * a / beta) / xi)
    return survival * (beta + xi * a) / (1.0 - xi)


def mixture_tail_risk(params, d, u):
    """R = (1 - pi) * R_base + pi * R_heavy  (expectation is linear in the mixture weights)."""
    xi_b, beta_b, xi_h, beta_h, pi = np.asarray(params, dtype=float).T
    a = d - u
    base = (1.0 - pi) * gpd_expected_excess(xi_b, beta_b, a)
    heavy = pi * gpd_expected_excess(xi_h, beta_h, a)
    return base + heavy, base, heavy


def allocate_from_risk(R, R_target, gamma=1.0, delta=1e-6):
    """R (one value per stock, ONE day) -> weights, cash, v, B, c."""
    R = np.asarray(R, dtype=float)
    scores = -gamma * np.log(R + delta)          # inverse-risk weighting in log space
    scores = np.exp(scores - np.max(scores))
    v = scores / scores.sum()
    B = float(np.sum(v * R))                     # risk of the fully invested portfolio
    c = min(1.0, R_target / B) if B > 0 else 1.0 # total exposure
    weights = c * v
    return weights, 1.0 - weights.sum(), v, B, c


def total_variation(a, b):
    """Total variation distance 0.5 * sum_i |a_i - b_i| between two weight vectors given as
    Series indexed by symbol. Symbols missing from one side count as weight 0."""
    a, b = a.align(b, fill_value=0.0)
    return 0.5 * float(np.abs(a.to_numpy() - b.to_numpy()).sum())


# =============================================================================
# 4. HISTORY HELPERS  (QuantConnect history DataFrame -> factor matrix)
# =============================================================================

def ticker_of(sym):
    """Ticker string of a QuantConnect Symbol (or of a plain string)."""
    return str(getattr(sym, "value", sym)).upper()


def factor_panel_from_history(hist, symbols, vix, sec=None):
    """QuantConnect daily history (MultiIndex symbol, time) -> {symbol: factor DataFrame}.
    sec: {ticker: SEC factor DataFrame} from load_sec_factors, or None."""
    sec = sec or {}
    panels = {}
    for sym in symbols:
        try:
            bars = hist.loc[sym]
        except KeyError:
            continue
        bars = bars[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
        if len(bars) >= 2:
            panels[sym] = compute_factor_panel(bars, vix, sec.get(ticker_of(sym)))
    return panels


def latest_factor_matrix(panels, feature_names, min_history=61):
    """Last row of each stock's factor panel -> (symbols, X). Drops stocks whose last bar
    is older than the newest bar (no data yesterday) or that have too little history."""
    if not panels:
        return [], np.empty((0, len(feature_names)))
    newest = max(p.index[-1] for p in panels.values())
    syms, rows = [], []
    for sym, p in panels.items():
        if p.index[-1] != newest or len(p) < min_history:
            continue
        syms.append(sym)
        rows.append(p.iloc[-1].reindex(feature_names).to_numpy(dtype=float))
    return syms, np.array(rows, dtype=float).reshape(len(rows), len(feature_names))


def series_from_history(hist, column_candidates=("close", "value")):
    """Single-symbol history -> Series indexed by time."""
    if hist is None or len(hist) == 0:
        return None
    if isinstance(hist.index, pd.MultiIndex):
        hist = hist.droplevel(0)
    for col in column_candidates:
        if col in hist.columns:
            return hist[col].astype(float).sort_index()
    return None


# =============================================================================
# 5. THE ALGORITHM
# =============================================================================

class GPDMixtureTailRiskAlgorithm(QCAlgorithm):

    def initialize(self):
        self.set_start_date(*START_DATE)
        self.set_end_date(*END_DATE)
        self.set_cash(CASH)
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)
        self.settings.free_portfolio_value_percentage = 0.01   # keep 1% buffer for fees

        self.symbols = [self.add_equity(t, Resolution.MINUTE).symbol for t in TICKERS]
        self.spy = self.add_equity("SPY", Resolution.MINUTE).symbol
        self.set_benchmark(self.spy)
        self.vix = self.add_data(CBOE, "VIX", Resolution.DAILY).symbol

        self.model = GPDMixtureModel(self.download_model_json())
        if not 0 <= self.model.u < D_LOSS:
            raise ValueError(f"need 0 <= u < d, got u={self.model.u}, d={D_LOSS}")
        self.log(f"model loaded: u = {self.model.u:.4f}, features = {self.model.feature_names}")
        missing = [n for n in self.model.feature_names
                   if n not in compute_factor_panel(_dummy_bars()).columns]
        if missing:
            self.log(f"WARNING: factors not computed by this algorithm (left NaN -> training median): {missing}")

        self.sec = self.download_sec_factors()      # needed before calibrate_r_target()

        self.r_target = R_TARGET if R_TARGET is not None else self.calibrate_r_target()
        self.log(f"R_target = {self.r_target:.6g}")
        self.checked_distribution = False

        if REBALANCE_REFERENCE not in ("previous_day", "last_rebalance"):
            raise ValueError(f"unknown REBALANCE_REFERENCE {REBALANCE_REFERENCE}")
        self.prev_v = None            # yesterday's relative weights (Series indexed by symbol)
        self.last_rebalance_v = None  # relative weights at the last actual rebalance
        self.n_days = 0
        self.n_rebalances = 0

        self.schedule.on(self.date_rules.every_day(self.spy),
                         self.time_rules.after_market_open(self.spy, REBALANCE_MINUTES_AFTER_OPEN),
                         self.rebalance)

    # ---------------------------------------------------------------- model
    def download_model_json(self):
        """Download the exported model from MODEL_URL and check that it is the right file."""
        text = self.download(MODEL_URL)
        if not text or not text.lstrip().startswith("{"):
            raise ValueError(f"could not download the model from {MODEL_URL} "
                             f"(got: {str(text)[:80]!r}). Check that the URL is the raw link "
                             f"and that the repository is public.")
        model_json = json.loads(text)
        if model_json.get("format") != "gpd_mixture_numpy_v1":
            raise ValueError(f"{MODEL_URL} is not a model exported by export_model_for_qc.py")
        self.log(f"model downloaded from {MODEL_URL} ({len(text) / 1e3:.0f} KB)")
        return model_json

    def download_sec_factors(self):
        """Download qc_sec_factors.csv from SEC_FACTORS_URL -> {ticker: DataFrame indexed by date}."""
        text = self.download(SEC_FACTORS_URL)
        if not text or not text.lstrip().lower().startswith("date"):
            raise ValueError(f"could not download the SEC factors from {SEC_FACTORS_URL} "
                             f"(got: {str(text)[:80]!r}). Check that the file is pushed to "
                             f"GITHUB_BRANCH and that the repository is public.")
        sec = load_sec_factors(text)
        first = min(d.index[0] for d in sec.values()).date()
        last = max(d.index[-1] for d in sec.values()).date()
        self.log(f"SEC factors downloaded: {len(sec)} tickers, {first} to {last}")
        uncovered = [t for t in TICKERS if t.upper() not in sec]
        if uncovered:
            self.log(f"WARNING: no SEC factors for {uncovered} -> training median is used for them")
        return sec

    # ---------------------------------------------------------------- data
    def vix_series(self, bars):
        try:
            return series_from_history(self.history(self.vix, bars, Resolution.DAILY))
        except Exception as e:
            self.debug(f"VIX history unavailable ({e}); vix factor -> training median")
            return None

    def factor_panels(self, bars):
        hist = self.history(self.symbols, bars, Resolution.DAILY)
        if hist is None or len(hist) == 0:
            return {}
        return factor_panel_from_history(hist, self.symbols, self.vix_series(bars), self.sec)

    # ---------------------------------------------------------------- calibration
    def calibrate_r_target(self):
        """R_target = TARGET_QUANTILE of the full-investment risk B_t over the CALIB_DAYS
        trading days before START_DATE (only data that was available before the backtest)."""
        panels = self.factor_panels(HISTORY_BARS + CALIB_DAYS)
        frames = []
        for sym, p in panels.items():
            q = p.iloc[-CALIB_DAYS:].copy()
            q = q[q.index >= p.index[min(len(p) - 1, 60)]]           # need >= 60 days of history
            q["symbol"] = str(sym)
            frames.append(q)
        if not frames:
            raise ValueError("no history for calibration; set R_TARGET to a number instead")
        panel = pd.concat(frames)
        X = panel.reindex(columns=self.model.feature_names).to_numpy(dtype=float)  # missing -> NaN
        params = self.model.forward(X)
        panel["R"] = mixture_tail_risk(params, D_LOSS, self.model.u)[0]
        B = panel.groupby(level=0)["R"].apply(
            lambda R: allocate_from_risk(R.to_numpy(), np.inf, GAMMA, DELTA)[3])
        self.log(f"calibration: {len(B)} days, B_t median {B.median():.6g}, "
                 f"10%-90% [{B.quantile(0.1):.6g}, {B.quantile(0.9):.6g}]")
        return float(B.quantile(TARGET_QUANTILE))

    # ---------------------------------------------------------------- daily rebalance
    def rebalance(self):
        if self.is_warming_up:
            return
        syms, X = latest_factor_matrix(self.factor_panels(HISTORY_BARS), self.model.feature_names)
        if not syms:
            self.debug("no stocks with data today; skipping")
            return
        keep = [i for i, s in enumerate(syms)
                if self.securities[s].has_data and self.securities[s].price > 0]
        if not keep:
            return
        syms, X = [syms[i] for i in keep], X[keep]

        if not self.checked_distribution:
            self.check_factor_distribution(X)
            self.checked_distribution = True

        if MODE == "gpd":
            params = self.model.forward(X)
            R, R_base, R_heavy = mixture_tail_risk(params, D_LOSS, self.model.u)
            weights, cash, v, B, c = allocate_from_risk(R, self.r_target, GAMMA, DELTA)
            # chart names must not clash with QuantConnect's built-in charts
            # ("Strategy Equity", "Exposure", "Benchmark", "Drawdown", "Capacity", ...)
            self.plot("GPD Invested", "invested c", c)
            self.plot("GPD Risk", "B (full investment)", B)
            self.plot("GPD Risk", "R_target", self.r_target)
            self.plot("GPD Heavy Share", "heavy share of R", float(np.mean(R_heavy / R)))
        elif MODE == "equal":
            weights = np.full(len(syms), 1.0 / len(syms))
            v = weights
        elif MODE == "inverse_vol":
            vol = X[:, self.model.feature_names.index("vol_20d")]
            inv = np.where(np.isfinite(vol) & (vol > 0), 1.0 / vol, np.nan)
            inv = np.where(np.isnan(inv), np.nanmedian(inv), inv)
            weights = inv / inv.sum()
            v = weights
        else:
            raise ValueError(f"unknown MODE {MODE}")

        # ---- turnover control: trade only if v moved enough (see REBALANCE_TV_THRESHOLD) ----
        v_today = pd.Series(np.asarray(v, dtype=float), index=syms)
        ref = self.prev_v if REBALANCE_REFERENCE == "previous_day" else self.last_rebalance_v
        self.prev_v = v_today
        self.n_days += 1
        if ref is not None and set(ref.index) == set(syms):
            tv = total_variation(v_today, ref)
            self.plot("GPD Rebalance", "TV distance of v", tv)
            self.plot("GPD Rebalance", "threshold", REBALANCE_TV_THRESHOLD)
            if tv <= REBALANCE_TV_THRESHOLD:
                return                      # keep current holdings, no orders today
        # first day, a stock entered / left, or v moved more than the threshold -> rebalance

        targets = [PortfolioTarget(s, float(w)) for s, w in zip(syms, weights)]
        self.set_holdings(targets, liquidate_existing_holdings=True)
        self.last_rebalance_v = v_today
        self.n_rebalances += 1

    def on_end_of_algorithm(self):
        self.log(f"rebalanced on {self.n_rebalances} of {self.n_days} trading days "
                 f"(TV threshold {REBALANCE_TV_THRESHOLD}, reference = {REBALANCE_REFERENCE})")

    def check_factor_distribution(self, X):
        """Warn if live factors look very different from the training distribution
        (usually a sign that a factor is defined differently from features.py)."""
        z = self.model.zscores(X)
        for j, name in enumerate(self.model.feature_names):
            col = z[:, j]
            if np.all(np.isnan(X[:, j])):
                continue
            mean_abs = float(np.nanmean(np.abs(col)))
            clipped = float(np.mean(np.abs(col) > self.model.clip_z))
            if mean_abs > 3 or clipped > 0.2:
                self.log(f"WARNING factor '{name}': mean |z| = {mean_abs:.1f}, {clipped:.0%} clipped "
                         f"-> check that its definition matches training")


def _dummy_bars(n=300):
    """Tiny synthetic bar frame, only used to list which factor columns compute_factor_panel produces."""
    idx = pd.date_range("2000-01-03", periods=n, freq="B")
    p = pd.Series(np.linspace(100, 110, n), index=idx)
    return pd.DataFrame({"open": p, "high": p * 1.01, "low": p * 0.99, "close": p, "volume": 1e6})
