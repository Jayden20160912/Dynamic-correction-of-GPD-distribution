"""
GPD-mixture tail-risk allocation strategy for QuantConnect (LEAN, Python).

Each trading day:
  1. take daily bars of the chosen stocks up to yesterday's close (day t)
  2. compute the 13 factors for every stock
  3. run the trained gated GPD-mixture network -> 5 parameters per stock, and the
     exceedance network -> p_u = P(loss > u)
  4. turn them into a catastrophic-risk score R = p_u * R_GPD, relative weights v and exposure c
  5. rebalance 30 minutes after today's open (day t+1); the rest stays in cash

How to run
  1. Locally:  python export_model_for_qc.py --artifacts ../artifacts
     -> writes gpd_mixture_model.json (weights + preprocessing statistics in one file;
        gpd_model.pt and preprocess.json are not needed on QuantConnect)
  2. Commit and push gpd_mixture_model.json and qc_sec_factors.csv to GitHub (public repository).
  3. New Python project on QuantConnect: paste this file as main.py and add a second file
     gpd_core.py (factors, model, formulas; QuantConnect limits each file to 32,000 characters).
     Edit the CONFIG section here,
     run a backtest. The model (MODEL_URL) and the SEC factors (SEC_FACTORS_URL) are downloaded
     from GitHub with self.download().

The network forward pass is written in numpy (no torch needed on QuantConnect);
export_model_for_qc.py checks that it matches the PyTorch model exactly.
"""

from AlgorithmImports import *

import json

import numpy as np
import pandas as pd

from gpd_core import *                      # factors, model, strategy formulas, helpers
from gpd_core import _dummy_bars

# =============================================================================
# CONFIG
# =============================================================================
# The 30 stocks covered by qc_sec_factors.csv
TICKERS = ["AAPL", "JPM", "XOM", "MRK", "PG", "DIS", "UNH", "NKE", "BA", "LULU",
           "ETSY", "CROX", "DAR", "CAKE", "ALB", "MTN", "POOL", "MOS", "ENPH", "BBWI",
           "SMP", "TILE", "HCSG", "ARLO", "PRDO", "MLKN", "LZB", "WGO", "HELE", "OI"]
# The model was trained on 2016-01-01 .. 2024-12-31 (data after 2023-03-30 was used for model
# selection), so 2025-01-01 onwards is genuinely out of sample.
START_DATE = (2025, 1, 1)
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

# ---- risk score: with or without the exceedance probability p_u ----
USE_PU = True           # True:  R = p_u(x) * E[(L-d)^+ | L > u]  = E[(L-d)^+]  (unconditional)
                        # False: R = E[(L-d)^+ | L > u]              (GPD part only, as before)
# The parameters below depend on the scale of R, so each setting of USE_PU has its own set.
# Switching USE_PU picks the matching set automatically (see select_risk_profile below);
# edit the numbers here, not further down.
#   GAMMA                   sensitivity of inverse-risk weighting
#   DELTA                   numerical stability only; must stay far below the smallest R
#   R_TARGET                fixed risk budget, or None = calibrate from the CALIB_DAYS before START_DATE
#                           (a manual value must match the scale of R: ~1e-4 with p_u, ~1e-3 without)
#   REBALANCE_TV_THRESHOLD  trade only if the TV distance of v exceeds this (0 = every day)
RISK_PROFILES = {
    True: dict(               # with p_u: R ~ 5e-5 for a typical stock, smallest R ~ 1e-7
        GAMMA=1.0,
        DELTA=1e-9,
        R_TARGET=None,
        REBALANCE_TV_THRESHOLD=0.12,
    ),
    False: dict(              # without p_u: R ~ 7e-4 for a typical stock, R >= pi_min * 0.0074 = 2.2e-4
        GAMMA=1.0,
        DELTA=1e-6,
        R_TARGET=None,
        REBALANCE_TV_THRESHOLD=0.12,
    ),
}


def select_risk_profile(use_pu):
    """Return (GAMMA, DELTA, R_TARGET, REBALANCE_TV_THRESHOLD) for the chosen USE_PU setting."""
    if use_pu not in RISK_PROFILES:
        raise ValueError(f"USE_PU must be True or False, got {use_pu!r}")
    prof = RISK_PROFILES[use_pu]
    if prof["GAMMA"] < 0 or prof["DELTA"] < 0 or prof["REBALANCE_TV_THRESHOLD"] < 0:
        raise ValueError(f"GAMMA, DELTA and REBALANCE_TV_THRESHOLD must be >= 0: {prof}")
    if prof["R_TARGET"] is not None and prof["R_TARGET"] <= 0:
        raise ValueError(f"R_TARGET must be None or > 0, got {prof['R_TARGET']}")
    return prof["GAMMA"], prof["DELTA"], prof["R_TARGET"], prof["REBALANCE_TV_THRESHOLD"]


GAMMA, DELTA, R_TARGET, REBALANCE_TV_THRESHOLD = select_risk_profile(USE_PU)

CALIB_DAYS = 252        # calibration window (trading days) when R_TARGET is None
TARGET_QUANTILE = 0.5   # R_target = this quantile of the full-investment risk B_t in the calibration window

# ---- how R_target is set (exposure c = min(1, R_target / B_t)) ----
R_TARGET_MODE = "rolling"   # "fixed":   one R_target for the whole backtest (R_TARGET, or calibrated
                            #            from the CALIB_DAYS before START_DATE at TARGET_QUANTILE)
                            # "rolling": every day R_target = ROLLING_QUANTILE of B over the previous
                            #            ROLLING_WINDOW trading days (past days only, no look-ahead), so
                            #            c only drops when risk is high relative to the last year
ROLLING_WINDOW = 252
ROLLING_QUANTILE = 0.7
ROLLING_MIN_DAYS = 60       # fewer past B values than this -> not enough history, use c = 1

# ---- market trend filter ----
TREND_FILTER = True         # True: if SPY's last close is below its TREND_SMA_DAYS-day average,
                            #       multiply the exposure by TREND_SCALE (GPD mode only)
TREND_SMA_DAYS = 200
TREND_SCALE = 0.5
REBALANCE_MINUTES_AFTER_OPEN = 30
HISTORY_BARS = 320      # daily bars needed per stock (echo_q looks back 255 returns)

# ---- turnover control ----
# Trade only when the relative weights v move enough. Distance between two days' weights:
#     TV(v_today, v_ref) = 0.5 * sum_i |v_today,i - v_ref,i|      (0 = identical, 1 = no overlap)
# Otherwise the current holdings are kept unchanged (no orders that day).
# REBALANCE_TV_THRESHOLD is set per USE_PU setting in RISK_PROFILES above.
# The TV distance only looks at the RELATIVE weights v. The total exposure c can change a lot while
# v stays almost the same (e.g. R rises for all stocks together), and without a second trigger the
# portfolio would then keep a stale exposure for weeks. So also rebalance when
#     | c_today - current invested fraction of the portfolio | > REBALANCE_C_THRESHOLD
REBALANCE_C_THRESHOLD = 0.05           # None = off (only the TV rule, the old behaviour)
REBALANCE_REFERENCE = "previous_day"   # "previous_day":   compare with yesterday's v
                                       # "last_rebalance": compare with v of the last actual rebalance
                                       #                   (catches slow drift that never jumps in one day)

# ---- comparison chart (strategy vs. 30-stock average vs. S&P 500) ----
COMPARISON_CHART = "Strategy vs Benchmarks"   # must not be a reserved chart name


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
        if USE_PU and not self.model.has_pu:
            raise ValueError(f"USE_PU = True but {MODEL_URL} has no exceedance network. Re-run "
                             f"export_model_for_qc.py (with pu_model.pt in artifacts/) and push "
                             f"the new gpd_mixture_model.json, or set USE_PU = False.")
        self.log(f"risk score: {'p_u * E[(L-d)+ | L>u]' if USE_PU else 'E[(L-d)+ | L>u] (no p_u)'}; "
                 f"USE_PU = {USE_PU} -> GAMMA = {GAMMA}, DELTA = {DELTA}, "
                 f"R_TARGET = {'auto' if R_TARGET is None else R_TARGET}, "
                 f"TV threshold = {REBALANCE_TV_THRESHOLD}")
        missing = [n for n in self.model.feature_names
                   if n not in compute_factor_panel(_dummy_bars()).columns]
        if missing:
            self.log(f"WARNING: factors not computed by this algorithm (left NaN -> training median): {missing}")

        self.sec = self.download_sec_factors()      # needed before calibration_B()

        if R_TARGET_MODE not in ("fixed", "rolling"):
            raise ValueError(f"unknown R_TARGET_MODE {R_TARGET_MODE!r}")
        self.B_history = []           # B_t of past days (rolling R_target)
        if MODE == "gpd" and (R_TARGET_MODE == "rolling" or R_TARGET is None):
            B_cal = self.calibration_B()
            self.B_history = [float(b) for b in B_cal.iloc[-ROLLING_WINDOW:]]
        if R_TARGET_MODE == "fixed":
            if R_TARGET is not None:
                self.r_target = R_TARGET
            elif MODE == "gpd":
                self.r_target = float(B_cal.quantile(TARGET_QUANTILE))
            else:
                self.r_target = np.inf        # benchmark modes do not use R_target
            self.log(f"R_target (fixed) = {self.r_target:.6g}")
        else:
            if R_TARGET is not None:
                self.log("NOTE: R_TARGET is ignored because R_TARGET_MODE = 'rolling'")
            self.r_target = self.rolling_r_target()
            self.log(f"R_target (rolling, q = {ROLLING_QUANTILE} of last {ROLLING_WINDOW} days) "
                     f"starts at {self.r_target:.6g}")
        self.log(f"trend filter: {'on' if TREND_FILTER else 'off'} "
                 f"(SPY < {TREND_SMA_DAYS}-day average -> exposure x {TREND_SCALE})")
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

        # ---- comparison chart: strategy vs. equal-weight 30 stocks vs. S&P 500 ----
        self.start_prices = {}        # first price seen for each stock / SPY (sets the scale to CASH)
        self.schedule.on(self.date_rules.every_day(self.spy),
                         self.time_rules.before_market_close(self.spy, 1),
                         self.plot_comparison)

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
    def calibration_B(self):
        """Full-investment risk B_t on each of the CALIB_DAYS trading days before START_DATE
        (only data that was available before the backtest). Seeds the fixed or rolling R_target."""
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
        panel["R"] = tail_risk_scores(self.model, X, D_LOSS, USE_PU)[0]
        B = panel.groupby(level=0)["R"].apply(
            lambda R: allocate_from_risk(R.to_numpy(), np.inf, GAMMA, DELTA)[3])
        self.log(f"calibration: {len(B)} days, B_t median {B.median():.6g}, "
                 f"10%-90% [{B.quantile(0.1):.6g}, {B.quantile(0.9):.6g}]")
        return B.sort_index()

    def rolling_r_target(self):
        """ROLLING_QUANTILE of B over the last ROLLING_WINDOW days (today excluded);
        inf (-> c = 1) while there are fewer than ROLLING_MIN_DAYS past values."""
        past = self.B_history[-ROLLING_WINDOW:]
        if len(past) < ROLLING_MIN_DAYS:
            return np.inf
        return float(np.quantile(past, ROLLING_QUANTILE))

    def trend_multiplier(self):
        """TREND_SCALE if SPY's last daily close is below its TREND_SMA_DAYS-day average, else 1."""
        if not TREND_FILTER:
            return 1.0
        try:
            px = series_from_history(self.history(self.spy, TREND_SMA_DAYS + 5, Resolution.DAILY))
        except Exception as e:
            self.debug(f"SPY history unavailable ({e}); trend filter off today")
            return 1.0
        px = px.dropna() if px is not None else None
        if px is None or len(px) < TREND_SMA_DAYS:
            return 1.0
        sma = float(px.iloc[-TREND_SMA_DAYS:].mean())
        return TREND_SCALE if float(px.iloc[-1]) < sma else 1.0

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
            R, R_base, R_heavy, p_u = tail_risk_scores(self.model, X, D_LOSS, USE_PU)
            if R_TARGET_MODE == "rolling":
                self.r_target = self.rolling_r_target()          # from past days only
            weights, cash, v, B, c = allocate_from_risk(R, self.r_target, GAMMA, DELTA)
            self.B_history.append(B)
            self.B_history = self.B_history[-ROLLING_WINDOW:]
            trend = self.trend_multiplier()
            weights, c = weights * trend, c * trend
            self.plot("GPD Invested", "trend multiplier", trend)
            # chart names must not clash with QuantConnect's built-in charts
            # ("Strategy Equity", "Exposure", "Benchmark", "Drawdown", "Capacity", ...)
            self.plot("GPD Invested", "invested c", c)
            self.plot("GPD Risk", "B (full investment)", B)
            if np.isfinite(self.r_target):
                self.plot("GPD Risk", "R_target", self.r_target)
            self.plot("GPD Heavy Share", "heavy share of R", float(np.mean(R_heavy / R)))
            if USE_PU:
                self.plot("GPD Exceedance", "mean p_u", float(np.mean(p_u)))
                self.plot("GPD Exceedance", "max p_u", float(np.max(p_u)))
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
            c_gap = abs(float(np.sum(weights)) - self.invested_fraction())
            self.plot("GPD Rebalance", "exposure gap |c - invested|", c_gap)
            c_moved = REBALANCE_C_THRESHOLD is not None and c_gap > REBALANCE_C_THRESHOLD
            if tv <= REBALANCE_TV_THRESHOLD and not c_moved:
                return                      # keep current holdings, no orders today
        # first day, a stock entered / left, v moved more than the threshold, or the target
        # exposure c differs from the current exposure by more than REBALANCE_C_THRESHOLD

        targets = [PortfolioTarget(s, float(w)) for s, w in zip(syms, weights)]
        self.set_holdings(targets, liquidate_existing_holdings=True)
        self.last_rebalance_v = v_today
        self.n_rebalances += 1

    # ---------------------------------------------------------------- comparison chart
    def plot_comparison(self):
        """Plot, once a day just before the close, three lines that all start at CASH:
           - the strategy's portfolio value;
           - the 30 stocks' average price, each price divided by its own first price
             (= an equal-weight buy-and-hold of the 30 stocks);
           - SPY, the S&P 500 ETF.
        A plain average of raw prices is not used: it would be dominated by high-priced
        stocks (e.g. a $500 stock would count 50x more than a $10 stock)."""
        if self.is_warming_up:
            return
        rel = []
        for s in self.symbols:
            px = self.securities[s].price
            if px > 0:
                p0 = self.start_prices.setdefault(s, px)
                rel.append(px / p0)
        spy_px = self.securities[self.spy].price
        if not rel or spy_px <= 0:
            return
        spy0 = self.start_prices.setdefault(self.spy, spy_px)
        self.plot(COMPARISON_CHART, "GPD strategy", self.portfolio.total_portfolio_value)
        self.plot(COMPARISON_CHART, "30 stocks average (equal weight)", CASH * float(np.mean(rel)))
        self.plot(COMPARISON_CHART, "S&P 500 (SPY)", CASH * spy_px / spy0)

    def invested_fraction(self):
        """Current long exposure of the portfolio (market value of holdings / total value)."""
        total = float(self.portfolio.total_portfolio_value)
        return float(self.portfolio.total_holdings_value) / total if total > 0 else 0.0

    def on_end_of_algorithm(self):
        self.log(f"rebalanced on {self.n_rebalances} of {self.n_days} trading days "
                 f"(TV threshold {REBALANCE_TV_THRESHOLD}, exposure threshold {REBALANCE_C_THRESHOLD}, "
                 f"reference = {REBALANCE_REFERENCE})")

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
