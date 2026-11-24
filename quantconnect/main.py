"""
GPD-mixture tail-risk allocation strategy for QuantConnect (LEAN, Python).

Each trading day:
  1. take daily bars of the chosen stocks up to yesterday's close (day t)
  2. compute the 13 factors for every stock                      (factor_core.py)
  3. run both networks: 5 GPD-mixture parameters + p_u per stock  (tail_model.py)
  4. risk score R = p_u * E[(L - d)^+ | L > u] -> relative weights v and total exposure c
                                                                  (strategy_core.py)
  5. rebalance 30 minutes after today's open (day t+1); the rest stays in cash

The three shared files are the SAME files the training code uses -- one implementation of the
factors, the forward pass and the allocation rule for the whole project.

How to run
  1. Locally: train ("TQT research machine learning part.py" writes quantconnect/gpd_mixture_model.json)
     or re-export with  python export_model_for_qc.py --artifacts ../artifacts
  2. QuantConnect -> Object Store: upload gpd_mixture_model.json (key MODEL_KEY below)
     and, optionally, qc_sec_factors.csv (key SEC_KEY)
  3. New Python project: main.py = this file, plus three more files copied unchanged from the
     repository root: factor_core.py, tail_model.py, strategy_core.py. Edit CONFIG, run a backtest.
"""

from AlgorithmImports import *

import numpy as np
import pandas as pd

from factor_core import FEATURE_NAMES, compute_factor_panel, load_sec_table
from strategy_core import allocate_from_risk
from tail_model import TwoStageTailModel


# =============================================================================
# CONFIG
# =============================================================================
TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "JNJ", "XOM", "PG"]
# The model was trained on 2016-01-01 .. 2024-12-31 (data after 2023-03-30 was used for model
# selection), so 2025-01-01 onwards is genuinely out of sample.
START_DATE = (2025, 1, 1)
END_DATE = (2026, 10, 30)
CASH = 1_000_000

MODEL_KEY = "gpd_mixture_model.json"   # Object Store key of the exported model
SEC_KEY = "qc_sec_factors.csv"         # Object Store key of n_8k_30d / filing_change (qc_sec_factors.py)

MODE = "gpd"            # "gpd" = this strategy; "equal" / "inverse_vol" = benchmarks on the same stocks
D_LOSS = 0.07           # disaster loss threshold d
USE_PU = True           # True:  R = p_u(x) * E[(L-d)^+ | L > u] = E[(L-d)^+]  (unconditional)
                        # False: R = E[(L-d)^+ | L > u]  (GPD part only; the old behaviour)
GAMMA = 1.0             # sensitivity of inverse-risk weighting
DELTA = 1e-6            # numerical stability
R_TARGET = None         # fixed risk budget, or None = calibrate from the CALIB_DAYS before START_DATE
CALIB_DAYS = 252        # calibration window (trading days) when R_TARGET is None
TARGET_QUANTILE = 0.5   # R_target = this quantile of the full-investment risk B_t in the calibration window
REBALANCE_MINUTES_AFTER_OPEN = 30
HISTORY_BARS = 320      # daily bars needed per stock (echo_q looks back 255 returns)


# =============================================================================
# 1-3. FACTORS, MODEL, ALLOCATION: factor_core.py, tail_model.py, strategy_core.py
# =============================================================================
# Same files as in the training repository (do not edit copies here).
# SEC factors: qc_sec_factors.csv in the Object Store (SEC_KEY); value on date d = known at the
# close of d. Missing file / stocks / dates -> NaN -> training median in the preprocessing.


# =============================================================================
# 4. HISTORY HELPERS  (QuantConnect history DataFrame -> factor matrix)
# =============================================================================

def factor_panel_from_history(hist, symbols, vix, sec_tables=None):
    """QuantConnect daily history (MultiIndex symbol, time) -> {symbol: factor DataFrame}."""
    panels = {}
    for sym in symbols:
        try:
            bars = hist.loc[sym]
        except KeyError:
            continue
        bars = bars[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
        if len(bars) >= 2:
            sec = (sec_tables or {}).get(sym.value)
            panels[sym] = compute_factor_panel(bars, vix, sec, end_stamped=True)
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

        if not self.object_store.contains_key(MODEL_KEY):
            raise ValueError(f"Object Store key '{MODEL_KEY}' not found. Upload the file written by "
                             f"export_model_for_qc.py first.")
        self.model = TwoStageTailModel.from_json(self.object_store.read(MODEL_KEY))
        if not 0 <= self.model.u < D_LOSS:
            raise ValueError(f"need 0 <= u < d, got u={self.model.u}, d={D_LOSS}")
        self.log(f"model loaded: u = {self.model.u:.4f}, pi_min = {self.model.pi_min}, "
                 f"p_u network: {self.model.has_pu}, features = {self.model.feature_names}")
        if USE_PU and not self.model.has_pu:
            raise ValueError("USE_PU = True but the model file has no exceedance network; re-export "
                             "it with export_model_for_qc.py (needs pu_model.pt) or set USE_PU = False")
        if list(self.model.feature_names) != FEATURE_NAMES:
            raise ValueError("model feature order differs from factor_core.FEATURE_NAMES")
        missing = [n for n in self.model.feature_names
                   if n not in compute_factor_panel(_dummy_bars()).columns]
        if missing:
            self.log(f"WARNING: factors not computed by this algorithm (left NaN -> training median): {missing}")

        self.sec_tables = {}
        if self.object_store.contains_key(SEC_KEY):
            self.sec_tables = load_sec_table(self.object_store.read(SEC_KEY))
            no_sec = [t for t in TICKERS if t not in self.sec_tables]
            self.log(f"SEC factors loaded for {len(self.sec_tables)} tickers"
                     + (f"; missing (-> training median): {no_sec}" if no_sec else ""))
        else:
            self.log(f"WARNING: Object Store key '{SEC_KEY}' not found; n_8k_30d and filing_change "
                     f"-> training median")

        self.r_target = R_TARGET if R_TARGET is not None else self.calibrate_r_target()
        self.log(f"R_target = {self.r_target:.6g}")
        self.checked_distribution = False

        self.schedule.on(self.date_rules.every_day(self.spy),
                         self.time_rules.after_market_open(self.spy, REBALANCE_MINUTES_AFTER_OPEN),
                         self.rebalance)

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
        return factor_panel_from_history(hist, self.symbols, self.vix_series(bars), self.sec_tables)

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
        panel["R"] = self.model.risk_score(X, D_LOSS, use_pu=USE_PU)
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
            R, R_base, R_heavy, p_u = self.model.risk_score(X, D_LOSS, use_pu=USE_PU, return_parts=True)
            weights, cash, v, B, c = allocate_from_risk(R, self.r_target, GAMMA, DELTA)
            self.plot("Exposure", "invested c", c)
            self.plot("Risk", "B (full investment)", B)
            self.plot("Risk", "R_target", self.r_target)
            self.plot("Risk", "heavy share of R", float(np.mean(R_heavy / R)))
            if USE_PU:
                self.plot("Exceedance", "mean p_u", float(np.mean(p_u)))
        elif MODE == "equal":
            weights = np.full(len(syms), 1.0 / len(syms))
        elif MODE == "inverse_vol":
            vol = X[:, self.model.feature_names.index("vol_20d")]
            inv = np.where(np.isfinite(vol) & (vol > 0), 1.0 / vol, np.nan)
            inv = np.where(np.isnan(inv), np.nanmedian(inv), inv)
            weights = inv / inv.sum()
        else:
            raise ValueError(f"unknown MODE {MODE}")

        targets = [PortfolioTarget(s, float(w)) for s, w in zip(syms, weights)]
        self.set_holdings(targets, liquidate_existing_holdings=True)

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
