"""
Computation for the GPD-mixture strategy (no QuantConnect API, no settings):
factors, numpy forward pass of the networks, risk score / allocation formulas, history helpers.
main.py (settings + the QuantConnect algorithm) imports everything from here.
On QuantConnect: add this file to the same project, named gpd_core.py.
"""

from io import StringIO

import numpy as np
import pandas as pd

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
        pu = model_json.get("pu")                       # stage-1 exceedance network (optional)
        self.has_pu = pu is not None
        if self.has_pu:
            self.pu_w = {k: np.asarray(v, dtype=float) for k, v in pu["weights"].items()}
            self.pu_eps = float(pu["bn_eps"])

    def zscores(self, X):
        """Raw factors (n, 13) -> standardised factors before clipping (used for diagnostics)."""
        X = np.asarray(X, dtype=float)
        X = np.where(np.isnan(X), self.median, X)
        return (X - self.mean) / self.std

    def preprocess(self, X):
        """Same as training: median fill -> z-score with training stats -> clip to +/- clip_z."""
        return np.clip(self.zscores(X), -self.clip_z, self.clip_z)

    def _bn(self, x, prefix, w=None, eps=None):
        w = self.w if w is None else w
        eps = self.eps if eps is None else eps
        return ((x - w[prefix + ".running_mean"]) / np.sqrt(w[prefix + ".running_var"] + eps)
                * w[prefix + ".weight"] + w[prefix + ".bias"])

    def exceedance_prob(self, X_raw):
        """Stage 1: raw factors (n, 13) -> p_u = P(L > u | x), shape (n,).
        Linear(13,224,no bias)-BN-ReLU-Linear(224,88,no bias)-BN-ReLU-Linear(88,1) -> sigmoid.
        Uses the same preprocessing (median fill, z-score, clip) as the GPD network."""
        if not self.has_pu:
            raise ValueError("the model JSON has no exceedance network ('pu'); "
                             "re-run export_model_for_qc.py with pu_model.pt in artifacts/")
        w, eps = self.pu_w, self.pu_eps
        x = self.preprocess(X_raw)
        h = np.maximum(self._bn(x @ w["trunk.0.weight"].T, "trunk.1", w, eps), 0.0)
        h = np.maximum(self._bn(h @ w["trunk.3.weight"].T, "trunk.4", w, eps), 0.0)
        return _sigmoid((h @ w["out.weight"].T + w["out.bias"])[:, 0])

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


def tail_risk_scores(model, X, d, use_pu=True):
    """Raw factors (n, 13) -> risk score per stock.

        R_cond = E[(L - d)^+ | L > u]       from the GPD mixture (five parameters)
        R      = p_u * R_cond = E[(L - d)^+]  when use_pu, because (L - d)^+ = 0 whenever L <= u < d

    Returns R, base part, heavy part (both already multiplied by p_u) and p_u."""
    params = model.forward(X)
    R, base, heavy = mixture_tail_risk(params, d, model.u)
    p_u = model.exceedance_prob(X) if use_pu else np.ones(len(R))
    return p_u * R, p_u * base, p_u * heavy, p_u


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


def _dummy_bars(n=300):
    """Tiny synthetic bar frame, only used to list which factor columns compute_factor_panel produces."""
    idx = pd.date_range("2000-01-03", periods=n, freq="B")
    p = pd.Series(np.linspace(100, 110, n), index=idx)
    return pd.DataFrame({"open": p, "high": p * 1.01, "low": p * 0.99, "close": p, "volume": 1e6})
