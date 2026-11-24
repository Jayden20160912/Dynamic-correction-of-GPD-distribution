"""
Parity tests for the shared implementations (factor_core.py, tail_model.py, strategy_core.py).

Run:  python tests/test_parity.py        (or: python -m pytest tests/)

1. factor_core reproduces the factor formulas the dataset was built with (frozen reference
   copies below) -- price factors, target, history factors, the live panel used by QuantConnect.
2. tail_model: JSON round trip is exact; if artifacts/dataset.csv exists, the validation
   metrics logged by training (preprocess.json, pu_meta.json) are reproduced.
3. If torch is installed: tail_model matches the PyTorch networks of the training script.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import factor_core as fc      # noqa: E402
import strategy_core as sc    # noqa: E402
import tail_model as tm       # noqa: E402

ART = ROOT / "artifacts"


# ---------------------------------------------------------------- frozen reference formulas
def ref_price_factors(px, target_mode="close_to_close"):
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
    y = (o.shift(-1) if target_mode == "overnight" else c.shift(-1)) / c - 1
    return f, y


def ref_add_history_features(df):
    df = df.sort_values(["ticker", "date"], kind="stable").copy()
    g = df.groupby("ticker", sort=False)["Y"]
    past = lambda k: g.shift(k)
    loss = -past(1)
    df["worst252"] = loss.groupby(df["ticker"]).transform(lambda s: s.rolling(252, min_periods=60).max())

    def window_max(center):
        return pd.concat([past(k).abs() for k in range(center - 3, center + 4)], axis=1).max(axis=1)

    windows = pd.concat([window_max(c) for c in (63, 126, 189, 252)], axis=1)
    df["e63"] = windows.iloc[:, 0]
    df["echo_q"] = windows.mean(axis=1)
    return df.sort_values("date", kind="stable").reset_index(drop=True)


def ref_qc_history(close):
    """The previous QuantConnect formulas for the history factors (rolling-window version)."""
    abs_r = close.pct_change().abs()
    ew = lambda lag: abs_r.rolling(7, min_periods=1).max().shift(lag - 4)
    return ((-close.pct_change()).rolling(252, min_periods=60).max(), ew(63),
            pd.concat([ew(L) for L in (63, 126, 189, 252)], axis=1).mean(axis=1))


def synthetic_bars(rng, n, start="2015-01-02"):
    idx = pd.bdate_range(start, periods=n)
    c = 50 * np.exp(np.cumsum(rng.standard_t(3, n) * 0.02))
    return pd.DataFrame({"open": c * (1 + rng.normal(0, .005, n)), "high": c * (1 + rng.random(n) * .03),
                         "low": c * (1 - rng.random(n) * .03), "close": c,
                         "volume": rng.lognormal(13, 0.6, n)}, index=idx)


def _same(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert (np.isnan(a) == np.isnan(b)).all(), "NaN pattern differs"
    assert np.allclose(a[~np.isnan(a)], b[~np.isnan(b)], rtol=0, atol=1e-12), "values differ"


# ---------------------------------------------------------------- 1. factors
def test_price_factors_and_target():
    rng = np.random.default_rng(0)
    for _ in range(3):
        b = synthetic_bars(rng, 700)
        f_ref, y_ref = ref_price_factors(b)
        f_new = fc.price_factors(b)
        for c in fc.PRICE_FEATURES:
            _same(f_ref[c], f_new[c])
        _same(y_ref, fc.next_day_return(b))
        _same(ref_price_factors(b, "overnight")[1], fc.next_day_return(b, "overnight"))


def test_history_factors_dataset_path():
    rng = np.random.default_rng(1)
    parts = []
    for t in "ABCD":
        b = synthetic_bars(rng, int(rng.integers(300, 800)))
        y = fc.next_day_return(b)[rng.random(len(b)) > 0.02]          # rows with gaps
        parts.append(pd.DataFrame({"date": y.index, "ticker": t, "Y": y.values}))
    df = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=0)
    ref, new = ref_add_history_features(df), fc.add_history_features(df)
    assert (ref[["date", "ticker"]].values == new[["date", "ticker"]].values).all()
    for c in fc.HISTORY_FEATURES:
        _same(ref[c], new[c])


def test_live_panel_matches_training_formulas():
    """compute_factor_panel (QuantConnect) = training formulas on the same bars."""
    rng = np.random.default_rng(2)
    b = synthetic_bars(rng, 600)
    end = b.copy()
    end.index = end.index + pd.Timedelta(hours=16)                      # QC end-stamped bars
    vix = pd.Series(rng.uniform(10, 40, 640), index=pd.bdate_range("2014-12-01", periods=640))
    cal = pd.date_range("2014-12-01", "2017-12-31")
    sec = pd.DataFrame({"n_8k_30d": rng.integers(0, 4, len(cal)).astype(float),
                        "filing_change": rng.random(len(cal))}, index=cal)
    p_end = fc.compute_factor_panel(end, vix.set_axis(vix.index + pd.Timedelta(hours=16)), sec, end_stamped=True)
    p_day = fc.compute_factor_panel(b, vix, sec, end_stamped=False)
    f_ref, _ = ref_price_factors(b)
    for c in fc.PRICE_FEATURES:
        _same(p_end[c], f_ref[c])
        _same(p_day[c], f_ref[c])
    _same(p_end["vix"], vix.reindex(b.index, method="ffill"))
    _same(p_end["n_8k_30d"], sec["n_8k_30d"].reindex(b.index))
    for col, ref in zip(fc.HISTORY_FEATURES, ref_qc_history(b["close"])):
        _same(p_end[col], ref)
    assert list(p_end.columns) == fc.FEATURE_NAMES


# ---------------------------------------------------------------- 2. model
def _model():
    if not (ART / "gpd_model.pt").exists():
        return None
    return tm.TwoStageTailModel.from_artifacts(ART, require_pu=False)


def test_json_round_trip():
    m = _model()
    if m is None:
        print("  (no artifacts: skipped)")
        return
    rng = np.random.default_rng(3)
    X = m.mean + m.std * rng.normal(scale=2, size=(2000, len(m.mean)))
    X[rng.random(X.shape) < 0.05] = np.nan
    back = tm.TwoStageTailModel.from_json(m.to_json())
    assert np.array_equal(back.gpd_params(X), m.gpd_params(X))
    if m.has_pu:
        assert np.array_equal(back.exceedance_prob(X), m.exceedance_prob(X))
    qc_json = ROOT / "quantconnect" / "gpd_mixture_model.json"
    if qc_json.exists():
        q = tm.TwoStageTailModel.from_json(qc_json)
        assert np.allclose(q.gpd_params(X), m.gpd_params(X), rtol=1e-12, atol=0), \
            "quantconnect/gpd_mixture_model.json is not the model in artifacts/ -- re-export it"


def test_reproduces_logged_validation_metrics():
    m = _model()
    if m is None or not (ART / "dataset.csv").exists():
        print("  (no artifacts/dataset.csv: skipped)")
        return
    df = pd.read_csv(ART / "dataset.csv", parse_dates=["date"])
    val = df[df["date"] >= pd.Timestamp(m.meta["train_val_split_date"])]
    L = -val["Y"].to_numpy()
    exc = L > m.u
    P = m.gpd_params(val[exc])
    y = L[exc] - m.u
    rec = m.meta["results"]["validation"]
    assert abs(-tm.mixture_logpdf(P, y).mean() - rec["nll"]) < 1e-4
    S = tm.mixture_survival(P, y)
    for a, v in rec["observed_share_S_below"].items():
        assert abs(np.mean(S < float(a)) - v) < 1e-6, f"S<{a}"
    if m.has_pu and (ART / "pu_meta.json").exists():
        pr = json.loads((ART / "pu_meta.json").read_text())["results"]["validation"]
        p = m.exceedance_prob(val)
        e = exc.astype(float)
        ll = -np.mean(e * np.log(np.clip(p, 1e-7, 1)) + (1 - e) * np.log(np.clip(1 - p, 1e-7, 1)))
        assert abs(ll - pr["logloss"]) < 1e-5 and abs(tm.auc_score(p, e) - pr["auc"]) < 1e-6


def test_matches_torch_networks():
    m = _model()
    try:
        import torch
    except ImportError:
        print("  (torch not installed: skipped)")
        return
    if m is None:
        return
    spec = importlib.util.spec_from_file_location("tr", ROOT / "TQT research machine learning part.py")
    tr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tr)
    rng = np.random.default_rng(4)
    z = rng.normal(size=(3000, len(m.mean))).clip(-5, 5)
    x = torch.from_numpy(z.astype("float32"))
    with torch.no_grad():
        P_t = tr.load_gpd_model(ART / "gpd_model.pt", m.scale, m.pi_min)(x).double().numpy()
    assert np.max(np.abs(m.gpd_params_z(z) - P_t) / np.maximum(np.abs(P_t), 1e-12)) < 1e-4
    if m.has_pu:
        p_t = tr.load_pu_model(ART / "pu_model.pt").predict_proba(x).double().numpy()
        assert np.max(np.abs(m.exceedance_prob_z(z) - p_t) / p_t) < 1e-4


# ---------------------------------------------------------------- 3. allocation
def test_allocation_rule():
    R = np.array([1e-4, 2e-4, 4e-4])
    w, cash, v, B, c = sc.allocate_from_risk(R, R_target=1e-4, gamma=1.0, delta=1e-12)
    assert np.isclose(v.sum(), 1) and v[0] > v[1] > v[2] and np.isclose(B, np.sum(v * R))
    assert np.isclose(c, min(1, 1e-4 / B)) and np.isclose(w.sum() + cash, 1)


if __name__ == "__main__":
    tests = [(k, f) for k, f in sorted(globals().items()) if k.startswith("test_") and callable(f)]
    for name, f in tests:
        f()
        print(f"ok  {name}")
    print(f"all {len(tests)} tests passed")
