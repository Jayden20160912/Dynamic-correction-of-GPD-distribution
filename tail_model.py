"""
Two-stage tail model: the ONLY inference implementation (forward pass + risk measures).

    P(L > l | x) = p_u(x) * S(l - u | x),   l > u        (L = -return of the NEXT day)
    Stage 1  p_u(x) = P(L > u | x)                         artifacts/pu_model.pt
    Stage 2  S(y|x) = (1-pi) S_GPD(y; xi_b, beta_b) + pi S_GPD(y; xi_h, beta_h)   artifacts/gpd_model.pt

Used by the training script (risk measures, backtest, numpy-vs-torch check), evaluate_tail.py,
gpd_predict.py, quantconnect/export_model_for_qc.py, quantconnect/main.py and the notebook.
Only numpy (pandas optional). Reads the PyTorch .pt files directly -- torch is not needed --
or the JSON written by to_json() (QuantConnect). Preprocessing and BatchNorm are exactly the
training-time inference path (training statistics, BN moving statistics).

    from tail_model import TwoStageTailModel
    m = TwoStageTailModel.from_artifacts("artifacts")     # or .from_json(text_or_dict)
    m.predict(df)              # xi_b, beta_b, xi_h, beta_h, pi, p_u   (df = 13 RAW factors)
    m.tail_prob(df, 0.10)      # P(next-day loss > 10%)
    m.var(df, 0.01), m.es(df, 0.01)
    m.risk_score(df, d=0.07)   # E[(L-d)^+] = p_u * E[(L-d)^+ | L > u]

CLI: python tail_model.py --selftest | --csv factors.csv --out predictions.csv
Losses are positive numbers (0.07 = 7% drop); y = L - u is the excess loss.
"""
from __future__ import annotations

import argparse
import json
import pickle
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

PARAM_COLS = ["xi_b", "beta_b", "xi_h", "beta_h", "pi"]
JSON_FORMAT = "gpd_mixture_numpy_v1"


# =============================================================================
# 1. PyTorch state_dict (.pt zip) -> numpy, without torch
# =============================================================================
_DTYPES = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16,
           "LongStorage": np.int64, "IntStorage": np.int32, "BoolStorage": np.bool_}


class _Storage:
    def __init__(self, name):
        self.dtype = _DTYPES[name]


def _rebuild(storage, offset, size, stride, *args):
    n = storage.itemsize
    return np.lib.stride_tricks.as_strided(storage[offset:], tuple(size), tuple(s * n for s in stride)).copy()


def load_state_dict(path):
    """torch.save(state_dict) file -> OrderedDict {name: float64 array}."""
    path = Path(path)
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        root = names[0].split("/")[0]
        bo = "<"
        if f"{root}/byteorder" in names and z.read(f"{root}/byteorder").decode().strip() == "big":
            bo = ">"

        class U(pickle.Unpickler):
            def find_class(self, module, name):
                if (module, name) == ("collections", "OrderedDict"):
                    return OrderedDict
                if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
                    return _rebuild
                if module == "torch" and name in _DTYPES:
                    return _Storage(name)
                raise pickle.UnpicklingError(f"unsupported object in {path.name}: {module}.{name}")

            def persistent_load(self, pid):
                _, st, key, _loc, _n = pid
                return np.frombuffer(z.read(f"{root}/data/{key}"), dtype=np.dtype(st.dtype).newbyteorder(bo))

        state = U(z.open(f"{root}/data.pkl")).load()
    return OrderedDict((k, np.asarray(v, dtype=np.float64)) for k, v in state.items())


# =============================================================================
# 2. Mixture GPD math on parameter arrays (..., 5) = [xi_b, beta_b, xi_h, beta_h, pi]
# =============================================================================
def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _softplus(x):
    return np.logaddexp(0.0, x)


def split_params(params):
    """(..., 5) array or DataFrame with PARAM_COLS -> xi_b, beta_b, xi_h, beta_h, pi."""
    if pd is not None and isinstance(params, pd.DataFrame):
        params = params[PARAM_COLS].to_numpy(dtype=float)
    P = np.asarray(params, dtype=np.float64)
    return tuple(P[..., j] for j in range(5))


def gpd_logsf(y, xi, beta):
    return -np.log1p(xi * y / beta) / xi


def gpd_logpdf(y, xi, beta):
    return -np.log(beta) - (1.0 / xi + 1.0) * np.log1p(xi * y / beta)


def mixture_logsf(params, y):
    """log S(y|x); y >= 0 (broadcasts against the parameter rows)."""
    xi_b, beta_b, xi_h, beta_h, pi = split_params(params)
    y = np.maximum(np.asarray(y, dtype=np.float64), 0.0)
    return np.logaddexp(np.log1p(-pi) + gpd_logsf(y, xi_b, beta_b), np.log(pi) + gpd_logsf(y, xi_h, beta_h))


def mixture_survival(params, y):
    """S(y|x) = P(L > u + y | L > u, x)."""
    return np.exp(mixture_logsf(params, y))


def mixture_logpdf(params, y):
    """log f(y|x), the training likelihood of the excess loss."""
    xi_b, beta_b, xi_h, beta_h, pi = split_params(params)
    y = np.maximum(np.asarray(y, dtype=np.float64), 0.0)
    return np.logaddexp(np.log1p(-pi) + gpd_logpdf(y, xi_b, beta_b), np.log(pi) + gpd_logpdf(y, xi_h, beta_h))


def mixture_pdf(params, y):
    return np.exp(mixture_logpdf(params, y))


def mixture_quantile_excess(params, target, iters=100):
    """Excess y with S(y|x) = target (0 < target <= 1), vectorised bisection."""
    xi_b, beta_b, xi_h, beta_h, pi = split_params(params)
    target = np.broadcast_to(np.asarray(target, dtype=np.float64), np.shape(xi_b))
    qb = beta_b / xi_b * (target ** (-xi_b) - 1)
    qh = beta_h / xi_h * (target ** (-xi_h) - 1)
    lo, hi = np.zeros_like(qb), np.maximum(qb, qh)          # S(hi) <= target
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        above = mixture_survival(params, mid) > target
        lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
    return 0.5 * (lo + hi)


def tail_probability(p_u, params, loss, u):
    """P(L > loss | x) = p_u * S(loss - u | x) for loss > u."""
    return np.asarray(p_u) * mixture_survival(params, np.asarray(loss) - u)


def var_es(p_u, params, alpha, u):
    """Unconditional VaR_(1-alpha), ES_(1-alpha) of the loss: p_u S(VaR - u) = alpha.
    NaN where p_u <= alpha (VaR below u, outside the GPD region)."""
    p_u = np.asarray(p_u, dtype=np.float64)
    ok = p_u > alpha
    y = mixture_quantile_excess(params, np.where(ok, alpha / np.where(ok, p_u, 1.0), 0.5))
    xi_b, beta_b, xi_h, beta_h, pi = split_params(params)
    sb, sh = np.exp(gpd_logsf(y, xi_b, beta_b)), np.exp(gpd_logsf(y, xi_h, beta_h))
    mean_excess = ((1 - pi) * sb * (beta_b + xi_b * y) / (1 - xi_b)
                   + pi * sh * (beta_h + xi_h * y) / (1 - xi_h)) / ((1 - pi) * sb + pi * sh)
    return np.where(ok, u + y, np.nan), np.where(ok, u + y + mean_excess, np.nan)


def gpd_expected_excess(xi, beta, a):
    """E[(Y - a)^+] of one GPD = S(a) (beta + xi a) / (1 - xi); needs 0 < xi < 1, a >= 0."""
    return np.exp(gpd_logsf(a, xi, beta)) * (beta + xi * a) / (1.0 - xi)


def mixture_tail_risk(params, d, u):
    """E[(L - d)^+ | L > u] = (1-pi) R_base + pi R_heavy; returns (R, base part, heavy part)."""
    xi_b, beta_b, xi_h, beta_h, pi = split_params(params)
    a = d - u
    base = (1 - pi) * gpd_expected_excess(xi_b, beta_b, a)
    heavy = pi * gpd_expected_excess(xi_h, beta_h, a)
    return base + heavy, base, heavy


# =============================================================================
# 3. Backtest helpers (numpy; used by training and evaluate_tail.py)
# =============================================================================
def auc_score(p, e):
    """ROC AUC via the rank (Mann-Whitney) formula, ties averaged."""
    from scipy.stats import rankdata
    p, pos = np.asarray(p, dtype=float), np.asarray(e, dtype=float) > 0.5
    n1, n0 = pos.sum(), (~pos).sum()
    return float((rankdata(p)[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def kupiec_test(breaches, alpha):
    """Kupiec unconditional-coverage LR test -> (LR, p-value)."""
    from scipy import stats
    n, x = len(breaches), int(np.sum(breaches))
    ph = x / n
    ll0 = (n - x) * np.log(1 - alpha) + x * np.log(alpha)
    ll1 = ((n - x) * np.log1p(-ph) if x < n else 0.0) + (x * np.log(ph) if x > 0 else 0.0)
    lr = float(-2 * (ll0 - ll1))
    return lr, float(stats.chi2.sf(lr, df=1))


def var_backtest(p_u, params, L, u, alphas=(0.001, 0.005, 0.01, 0.02)):
    """Breach rates of the two-stage VaR; rows with p_u <= alpha (VaR < u) breach when L > u."""
    out = {}
    for a in alphas:
        var, _ = var_es(p_u, params, a, u)
        below = np.isnan(var)
        breach = np.where(below, L > u, L > np.nan_to_num(var, nan=np.inf))
        lr, pval = kupiec_test(breach, a)
        out[str(a)] = {"expected": a, "breach_rate": float(breach.mean()), "n_breaches": int(breach.sum()),
                       "n": int(len(breach)), "rows_var_below_u": int(below.sum()),
                       "kupiec_lr": lr, "kupiec_p": pval}
    return out


def var99_by_decile(p_u, params, L, u, n_bins=10, constant_p_u=0.10):
    """VaR99 breach rate per p_u decile: model vs a constant p_u."""
    p_u = np.asarray(p_u, dtype=float)
    bins = pd.qcut(p_u, n_bins, labels=False, duplicates="drop")
    var_m, _ = var_es(p_u, params, 0.01, u)
    var_c, _ = var_es(np.full_like(p_u, constant_p_u), params, 0.01, u)
    rows = []
    for b in np.unique(bins):
        m = bins == b
        rows.append({"decile": int(b) + 1, "mean_p_u": float(p_u[m].mean()),
                     "breach_model": float(np.mean(L[m] > np.nan_to_num(var_m[m], nan=u))),
                     "breach_constant_p_u": float(np.mean(L[m] > var_c[m]))})
    return rows


# =============================================================================
# 4. The model
# =============================================================================
class TwoStageTailModel:
    """Both networks + preprocessing. Attributes: feature_names, u, scale, pi_min, meta."""

    def __init__(self, gpd_weights, pu_weights, meta, pu_bn_eps=None):
        self.meta = meta
        self.feature_names = list(meta["feature_names"])
        self.u = float(meta["threshold_u_loss"])
        self.pi_min = float(meta["pi_min"])
        self.clip_z = float(meta["clip_z"])
        self.bn_eps = float(meta.get("bn_eps", meta.get("batchnorm", {}).get("eps", 1e-3)))
        self.pu_bn_eps = self.bn_eps if pu_bn_eps is None else float(pu_bn_eps)
        self.median = np.asarray(meta["feature_median_fill"], dtype=float)
        self.mean = np.asarray(meta["feature_mean"], dtype=float)
        self.std = np.asarray(meta["feature_std"], dtype=float)
        self.g = {k: np.asarray(v, dtype=np.float64) for k, v in gpd_weights.items()}
        self.p = None if pu_weights is None else {k: np.asarray(v, dtype=np.float64) for k, v in pu_weights.items()}
        self.scale = float(self.g["scale"].reshape(()))
        n = len(self.feature_names)
        for name, w in (("gpd", self.g), ("p_u", self.p)):
            if w is not None and w["trunk.0.weight"].shape[1] != n:
                raise ValueError(f"{name} network expects {w['trunk.0.weight'].shape[1]} inputs, meta lists {n}")

    @property
    def has_pu(self):
        return self.p is not None

    # ---------------------------------------------------------------- loading / saving
    @classmethod
    def from_artifacts(cls, artifact_dir="artifacts", require_pu=True):
        """preprocess.json + gpd_model.pt (+ pu_model.pt) from one folder."""
        d = Path(artifact_dir)
        meta = json.loads((d / "preprocess.json").read_text())
        gpd = load_state_dict(d / meta.get("model_file", "gpd_model.pt"))
        pu_path = d / "pu_model.pt"
        if not pu_path.exists() and require_pu:
            raise FileNotFoundError(f"{pu_path} not found (stage 1 network)")
        return cls(gpd, load_state_dict(pu_path) if pu_path.exists() else None, meta)

    def to_dict(self):
        """Plain-JSON model (weights + preprocessing), format gpd_mixture_numpy_v1."""
        out = {"format": JSON_FORMAT,
               "meta": {"feature_names": self.feature_names,
                        "feature_median_fill": self.median.tolist(), "feature_mean": self.mean.tolist(),
                        "feature_std": self.std.tolist(), "clip_z": self.clip_z,
                        "threshold_u_loss": self.u, "pi_min": self.pi_min, "bn_eps": self.bn_eps},
               "weights": {k: v.tolist() for k, v in self.g.items()}}
        if self.has_pu:
            out["pu"] = {"bn_eps": self.pu_bn_eps, "weights": {k: v.tolist() for k, v in self.p.items()}}
        return out

    def to_json(self, path=None):
        """JSON text of to_dict(); also written to `path` if given."""
        text = json.dumps(self.to_dict())
        if path is not None:
            Path(path).write_text(text)
        return text

    @classmethod
    def from_json(cls, obj):
        """From to_json() output: JSON text, a dict, or a file path."""
        if isinstance(obj, (str, Path)) and not str(obj).lstrip().startswith("{"):
            obj = Path(obj).read_text()
        if isinstance(obj, (str, bytes)):
            obj = json.loads(obj)
        if obj.get("format") != JSON_FORMAT:
            raise ValueError(f"not a {JSON_FORMAT} model file")
        pu = obj.get("pu")
        return cls(obj["weights"], pu["weights"] if pu else None, obj["meta"],
                   pu_bn_eps=pu.get("bn_eps") if pu else None)

    # ---------------------------------------------------------------- inputs
    def to_matrix(self, X):
        """DataFrame / dict / (n, 13) array of RAW factors -> float array in feature_names order."""
        if pd is not None and isinstance(X, pd.DataFrame):
            missing = [c for c in self.feature_names if c not in X.columns]
            if missing:
                raise KeyError(f"missing factor columns: {missing}")
            return X[self.feature_names].to_numpy(dtype=float)
        if isinstance(X, dict):
            missing = [c for c in self.feature_names if c not in X]
            if missing:
                raise KeyError(f"missing factors: {missing}")
            return np.column_stack(np.broadcast_arrays(*[np.atleast_1d(np.asarray(X[c], dtype=float))
                                                         for c in self.feature_names]))
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != len(self.feature_names):
            raise ValueError(f"expected {len(self.feature_names)} columns {self.feature_names}, got {X.shape}")
        return X

    def zscores(self, X):
        """Median fill + z-score with training statistics, before clipping (diagnostics)."""
        X = self.to_matrix(X)
        return (np.where(np.isnan(X), self.median, X) - self.mean) / self.std

    def preprocess(self, X):
        """Network input: zscores clipped to +/- clip_z (exactly as in training)."""
        return np.clip(self.zscores(X), -self.clip_z, self.clip_z)

    # ---------------------------------------------------------------- networks
    @staticmethod
    def _bn(x, w, prefix, eps):
        return ((x - w[prefix + ".running_mean"]) / np.sqrt(w[prefix + ".running_var"] + eps)
                * w[prefix + ".weight"] + w[prefix + ".bias"])

    def _trunk(self, z, w, eps):
        h = np.maximum(self._bn(z @ w["trunk.0.weight"].T, w, "trunk.1", eps), 0.0)
        return np.maximum(self._bn(h @ w["trunk.3.weight"].T, w, "trunk.4", eps), 0.0)

    def gpd_params_z(self, z):
        """Stage 2 on PREPROCESSED input z -> (n, 5)."""
        w, eps = self.g, self.bn_eps
        h = self._trunk(np.atleast_2d(z), w, eps)
        a = h @ w["base_head.weight"].T + w["base_head.bias"]
        xi_b = 0.01 + 0.34 * _sigmoid(a[:, 0])
        beta_b = self.scale * _softplus(a[:, 1]) + 1e-6
        g = np.maximum(self._bn(h @ w["gate.0.weight"].T, w, "gate.1", eps), 0.0)
        g = (g @ w["gate.3.weight"].T + w["gate.3.bias"])[:, 0]
        pi = np.clip(self.pi_min + (1.0 - self.pi_min) * _sigmoid(g), 1e-6, 1.0 - 1e-6)
        one = np.ones_like(xi_b)
        xi_h = (0.35 + 0.45 * _sigmoid(float(w["theta_xi"].reshape(())))) * one
        beta_h = (self.scale * _softplus(float(w["theta_beta"].reshape(()))) + 1e-6) * one
        return np.column_stack([xi_b, beta_b, xi_h, beta_h, pi])

    def exceedance_prob_z(self, z):
        """Stage 1 on PREPROCESSED input z -> p_u, shape (n,)."""
        if not self.has_pu:
            raise ValueError("this model has no exceedance network (pu_model.pt / 'pu' in the JSON)")
        w = self.p
        h = self._trunk(np.atleast_2d(z), w, self.pu_bn_eps)
        return _sigmoid((h @ w["out.weight"].T + w["out.bias"])[:, 0])

    def gpd_params(self, X):
        """Stage 2 on RAW factors -> (n, 5) = [xi_b, beta_b, xi_h, beta_h, pi]."""
        return self.gpd_params_z(self.preprocess(X))

    def exceedance_prob(self, X):
        """Stage 1 on RAW factors -> p_u = P(L > u | x)."""
        return self.exceedance_prob_z(self.preprocess(X))

    def predict(self, X):
        """DataFrame (dict without pandas): xi_b, beta_b, xi_h, beta_h, pi, p_u (if available)."""
        z = self.preprocess(X)
        P = self.gpd_params_z(z)
        out = {c: P[:, j] for j, c in enumerate(PARAM_COLS)}
        if self.has_pu:
            out["p_u"] = self.exceedance_prob_z(z)
        if pd is None:
            return out
        return pd.DataFrame(out, index=X.index if isinstance(X, pd.DataFrame) else None)

    # ---------------------------------------------------------------- risk measures (RAW factors)
    def cond_survival(self, X, y):
        """S(y|x) = P(L > u + y | L > u, x)."""
        return mixture_survival(self.gpd_params(X), y)

    def cond_logpdf(self, X, y):
        return mixture_logpdf(self.gpd_params(X), y)

    def cond_quantile(self, X, q=0.99):
        """Loss level with P(L <= level | L > u, x) = q (the q99 line of the tail plots)."""
        return self.u + mixture_quantile_excess(self.gpd_params(X), 1.0 - q)

    def tail_prob(self, X, loss):
        """P(L > loss | x); NaN for loss <= u."""
        z = self.preprocess(X)
        loss = np.asarray(loss, dtype=float)
        pr = tail_probability(self.exceedance_prob_z(z), self.gpd_params_z(z), loss, self.u)
        return np.where(loss > self.u, pr, np.nan)

    def var(self, X, alpha=0.01):
        """Unconditional VaR_(1-alpha) (positive loss); NaN where p_u <= alpha."""
        z = self.preprocess(X)
        return var_es(self.exceedance_prob_z(z), self.gpd_params_z(z), alpha, self.u)[0]

    def es(self, X, alpha=0.01):
        """Unconditional ES_(1-alpha); NaN where p_u <= alpha."""
        z = self.preprocess(X)
        return var_es(self.exceedance_prob_z(z), self.gpd_params_z(z), alpha, self.u)[1]

    def risk_score(self, X, d=0.07, use_pu=True, return_parts=False):
        """R = E[(L - d)^+] = p_u * E[(L - d)^+ | L > u]  (use_pu=False: conditional part only).
        return_parts=True -> (R, base part, heavy part, p_u), parts already multiplied by p_u."""
        if not d > self.u:
            raise ValueError(f"d must exceed u = {self.u:.4f}, got {d}")
        z = self.preprocess(X)
        R, base, heavy = mixture_tail_risk(self.gpd_params_z(z), d, self.u)
        pu = self.exceedance_prob_z(z) if use_pu else np.ones_like(R)
        return (pu * R, pu * base, pu * heavy, pu) if return_parts else pu * R


# =============================================================================
# 5. Command line
# =============================================================================
def _selftest(m):
    rng = np.random.default_rng(0)
    X = m.median + rng.normal(0, 1, (64, len(m.feature_names))) * m.std
    X[0, 3] = np.nan
    P = m.gpd_params(X)
    assert P.shape == (64, 5) and np.isfinite(P).all() and (P[:, 4] >= m.pi_min - 1e-9).all()
    m2 = TwoStageTailModel.from_json(m.to_json())
    assert np.array_equal(m2.gpd_params(X), P), "JSON round trip changed the GPD output"
    if m.has_pu:
        pu = m.exceedance_prob(X)
        assert ((pu > 0) & (pu < 1)).all() and np.array_equal(m2.exceedance_prob(X), pu)
        v = m.var(X, 0.01)
        ok = np.isfinite(v)
        assert np.allclose(m.tail_prob(X[ok], v[ok]), 0.01, rtol=1e-4) and (m.es(X[ok], 0.01) > v[ok]).all()
    print(f"self-test passed: u = {m.u:.4f}, pi_min = {m.pi_min}, xi_h = {P[0, 2]:.3f}, "
          f"beta_h = {P[0, 3]:.4f}, p_u network: {m.has_pu}")


def main():
    ap = argparse.ArgumentParser(description="Two-stage tail model: forward pass and risk measures")
    ap.add_argument("--artifacts", default=str(Path(__file__).resolve().parent / "artifacts"))
    ap.add_argument("--csv", help="CSV with the 13 raw factor columns (other columns are kept)")
    ap.add_argument("--out", help="output CSV (default: print the first rows)")
    ap.add_argument("--d", type=float, default=0.07, help="disaster loss for the risk score")
    ap.add_argument("--alpha", type=float, default=0.01, help="VaR / ES level (0.01 -> 99%%)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    m = TwoStageTailModel.from_artifacts(args.artifacts)
    if args.selftest or not args.csv:
        _selftest(m)
        if not args.csv:
            return
    df = pd.read_csv(args.csv)
    out = pd.concat([df, m.predict(df)], axis=1)
    out["risk_score"] = m.risk_score(df, d=args.d)
    out[f"var_{1 - args.alpha:g}"] = m.var(df, args.alpha)
    out[f"es_{1 - args.alpha:g}"] = m.es(df, args.alpha)
    out["cond_q99"] = m.cond_quantile(df, 0.99)
    if args.out:
        out.to_csv(args.out, index=False)
        print(f"wrote {len(out)} rows to {args.out}")
    else:
        print(out.head().to_string())


if __name__ == "__main__":
    main()
