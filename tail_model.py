"""
Plug-and-play forward pass for the two-stage dynamic tail-risk model.

    P(L > l | x) = p_u(x) * S(l - u | x),   l > u          (L = -return of the NEXT day)

    Stage 1  exceedance network   p_u(x) = P(L > u | x)                 (artifacts/pu_model.pt)
    Stage 2  gated GPD mixture    S(y | x) = P(L > u + y | L > u, x)    (artifacts/gpd_model.pt)
             S(y|x) = (1 - pi) S_GPD(y; xi_b, beta_b) + pi S_GPD(y; xi_h, beta_h)

Only needs numpy (pandas optional). It reads the trained PyTorch weights (.pt) directly --
torch is NOT required -- together with artifacts/preprocess.json (feature order, training
median / mean / std, clip, threshold u, scale s, pi_min). Preprocessing and BatchNorm are done
exactly as at inference time in training (training statistics, BN moving statistics).

Input: the 13 RAW factors of day t (unscaled), in any of these forms
    * numpy array (n, 13) in model.feature_names order
    * pandas DataFrame with (at least) the 13 factor columns, any column order
    * dict {factor name: value or array}
Missing values (NaN) are replaced with the training median, like in training.

Quick start
    from tail_model import TwoStageTailModel
    m = TwoStageTailModel.from_artifacts("artifacts")
    out = m.predict(df)                      # DataFrame: xi_b, beta_b, xi_h, beta_h, pi, p_u
    m.tail_prob(df, loss=0.10)               # P(next-day loss > 10%)
    m.var(df, alpha=0.01)                    # unconditional VaR99 (loss, NaN if VaR < u)
    m.es(df, alpha=0.01)                     # unconditional ES99
    m.risk_score(df, d=0.07)                 # E[(L - d)^+] = p_u * E[(L - d)^+ | L > u]

Command line
    python tail_model.py --csv factors.csv --out predictions.csv [--d 0.07] [--alpha 0.01]
    python tail_model.py --selftest

Conventions: losses are positive numbers (0.07 = a 7% drop). y = L - u is the excess loss.
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
except ImportError:          # pandas is only needed for DataFrame input / output
    pd = None

PARAM_COLS = ["xi_b", "beta_b", "xi_h", "beta_h", "pi"]


# =============================================================================
# 1. Read a PyTorch state_dict (.pt, zip format) with numpy only
# =============================================================================
_DTYPES = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16,
           "LongStorage": np.int64, "IntStorage": np.int32, "ShortStorage": np.int16,
           "CharStorage": np.int8, "ByteStorage": np.uint8, "BoolStorage": np.bool_}


class _StorageType:
    def __init__(self, name):
        self.dtype = _DTYPES[name]


def _rebuild_tensor(storage, offset, size, stride, *args):
    itemsize = storage.itemsize
    return np.lib.stride_tricks.as_strided(storage[offset:], shape=tuple(size),
                                           strides=tuple(s * itemsize for s in stride)).copy()


def load_state_dict(path):
    """PyTorch .pt (torch.save of a state_dict, zip format) -> OrderedDict {name: np.ndarray}."""
    path = Path(path)
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        root = names[0].split("/")[0]
        byteorder = z.read(f"{root}/byteorder").decode().strip() if f"{root}/byteorder" in names else "little"
        bo = "<" if byteorder == "little" else ">"

        class _Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "collections" and name == "OrderedDict":
                    return OrderedDict
                if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
                    return _rebuild_tensor
                if module == "torch" and name in _DTYPES:
                    return _StorageType(name)
                raise pickle.UnpicklingError(f"unsupported object in {path.name}: {module}.{name}")

            def persistent_load(self, pid):
                _, stype, key, _location, _numel = pid
                raw = z.read(f"{root}/data/{key}")
                return np.frombuffer(raw, dtype=np.dtype(stype.dtype).newbyteorder(bo))

        state = _Unpickler(z.open(f"{root}/data.pkl")).load()
    return OrderedDict((k, np.asarray(v, dtype=np.float64)) for k, v in state.items())


# =============================================================================
# 2. Small math helpers
# =============================================================================
def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _softplus(x):
    return np.logaddexp(0.0, x)


def gpd_logsf(y, xi, beta):
    """log S_GPD(y; xi, beta) = -(1/xi) log(1 + xi y / beta), xi > 0, y >= 0."""
    return -np.log1p(xi * y / beta) / xi


def gpd_expected_excess(xi, beta, a):
    """E[(Y - a)^+] for one GPD = S(a) * (beta + xi a) / (1 - xi); needs 0 < xi < 1, a >= 0."""
    return np.exp(gpd_logsf(a, xi, beta)) * (beta + xi * a) / (1.0 - xi)


# =============================================================================
# 3. The model
# =============================================================================
class TwoStageTailModel:
    """Forward pass of both networks + the risk measures built from them.

    Attributes: feature_names (13), u (threshold, as a positive loss), scale s, pi_min, meta.
    """

    def __init__(self, gpd_state, pu_state, meta):
        self.meta = meta
        self.feature_names = list(meta["feature_names"])
        self.u = float(meta["threshold_u_loss"])
        self.pi_min = float(meta["pi_min"])
        self.clip_z = float(meta["clip_z"])
        self.bn_eps = float(meta.get("batchnorm", {}).get("eps", 1e-3))
        self.median = np.asarray(meta["feature_median_fill"], dtype=float)
        self.mean = np.asarray(meta["feature_mean"], dtype=float)
        self.std = np.asarray(meta["feature_std"], dtype=float)
        self.g = gpd_state
        self.p = pu_state
        self.scale = float(np.asarray(gpd_state["scale"]).reshape(()))
        self._check_shapes()

    # ------------------------------------------------------------ loading
    @classmethod
    def from_artifacts(cls, artifact_dir="artifacts", gpd_file=None, pu_file=None):
        """Load preprocess.json + gpd_model.pt + pu_model.pt from one folder."""
        d = Path(artifact_dir)
        meta = json.loads((d / "preprocess.json").read_text())
        gpd = load_state_dict(d / (gpd_file or meta.get("model_file", "gpd_model.pt")))
        pu_path = d / (pu_file or "pu_model.pt")
        if not pu_path.exists():
            raise FileNotFoundError(f"{pu_path} not found: the exceedance network (stage 1) is required")
        return cls(gpd, load_state_dict(pu_path), meta)

    def _check_shapes(self):
        n = len(self.feature_names)
        for name, w in (("GPD", self.g), ("p_u", self.p)):
            if w["trunk.0.weight"].shape[1] != n:
                raise ValueError(f"{name} network expects {w['trunk.0.weight'].shape[1]} inputs, "
                                 f"preprocess.json lists {n} features")
        need_g = ["trunk.0.weight", "trunk.3.weight", "base_head.weight", "gate.0.weight",
                  "gate.3.weight", "theta_xi", "theta_beta", "scale"]
        need_p = ["trunk.0.weight", "trunk.3.weight", "out.weight", "out.bias"]
        for k in need_g:
            if k not in self.g:
                raise KeyError(f"gpd_model.pt has no '{k}' (keys: {list(self.g)[:8]}...)")
        for k in need_p:
            if k not in self.p:
                raise KeyError(f"pu_model.pt has no '{k}' (keys: {list(self.p)[:8]}...)")

    # ------------------------------------------------------------ input handling
    def to_matrix(self, X):
        """DataFrame / dict / array -> float array (n, 13) in feature_names order."""
        if pd is not None and isinstance(X, pd.DataFrame):
            missing = [c for c in self.feature_names if c not in X.columns]
            if missing:
                raise KeyError(f"missing factor columns: {missing}")
            return X[self.feature_names].to_numpy(dtype=float)
        if isinstance(X, dict):
            missing = [c for c in self.feature_names if c not in X]
            if missing:
                raise KeyError(f"missing factors: {missing}")
            cols = [np.atleast_1d(np.asarray(X[c], dtype=float)) for c in self.feature_names]
            return np.column_stack(np.broadcast_arrays(*cols))
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != len(self.feature_names):
            raise ValueError(f"expected {len(self.feature_names)} columns in the order "
                             f"{self.feature_names}, got shape {X.shape}")
        return X

    def preprocess(self, X):
        """Raw factors -> network input: NaN -> training median, z-score with training
        mean / std, clip to +/- clip_z. Same as in training."""
        X = self.to_matrix(X)
        X = np.where(np.isnan(X), self.median, X)
        return np.clip((X - self.mean) / self.std, -self.clip_z, self.clip_z)

    # ------------------------------------------------------------ networks
    def _bn(self, x, w, prefix):
        """BatchNorm in inference mode (trained moving statistics)."""
        return ((x - w[prefix + ".running_mean"]) / np.sqrt(w[prefix + ".running_var"] + self.bn_eps)
                * w[prefix + ".weight"] + w[prefix + ".bias"])

    def _trunk(self, z, w):
        h = np.maximum(self._bn(z @ w["trunk.0.weight"].T, w, "trunk.1"), 0.0)
        return np.maximum(self._bn(h @ w["trunk.3.weight"].T, w, "trunk.4"), 0.0)

    def gpd_params(self, X):
        """Stage 2: -> array (n, 5) = [xi_b, beta_b, xi_h, beta_h, pi]."""
        w = self.g
        h = self._trunk(self.preprocess(X), w)
        a = h @ w["base_head.weight"].T + w["base_head.bias"]
        xi_b = 0.01 + 0.34 * _sigmoid(a[:, 0])
        beta_b = self.scale * _softplus(a[:, 1]) + 1e-6
        g = np.maximum(self._bn(h @ w["gate.0.weight"].T, w, "gate.1"), 0.0)
        g = (g @ w["gate.3.weight"].T + w["gate.3.bias"])[:, 0]
        pi = np.clip(self.pi_min + (1.0 - self.pi_min) * _sigmoid(g), 1e-6, 1.0 - 1e-6)
        one = np.ones_like(xi_b)
        xi_h = (0.35 + 0.45 * _sigmoid(float(np.asarray(w["theta_xi"]).reshape(())))) * one
        beta_h = (self.scale * _softplus(float(np.asarray(w["theta_beta"]).reshape(()))) + 1e-6) * one
        return np.column_stack([xi_b, beta_b, xi_h, beta_h, pi])

    def exceedance_prob(self, X):
        """Stage 1: -> p_u = P(L > u | x), shape (n,)."""
        w = self.p
        h = self._trunk(self.preprocess(X), w)
        return _sigmoid((h @ w["out.weight"].T + w["out.bias"])[:, 0])

    def predict(self, X):
        """Both stages: DataFrame (or dict of arrays without pandas) with
        xi_b, beta_b, xi_h, beta_h, pi, p_u. Keeps the input's index for DataFrame input."""
        P = self.gpd_params(X)
        out = {c: P[:, j] for j, c in enumerate(PARAM_COLS)}
        out["p_u"] = self.exceedance_prob(X)
        if pd is None:
            return out
        index = X.index if isinstance(X, pd.DataFrame) else None
        return pd.DataFrame(out, index=index)

    # ------------------------------------------------------------ distribution (given L > u)
    @staticmethod
    def _split(params):
        P = np.asarray(params, dtype=float)
        return P[:, 0], P[:, 1], P[:, 2], P[:, 3], P[:, 4]

    def cond_survival(self, X, y, params=None):
        """S(y | x) = P(L > u + y | L > u, x) for excess loss y >= 0 (scalar or per-row array)."""
        xi_b, beta_b, xi_h, beta_h, pi = self._split(self.gpd_params(X) if params is None else params)
        y = np.maximum(np.asarray(y, dtype=float), 0.0)
        return np.exp(np.logaddexp(np.log1p(-pi) + gpd_logsf(y, xi_b, beta_b),
                                   np.log(pi) + gpd_logsf(y, xi_h, beta_h)))

    def cond_logpdf(self, X, y, params=None):
        """log f(y | x) of the excess loss given L > u (the training likelihood)."""
        xi_b, beta_b, xi_h, beta_h, pi = self._split(self.gpd_params(X) if params is None else params)
        y = np.maximum(np.asarray(y, dtype=float), 0.0)

        def lpdf(xi, b):
            return -np.log(b) - (1.0 / xi + 1.0) * np.log1p(xi * y / b)
        return np.logaddexp(np.log1p(-pi) + lpdf(xi_b, beta_b), np.log(pi) + lpdf(xi_h, beta_h))

    def _solve_excess(self, params, s_target, iters=80):
        """Excess y with S(y|x) = s_target (per row, bisection in log y)."""
        s_target = np.broadcast_to(np.asarray(s_target, dtype=float), (len(params),))
        lo, hi = np.full(len(params), np.log(1e-10)), np.full(len(params), np.log(50.0))
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            above = self.cond_survival(None, np.exp(mid), params) > s_target
            lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
        return np.exp(0.5 * (lo + hi))

    def cond_quantile(self, X, q=0.99, params=None):
        """Loss level with P(L <= level | L > u, x) = q, e.g. the q99 line of the tail plots."""
        params = self.gpd_params(X) if params is None else params
        return self.u + self._solve_excess(params, 1.0 - q)

    # ------------------------------------------------------------ unconditional risk measures
    def tail_prob(self, X, loss):
        """P(L > loss | x) = p_u(x) * S(loss - u | x). loss: scalar or per-row array; NaN for loss <= u."""
        loss = np.asarray(loss, dtype=float)
        pr = self.exceedance_prob(X) * self.cond_survival(X, loss - self.u)
        return np.where(loss > self.u, pr, np.nan)

    def var(self, X, alpha=0.01):
        """Unconditional VaR_(1-alpha) as a positive loss: p_u(x) * S(VaR - u | x) = alpha.
        NaN where p_u(x) <= alpha (the VaR is then below u, outside the GPD region)."""
        params, pu = self.gpd_params(X), self.exceedance_prob(X)
        y = self._solve_excess(params, np.minimum(alpha / np.maximum(pu, 1e-300), 1.0))
        return np.where(pu > alpha, self.u + y, np.nan)

    def es(self, X, alpha=0.01):
        """Unconditional ES_(1-alpha) = E[L | L > VaR_(1-alpha)], closed form (xi < 1).
        NaN where VaR is below u."""
        params, pu = self.gpd_params(X), self.exceedance_prob(X)
        xi_b, beta_b, xi_h, beta_h, pi = self._split(params)
        y = self._solve_excess(params, np.minimum(alpha / np.maximum(pu, 1e-300), 1.0))
        S_b, S_h = np.exp(gpd_logsf(y, xi_b, beta_b)), np.exp(gpd_logsf(y, xi_h, beta_h))
        S = (1 - pi) * S_b + pi * S_h
        excess = ((1 - pi) * S_b * (beta_b + xi_b * y) / (1 - xi_b)
                  + pi * S_h * (beta_h + xi_h * y) / (1 - xi_h)) / S
        return np.where(pu > alpha, self.u + y + excess, np.nan)

    def risk_score(self, X, d=0.07, use_pu=True, return_parts=False):
        """Catastrophic-risk score R = E[(L - d)^+] (expected loss beyond d), d > u.
        use_pu=True : R = p_u * E[(L - d)^+ | L > u]   (unconditional, recommended)
        use_pu=False: R = E[(L - d)^+ | L > u]          (conditional on L > u only)
        return_parts=True also returns the base and heavy contributions and p_u."""
        if not d > self.u:
            raise ValueError(f"d must exceed u = {self.u:.4f}, got {d}")
        xi_b, beta_b, xi_h, beta_h, pi = self._split(self.gpd_params(X))
        a = d - self.u
        base = (1 - pi) * gpd_expected_excess(xi_b, beta_b, a)
        heavy = pi * gpd_expected_excess(xi_h, beta_h, a)
        pu = self.exceedance_prob(X) if use_pu else np.ones_like(base)
        R = pu * (base + heavy)
        return (R, pu * base, pu * heavy, pu) if return_parts else R


# =============================================================================
# 4. Command line
# =============================================================================
def _selftest(model):
    rng = np.random.default_rng(0)
    X = model.median + rng.normal(0, 1, (64, len(model.feature_names))) * model.std
    X[0, 3] = np.nan
    P = model.gpd_params(X)
    pu = model.exceedance_prob(X)
    assert P.shape == (64, 5) and np.isfinite(P).all()
    assert (P[:, 4] >= model.pi_min - 1e-9).all() and (P[:, 0] > 0).all() and (P[:, 2] < 1).all()
    assert ((pu > 0) & (pu < 1)).all()
    v = model.var(X, 0.01)
    ok = np.isfinite(v)
    assert np.allclose(model.tail_prob(X[ok], v[ok]), 0.01, rtol=1e-4), "VaR does not invert tail_prob"
    assert (model.es(X[ok], 0.01) > v[ok]).all()
    S = model.cond_survival(X, np.linspace(0, 0.3, 64))
    assert ((S > 0) & (S <= 1)).all()
    one = model.predict({k: float(model.median[j]) for j, k in enumerate(model.feature_names)})
    print(f"self-test passed: u = {model.u:.4f}, pi_min = {model.pi_min}, "
          f"xi_h = {P[0, 2]:.3f}, beta_h = {P[0, 3]:.4f}; at the median factors p_u = "
          f"{float(np.asarray(one['p_u'])[0]):.3f}, pi = {float(np.asarray(one['pi'])[0]):.3f}")


def main():
    ap = argparse.ArgumentParser(description="Two-stage tail model: forward pass and risk measures")
    ap.add_argument("--artifacts", default=str(Path(__file__).resolve().parent / "artifacts"))
    ap.add_argument("--csv", help="CSV with the 13 raw factor columns (other columns are kept)")
    ap.add_argument("--out", help="output CSV (default: print the first rows)")
    ap.add_argument("--d", type=float, default=0.07, help="disaster loss for the risk score")
    ap.add_argument("--alpha", type=float, default=0.01, help="VaR / ES level (0.01 -> 99%%)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    model = TwoStageTailModel.from_artifacts(args.artifacts)
    if args.selftest or not args.csv:
        _selftest(model)
        if not args.csv:
            return
    if pd is None:
        raise SystemExit("pandas is needed for --csv")
    df = pd.read_csv(args.csv)
    out = pd.concat([df, model.predict(df)], axis=1)
    out["risk_score"] = model.risk_score(df, d=args.d)
    out[f"var_{1 - args.alpha:g}"] = model.var(df, args.alpha)
    out[f"es_{1 - args.alpha:g}"] = model.es(df, args.alpha)
    out["cond_q99"] = model.cond_quantile(df, 0.99)
    if args.out:
        out.to_csv(args.out, index=False)
        print(f"wrote {len(out)} rows to {args.out}")
    else:
        print(out.head().to_string())


if __name__ == "__main__":
    main()
