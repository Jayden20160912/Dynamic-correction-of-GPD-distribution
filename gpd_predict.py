"""
Standalone inference for the trained gated GPD-mixture model.

Self-contained: only needs torch / numpy / pandas plus artifacts/gpd_model.pt and
artifacts/preprocess.json. It does NOT import the training script or features.py and
never calls an API.

Model (gate_bn32 + 3 history features + pi_min = 0.03), input = 13 raw factors:
    trunk   Linear(13,224,no bias)-BN-ReLU-Linear(224,88,no bias)-BN-ReLU -> h
    base    Linear(88,2):  xi_b = 0.01 + 0.34*sigmoid, beta_b = s*softplus + 1e-6
    gate    Linear(88,32,no bias)-BN-ReLU-Linear(32,1): pi = pi_min + (1-pi_min)*sigmoid
    heavy   global scalars: xi_h = 0.35 + 0.45*sigmoid(theta_xi), beta_h = s*softplus(theta_beta) + 1e-6
For the excess loss y = L - u (L = -return, given L > u):
    S(y|x) = (1-pi) (1 + xi_b y/beta_b)^(-1/xi_b) + pi (1 + xi_h y/beta_h)^(-1/xi_h)

Python use:
    from gpd_predict import GPDPredictor
    p = GPDPredictor()                         # loads artifacts/
    params = p.predict(df)                     # df has the 13 factor columns (raw, unscaled)
    p.survival(params, loss=0.10)              # P(loss > 10% | loss > u)
    p.quantile(params, 0.99)                   # 99% quantile of the LOSS given loss > u

Command line (rows taken from artifacts/dataset.csv):
    .venv/bin/python gpd_predict.py --ticker AAPL --date 2024-08-02
    .venv/bin/python gpd_predict.py --ticker AAPL                 # latest date for AAPL
    .venv/bin/python gpd_predict.py --csv my_factors.csv          # any CSV with the 13 factor columns
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"
PARAM_COLS = ["xi_b", "beta_b", "xi_h", "beta_h", "pi"]


# ==========================================
# 1. Data preprocessing (same transform as training, using TRAINING-set statistics)
# ==========================================

def load_meta(artifact_dir=ARTIFACT_DIR):
    """preprocess.json: feature order, train median / mean / std, clip, threshold u, model settings."""
    return json.loads((Path(artifact_dir) / "preprocess.json").read_text())


def preprocess(X, meta):
    """Raw factors (n, 13) in meta["feature_names"] order -> model input (n, 13).

    1. fill missing values with the training-set median of each factor
    2. standardise with the training-set mean and std (z-score)
    3. clip to +/- clip_z (5)
    """
    X = np.asarray(X, dtype="float64")
    X = np.where(np.isnan(X), np.asarray(meta["feature_median_fill"]), X)
    Z = (X - np.asarray(meta["feature_mean"])) / np.asarray(meta["feature_std"])
    return np.clip(Z, -meta["clip_z"], meta["clip_z"]).astype("float32")


# ==========================================
# 2. Network (identical to the training script, so the saved state_dict loads strictly)
# ==========================================

class KerasBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-3, momentum=0.99):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        if self.training:
            mean, var = x.mean(dim=0), x.var(dim=0, unbiased=False)
            with torch.no_grad():
                self.running_mean.mul_(self.momentum).add_((1.0 - self.momentum) * mean)
                self.running_var.mul_(self.momentum).add_((1.0 - self.momentum) * var)
        else:
            mean, var = self.running_mean, self.running_var
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class GateMixtureGPDNet(nn.Module):
    """forward(x) -> (B, 5) = [xi_b, beta_b, xi_h, beta_h, pi]; x is the preprocessed input."""

    def __init__(self, scale, input_dim=13, hidden_units=(224, 88), gate_units=32, pi_min=0.03):
        super().__init__()
        h1, h2 = hidden_units
        self.pi_min = pi_min
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, h1, bias=False), KerasBatchNorm1d(h1), nn.ReLU(),
            nn.Linear(h1, h2, bias=False), KerasBatchNorm1d(h2), nn.ReLU(),
        )
        self.base_head = nn.Linear(h2, 2, bias=True)
        self.gate = nn.Sequential(
            nn.Linear(h2, gate_units, bias=False), KerasBatchNorm1d(gate_units), nn.ReLU(),
            nn.Linear(gate_units, 1, bias=True),
        )
        self.theta_xi = nn.Parameter(torch.zeros(()))
        self.theta_beta = nn.Parameter(torch.zeros(()))
        self.register_buffer("scale", torch.tensor(float(scale), dtype=torch.float32))

    def forward(self, x):
        h = self.trunk(x)
        a = self.base_head(h)
        xi_b = 0.01 + 0.34 * torch.sigmoid(a[:, 0])
        beta_b = self.scale * F.softplus(a[:, 1]) + 1e-6
        pi = self.pi_min + (1.0 - self.pi_min) * torch.sigmoid(self.gate(h).squeeze(-1))
        pi = pi.clamp(1e-6, 1 - 1e-6)
        ones = torch.ones_like(xi_b)
        xi_h = (0.35 + 0.45 * torch.sigmoid(self.theta_xi)) * ones
        beta_h = (self.scale * F.softplus(self.theta_beta) + 1e-6) * ones
        return torch.stack([xi_b, beta_b, xi_h, beta_h, pi], dim=1)


# ==========================================
# 3. Predictor: preprocessing + forward pass + mixture formulas
# ==========================================

class GPDPredictor:

    def __init__(self, artifact_dir=ARTIFACT_DIR):
        artifact_dir = Path(artifact_dir)
        self.meta = load_meta(artifact_dir)
        m = self.meta
        self.feature_names = m["feature_names"]
        self.u = float(m["threshold_u_loss"])
        self.model = GateMixtureGPDNet(m["gpd_scale"], input_dim=len(self.feature_names),
                                       hidden_units=tuple(m["hidden_units"]),
                                       gate_units=m["gate_units"], pi_min=m["pi_min"])
        state = torch.load(artifact_dir / m["model_file"], map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()                       # BatchNorm uses the trained moving statistics

    def preprocess(self, X):
        """Section 1's preprocess() with this model's saved statistics."""
        return preprocess(X, self.meta)

    @torch.no_grad()
    def forward(self, X_raw):
        """Raw factor matrix (n, 13) in feature_names order -> parameter array (n, 5)."""
        x = torch.from_numpy(self.preprocess(X_raw))
        return self.model(x).numpy().astype("float64")

    def predict(self, df):
        """DataFrame with the 13 factor columns -> copy with xi_b, beta_b, xi_h, beta_h, pi added."""
        missing = [c for c in self.feature_names if c not in df.columns]
        if missing:
            raise ValueError(f"missing factor columns: {missing}")
        P = self.forward(df[self.feature_names].values)
        out = df.copy()
        for i, c in enumerate(PARAM_COLS):
            out[c] = P[:, i]
        out["q99_loss"] = self.quantile(out, 0.99)
        return out

    @staticmethod
    def _params(params):
        return [np.asarray(params[c], dtype="float64") for c in PARAM_COLS]

    def survival(self, params, loss):
        """P(L > loss | L > u) for a loss level (e.g. 0.10 = 10%); loss <= u gives 1."""
        xi_b, beta_b, xi_h, beta_h, pi = self._params(params)
        y = np.maximum(np.asarray(loss, dtype="float64") - self.u, 0.0)
        sf = lambda xi, b: np.exp(-np.log1p(xi * y / b) / xi)
        return (1 - pi) * sf(xi_b, beta_b) + pi * sf(xi_h, beta_h)

    def quantile(self, params, q):
        """Loss level with P(L <= level | L > u) = q, solved by bisection (vectorised)."""
        xi_b, beta_b, xi_h, beta_h, pi = self._params(params)
        target = 1.0 - q
        qb = beta_b / xi_b * ((1 - q) ** (-xi_b) - 1)
        qh = beta_h / xi_h * ((1 - q) ** (-xi_h) - 1)
        lo, hi = np.zeros_like(qb), np.maximum(qb, qh)          # S(hi) <= 1 - q
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            s = (1 - pi) * np.exp(-np.log1p(xi_b * mid / beta_b) / xi_b) \
                + pi * np.exp(-np.log1p(xi_h * mid / beta_h) / xi_h)
            lo, hi = np.where(s > target, mid, lo), np.where(s > target, hi, mid)
        return self.u + 0.5 * (lo + hi)


# ==========================================
# 4. Command line
# ==========================================

def main():
    ap = argparse.ArgumentParser(description="Forward pass of the gated GPD-mixture model")
    ap.add_argument("--ticker", type=str, default=None)
    ap.add_argument("--date", type=str, default=None, help="factor date t (loss is on day t+1)")
    ap.add_argument("--csv", type=str, default=None,
                    help="CSV with the 13 factor columns (default: artifacts/dataset.csv)")
    args = ap.parse_args()

    p = GPDPredictor()
    df = pd.read_csv(args.csv or ARTIFACT_DIR / "dataset.csv")
    if args.ticker:
        df = df[df["ticker"] == args.ticker.upper()]
        if args.date is None and "date" in df:
            df = df[df["date"] == df["date"].max()]
    if args.date:
        df = df[df["date"] == args.date]
    if df.empty:
        raise SystemExit("no matching rows")
    if len(df) > 20 and not args.csv:
        df = df.tail(20)
        print("(showing the last 20 matching rows)")

    out = p.predict(df)
    print(f"Threshold u = {p.u:.2%} loss. All probabilities are conditional on loss > u.")
    show = [c for c in ("date", "ticker") if c in out] + PARAM_COLS + ["q99_loss"]
    for lvl in (0.10, 0.20):
        out[f"P(L>{lvl:.0%})"] = p.survival(out, lvl)
        show.append(f"P(L>{lvl:.0%})")
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        print(out[show].to_string(index=False))


if __name__ == "__main__":
    main()
