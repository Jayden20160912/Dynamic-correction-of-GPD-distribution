"""
Quick look-up of the model's predictions (kept for existing callers; the model is tail_model.py).

All computation is in tail_model.py -- the single inference implementation of the project.
This file keeps the old GPDPredictor API and command line on top of it (no torch needed).

Python use:
    from gpd_predict import GPDPredictor
    p = GPDPredictor()                         # loads artifacts/ (both stages)
    params = p.predict(df)                     # df has the 13 factor columns (raw, unscaled)
    p.survival(params, loss=0.10)              # P(loss > 10% | loss > u)
    p.quantile(params, 0.99)                   # 99% quantile of the LOSS given loss > u
    p.model                                    # the tail_model.TwoStageTailModel (VaR, ES, risk score)

Command line (rows taken from artifacts/dataset.csv):
    .venv/bin/python gpd_predict.py --ticker AAPL --date 2024-08-02
    .venv/bin/python gpd_predict.py --ticker AAPL                 # latest date for AAPL
    .venv/bin/python gpd_predict.py --csv my_factors.csv          # any CSV with the 13 factor columns
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import tail_model
from tail_model import PARAM_COLS

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


class GPDPredictor:
    """Thin wrapper around tail_model.TwoStageTailModel with the old method names."""

    def __init__(self, artifact_dir=ARTIFACT_DIR):
        self.model = tail_model.TwoStageTailModel.from_artifacts(artifact_dir, require_pu=False)
        self.meta = self.model.meta
        self.feature_names = self.model.feature_names
        self.u = self.model.u

    def preprocess(self, X):
        return self.model.preprocess(X)

    def forward(self, X_raw):
        """Raw factor matrix (n, 13) in feature_names order -> parameter array (n, 5)."""
        return self.model.gpd_params(X_raw)

    def predict(self, df):
        """DataFrame with the 13 factor columns -> copy with xi_b, beta_b, xi_h, beta_h, pi
        (+ p_u when the stage 1 network is available) and q99_loss added."""
        pred = self.model.predict(df)
        out = df.copy()
        for c in pred.columns:
            out[c] = pred[c].to_numpy()
        out["q99_loss"] = self.quantile(out, 0.99)
        return out

    def survival(self, params, loss):
        """P(L > loss | L > u) for a loss level (e.g. 0.10 = 10%); loss <= u gives 1."""
        return tail_model.mixture_survival(params, np.asarray(loss, dtype=float) - self.u)

    def quantile(self, params, q):
        """Loss level with P(L <= level | L > u) = q."""
        return self.u + tail_model.mixture_quantile_excess(params, 1.0 - q)


def main():
    ap = argparse.ArgumentParser(description="Predictions of the two-stage tail model")
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
    print(f"Threshold u = {p.u:.2%} loss. P(L>x) columns are conditional on loss > u; "
          f"p_u = P(loss > u).")
    show = [c for c in ("date", "ticker") if c in out] + PARAM_COLS
    show += ["p_u"] if "p_u" in out else []
    show += ["q99_loss"]
    for lvl in (0.10, 0.20):
        out[f"P(L>{lvl:.0%})"] = p.survival(out, lvl)
        show.append(f"P(L>{lvl:.0%})")
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        print(out[show].to_string(index=False))


if __name__ == "__main__":
    main()
