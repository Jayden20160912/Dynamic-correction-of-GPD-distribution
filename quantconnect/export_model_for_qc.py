"""
Export the trained two-stage model for QuantConnect.

Reads   <artifacts>/preprocess.json, gpd_model.pt, pu_model.pt
Writes  gpd_mixture_model.json  (both networks + preprocessing, plain JSON, tail_model.to_json)

The training script already writes this file after every training run; use this script to
re-export by hand. Everything goes through tail_model.py, the single inference implementation
that quantconnect/main.py also runs, so no torch is needed. Checks:
  1. the JSON read back gives exactly the same outputs as the .pt files;
  2. if torch is installed, the numpy forward pass matches the PyTorch networks of the
     training script.

Usage:
    python export_model_for_qc.py --artifacts ../artifacts
Then upload gpd_mixture_model.json to QuantConnect's Object Store (key MODEL_KEY in main.py).
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import tail_model  # noqa: E402


def test_inputs(model, n=5000, seed=0):
    """Raw factors around the training distribution, incl. clipped and missing values."""
    rng = np.random.default_rng(seed)
    X = model.mean + model.std * rng.normal(scale=2.0, size=(n, len(model.mean)))
    X[rng.random(X.shape) < 0.05] = np.nan
    return X


def check_against_torch(model, X):
    """Compare with the PyTorch networks of the training script (skipped without torch)."""
    try:
        import torch
    except ImportError:
        print("torch not installed: skipped the numpy-vs-torch check (the training script runs it)")
        return
    spec = importlib.util.spec_from_file_location("gpd_training", ROOT / "TQT research machine learning part.py")
    tr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tr)
    art = Path(model.meta.get("_artifact_dir", ROOT / "artifacts"))
    net = tr.load_gpd_model(art / "gpd_model.pt", model.scale, model.pi_min)
    x = torch.from_numpy(model.preprocess(X).astype("float32"))
    with torch.no_grad():
        P_t = net(x).double().numpy()
    rel = np.max(np.abs(model.gpd_params(X) - P_t) / np.maximum(np.abs(P_t), 1e-12), axis=0)
    worst = dict(zip(tail_model.PARAM_COLS, rel))
    if model.has_pu:
        pu_t = tr.load_pu_model(art / "pu_model.pt").predict_proba(x).double().numpy()
        worst["p_u"] = float(np.max(np.abs(model.exceedance_prob(X) - pu_t) / np.maximum(pu_t, 1e-12)))
    print("max relative difference numpy vs torch:", {k: f"{v:.1e}" for k, v in worst.items()})
    if max(worst.values()) > 1e-4:
        raise SystemExit("MISMATCH between tail_model.py and the PyTorch networks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"),
                    help="folder with preprocess.json, gpd_model.pt, pu_model.pt")
    ap.add_argument("--out", default=str(HERE / "gpd_mixture_model.json"))
    args = ap.parse_args()

    model = tail_model.TwoStageTailModel.from_artifacts(args.artifacts)
    model.meta["_artifact_dir"] = args.artifacts
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.to_json(out)
    print(f"wrote {out}  ({out.stat().st_size / 1e3:.0f} KB; stages: GPD mixture"
          f"{' + p_u' if model.has_pu else ''})")

    X = test_inputs(model)
    back = tail_model.TwoStageTailModel.from_json(out)
    same = np.array_equal(back.gpd_params(X), model.gpd_params(X))
    if model.has_pu:
        same &= np.array_equal(back.exceedance_prob(X), model.exceedance_prob(X))
    if not same:
        raise SystemExit("MISMATCH: the exported JSON does not reproduce the .pt files")
    print("OK: the JSON gives exactly the same outputs as the .pt files")
    check_against_torch(model, X)


if __name__ == "__main__":
    main()
