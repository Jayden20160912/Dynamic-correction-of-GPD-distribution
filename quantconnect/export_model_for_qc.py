"""
Export the trained GPD-mixture model for QuantConnect.

Reads   <artifacts>/gpd_model.pt  and  <artifacts>/preprocess.json
Writes  gpd_mixture_model.json  (network weights + preprocessing statistics, plain JSON)

Then checks that the numpy forward pass in main.py gives the same five parameters as the
PyTorch model on random inputs, so what runs on QuantConnect is exactly the trained model.

Usage (run locally, needs torch):
    python export_model_for_qc.py --artifacts ../artifacts
Upload the resulting gpd_mixture_model.json to QuantConnect's Object Store.
"""

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
BN_EPS = 1e-3          # KerasBatchNorm1d default used in training (not stored in the state_dict)


# ---- PyTorch model, identical to the training / inference code (used only for the check) ----
class KerasBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=BN_EPS, momentum=0.99):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        return (x - self.running_mean) / torch.sqrt(self.running_var + self.eps) * self.weight + self.bias


class GateMixtureGPDNet(nn.Module):
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


def load_main_module():
    """Import main.py outside QuantConnect by stubbing the AlgorithmImports module."""
    stub = types.ModuleType("AlgorithmImports")
    stub.QCAlgorithm = object
    sys.modules.setdefault("AlgorithmImports", stub)
    spec = importlib.util.spec_from_file_location("qc_main", HERE / "main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default=str(HERE.parent / "artifacts"),
                    help="folder with gpd_model.pt and preprocess.json")
    ap.add_argument("--out", default=str(HERE / "gpd_mixture_model.json"))
    args = ap.parse_args()

    art = Path(args.artifacts)
    meta = json.loads((art / "preprocess.json").read_text())
    state = torch.load(art / meta["model_file"], map_location="cpu", weights_only=True)

    net = GateMixtureGPDNet(meta["gpd_scale"], input_dim=len(meta["feature_names"]),
                            hidden_units=tuple(meta["hidden_units"]),
                            gate_units=meta["gate_units"], pi_min=meta["pi_min"])
    net.load_state_dict(state, strict=True)
    net.eval()

    export = {
        "format": "gpd_mixture_numpy_v1",
        "meta": {
            "feature_names": meta["feature_names"],
            "feature_median_fill": meta["feature_median_fill"],
            "feature_mean": meta["feature_mean"],
            "feature_std": meta["feature_std"],
            "clip_z": meta["clip_z"],
            "threshold_u_loss": meta["threshold_u_loss"],
            "pi_min": meta["pi_min"],
            "bn_eps": float(meta.get("batchnorm", {}).get("eps", BN_EPS)),
        },
        "weights": {k: v.detach().cpu().double().numpy().tolist() for k, v in state.items()},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(export))
    print(f"wrote {out}  ({out.stat().st_size / 1e3:.0f} KB, {len(export['weights'])} tensors)")

    # ---- check: numpy forward in main.py == PyTorch forward ----
    qc = load_main_module()
    model = qc.GPDMixtureModel(json.loads(out.read_text()))
    rng = np.random.default_rng(0)
    std = np.asarray(meta["feature_std"]); mean = np.asarray(meta["feature_mean"])
    X = mean + std * rng.normal(scale=2.0, size=(5000, len(mean)))      # includes clipped values
    X[rng.random(X.shape) < 0.05] = np.nan                               # includes missing values
    x_t = torch.from_numpy(model.preprocess(X).astype("float32"))
    with torch.no_grad():
        P_torch = net(x_t).double().numpy()
    P_np = model.forward(X)
    rel = np.max(np.abs(P_np - P_torch) / np.maximum(np.abs(P_torch), 1e-12), axis=0)
    print("max relative difference numpy vs torch per parameter:",
          dict(zip(qc.PARAM_COLS, np.round(rel, 8))))
    if np.max(rel) > 1e-4:
        raise SystemExit("MISMATCH between numpy and torch forward pass")
    print("OK: numpy forward pass matches the PyTorch model")


if __name__ == "__main__":
    main()
