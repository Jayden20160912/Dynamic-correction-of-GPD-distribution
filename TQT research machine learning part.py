"""
Dynamic GPD tail-risk model -- local training on artifacts/dataset.csv (PyTorch).

Architecture: gate_bn32 + 3 history features + pi_min = 0.03  (26,341 trainable parameters)

    x (B, 13)
      Shared trunk:  Linear(13->224, no bias) -> BN(224) -> ReLU
                     Linear(224->88, no bias) -> BN(88)  -> ReLU          -> h(x) (B, 88)
      Base GPD head: Linear(88->2, bias)
                     xi_b(x)   = 0.01 + 0.34 * sigmoid(a_xi(x))
                     beta_b(x) = s * softplus(a_beta(x)) + 1e-6
      BN gate head:  Linear(88->32, no bias) -> BN(32) -> ReLU -> Linear(32->1, bias)
                     pi(x)     = pi_min + (1 - pi_min) * sigmoid(g(x)),  pi_min = 0.03
      Global heavy:  learned scalars theta_xi, theta_beta
                     xi_h      = 0.35 + 0.45 * sigmoid(theta_xi)
                     beta_h    = s * softplus(theta_beta) + 1e-6
    output (B, 5) = [xi_b, beta_b, xi_h, beta_h, pi]

Distribution of the excess loss y = L - u (L = -Y, u = 90% quantile of training losses):
    f(y|x) = (1-pi) f_GPD(y; xi_b, beta_b) + pi f_GPD(y; xi_h, beta_h)
    S(y|x) = (1-pi) S_GPD(y; xi_b, beta_b) + pi S_GPD(y; xi_h, beta_h)
Loss: mixture NLL (logsumexp) + L2 * sum ||W||^2 over the Linear weight matrices.

Run:
    .venv/bin/python "TQT research machine learning part.py"            # reuse artifacts/dataset.csv
    .venv/bin/python "TQT research machine learning part.py" --rebuild  # rebuild the dataset first

Outputs (artifacts/):
    gpd_model.pt             model state_dict (weights + BatchNorm moving statistics + scale)
    preprocess.json          everything needed to turn raw factors into model inputs
"""

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

import features
from features import FEATURE_NAMES, TICKERS, TARGET_MODE, DATA_START, DATA_END


INPUT_DIM = len(FEATURE_NAMES)        # 13
HIDDEN_UNITS = (224, 88)
GATE_UNITS = 32
PI_MIN = 0.03
EXPECTED_TRAINABLE_PARAMS = 26_341

L2 = 1e-4                     # Keras-style kernel L2: L2 * sum(w^2) on every Linear weight matrix
BN_EPS = 1e-3                 # Keras BatchNormalization defaults
BN_MOMENTUM = 0.99            # keep 99% of the old moving statistics

# GPD peaks-over-threshold: loss L = -Y, threshold = this quantile of TRAINING losses
TAIL_QUANTILE = 0.90
TRAIN_FRACTION = 0.8          # chronological split by date: train < split_date <= validation
EARLY_STOP_FRACTION = 0.125   # last 12.5% of TRAINING dates = early-stopping set (validation unused)
CLIP_Z = 5.0
SEED = 42

LEARNING_RATE = 1e-3
CLIPNORM = 1.0                # per-tensor gradient norm clipping (Keras clipnorm)
BATCH_SIZE = 512
MAX_EPOCHS = 150
PATIENCE = 15

CALIBRATION_LEVELS = (0.01, 0.05, 0.10)

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_FILE = "gpd_model.pt"


# ==========================================
# 1. Network
# ==========================================

class KerasBatchNorm1d(nn.Module):
    """BatchNorm matching tf.keras.layers.BatchNormalization.

    Training: normalise with the batch mean and the population (biased) batch variance;
    moving_stat = momentum * moving_stat + (1 - momentum) * batch_stat.
    Inference: normalise with the moving statistics.
    """

    def __init__(self, num_features, eps=BN_EPS, momentum=BN_MOMENTUM):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))   # gamma
        self.bias = nn.Parameter(torch.zeros(num_features))    # beta (offset)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            with torch.no_grad():
                self.running_mean.mul_(self.momentum).add_((1.0 - self.momentum) * mean)
                self.running_var.mul_(self.momentum).add_((1.0 - self.momentum) * var)
        else:
            mean, var = self.running_mean, self.running_var
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return x_hat * self.weight + self.bias


class GateMixtureGPDNet(nn.Module):
    """Shared trunk + input-dependent base GPD + BN gate + global heavy GPD.

    forward(x) -> (B, 5) = [xi_b, beta_b, xi_h, beta_h, pi]
    """

    def __init__(self, scale, input_dim=INPUT_DIM, hidden_units=HIDDEN_UNITS,
                 gate_units=GATE_UNITS, pi_min=PI_MIN):
        super().__init__()
        h1, h2 = hidden_units
        self.pi_min = pi_min

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, h1, bias=False), KerasBatchNorm1d(h1), nn.ReLU(),
            nn.Linear(h1, h2, bias=False), KerasBatchNorm1d(h2), nn.ReLU(),
        )
        self.base_head = nn.Linear(h2, 2, bias=True)            # -> [a_xi, a_beta]
        self.gate = nn.Sequential(
            nn.Linear(h2, gate_units, bias=False), KerasBatchNorm1d(gate_units), nn.ReLU(),
            nn.Linear(gate_units, 1, bias=True),                # -> g(x)
        )
        self.theta_xi = nn.Parameter(torch.zeros(()))           # global heavy component
        self.theta_beta = nn.Parameter(torch.zeros(()))

        # Fixed scale s = median training excess; saved with the model, never trained
        self.register_buffer("scale", torch.tensor(float(scale), dtype=torch.float32))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def kernel_weights(self):
        """All Linear weight matrices (the ones that get the L2 penalty)."""
        return [m.weight for m in self.modules() if isinstance(m, nn.Linear)]

    def forward(self, x):
        h = self.trunk(x)

        a = self.base_head(h)
        xi_b = 0.01 + 0.34 * torch.sigmoid(a[:, 0])
        beta_b = self.scale * F.softplus(a[:, 1]) + 1e-6

        g = self.gate(h).squeeze(-1)
        pi = self.pi_min + (1.0 - self.pi_min) * torch.sigmoid(g)
        pi = pi.clamp(1e-6, 1 - 1e-6)

        ones = torch.ones_like(xi_b)
        xi_h = (0.35 + 0.45 * torch.sigmoid(self.theta_xi)) * ones
        beta_h = (self.scale * F.softplus(self.theta_beta) + 1e-6) * ones

        return torch.stack([xi_b, beta_b, xi_h, beta_h, pi], dim=1)


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_model(model):
    """Parameter count, output shape and output constraints."""
    n = count_trainable_params(model)
    assert n == EXPECTED_TRAINABLE_PARAMS, f"trainable params {n} != {EXPECTED_TRAINABLE_PARAMS}"

    was_training = model.training
    g = torch.Generator().manual_seed(0)
    x = torch.cat([torch.randn(256, INPUT_DIM, generator=g),
                   torch.full((4, INPUT_DIM), 1e3), torch.full((4, INPUT_DIM), -1e3)])
    with torch.no_grad():
        for mode in (True, False):
            model.train(mode)
            snapshot = {k: v.clone() for k, v in model.state_dict().items()}
            out = model(x)
            model.load_state_dict(snapshot)        # do not let the check move BN statistics
            assert out.shape == (len(x), 5), f"output shape {tuple(out.shape)}"
            assert torch.isfinite(out).all(), "non-finite output"
            xi_b, beta_b, xi_h, beta_h, pi = out.unbind(1)
            # float32 sigmoid saturation can hit the end points
            assert ((xi_b >= 0.01) & (xi_b <= 0.35)).all(), "xi_b out of range"
            assert ((xi_h >= 0.35) & (xi_h <= 0.80)).all(), "xi_h out of range"
            assert (beta_b > 0).all() and (beta_h > 0).all(), "beta not positive"
            assert ((pi >= PI_MIN - 1e-6) & (pi < 1)).all(), "pi out of range"
    model.train(was_training)
    print(f"Model check passed: {n:,} trainable parameters, output shape (batch, 5), "
          f"xi_b in [0.01, 0.35], xi_h in [0.35, 0.80], beta > 0, pi in [{PI_MIN}, 1)")


def load_gpd_model(path, scale):
    """Rebuild the network and load saved weights; returns the model in eval mode."""
    model = GateMixtureGPDNet(scale)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


# ==========================================
# 2. Prepare data (artifacts/dataset.csv, 13 factors)
# ==========================================

def prepare_data(rebuild=False):

    X, Y, dates, tickers = features.get_dataset(target_mode=TARGET_MODE, rebuild=rebuild)
    assert X.shape[1] == INPUT_DIM, f"dataset has {X.shape[1]} factors, model expects {INPUT_DIM}"
    print(f"X shape {X.shape}, Y shape {Y.shape}")

    # Chronological split by DATE (no same-day leakage across stocks)
    unique_dates = np.unique(dates)
    split_date = unique_dates[int(TRAIN_FRACTION * len(unique_dates))]
    is_train = dates < split_date
    train_dates = unique_dates[unique_dates < split_date]
    es_date = train_dates[int((1 - EARLY_STOP_FRACTION) * len(train_dates))]
    print(f"Fit < {pd.Timestamp(es_date).date()} <= Early-stop < "
          f"{pd.Timestamp(split_date).date()} <= Validation")

    X_train, X_val = X[is_train].copy(), X[~is_train].copy()
    Y_train, Y_val = Y[is_train], Y[~is_train]
    dates_train = dates[is_train]

    # Fill missing values with TRAIN median, then standardize with TRAIN stats
    median = np.nanmedian(X_train, axis=0)
    for arr in (X_train, X_val):
        rows, cols = np.where(np.isnan(arr))
        arr[rows, cols] = median[cols]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8

    X_train = np.clip((X_train - mean) / std, -CLIP_Z, CLIP_Z)
    X_val = np.clip((X_val - mean) / std, -CLIP_Z, CLIP_Z)

    # Peaks-over-threshold: u from the whole training period
    L_train, L_val = -Y_train[:, 0], -Y_val[:, 0]
    u = float(np.quantile(L_train, TAIL_QUANTILE))
    exc_train, exc_val = L_train > u, L_val > u
    fit_mask = exc_train & (dates_train < es_date)
    es_mask = exc_train & (dates_train >= es_date)

    data = {
        "X_val_all": X_val,
        "dates_val": dates[~is_train],
        "tickers_val": tickers[~is_train],
        "Xg_fit": X_train[fit_mask], "yg_fit": (L_train[fit_mask] - u).astype("float32"),
        "Xg_es": X_train[es_mask], "yg_es": (L_train[es_mask] - u).astype("float32"),
        "Xg_val": X_val[exc_val], "yg_val": (L_val[exc_val] - u).astype("float32"),
        "yg_train": (L_train[exc_train] - u).astype("float32"),
        "mean": mean, "std": std, "median": median,
        "u": u, "split_date": split_date, "early_stop_date": es_date,
    }
    print(f"Threshold u = {u:.4%} loss; exceedances fit {len(data['yg_fit'])}, "
          f"early-stop {len(data['yg_es'])}, validation {len(data['yg_val'])}")
    return data


# ==========================================
# 3. Mixture GPD likelihood
# ==========================================

def gpd_logpdf(y, xi, beta):
    return -torch.log(beta) - (1.0 + 1.0 / xi) * torch.log1p(xi * y / beta)


def gpd_logsf(y, xi, beta):
    return -torch.log1p(xi * y / beta) / xi


def mixture_nll(y, params):
    """Per-sample negative log-likelihood of the two-component GPD mixture (logsumexp)."""
    y = y.reshape(-1)
    xi_b, beta_b, xi_h, beta_h, pi = params.unbind(1)
    log_f = torch.logsumexp(torch.stack([
        torch.log1p(-pi) + gpd_logpdf(y, xi_b, beta_b),
        torch.log(pi) + gpd_logpdf(y, xi_h, beta_h),
    ]), dim=0)
    return -log_f


def mixture_survival(y, params):
    """S(y|x) = (1-pi) S_GPD(y; xi_b, beta_b) + pi S_GPD(y; xi_h, beta_h)."""
    y = y.reshape(-1)
    xi_b, beta_b, xi_h, beta_h, pi = params.unbind(1)
    return (1 - pi) * torch.exp(gpd_logsf(y, xi_b, beta_b)) + pi * torch.exp(gpd_logsf(y, xi_h, beta_h))


def l2_penalty(model):
    """Keras-style kernel_regularizer l2(L2): L2 * sum(w^2) on every Linear weight matrix."""
    return L2 * sum((w ** 2).sum() for w in model.kernel_weights())


# ==========================================
# 4. Evaluation metrics
# ==========================================

@torch.no_grad()
def evaluate(model, X, y):
    """Eval-mode mixture NLL (early-stopping monitor) and calibration of S(y)."""
    model.eval()
    params = model(X)
    nll = mixture_nll(y, params).mean().item()
    S = mixture_survival(y, params)
    observed = {a: (S < a).float().mean().item() for a in CALIBRATION_LEVELS}
    mae_pp = 100.0 * float(np.mean([abs(observed[a] - a) for a in CALIBRATION_LEVELS]))
    return {"nll": nll, "calibration_mae_pp": mae_pp, "observed_share": observed,
            "min_S": S.min().item()}


def _tensor(a):
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


# ==========================================
# 5. Training loop (Adam, per-tensor clipnorm, early stopping on the early-stop set)
# ==========================================

def train(model, data):

    X_fit, y_fit = _tensor(data["Xg_fit"]), _tensor(data["yg_fit"])
    X_es, y_es = _tensor(data["Xg_es"]), _tensor(data["yg_es"])

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 betas=(0.9, 0.999), eps=1e-7)
    shuffle_gen = torch.Generator().manual_seed(SEED)

    history = {"loss": [], "es_nll": [], "es_calibration_mae_pp": []}
    best_nll, best_state, best_epoch, wait = float("inf"), None, 0, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(X_fit), generator=shuffle_gen)
        total, count = 0.0, 0
        for start in range(0, len(perm), BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            if len(idx) < 2:                               # BatchNorm needs > 1 sample
                continue
            loss = mixture_nll(y_fit[idx], model(X_fit[idx])).mean() + l2_penalty(model)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for p in model.parameters():                 # Keras clipnorm is per tensor
                if p.grad is not None:
                    torch.nn.utils.clip_grad_norm_([p], CLIPNORM)
            optimizer.step()

            total += loss.item() * len(idx)
            count += len(idx)

        es = evaluate(model, X_es, y_es)
        history["loss"].append(total / count)
        history["es_nll"].append(es["nll"])
        history["es_calibration_mae_pp"].append(es["calibration_mae_pp"])
        print(f"Epoch {epoch:3d}  loss {total / count:.4f}  early-stop NLL {es['nll']:.4f}  "
              f"calib MAE {es['calibration_mae_pp']:.2f} pp")

        if es["nll"] < best_nll:
            best_nll, best_epoch, wait = es["nll"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())   # includes BN moving statistics
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    return history, best_epoch


# ==========================================
# 6. Save model + preprocessing
# ==========================================

def save_artifacts(model, history, best_epoch, results, data, scale):

    ARTIFACT_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), ARTIFACT_DIR / MODEL_FILE)

    with torch.no_grad():
        heavy = model(torch.zeros(2, INPUT_DIM))[0]
    meta = {
        "framework": "pytorch",
        "model_file": MODEL_FILE,
        "model_class": "GateMixtureGPDNet",
        "architecture": "gate_bn32 + hist3 + pi_min: trunk Linear(13,224,no bias)-BN-ReLU-"
                        "Linear(224,88,no bias)-BN-ReLU; base head Linear(88,2) xi_b=0.01+0.34*sigmoid, "
                        "beta_b=s*softplus+1e-6; gate Linear(88,32,no bias)-BN-ReLU-Linear(32,1) "
                        "pi=pi_min+(1-pi_min)*sigmoid; heavy xi_h=0.35+0.45*sigmoid(theta), "
                        "beta_h=s*softplus(theta)+1e-6",
        "output_columns": ["xi_b", "beta_b", "xi_h", "beta_h", "pi"],
        "trainable_params": count_trainable_params(model),
        "pi_min": PI_MIN,
        "heavy_xi": float(heavy[2]),
        "heavy_beta": float(heavy[3]),
        "feature_names": FEATURE_NAMES,
        "feature_mean": [float(x) for x in data["mean"]],
        "feature_std": [float(x) for x in data["std"]],
        "feature_median_fill": [float(x) for x in data["median"]],
        "clip_z": CLIP_Z,
        "threshold_u_loss": float(data["u"]),
        "gpd_scale": float(scale),
        "tail_quantile": TAIL_QUANTILE,
        "target_mode": TARGET_MODE,
        "data_start": DATA_START,
        "data_end": DATA_END,
        "train_val_split_date": str(pd.Timestamp(data["split_date"]).date()),
        "early_stop_start_date": str(pd.Timestamp(data["early_stop_date"]).date()),
        "n_exceedances_fit": int(len(data["yg_fit"])),
        "n_exceedances_early_stop": int(len(data["yg_es"])),
        "n_exceedances_val": int(len(data["yg_val"])),
        "tickers": TICKERS,
        "hidden_units": list(HIDDEN_UNITS),
        "gate_units": GATE_UNITS,
        "batchnorm": {"eps": BN_EPS, "momentum_keras": BN_MOMENTUM, "variance": "population"},
        "l2": L2,
        "dropout": 0.0,
        "learning_rate": LEARNING_RATE,
        "clipnorm": CLIPNORM,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "seed": SEED,
        "data_sources": "artifacts/dataset.csv (Alpaca, FRED VIXCLS, SEC EDGAR)",
        "epochs_run": len(history["loss"]),
        "best_epoch": best_epoch,
        "results": results,
        "loss_history": [float(x) for x in history["loss"]],
        "early_stop_nll_history": [float(x) for x in history["es_nll"]],
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    (ARTIFACT_DIR / "preprocess.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved to {ARTIFACT_DIR}: {MODEL_FILE}, preprocess.json")


def main():

    parser = argparse.ArgumentParser(description="Train the gated GPD mixture model")
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild artifacts/dataset.csv from data_cache/ / the APIs before training")
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)

    data = prepare_data(rebuild=args.rebuild)

    # Fixed scale s: median of training excess losses
    SCALE = max(float(np.median(data["yg_train"])), 1e-4)
    print(f"scale s = {SCALE:.10f}")

    model = GateMixtureGPDNet(SCALE)
    check_model(model)
    print(model)

    history, best_epoch = train(model, data)

    results = {}
    for name in ("es", "val"):
        r = evaluate(model, _tensor(data[f"Xg_{name}"]), _tensor(data[f"yg_{name}"]))
        results[{"es": "early_stop", "val": "validation"}[name]] = {
            "nll": r["nll"], "calibration_mae_pp": r["calibration_mae_pp"], "min_S": r["min_S"],
            "observed_share_S_below": {str(k): v for k, v in r["observed_share"].items()}}
    print(f"\nEpochs run: {len(history['loss'])}, best epoch: {best_epoch}")
    for name, r in results.items():
        print(f"{name:>10}: mixture NLL {r['nll']:.4f}  calib MAE {r['calibration_mae_pp']:.2f} pp  "
              + "  ".join(f"S<{k}: {v:.2%}" for k, v in r["observed_share_S_below"].items())
              + f"  min S {r['min_S']:.2e}")

    with torch.no_grad():
        preds = model(_tensor(data["X_val_all"][-5:])).numpy()
    print(f"Heavy component: xi_h = {preds[0, 2]:.4f}, beta_h = {preds[0, 3]:.5f}")
    for i in range(len(preds)):
        print(f"{data['tickers_val'][-5 + i]} {pd.Timestamp(data['dates_val'][-5 + i]).date()} "
              f"xi_b={preds[i, 0]:.4f} beta_b={preds[i, 1]:.5f} pi={preds[i, 4]:.3f}")

    save_artifacts(model, history, best_epoch, results, data, SCALE)


if __name__ == "__main__":
    main()
