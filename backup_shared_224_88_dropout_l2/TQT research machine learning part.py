"""
Dynamic GPD tail-risk model -- local training on Alpaca + FRED + SEC EDGAR data (PyTorch).

Architecture: Shared (224, 88)
    one MLP whose two hidden layers are shared by both GPD parameters:
    Linear(13->224, no bias) -> BN -> ReLU -> Dropout -> Linear(224->88, no bias) -> BN -> ReLU
    -> Dropout -> Linear(88->2) -> [xi_raw, beta_raw]
    xi   = 0.01 + 0.79 * sigmoid(xi_raw)
    beta = scale * softplus(beta_raw) + 1e-6

Run:
    .venv/bin/python "TQT research machine learning part.py"            # reuse artifacts/dataset.csv
    .venv/bin/python "TQT research machine learning part.py" --rebuild  # rebuild the dataset first

By default the saved artifacts/dataset.csv is reused, so no API is called. With --rebuild
(or when dataset.csv is missing / built with a different TARGET_MODE) the dataset is rebuilt
from data_cache/, downloading only what is not cached yet.

Outputs (artifacts/):
    dataset.csv              full X / Y table
    gpd_model.pt             model state_dict (weights + BatchNorm moving statistics + scale)
    preprocess.json          everything needed to turn raw factors into model inputs

Load later with:
    load_gpd_model("artifacts/gpd_model.pt", meta["gpd_scale"])   # returns the model in eval mode
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


INPUT_DIM = 10
HIDDEN_UNITS = (224, 88)
EXPECTED_TRAINABLE_PARAMS = 23_426   # 13 inputs
L2 = 1e-2                     # L2 on the hidden-layer weights (Keras-style: L2 * sum(w^2)); was 1e-4
DROPOUT = 0.2                 # dropout rate after each hidden ReLU (0 = no dropout)

BN_EPS = 1e-3                 # Keras BatchNormalization defaults
BN_MOMENTUM = 0.99            # Keras convention: keep 99% of the old moving statistics

LAMBDA_TAIL = 0.0
LAMBDA_SEVERITY = 1.0

# GPD peaks-over-threshold: loss L = -Y, threshold = this quantile of TRAINING losses
TAIL_QUANTILE = 0.90
TRAIN_FRACTION = 0.8         # chronological split by date
CLIP_Z = 5.0
SEED = 42

LEARNING_RATE = 1e-3
CLIPNORM = 1.0               # Keras clipnorm: each gradient tensor is clipped to this norm separately
BATCH_SIZE = 32
MAX_EPOCHS = 150
PATIENCE = 15

CALIBRATION_LEVELS = (0.01, 0.05, 0.10)

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_FILE = "gpd_model.pt"


# ==========================================
# 1. Network: Keras-compatible BatchNorm + Shared (224, 88) MLP
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
        self.weight = nn.Parameter(torch.ones(num_features))   # gamma (scale)
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


class SharedGPDNet(nn.Module):
    """Shared (224, 88) MLP that outputs the conditional GPD parameters [xi, beta]."""

    def __init__(self, scale, input_dim=INPUT_DIM, hidden_units=HIDDEN_UNITS, dropout=None):
        super().__init__()
        h1, h2 = hidden_units
        p = DROPOUT if dropout is None else dropout
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, h1, bias=False),   # hidden layer 1
            KerasBatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(p),                          # active in train mode only; no parameters
            nn.Linear(h1, h2, bias=False),          # hidden layer 2
            KerasBatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(p),
        )
        self.output = nn.Linear(h2, 2, bias=True)   # -> [xi_raw, beta_raw]

        # Fixed scale computed from training data only; saved with the model, never trained
        self.register_buffer("scale", torch.tensor(float(scale), dtype=torch.float32))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def hidden_weights(self):
        """Hidden-layer weight matrices (the ones that get the L2 penalty)."""
        return [m.weight for m in self.hidden if isinstance(m, nn.Linear)]

    def forward(self, x):
        raw = self.output(self.hidden(x))
        xi_raw, beta_raw = raw[:, 0:1], raw[:, 1:2]
        xi = 0.01 + 0.79 * torch.sigmoid(xi_raw)               # shape parameter in (0.01, 0.80)
        beta = self.scale * F.softplus(beta_raw) + 1e-6        # scale parameter > 0
        return torch.cat([xi, beta], dim=1)                    # (batch_size, 2): [xi, beta]


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
            assert out.shape == (len(x), 2), f"output shape {tuple(out.shape)}"
            xi, beta = out[:, 0], out[:, 1]
            assert torch.isfinite(out).all(), "non-finite output"
            # (0.01, 0.80) mathematically; float32 sigmoid saturation can hit the end points
            assert ((xi >= 0.01) & (xi <= 0.80)).all(), "xi out of range"
            assert (beta > 0).all(), "beta not positive"
    model.train(was_training)
    print(f"Model check passed: {n:,} trainable parameters, output shape (batch, 2), "
          f"xi in [0.01, 0.80], beta > 0")


def load_gpd_model(path, scale):
    """Rebuild the network and load saved weights; returns the model in eval mode."""
    model = SharedGPDNet(scale)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


# ==========================================
# 2. Prepare Data (real data: Alpaca + FRED + SEC EDGAR) -- unchanged
# ==========================================

def prepare_data(rebuild=False):

    X, Y, dates, tickers = features.get_dataset(target_mode=TARGET_MODE, rebuild=rebuild)
    print(f"X shape {X.shape}, Y shape {Y.shape}")

    # Chronological split by DATE (no same-day leakage across stocks)
    unique_dates = np.unique(dates)
    split_date = unique_dates[int(TRAIN_FRACTION * len(unique_dates))]
    is_train = dates < split_date
    print(f"Train < {pd.Timestamp(split_date).date()} <= Validation")

    X_train, X_val = X[is_train].copy(), X[~is_train].copy()
    Y_train, Y_val = Y[is_train], Y[~is_train]

    # Fill missing SEC similarity with TRAIN median, then standardize with TRAIN stats
    median = np.nanmedian(X_train, axis=0)
    for arr in (X_train, X_val):
        rows, cols = np.where(np.isnan(arr))
        arr[rows, cols] = median[cols]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8

    X_train = np.clip((X_train - mean) / std, -CLIP_Z, CLIP_Z)
    X_val = np.clip((X_val - mean) / std, -CLIP_Z, CLIP_Z)

    # Peaks-over-threshold: GPD is fitted on loss exceedances, not raw returns
    L_train, L_val = -Y_train[:, 0], -Y_val[:, 0]
    u = float(np.quantile(L_train, TAIL_QUANTILE))
    train_mask, val_mask = L_train > u, L_val > u

    data = {
        "X_val_all": X_val,
        "dates_val": dates[~is_train],
        "tickers_val": tickers[~is_train],
        "Xg_train": X_train[train_mask],
        "yg_train": (L_train[train_mask] - u).astype("float32"),
        "Xg_val": X_val[val_mask],
        "yg_val": (L_val[val_mask] - u).astype("float32"),
        "mean": mean, "std": std, "median": median,
        "u": u, "split_date": split_date,
    }
    print(f"Threshold u = {u:.4%} loss; exceedances train {len(data['yg_train'])}, "
          f"val {len(data['yg_val'])}")
    return data


# ==========================================
# 3. GPD NLL + Severity-Weighted Tail Loss (same math as before)
# ==========================================

def gpd_nll(y, params):
    """Per-sample GPD negative log-likelihood (pure NLL, no penalty)."""
    y = y.reshape(-1, 1)
    xi, beta = params[:, 0:1], params[:, 1:2]
    log_term = torch.log1p(xi * y / beta)
    return (torch.log(beta) + (1.0 + 1.0 / xi) * log_term).squeeze(-1)


def gpd_loss(y, params, scale):
    """Per-sample training loss: NLL + LAMBDA_TAIL * severity weight * tail penalty."""
    y = y.reshape(-1, 1)
    xi, beta = params[:, 0:1], params[:, 1:2]

    # GPD Negative Log-Likelihood
    log_term = torch.log1p(xi * y / beta)
    nll = torch.log(beta) + (1.0 + 1.0 / xi) * log_term

    # Tail Probability Penalty
    tail_loss = log_term / xi

    # Severity Weight
    weight = 1.0 + LAMBDA_SEVERITY * y / scale

    # Combined Loss
    return (nll + LAMBDA_TAIL * weight * tail_loss).squeeze(-1)


def l2_penalty(model):
    """Keras-style kernel_regularizer l2(L2): L2 * sum(w^2) on the hidden-layer weights."""
    return L2 * sum((w ** 2).sum() for w in model.hidden_weights())


# ==========================================
# 4. Evaluation metrics used for model comparison
# ==========================================

def gpd_survival_torch(y, params):
    """S(y) = P(excess > y) = (1 + xi*y/beta)^(-1/xi)."""
    xi, beta = params[:, 0], params[:, 1]
    return torch.exp(-torch.log1p(xi * y / beta) / xi)


@torch.no_grad()
def evaluate(model, X, y, scale):
    """Eval-mode objective (as monitored by early stopping), pure NLL and calibration."""
    model.eval()
    params = model(X)
    objective = (gpd_loss(y, params, scale).mean() + l2_penalty(model)).item()
    nll = gpd_nll(y, params).mean().item()
    S = gpd_survival_torch(y, params)
    observed = {a: (S < a).float().mean().item() for a in CALIBRATION_LEVELS}
    mae_pp = 100.0 * float(np.mean([abs(observed[a] - a) for a in CALIBRATION_LEVELS]))
    return {"objective": objective, "nll": nll, "calibration_mae_pp": mae_pp,
            "observed_share": observed}


# ==========================================
# 5. Training loop (Adam, per-tensor clipnorm, early stopping with best-weight restore)
# ==========================================

def train(model, data, scale):

    X_tr = torch.from_numpy(np.ascontiguousarray(data["Xg_train"], dtype=np.float32))
    y_tr = torch.from_numpy(data["yg_train"])
    X_va = torch.from_numpy(np.ascontiguousarray(data["Xg_val"], dtype=np.float32))
    y_va = torch.from_numpy(data["yg_val"])

    # Keras Adam defaults: beta_1=0.9, beta_2=0.999, epsilon=1e-7
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 betas=(0.9, 0.999), eps=1e-7)
    shuffle_gen = torch.Generator().manual_seed(SEED)

    history = {"loss": [], "val_loss": [], "val_nll": [], "val_calibration_mae_pp": []}
    best_val, best_state, best_epoch, wait = float("inf"), None, 0, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(X_tr), generator=shuffle_gen)
        total, count = 0.0, 0
        for start in range(0, len(perm), BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            params = model(X_tr[idx])
            loss = gpd_loss(y_tr[idx], params, scale).mean() + l2_penalty(model)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for p in model.parameters():                 # Keras clipnorm is per tensor
                if p.grad is not None:
                    torch.nn.utils.clip_grad_norm_([p], CLIPNORM)
            optimizer.step()

            total += loss.item() * len(idx)
            count += len(idx)

        val = evaluate(model, X_va, y_va, scale)
        history["loss"].append(total / count)
        history["val_loss"].append(val["objective"])
        history["val_nll"].append(val["nll"])
        history["val_calibration_mae_pp"].append(val["calibration_mae_pp"])
        print(f"Epoch {epoch:3d}  loss {total / count:.4f}  val_loss {val['objective']:.4f}  "
              f"val_nll {val['nll']:.4f}  val_calib_MAE {val['calibration_mae_pp']:.2f} pp")

        # EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
        if val["objective"] < best_val:
            best_val, best_epoch, wait = val["objective"], epoch, 0
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

def save_artifacts(model, history, best_epoch, final_val, data, scale):

    ARTIFACT_DIR.mkdir(exist_ok=True)

    # Model weights + BN moving statistics + fixed scale (replaces the old Keras files)
    torch.save(model.state_dict(), ARTIFACT_DIR / MODEL_FILE)
    for old in ("gpd_model.keras", "gpd_model.weights.h5"):
        (ARTIFACT_DIR / old).unlink(missing_ok=True)

    meta = {
        "framework": "pytorch",
        "model_file": MODEL_FILE,
        "architecture": "Shared (224, 88): Linear(13,224,no bias)-BN-ReLU-Linear(224,88,no bias)"
                        "-BN-ReLU-Linear(88,2), dropout after each hidden ReLU; xi=0.01+0.79*sigmoid, beta=scale*softplus+1e-6",
        "trainable_params": count_trainable_params(model),
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
        "n_exceedances_train": int(len(data["yg_train"])),
        "n_exceedances_val": int(len(data["yg_val"])),
        "tickers": TICKERS,
        "hidden_units": list(HIDDEN_UNITS),
        "batchnorm": {"eps": BN_EPS, "momentum_keras": BN_MOMENTUM, "variance": "population"},
        "l2": L2,
        "dropout": DROPOUT,
        "lambda_tail": LAMBDA_TAIL,
        "lambda_severity": LAMBDA_SEVERITY,
        "learning_rate": LEARNING_RATE,
        "clipnorm": CLIPNORM,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "seed": SEED,
        "data_sources": "Alpaca (sip, adjustment=all), FRED VIXCLS, SEC EDGAR",
        "epochs_run": len(history["loss"]),
        "best_epoch": best_epoch,
        "best_val_loss": float(min(history["val_loss"])),
        "val_nll": final_val["nll"],
        "val_calibration_mae_pp": final_val["calibration_mae_pp"],
        "val_observed_share_S_below": {str(k): v for k, v in final_val["observed_share"].items()},
        "loss_history": [float(x) for x in history["loss"]],
        "val_loss_history": [float(x) for x in history["val_loss"]],
        "val_nll_history": [float(x) for x in history["val_nll"]],
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    (ARTIFACT_DIR / "preprocess.json").write_text(json.dumps(meta, indent=2))

    print(f"Saved to {ARTIFACT_DIR}: {MODEL_FILE}, preprocess.json "
          f"(old gpd_model.keras / gpd_model.weights.h5 removed)")


def main():

    parser = argparse.ArgumentParser(description="Train the dynamic GPD model")
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild artifacts/dataset.csv from data_cache/ / the APIs before training")
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)

    data = prepare_data(rebuild=args.rebuild)

    # Reference scale for GPD parameters and loss (training exceedances only)
    SCALE = max(float(np.median(data["yg_train"])), 1e-4)
    print(f"scale = {SCALE:.10f}")

    model = SharedGPDNet(SCALE)
    check_model(model)
    print(model)

    history, best_epoch = train(model, data, SCALE)

    X_va = torch.from_numpy(np.ascontiguousarray(data["Xg_val"], dtype=np.float32))
    final_val = evaluate(model, X_va, torch.from_numpy(data["yg_val"]), SCALE)
    print(f"\nEpochs run: {len(history['loss'])}, best epoch: {best_epoch}, "
          f"best val_loss: {min(history['val_loss']):.4f}")
    print(f"Validation pure GPD NLL: {final_val['nll']:.4f}")
    print(f"Validation calibration MAE (1%, 5%, 10%): {final_val['calibration_mae_pp']:.2f} pp  "
          + "  ".join(f"S<{a:.0%}: {s:.2%}" for a, s in final_val["observed_share"].items()))

    # Dynamic GPD parameters for the latest validation rows
    with torch.no_grad():
        preds = model(torch.from_numpy(np.ascontiguousarray(data["X_val_all"][-5:],
                                                            dtype=np.float32))).numpy()
    for i in range(len(preds)):
        print(f"{data['tickers_val'][-5 + i]} {pd.Timestamp(data['dates_val'][-5 + i]).date()} "
              f"epsilon={preds[i, 0]:.4f} beta={preds[i, 1]:.5f}")

    save_artifacts(model, history, best_epoch, final_val, data, SCALE)


if __name__ == "__main__":
    main()
