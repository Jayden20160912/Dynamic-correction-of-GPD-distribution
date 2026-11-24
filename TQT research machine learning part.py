"""
Two-stage dynamic tail-risk model -- local training on artifacts/dataset.csv (PyTorch).

    P(L > l | x) = p_u(x) * S(l - u | x),   l > u          (L = -Y = next-day loss)

Stage 1  exceedance network p_u(x) = P(L > u | x)            23,337 trainable parameters
    x (B, 13) -> Linear(13->224, no bias) -> BN(224) -> ReLU
              -> Linear(224->88, no bias) -> BN(88)  -> ReLU
              -> Linear(88->1, bias, init logit(train exceedance rate)) -> sigmoid
    trained on ALL stock-days of the fit period, binary cross-entropy + L2 1e-4

Stage 2  gated GPD mixture S(y | x), y = L - u given L > u    26,341 trainable parameters
    Shared trunk:  Linear(13->224, no bias) -> BN(224) -> ReLU
                   Linear(224->88, no bias) -> BN(88)  -> ReLU          -> h(x) (B, 88)
    Base GPD head: Linear(88->2, bias)
                   xi_b(x)   = 0.01 + 0.34 * sigmoid(a_xi(x))
                   beta_b(x) = s * softplus(a_beta(x)) + 1e-6
    BN gate head:  Linear(88->32, no bias) -> BN(32) -> ReLU -> Linear(32->1, bias init -2)
                   pi(x)     = pi_min + (1 - pi_min) * sigmoid(g(x)),  pi_min = 0.01
    Global heavy:  learned scalars theta_xi (init 0), theta_beta (init 1.85 -> wide heavy solution)
                   xi_h      = 0.35 + 0.45 * sigmoid(theta_xi)
                   beta_h    = s * softplus(theta_beta) + 1e-6
    output (B, 5) = [xi_b, beta_b, xi_h, beta_h, pi]
    S(y|x) = (1-pi) S_GPD(y; xi_b, beta_b) + pi S_GPD(y; xi_h, beta_h)
    trained on exceedances (L > u) of the fit period, mixture NLL (logsumexp) + L2 1e-4

The two networks share no parameters and the likelihood factorises (Bernoulli x GPD), so
fitting them separately equals fitting them jointly. Both use the same preprocessing, threshold
u and date split, stored in preprocess.json.

Risk measures (section 8):
    tail probability  P(L > l | x) = p_u(x) S(l - u | x)
    VaR_alpha         solve p_u(x) S(y|x) = alpha, VaR = u + y  (NaN if p_u(x) <= alpha)
    ES_alpha          VaR + sum_k w_k S_k(y)(beta_k + xi_k y)/(1 - xi_k) / S(y), w_b = 1-pi, w_h = pi

Run:
    .venv/bin/python "TQT research machine learning part.py"              # stage 2, stage 1, validation
    .venv/bin/python "TQT research machine learning part.py" --stage gpd  # stage 2 only
    .venv/bin/python "TQT research machine learning part.py" --stage pu   # stage 1 + validation
                                                                          # (needs stage 2 artifacts)
    add --rebuild to rebuild artifacts/dataset.csv first

Outputs (artifacts/):
    gpd_model.pt      stage 2 state_dict (weights + BatchNorm moving statistics + scale)
    preprocess.json   preprocessing, threshold u, split dates, stage 2 settings and results
    pu_model.pt       stage 1 state_dict
    pu_meta.json      stage 1 settings and results + two-stage VaR backtest
quantconnect/gpd_mixture_model.json   both stages as plain JSON (tail_model.to_json), for QuantConnect

Single implementations shared with the rest of the project:
    factor_core.py   the 13 factors (via features.py)
    tail_model.py    numpy inference: forward pass, S, VaR, ES, risk score, backtest helpers.
                     After training, check_numpy_forward() verifies that tail_model.py reproduces
                     the PyTorch networks below; change both together if the architecture changes.
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
from scipy import stats

import features
import tail_model
from features import FEATURE_NAMES, TICKERS, TARGET_MODE, DATA_START, DATA_END


INPUT_DIM = len(FEATURE_NAMES)        # 13
HIDDEN_UNITS = (224, 88)

L2 = 1e-4                     # Keras-style kernel L2: L2 * sum(w^2) on every Linear weight matrix
BN_EPS = 1e-3                 # Keras BatchNormalization defaults
BN_MOMENTUM = 0.99            # keep 99% of the old moving statistics
CLIPNORM = 1.0                # per-tensor gradient norm clipping (Keras clipnorm)

# Peaks-over-threshold: loss L = -Y, threshold = this quantile of TRAINING losses (all rows)
TAIL_QUANTILE = 0.90
TRAIN_FRACTION = 0.8          # chronological split by date: train < split_date <= validation
EARLY_STOP_FRACTION = 0.125   # last 12.5% of TRAINING dates = early-stopping set (validation unused)
CLIP_Z = 5.0
SEED = 42

# Stage 2: gated GPD mixture
GATE_UNITS = 32
PI_MIN = 0.01
THETA_BETA_INIT = 1.85        # beta_h starts at ~2s -> converges to the wide heavy solution
GATE_BIAS_INIT = -2.0
EXPECTED_TRAINABLE_PARAMS = 26_341
LEARNING_RATE = 3e-4
BATCH_SIZE = 512
MAX_EPOCHS = 150
PATIENCE = 15
CALIBRATION_LEVELS = (0.01, 0.025, 0.05, 0.10, 0.25, 0.50)
MAE_LEVELS = (0.01, 0.05, 0.10)   # levels averaged into the "calib MAE" printed per epoch

# Stage 1: exceedance network
PU_EXPECTED_TRAINABLE_PARAMS = 23_337
PU_LEARNING_RATE = 1e-3
PU_BATCH_SIZE = 2048
PU_MAX_EPOCHS = 60
PU_PATIENCE = 6

# Two-stage VaR backtest levels (alpha = expected breach rate)
VAR_ALPHAS = (0.001, 0.005, 0.01, 0.02)

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_FILE = "gpd_model.pt"
PU_MODEL_FILE = "pu_model.pt"
QC_MODEL_JSON = Path(__file__).resolve().parent / "quantconnect" / "gpd_mixture_model.json"


# ==========================================
# 1. Networks
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


def _xavier_init(module):
    """Xavier-uniform weights and zero biases on every Linear layer."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def _kernel_weights(module):
    """All Linear weight matrices (the ones that get the L2 penalty)."""
    return [m.weight for m in module.modules() if isinstance(m, nn.Linear)]


def _trunk(input_dim, hidden_units):
    h1, h2 = hidden_units
    return nn.Sequential(
        nn.Linear(input_dim, h1, bias=False), KerasBatchNorm1d(h1), nn.ReLU(),
        nn.Linear(h1, h2, bias=False), KerasBatchNorm1d(h2), nn.ReLU(),
    )


class ExceedanceNet(nn.Module):
    """Stage 1: forward(x) -> logit of p_u(x) = P(L > u | x), shape (B,)."""

    def __init__(self, base_rate=1 - TAIL_QUANTILE, input_dim=INPUT_DIM,
                 hidden_units=HIDDEN_UNITS):
        super().__init__()
        self.trunk = _trunk(input_dim, hidden_units)
        self.out = nn.Linear(hidden_units[1], 1, bias=True)
        _xavier_init(self)
        with torch.no_grad():                  # initial prediction = average exceedance rate
            self.out.bias.fill_(float(np.log(base_rate / (1.0 - base_rate))))

    def kernel_weights(self):
        return _kernel_weights(self)

    def forward(self, x):
        return self.out(self.trunk(x)).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, x):
        self.eval()
        return torch.sigmoid(self(x))


class GateMixtureGPDNet(nn.Module):
    """Stage 2: shared trunk + input-dependent base GPD + BN gate + global heavy GPD.

    forward(x) -> (B, 5) = [xi_b, beta_b, xi_h, beta_h, pi]
    """

    def __init__(self, scale, input_dim=INPUT_DIM, hidden_units=HIDDEN_UNITS,
                 gate_units=GATE_UNITS, pi_min=PI_MIN):
        super().__init__()
        h2 = hidden_units[1]
        self.pi_min = pi_min

        self.trunk = _trunk(input_dim, hidden_units)
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
        _xavier_init(self)
        with torch.no_grad():
            self.gate[-1].bias.fill_(GATE_BIAS_INIT)            # pi starts small
            self.theta_xi.fill_(0.0)
            self.theta_beta.fill_(THETA_BETA_INIT)              # wide heavy component

    def kernel_weights(self):
        return _kernel_weights(self)

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


def _check_inputs():
    g = torch.Generator().manual_seed(0)
    return torch.cat([torch.randn(256, INPUT_DIM, generator=g),
                      torch.full((4, INPUT_DIM), 1e3), torch.full((4, INPUT_DIM), -1e3)])


def _run_both_modes(model, x, check):
    """Run check(model(x)) in train and eval mode without moving BN statistics."""
    was_training = model.training
    with torch.no_grad():
        for mode in (True, False):
            model.train(mode)
            snapshot = {k: v.clone() for k, v in model.state_dict().items()}
            out = model(x)
            model.load_state_dict(snapshot)
            check(out)
    model.train(was_training)


def check_model(model):
    """Stage 2: parameter count, output shape and output constraints."""
    n = count_trainable_params(model)
    assert n == EXPECTED_TRAINABLE_PARAMS, f"trainable params {n} != {EXPECTED_TRAINABLE_PARAMS}"
    x = _check_inputs()

    def check(out):
        assert out.shape == (len(x), 5), f"output shape {tuple(out.shape)}"
        assert torch.isfinite(out).all(), "non-finite output"
        xi_b, beta_b, xi_h, beta_h, pi = out.unbind(1)
        # float32 sigmoid saturation can hit the end points
        assert ((xi_b >= 0.01) & (xi_b <= 0.35)).all(), "xi_b out of range"
        assert ((xi_h >= 0.35) & (xi_h <= 0.80)).all(), "xi_h out of range"
        assert (beta_b > 0).all() and (beta_h > 0).all(), "beta not positive"
        assert ((pi >= PI_MIN - 1e-6) & (pi < 1)).all(), "pi out of range"

    _run_both_modes(model, x, check)
    print(f"Stage 2 model check passed: {n:,} trainable parameters, output shape (batch, 5), "
          f"xi_b in [0.01, 0.35], xi_h in [0.35, 0.80], beta > 0, pi in [{PI_MIN}, 1)")


def check_pu_model(model):
    """Stage 1: parameter count, output shape, finite logits."""
    n = count_trainable_params(model)
    assert n == PU_EXPECTED_TRAINABLE_PARAMS, f"trainable params {n} != {PU_EXPECTED_TRAINABLE_PARAMS}"
    x = _check_inputs()

    def check(out):
        assert out.shape == (len(x),), f"output shape {tuple(out.shape)}"
        assert torch.isfinite(out).all(), "non-finite output"

    _run_both_modes(model, x, check)
    print(f"Stage 1 model check passed: {n:,} trainable parameters, output shape (batch,)")


def load_gpd_model(path, scale, pi_min=PI_MIN):
    """Rebuild the stage 2 network and load saved weights; returns the model in eval mode."""
    model = GateMixtureGPDNet(scale, pi_min=pi_min)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def load_pu_model(path):
    """Rebuild the stage 1 network and load saved weights; returns the model in eval mode."""
    model = ExceedanceNet()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


# ==========================================
# 2. Prepare data (artifacts/dataset.csv, 13 factors) -- shared by both stages
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

    # Peaks-over-threshold: u from ALL rows of the training period
    L_train, L_val = -Y_train[:, 0], -Y_val[:, 0]
    u = float(np.quantile(L_train, TAIL_QUANTILE))
    exc_train, exc_val = L_train > u, L_val > u
    is_fit = dates_train < es_date
    fit_mask = exc_train & is_fit
    es_mask = exc_train & ~is_fit

    data = {
        # stage 1: all rows, label e = 1{L > u}
        "X_fit_all": X_train[is_fit], "e_fit": exc_train[is_fit].astype("float32"),
        "X_es_all": X_train[~is_fit], "e_es": exc_train[~is_fit].astype("float32"),
        "X_val_all": X_val, "L_val_all": L_val, "e_val": exc_val.astype("float32"),
        "dates_val": dates[~is_train],
        "tickers_val": tickers[~is_train],
        # stage 2: exceedances only, target y = L - u
        "Xg_fit": X_train[fit_mask], "yg_fit": (L_train[fit_mask] - u).astype("float32"),
        "Xg_es": X_train[es_mask], "yg_es": (L_train[es_mask] - u).astype("float32"),
        "Xg_val": X_val[exc_val], "yg_val": (L_val[exc_val] - u).astype("float32"),
        "yg_train": (L_train[exc_train] - u).astype("float32"),
        "mean": mean, "std": std, "median": median,
        "u": u, "split_date": split_date, "early_stop_date": es_date,
    }
    print(f"Threshold u = {u:.4%} loss; rows fit {len(data['e_fit'])}, early-stop "
          f"{len(data['e_es'])}, validation {len(data['e_val'])}; exceedances fit "
          f"{len(data['yg_fit'])}, early-stop {len(data['yg_es'])}, validation {len(data['yg_val'])}")
    return data


def _tensor(a):
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


def l2_penalty(model):
    """Keras-style kernel_regularizer l2(L2): L2 * sum(w^2) on every Linear weight matrix."""
    return L2 * sum((w ** 2).sum() for w in model.kernel_weights())


def _clip_per_tensor(model):
    for p in model.parameters():                       # Keras clipnorm is per tensor
        if p.grad is not None:
            torch.nn.utils.clip_grad_norm_([p], CLIPNORM)


# ==========================================
# 3. Mixture GPD likelihood (stage 2)
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


# ==========================================
# 4. Stage 2 evaluation
# ==========================================

@torch.no_grad()
def evaluate(model, X, y):
    """Eval-mode mixture NLL (early-stopping monitor) and calibration of S(y)."""
    model.eval()
    params = model(X)
    nll = mixture_nll(y, params).mean().item()
    S = mixture_survival(y, params)
    observed = {a: (S < a).float().mean().item() for a in CALIBRATION_LEVELS}
    mae_pp = 100.0 * float(np.mean([abs(observed[a] - a) for a in MAE_LEVELS]))
    return {"nll": nll, "calibration_mae_pp": mae_pp, "observed_share": observed,
            "min_S": S.min().item(), "params": params.numpy().astype("float64")}


def worst_days_within_q99(params, y, worst_share=0.01):
    """Share of the worst `worst_share` exceedances (largest y) that fall inside the model's q99."""
    q99 = mixture_quantile_excess(params, 0.01)
    worst = y >= np.quantile(y, 1 - worst_share)
    return float((y[worst] <= q99[worst]).mean())


# ==========================================
# 5. Stage 2 training (Adam, per-tensor clipnorm, early stopping on the early-stop set)
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
            _clip_per_tensor(model)
            optimizer.step()

            total += loss.item() * len(idx)
            count += len(idx)

        es = evaluate(model, X_es, y_es)
        history["loss"].append(total / count)
        history["es_nll"].append(es["nll"])
        history["es_calibration_mae_pp"].append(es["calibration_mae_pp"])
        print(f"[GPD] Epoch {epoch:3d}  loss {total / count:.4f}  early-stop NLL {es['nll']:.4f}  "
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
# 6. Stage 1 training and evaluation (exceedance network)
# ==========================================

auc_score = tail_model.auc_score          # ROC AUC (rank-sum formula), tail_model.py


def evaluate_pu(model, X, e, n_bins=10):
    """Log-loss, AUC and 10-bin (decile of p_u) calibration error of the exceedance network."""
    p = model.predict_proba(X).numpy().astype("float64")
    e = np.asarray(e, dtype="float64")
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    logloss = float(-np.mean(e * np.log(pc) + (1 - e) * np.log(1 - pc)))
    base = e.mean()
    logloss_const = float(-(base * np.log(1 - TAIL_QUANTILE)
                            + (1 - base) * np.log(TAIL_QUANTILE)))
    bins = pd.qcut(p, n_bins, labels=False, duplicates="drop")
    tab = pd.DataFrame({"p": p, "e": e, "bin": bins}).groupby("bin").mean()
    err_pp = 100 * (tab["p"] - tab["e"]).abs()
    return {"logloss": logloss, "logloss_constant_10pct": logloss_const, "auc": auc_score(p, e),
            "decile_calibration_error_pp_mean": float(err_pp.mean()),
            "decile_calibration_error_pp_max": float(err_pp.max()),
            "decile_mean_p": [float(v) for v in tab["p"]],
            "decile_observed_rate": [float(v) for v in tab["e"]],
            "p": p}


def train_pu(model, data):

    X_fit, e_fit = _tensor(data["X_fit_all"]), _tensor(data["e_fit"])
    X_es, e_es = _tensor(data["X_es_all"]), data["e_es"]

    optimizer = torch.optim.Adam(model.parameters(), lr=PU_LEARNING_RATE,
                                 betas=(0.9, 0.999), eps=1e-7)
    shuffle_gen = torch.Generator().manual_seed(SEED)

    history = {"loss": [], "es_logloss": [], "es_auc": []}
    best_ll, best_state, best_epoch, wait = float("inf"), None, 0, 0

    for epoch in range(1, PU_MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(X_fit), generator=shuffle_gen)
        total, count = 0.0, 0
        for start in range(0, len(perm), PU_BATCH_SIZE):
            idx = perm[start:start + PU_BATCH_SIZE]
            if len(idx) < 2:
                continue
            loss = F.binary_cross_entropy_with_logits(model(X_fit[idx]), e_fit[idx]) \
                + l2_penalty(model)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            _clip_per_tensor(model)
            optimizer.step()

            total += loss.item() * len(idx)
            count += len(idx)

        es = evaluate_pu(model, X_es, e_es)
        history["loss"].append(total / count)
        history["es_logloss"].append(es["logloss"])
        history["es_auc"].append(es["auc"])
        print(f"[p_u] Epoch {epoch:3d}  loss {total / count:.4f}  early-stop log-loss "
              f"{es['logloss']:.4f}  AUC {es['auc']:.4f}")

        if es["logloss"] < best_ll:
            best_ll, best_epoch, wait = es["logloss"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= PU_PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    return history, best_epoch


# ==========================================
# 7. Mixture formulas in numpy (float64) for quantiles, VaR, ES and backtests
# ==========================================
# Single implementation in tail_model.py (also used by evaluate_tail.py, gpd_predict.py,
# quantconnect/ and the notebook). Kept under the old names for existing callers.
mixture_sf_np = lambda params, y: tail_model.mixture_survival(params, y)
mixture_quantile_excess = tail_model.mixture_quantile_excess
tail_probability = tail_model.tail_probability
var_es = tail_model.var_es
kupiec_test = tail_model.kupiec_test


def var_backtest(p_u, params, L, u, alphas=VAR_ALPHAS):
    """Breach rates of the two-stage VaR on all validation stock-days (tail_model.var_backtest)."""
    return tail_model.var_backtest(p_u, params, L, u, alphas)


def var99_by_decile(p_u, params, L, u, n_bins=10):
    """VaR99 breach rate per p_u decile: model vs a constant 10% (tail_model.var99_by_decile)."""
    return tail_model.var99_by_decile(p_u, params, L, u, n_bins, constant_p_u=1 - TAIL_QUANTILE)


def check_numpy_forward(data, gpd_model, pu_model=None, tol=1e-4):
    """The saved artifacts, run through tail_model.py (the numpy inference used everywhere else),
    must reproduce the PyTorch networks on the validation inputs. Then write the QuantConnect JSON."""
    m = tail_model.TwoStageTailModel.from_artifacts(ARTIFACT_DIR, require_pu=pu_model is not None)
    z = np.asarray(data["X_val_all"], dtype="float64")
    x = _tensor(data["X_val_all"])
    gpd_model.eval()
    with torch.no_grad():
        P_torch = gpd_model(x).double().numpy()
    rel = np.abs(m.gpd_params_z(z) - P_torch) / np.maximum(np.abs(P_torch), 1e-12)
    worst = {c: float(v) for c, v in zip(tail_model.PARAM_COLS, rel.max(axis=0))}
    if pu_model is not None:
        p_torch = pu_model.predict_proba(x).double().numpy()
        worst["p_u"] = float(np.max(np.abs(m.exceedance_prob_z(z) - p_torch) / np.maximum(p_torch, 1e-12)))
    print("numpy (tail_model.py) vs torch, max relative difference:",
          {k: f"{v:.1e}" for k, v in worst.items()})
    if max(worst.values()) > tol:
        raise RuntimeError("tail_model.py does not reproduce the trained networks -- update it "
                           "together with the network definition in this file")
    if m.has_pu:
        QC_MODEL_JSON.parent.mkdir(exist_ok=True)
        m.to_json(QC_MODEL_JSON)
        print(f"Wrote {QC_MODEL_JSON} (both stages, for quantconnect/main.py)")


# ==========================================
# 8. Save artifacts
# ==========================================

def save_artifacts(model, history, best_epoch, results, data, scale):
    """Stage 2 weights + preprocess.json (shared preprocessing, used by both stages)."""

    ARTIFACT_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), ARTIFACT_DIR / MODEL_FILE)

    with torch.no_grad():
        model.eval()
        heavy = model(torch.zeros(2, INPUT_DIM))[0]
    meta = {
        "framework": "pytorch",
        "model_file": MODEL_FILE,
        "model_class": "GateMixtureGPDNet",
        "architecture": "two-stage model, stage 2 = gated GPD mixture: trunk Linear(13,224,no bias)-BN-"
                        "ReLU-Linear(224,88,no bias)-BN-ReLU; base head Linear(88,2) xi_b=0.01+0.34*sigmoid, "
                        "beta_b=s*softplus+1e-6; gate Linear(88,32,no bias)-BN-ReLU-Linear(32,1) "
                        "pi=pi_min+(1-pi_min)*sigmoid; heavy xi_h=0.35+0.45*sigmoid(theta), "
                        "beta_h=s*softplus(theta)+1e-6",
        "output_columns": ["xi_b", "beta_b", "xi_h", "beta_h", "pi"],
        "trainable_params": count_trainable_params(model),
        "pi_min": PI_MIN,
        "theta_beta_init": THETA_BETA_INIT,
        "gate_bias_init": GATE_BIAS_INIT,
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


def save_pu_artifacts(model, history, best_epoch, results, base_rate):
    torch.save(model.state_dict(), ARTIFACT_DIR / PU_MODEL_FILE)
    meta = {
        "framework": "pytorch",
        "model_file": PU_MODEL_FILE,
        "model_class": "ExceedanceNet",
        "architecture": "stage 1 = exceedance network: Linear(13,224,no bias)-BN-ReLU-"
                        "Linear(224,88,no bias)-BN-ReLU-Linear(88,1) -> sigmoid = P(L > u | x)",
        "preprocess_file": "preprocess.json",
        "trainable_params": count_trainable_params(model),
        "hidden_units": list(HIDDEN_UNITS),
        "output_bias_init": "logit(fit-period exceedance rate)",
        "fit_exceedance_rate": float(base_rate),
        "loss": "binary cross-entropy + L2 on Linear weights",
        "l2": L2,
        "learning_rate": PU_LEARNING_RATE,
        "clipnorm": CLIPNORM,
        "batch_size": PU_BATCH_SIZE,
        "max_epochs": PU_MAX_EPOCHS,
        "early_stopping_patience": PU_PATIENCE,
        "seed": SEED,
        "epochs_run": len(history["loss"]),
        "best_epoch": best_epoch,
        "results": results,
        "loss_history": [float(x) for x in history["loss"]],
        "early_stop_logloss_history": [float(x) for x in history["es_logloss"]],
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    (ARTIFACT_DIR / "pu_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved to {ARTIFACT_DIR}: {PU_MODEL_FILE}, pu_meta.json")


# ==========================================
# 9. Pipeline
# ==========================================

def _seed():
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def run_stage2(data):
    """Train the gated GPD mixture, save gpd_model.pt + preprocess.json, return the model."""
    _seed()
    scale = max(float(np.median(data["yg_train"])), 1e-4)   # fixed s: median training excess
    print(f"scale s = {scale:.10f}")

    model = GateMixtureGPDNet(scale)
    check_model(model)
    print(model)

    history, best_epoch = train(model, data)

    results = {}
    for name, label in (("es", "early_stop"), ("val", "validation")):
        r = evaluate(model, _tensor(data[f"Xg_{name}"]), _tensor(data[f"yg_{name}"]))
        results[label] = {
            "nll": r["nll"], "calibration_mae_pp": r["calibration_mae_pp"], "min_S": r["min_S"],
            "observed_share_S_below": {str(k): v for k, v in r["observed_share"].items()},
            "worst_1pct_within_q99": worst_days_within_q99(r["params"], data[f"yg_{name}"])}
    print(f"\n[GPD] Epochs run: {len(history['loss'])}, best epoch: {best_epoch}")
    for name, r in results.items():
        print(f"{name:>10}: mixture NLL {r['nll']:.4f}  min S {r['min_S']:.2e}  "
              f"worst 1% within q99 {r['worst_1pct_within_q99']:.0%}\n"
              f"{'':>12}" + "  ".join(f"S<{k}: {v:.2%}" for k, v in r["observed_share_S_below"].items()))

    with torch.no_grad():
        heavy = model(torch.zeros(2, INPUT_DIM))[0]
    print(f"Heavy component: xi_h = {heavy[2]:.4f}, beta_h = {heavy[3]:.5f}")

    save_artifacts(model, history, best_epoch, results, data, scale)
    return model


def run_stage1(data, gpd_model):
    """Train the exceedance network, validate it and the combined two-stage VaR."""
    _seed()
    base_rate = float(data["e_fit"].mean())
    model = ExceedanceNet(base_rate)
    check_pu_model(model)
    print(model)

    history, best_epoch = train_pu(model, data)

    results = {}
    for name, label in (("es", "early_stop"), ("val", "validation")):
        X = data["X_es_all"] if name == "es" else data["X_val_all"]
        r = evaluate_pu(model, _tensor(X), data[f"e_{name}"])
        r.pop("p")
        results[label] = r
    print(f"\n[p_u] Epochs run: {len(history['loss'])}, best epoch: {best_epoch}")
    for name, r in results.items():
        print(f"{name:>10}: log-loss {r['logloss']:.4f} (constant 10%: "
              f"{r['logloss_constant_10pct']:.4f})  AUC {r['auc']:.4f}  decile calib error "
              f"mean {r['decile_calibration_error_pp_mean']:.2f} pp, "
              f"max {r['decile_calibration_error_pp_max']:.2f} pp")

    # Combined model on ALL validation stock-days
    u = data["u"]
    X_val = _tensor(data["X_val_all"])
    p_u = model.predict_proba(X_val).numpy().astype("float64")
    gpd_model.eval()
    with torch.no_grad():
        params = gpd_model(X_val).numpy().astype("float64")
    L = data["L_val_all"]
    backtest = var_backtest(p_u, params, L, u)
    deciles = var99_by_decile(p_u, params, L, u)
    results["two_stage_var_backtest_validation"] = backtest
    results["var99_breach_by_p_u_decile"] = deciles

    print("\nTwo-stage unconditional VaR backtest (all validation stock-days)")
    for a, r in backtest.items():
        print(f"  VaR{100 * (1 - float(a)):g}: breach {r['breach_rate']:.3%} (expected "
              f"{float(a):.1%}), {r['n_breaches']}/{r['n']}, Kupiec p {r['kupiec_p']:.3f}, "
              f"rows with VaR < u: {r['rows_var_below_u']}")
    print("  VaR99 breach by p_u decile (model / constant 10% instead of p_u):")
    for r in deciles:
        print(f"    decile {r['decile']:2d}  mean p_u {r['mean_p_u']:.3f}  "
              f"{r['breach_model']:.2%} / {r['breach_constant_p_u']:.2%}")

    save_pu_artifacts(model, history, best_epoch, results, base_rate)
    return model


def main():

    parser = argparse.ArgumentParser(description="Train the two-stage tail-risk model")
    parser.add_argument("--stage", choices=("all", "gpd", "pu"), default="all",
                        help="all: stage 2 then stage 1 (default); gpd: stage 2 only; "
                             "pu: stage 1 + validation using the saved stage 2 model")
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild artifacts/dataset.csv from data_cache/ / the APIs before training")
    args = parser.parse_args()

    torch.use_deterministic_algorithms(True)
    data = prepare_data(rebuild=args.rebuild)

    if args.stage in ("all", "gpd"):
        gpd_model = run_stage2(data)
    else:
        meta = json.loads((ARTIFACT_DIR / "preprocess.json").read_text())
        assert abs(meta["threshold_u_loss"] - data["u"]) < 1e-12, "preprocess.json does not match data"
        gpd_model = load_gpd_model(ARTIFACT_DIR / MODEL_FILE, meta["gpd_scale"], meta["pi_min"])

    pu_model = run_stage1(data, gpd_model) if args.stage in ("all", "pu") else None
    check_numpy_forward(data, gpd_model, pu_model)


if __name__ == "__main__":
    main()
