"""
Dynamic GPD tail-risk model -- local training on Alpaca + FRED + SEC EDGAR data.

Run:
    .venv/bin/python "TQT research machine learning part.py"

Outputs (artifacts/):
    dataset.csv              full X / Y table
    gpd_model.keras          full model
    gpd_model.weights.h5     weights only
    preprocess.json          everything needed to turn raw factors into model inputs

Load later with:
    tf.keras.models.load_model("artifacts/gpd_model.keras", compile=False)
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential, regularizers

import features
from features import FEATURE_NAMES, TICKERS, TARGET_MODE, DATA_START, DATA_END


INPUT_DIM = 10
HIDDEN_UNITS = (16, 8)
L2 = 1e-4

LAMBDA_TAIL = 0.2
LAMBDA_SEVERITY = 1.0

# GPD peaks-over-threshold: loss L = -Y, threshold = this quantile of TRAINING losses
TAIL_QUANTILE = 0.90
TRAIN_FRACTION = 0.8         # chronological split by date
CLIP_Z = 5.0
SEED = 42

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


#Build Dense Neural Network


def build_network(name, neurons, use_bn=True):

    model = Sequential(name=name)
    model.add(layers.Input(shape=(INPUT_DIM,)))

    for units in neurons:
        model.add(layers.Dense(
            units,
            use_bias=not use_bn,
            kernel_regularizer=regularizers.l2(L2)
        ))

        if use_bn:
            model.add(layers.BatchNormalization())

        model.add(layers.Activation("relu"))

    model.add(layers.Dense(1, activation="linear"))

    return model


# ==========================================
# 4. Prepare Data (real data: Alpaca + FRED + SEC EDGAR)
# ==========================================

def prepare_data():

    X, Y, dates, tickers = features.build_dataset(target_mode=TARGET_MODE)
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
# 5. Combine Networks into Dynamic GPD
# ==========================================

def build_gpd_model(scale):

    # Two Independent Networks
    epsilon_network = build_network("Epsilon_Network", HIDDEN_UNITS)
    beta_network = build_network("Beta_Network", HIDDEN_UNITS)

    inputs = layers.Input(shape=(INPUT_DIM,))

    epsilon_raw = epsilon_network(inputs)
    beta_raw = beta_network(inputs)

    # Shape parameter: 0.01 < xi < 0.8
    epsilon = layers.Rescaling(0.79, offset=0.01)(
        layers.Activation("sigmoid")(epsilon_raw)
    )

    # Scale parameter: beta > 0
    beta = layers.Rescaling(scale, offset=1e-6)(
        layers.Activation("softplus")(beta_raw)
    )

    outputs = layers.Concatenate()([epsilon, beta])

    return Model(inputs, outputs, name="Dynamic_GPD")


# ==========================================
# 6. GPD NLL + Severity-Weighted Tail Loss
# ==========================================

def make_gpd_loss(scale):

    def gpd_loss(y_true, y_pred):

        y = tf.reshape(
            tf.cast(y_true, y_pred.dtype), (-1, 1)
        )

        epsilon = y_pred[:, 0:1]
        beta = y_pred[:, 1:2]

        # GPD Negative Log-Likelihood
        log_term = tf.math.log1p(epsilon * y / beta)

        nll = (
            tf.math.log(beta)
            + (1.0 + 1.0 / epsilon) * log_term
        )

        # Tail Probability Penalty
        tail_loss = log_term / epsilon

        # Severity Weight
        weight = 1.0 + LAMBDA_SEVERITY * y / scale

        # Combined Loss
        total_loss = nll + LAMBDA_TAIL * weight * tail_loss

        return tf.squeeze(total_loss, axis=-1)

    return gpd_loss


# ==========================================
# 9. Save model + weights + preprocessing
# ==========================================

def save_artifacts(gpd_model, history, data, scale):

    ARTIFACT_DIR.mkdir(exist_ok=True)

    # 1) full model (architecture + weights)
    gpd_model.save(ARTIFACT_DIR / "gpd_model.keras")

    # 2) weights only
    gpd_model.save_weights(ARTIFACT_DIR / "gpd_model.weights.h5")

    # 3) everything needed to turn raw factors into model inputs
    meta = {
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
        "l2": L2,
        "lambda_tail": LAMBDA_TAIL,
        "lambda_severity": LAMBDA_SEVERITY,
        "learning_rate": 0.001,
        "clipnorm": 1.0,
        "batch_size": 32,
        "max_epochs": 150,
        "early_stopping_patience": 15,
        "seed": SEED,
        "data_sources": "Alpaca (sip, adjustment=all), FRED VIXCLS, SEC EDGAR",
        "epochs_run": len(history.history["loss"]),
        "best_val_loss": float(min(history.history["val_loss"])),
        "loss_history": [float(x) for x in history.history["loss"]],
        "val_loss_history": [float(x) for x in history.history["val_loss"]],
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    (ARTIFACT_DIR / "preprocess.json").write_text(json.dumps(meta, indent=2))

    print(f"Saved to {ARTIFACT_DIR}: gpd_model.keras, gpd_model.weights.h5, preprocess.json")


def main():

    tf.keras.utils.set_random_seed(SEED)

    data = prepare_data()

    # Reference scale for GPD parameters and loss
    SCALE = max(float(np.median(data["yg_train"])), 1e-4)

    gpd_model = build_gpd_model(SCALE)

    # ==========================================
    # 7. Compile and Train
    # ==========================================

    gpd_model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001,
            clipnorm=1.0
        ),
        loss=make_gpd_loss(SCALE)
    )

    history = gpd_model.fit(
        data["Xg_train"],
        data["yg_train"],
        validation_data=(data["Xg_val"], data["yg_val"]),
        epochs=150,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=15,
                restore_best_weights=True
            )
        ],
        verbose=2
    )
    print(f"Epochs run: {len(history.history['loss'])}, "
          f"best val_loss: {min(history.history['val_loss']):.4f}")

    # ==========================================
    # 8. Predict Dynamic GPD Parameters (latest validation rows)
    # ==========================================

    predictions = gpd_model.predict(data["X_val_all"][-5:], verbose=0)
    for i in range(len(predictions)):
        print(f"{data['tickers_val'][-5 + i]} {pd.Timestamp(data['dates_val'][-5 + i]).date()} "
              f"epsilon={predictions[i, 0]:.4f} beta={predictions[i, 1]:.5f}")

    gpd_model.summary()

    save_artifacts(gpd_model, history, data, SCALE)


if __name__ == "__main__":
    main()
