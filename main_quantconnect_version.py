# region imports
from AlgorithmImports import *
# endregion

import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential, regularizers
from sklearn.feature_extraction.text import HashingVectorizer


# ==========================================
# 0. Config
# ==========================================

INPUT_DIM = 10
HIDDEN_UNITS = (16, 8)
L2 = 1e-4

LAMBDA_TAIL = 0.2
LAMBDA_SEVERITY = 1.0

# Sample window (all X rows / Y rows come from this window)
DATA_START = datetime(2016, 1, 1)
DATA_END = datetime(2024, 12, 31)
WARMUP_DAYS = 120            # extra calendar days of price history for rolling factors

# Y definition:
#   "close_to_close": Y_t = Close(t+1) / Close(t) - 1   (default)
#   "overnight"     : Y_t = Open(t+1)  / Close(t) - 1
TARGET_MODE = "close_to_close"

# GPD peaks-over-threshold: loss L = -Y, threshold = this quantile of TRAINING losses
TAIL_QUANTILE = 0.90

TRAIN_FRACTION = 0.8         # chronological split by date

# Fixed cross-sector large-cap universe that traded through the whole window
TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "INTC", "CSCO", "ORCL", "IBM",
    "JPM", "BAC", "WFC", "GS",
    "XOM", "CVX",
    "JNJ", "PFE", "MRK", "UNH",
    "PG", "KO", "PEP", "WMT",
    "HD", "MCD", "NKE", "DIS",
    "CAT", "BA", "MMM",
    "T", "VZ",
]

FEATURE_NAMES = [
    "vol_20d",          # 1  20-day annualised realised volatility
    "downside_vol_20d", # 2  20-day annualised downside semi-volatility
    "kurtosis_60d",     # 3  60-day excess kurtosis of daily log returns
    "momentum_60d",     # 4  60-day price momentum
    "reversal_5d",      # 5  5-day return (short-term reversal)
    "range_5d",         # 6  5-day mean (High-Low)/Close
    "abn_volume",       # 7  log(5-day avg volume / 60-day avg volume)
    "vix",              # 8  CBOE VIX close
    "n_8k_30d",         # 9  number of 8-K filings in the past 30 calendar days   (SEC)
    "filing_change",    # 10 1 - cosine similarity of latest 10-K/10-Q vs previous same-type filing (SEC, "Lazy Prices")
]


# ==========================================
# 1. Build Dense Neural Network (unchanged)
# ==========================================

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
# 2. Combine Networks into Dynamic GPD
# ==========================================

def build_gpd_model(scale):

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
# 3. GPD NLL + Severity-Weighted Tail Loss (unchanged math)
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
# 4. SEC text helpers
# ==========================================

_TAG_RE = re.compile(r"<[^>]+>")
_HASHER = HashingVectorizer(
    n_features=2 ** 18,
    alternate_sign=False,
    norm="l2",
    lowercase=True,
    stop_words="english",
    token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
)
MAX_TEXT_CHARS = 3_000_000


def _available_date(data_point):
    """Date on which a filing could first be used (point-in-time, no look-ahead)."""
    return pd.Timestamp(data_point.end_time).normalize()


def _filing_text(data_point, form_prefix):
    """Main document text of a 10-K / 10-Q, HTML stripped."""
    sub = data_point.report
    docs = list(sub.documents) if sub is not None and sub.documents is not None else []
    if not docs:
        return ""
    main = [d for d in docs
            if str(getattr(d, "form_type", "") or "").upper().startswith(form_prefix)]
    if not main:
        main = docs[:1]
    text = " ".join(str(d.text or "") for d in main)[:MAX_TEXT_CHARS]
    return _TAG_RE.sub(" ", text)


def _accession(data_point):
    sub = data_point.report
    return str(getattr(sub, "accession_number", "") or data_point.end_time)


# ==========================================
# 5. Algorithm: build X / Y from QC data, then train
# ==========================================

class DynamicGPDTraining(QCAlgorithm):

    def initialize(self):
        # Backtest clock sits AFTER the data window so history() can reach DATA_END
        self.set_start_date(2025, 1, 2)
        self.set_end_date(2025, 1, 10)
        self.set_cash(100000)

        self.equities = {}
        self.sec = {}
        for t in TICKERS:
            sym = self.add_equity(t, Resolution.DAILY).symbol
            self.equities[t] = sym
            self.sec[t] = {
                "8K": self.add_data(SECReport8K, sym).symbol,
                "10K": self.add_data(SECReport10K, sym).symbol,
                "10Q": self.add_data(SECReport10Q, sym).symbol,
            }

        self.vix = self.add_data(CBOE, "VIX", Resolution.DAILY).symbol
        self.spy = self.add_equity("SPY", Resolution.DAILY).symbol

        # self.train gives a longer time budget than initialize
        self.train(self.run_pipeline)

    # ---------- market factor ----------
    def _vix_series(self):
        start = DATA_START - timedelta(days=WARMUP_DAYS)
        df = self.history(self.vix, start, DATA_END, Resolution.DAILY)
        if df is not None and not df.empty and "close" in df.columns:
            s = df["close"].droplevel(0)
        else:
            # Fallback: SPY 20d realised vol in VIX units
            self.debug("VIX history empty, falling back to SPY realised vol")
            spy = self.history(self.spy, start, DATA_END, Resolution.DAILY)["close"].droplevel(0)
            s = np.log(spy).diff().rolling(20).std() * np.sqrt(252) * 100
        s.index = pd.to_datetime(s.index).normalize()
        return s.groupby(level=0).last()

    # ---------- SEC factors for one stock ----------
    def _sec_factors(self, ticker, dates):
        start = DATA_START - timedelta(days=550)   # need one prior 10-K for the first similarity
        syms = self.sec[ticker]

        # (a) 8-K event intensity
        k8_dates = []
        for dp in self.history[SECReport8K](syms["8K"], start, DATA_END, Resolution.DAILY):
            k8_dates.append(_available_date(dp))
        k8 = np.array(sorted(set(k8_dates)), dtype="datetime64[ns]")

        d = dates.values.astype("datetime64[ns]")
        d30 = (dates - pd.Timedelta(days=30)).values.astype("datetime64[ns]")
        n_8k = (np.searchsorted(k8, d, side="right")
                - np.searchsorted(k8, d30, side="right")).astype(float)

        # (b) Lazy-Prices change score: 1 - cos(latest filing, previous same-type filing)
        events = []   # (available_date, change_score)
        for key, cls, prefix in (("10K", SECReport10K, "10-K"), ("10Q", SECReport10Q, "10-Q")):
            seen, filings = set(), []
            for dp in self.history[cls](syms[key], start, DATA_END, Resolution.DAILY):
                acc = _accession(dp)
                if acc in seen:
                    continue
                seen.add(acc)
                text = _filing_text(dp, prefix)
                if len(text) < 1000:
                    continue
                filings.append((_available_date(dp), _HASHER.transform([text])))
            filings.sort(key=lambda x: x[0])
            for i in range(1, len(filings)):
                sim = float(filings[i][1].multiply(filings[i - 1][1]).sum())
                events.append((filings[i][0], 1.0 - sim))

        change = np.full(len(dates), np.nan)
        if events:
            events.sort(key=lambda x: x[0])
            ev_dates = np.array([e[0] for e in events], dtype="datetime64[ns]")
            ev_vals = np.array([e[1] for e in events])
            idx = np.searchsorted(ev_dates, d, side="right") - 1
            ok = idx >= 0
            change[ok] = ev_vals[idx[ok]]

        return n_8k, change

    # ---------- price factors + target for one stock ----------
    @staticmethod
    def _price_factors(px):
        c, h, l, o, v = px["close"], px["high"], px["low"], px["open"], px["volume"]
        r = np.log(c).diff()
        f = pd.DataFrame(index=px.index)
        f["vol_20d"] = r.rolling(20).std() * np.sqrt(252)
        f["downside_vol_20d"] = np.sqrt((r.clip(upper=0) ** 2).rolling(20).mean()) * np.sqrt(252)
        f["kurtosis_60d"] = r.rolling(60).kurt()
        f["momentum_60d"] = c / c.shift(60) - 1
        f["reversal_5d"] = c / c.shift(5) - 1
        f["range_5d"] = ((h - l) / c).rolling(5).mean()
        f["abn_volume"] = np.log((v.rolling(5).mean() + 1) / (v.rolling(60).mean() + 1))

        if TARGET_MODE == "overnight":
            y = o.shift(-1) / c - 1
        else:
            y = c.shift(-1) / c - 1
        return f, y

    # ---------- assemble X, Y ----------
    def build_dataset(self):
        vix = self._vix_series()
        start = DATA_START - timedelta(days=WARMUP_DAYS)

        X_parts, Y_parts, D_parts, T_parts = [], [], [], []
        for t in TICKERS:
            sym = self.equities[t]
            hist = self.history(sym, start, DATA_END, Resolution.DAILY)
            if hist is None or hist.empty:
                self.debug(f"{t}: no price history, skipped")
                continue
            px = hist.droplevel(0)
            px.index = pd.to_datetime(px.index).normalize()
            px = px[~px.index.duplicated(keep="last")].sort_index()

            f, y = self._price_factors(px)
            f["vix"] = vix.reindex(f.index, method="ffill")

            n_8k, change = self._sec_factors(t, f.index.to_series())
            f["n_8k_30d"] = n_8k
            f["filing_change"] = change

            f["__y"] = y
            f = f[(f.index >= pd.Timestamp(DATA_START))]
            # price/market factors and target must exist; SEC similarity may be NaN early on
            f = f.dropna(subset=FEATURE_NAMES[:8] + ["__y"])

            X_parts.append(f[FEATURE_NAMES].values)
            Y_parts.append(f["__y"].values)
            D_parts.append(f.index.values)
            T_parts.append(np.array([t] * len(f)))
            self.debug(f"{t}: {len(f)} rows")

        X = np.vstack(X_parts).astype("float32")
        Y = np.concatenate(Y_parts).astype("float32").reshape(-1, 1)   # (N, 1)
        dates = np.concatenate(D_parts)
        tickers = np.concatenate(T_parts)

        order = np.argsort(dates, kind="stable")   # chronological, stocks interleaved
        return X[order], Y[order], dates[order], tickers[order]

    # ---------- full pipeline ----------
    def run_pipeline(self):
        tf.random.set_seed(42)
        np.random.seed(42)

        X, Y, dates, tickers = self.build_dataset()
        self.debug(f"X shape {X.shape}, Y shape {Y.shape}")

        # Chronological split by DATE (no same-day leakage across stocks)
        unique_dates = np.unique(dates)
        split_date = unique_dates[int(TRAIN_FRACTION * len(unique_dates))]
        is_train = dates < split_date
        self.debug(f"Train < {pd.Timestamp(split_date).date()} <= Validation")

        X_train, X_val = X[is_train].copy(), X[~is_train].copy()
        Y_train, Y_val = Y[is_train], Y[~is_train]

        # Fill missing SEC similarity with TRAIN median, then standardize with TRAIN stats
        med = np.nanmedian(X_train, axis=0)
        for arr in (X_train, X_val):
            r_, c_ = np.where(np.isnan(arr))
            arr[r_, c_] = med[c_]

        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8
        X_train = np.clip((X_train - mean) / std, -5, 5)
        X_val = np.clip((X_val - mean) / std, -5, 5)

        # Peaks-over-threshold: GPD is fitted on loss exceedances, not raw returns
        L_train, L_val = -Y_train[:, 0], -Y_val[:, 0]
        u = float(np.quantile(L_train, TAIL_QUANTILE))
        tr_mask, va_mask = L_train > u, L_val > u

        Xg_train, yg_train = X_train[tr_mask], (L_train[tr_mask] - u).astype("float32")
        Xg_val, yg_val = X_val[va_mask], (L_val[va_mask] - u).astype("float32")
        self.debug(f"Threshold u = {u:.4%} loss; exceedances train {len(yg_train)}, val {len(yg_val)}")

        # Reference scale for GPD parameters and loss
        scale = max(float(np.median(yg_train)), 1e-4)

        gpd_model = build_gpd_model(scale)
        gpd_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
            loss=make_gpd_loss(scale),
        )

        history = gpd_model.fit(
            Xg_train, yg_train,
            validation_data=(Xg_val, yg_val),
            epochs=150,
            batch_size=32,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=15, restore_best_weights=True
                )
            ],
            verbose=0,
        )
        self.debug(f"Epochs run: {len(history.history['loss'])}, "
                   f"best val_loss: {min(history.history['val_loss']):.4f}")

        # Dynamic GPD parameters for the latest validation rows
        preds = gpd_model.predict(X_val[-5:], verbose=0)
        for i in range(len(preds)):
            self.debug(f"{tickers[~is_train][-5 + i]} {pd.Timestamp(dates[~is_train][-5 + i]).date()} "
                       f"epsilon={preds[i, 0]:.4f} beta={preds[i, 1]:.5f}")

        # Keep everything for later use (research notebook / live trading)
        self.gpd_model = gpd_model
        self.feature_mean, self.feature_std, self.feature_median = mean, std, med
        self.threshold_u, self.gpd_scale = u, scale
        self._save_dataset(X, Y, dates, tickers)
        self._save_model(gpd_model, history, split_date, len(yg_train), len(yg_val))

    def _save_model(self, model, history, split_date, n_train, n_val):
        """Persist model + weights + preprocessing so the model can be reused without retraining."""
        import json

        # 1) full model (architecture + weights)
        model_path = self.object_store.get_file_path("gpd/gpd_model.keras")
        model.save(model_path)
        with open(model_path, "rb") as fh:
            self.object_store.save_bytes("gpd/gpd_model.keras", bytearray(fh.read()))

        # 2) weights only
        w_path = self.object_store.get_file_path("gpd/gpd_model.weights.h5")
        model.save_weights(w_path)
        with open(w_path, "rb") as fh:
            self.object_store.save_bytes("gpd/gpd_model.weights.h5", bytearray(fh.read()))

        # 3) everything needed to turn raw factors into model inputs
        meta = {
            "feature_names": FEATURE_NAMES,
            "feature_mean": [float(x) for x in self.feature_mean],
            "feature_std": [float(x) for x in self.feature_std],
            "feature_median_fill": [float(x) for x in self.feature_median],
            "clip_z": 5.0,
            "threshold_u_loss": float(self.threshold_u),
            "gpd_scale": float(self.gpd_scale),
            "tail_quantile": TAIL_QUANTILE,
            "target_mode": TARGET_MODE,
            "data_start": str(DATA_START.date()),
            "data_end": str(DATA_END.date()),
            "train_val_split_date": str(pd.Timestamp(split_date).date()),
            "n_exceedances_train": int(n_train),
            "n_exceedances_val": int(n_val),
            "tickers": TICKERS,
            "hidden_units": list(HIDDEN_UNITS),
            "l2": L2,
            "lambda_tail": LAMBDA_TAIL,
            "lambda_severity": LAMBDA_SEVERITY,
            "loss_history": [float(x) for x in history.history["loss"]],
            "val_loss_history": [float(x) for x in history.history["val_loss"]],
            "saved_at": str(self.time),
        }
        self.object_store.save("gpd/preprocess.json", json.dumps(meta, indent=2))
        self.debug("Saved model to Object Store: gpd/gpd_model.keras, "
                   "gpd/gpd_model.weights.h5, gpd/preprocess.json")

    def _save_dataset(self, X, Y, dates, tickers):
        df = pd.DataFrame(X, columns=FEATURE_NAMES)
        df.insert(0, "ticker", tickers)
        df.insert(0, "date", pd.to_datetime(dates).strftime("%Y-%m-%d"))
        df["Y"] = Y[:, 0]
        self.object_store.save("gpd/dataset.csv", df.to_csv(index=False))
        self.debug("Saved X/Y to Object Store key gpd/dataset.csv")

    def on_data(self, data: Slice):
        pass
