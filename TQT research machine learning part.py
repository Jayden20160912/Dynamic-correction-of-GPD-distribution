
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential, regularizers


INPUT_DIM = 10
HIDDEN_UNITS = (16, 8)   
L2 = 1e-4

LAMBDA_TAIL = 0.2
LAMBDA_SEVERITY = 1.0

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

# 3. Two Independent Networks

epsilon_network = build_network("Epsilon_Network", HIDDEN_UNITS)
beta_network = build_network("Beta_Network", HIDDEN_UNITS)


# 4. Prepare Data
# ==========================================

# Demo data: replace X and y with your real data.
# X: (N, 10) market features known before the loss.
# y: (N,) nonnegative excess losses above threshold.

rng = np.random.default_rng(42)
X = rng.normal(size=(600, 10)).astype("float32")

# Simulated GPD exceedances, for demonstration only
true_beta = 0.02 * np.exp(0.2 * X[:, 0])
u = rng.uniform(size=len(X))
y = (true_beta / 0.2) * ((1 - u)**(-0.2) - 1)
y = y.astype("float32")

# Chronological split: 80% train, 20% validation
split = int(0.8 * len(X))

X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# Standardize features using training data only
mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-8

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std

# Reference scale for GPD parameters and loss
SCALE = max(float(np.median(y_train)), 1e-4)


# 5. Combine Networks into Dynamic GPD


inputs = layers.Input(shape=(INPUT_DIM,))

epsilon_raw = epsilon_network(inputs)
beta_raw = beta_network(inputs)

# Shape parameter: 0.01 < xi < 0.8
epsilon = layers.Rescaling(0.79, offset=0.01)(
    layers.Activation("sigmoid")(epsilon_raw)
)

# Scale parameter: beta > 0
beta = layers.Rescaling(SCALE, offset=1e-6)(
    layers.Activation("softplus")(beta_raw)
)

outputs = layers.Concatenate()([epsilon, beta])

gpd_model = Model(inputs, outputs, name="Dynamic_GPD")


# ==========================================
# 6. GPD NLL + Severity-Weighted Tail Loss
# ==========================================

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
    weight = 1.0 + LAMBDA_SEVERITY * y / SCALE

    # Combined Loss
    total_loss = nll + LAMBDA_TAIL * weight * tail_loss

    return tf.squeeze(total_loss, axis=-1)


# ==========================================
# 7. Compile and Train
# ==========================================

gpd_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001,
        clipnorm=1.0
    ),
    loss=gpd_loss
)

history = gpd_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=32,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True
        )
    ],
    verbose=1
)


# ==========================================
# 8. Predict Dynamic GPD Parameters
# ==========================================

predictions = gpd_model.predict(X_val[:5], verbose=0)

print("Predicted Shape (epsilon):")
print(predictions[:, 0])

print("\nPredicted Scale (beta):")
print(predictions[:, 1])

gpd_model.summary()
