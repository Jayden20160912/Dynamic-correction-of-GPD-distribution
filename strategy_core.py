"""
Allocation rule of the GPD tail-risk strategy -- the ONLY implementation, shared by
quantconnect/main.py and trading_strategy.ipynb (add this file to the QuantConnect project).

    risk score R_i   from tail_model.TwoStageTailModel.risk_score (E[(L - d)^+], with p_u)
    v_i = (R_i + delta)^(-gamma) / sum_j (R_j + delta)^(-gamma)      relative weights
    B   = sum_i v_i R_i                                             risk of the fully invested portfolio
    c   = min(1, R_target / B)                                      total exposure, rest in cash
    w_i = c v_i
"""
import numpy as np


def allocate_from_risk(R, R_target, gamma=1.0, delta=1e-6):
    """Risk scores of ONE day (one per stock) -> weights, cash, v, B, c."""
    R = np.asarray(R, dtype=float)
    if R.ndim != 1 or len(R) == 0 or not np.all(np.isfinite(R)) or np.any(R < 0):
        raise ValueError("R must be a non-empty 1-D array of finite, non-negative risk scores")
    if R_target < 0 or gamma < 0 or delta <= 0:
        raise ValueError("need R_target >= 0, gamma >= 0, delta > 0")
    scores = -gamma * np.log(R + delta)              # inverse-risk weighting in log space
    scores = np.exp(scores - np.max(scores))         # subtract the max: no overflow
    v = scores / scores.sum()
    B = float(np.sum(v * R))
    c = min(1.0, R_target / B) if B > 0 else 1.0
    weights = c * v
    return weights, 1.0 - weights.sum(), v, B, c
