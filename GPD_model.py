"""
Generalized Pareto Distribution (GPD) model -- models the exceedance losses in the
left tail (loss side) of the return distribution

Probability density function:
    f(y; xi, beta) = (1/beta) * (1 + xi * y / beta) ** (-1/xi - 1)

Variable definitions:
    r          : raw daily return (negative values are losses)
    u          : loss threshold (negative, e.g. -0.02 means -2%)
    y          : distance between a return left of the threshold and the threshold,
                 y = |r - u|, only for r <= u, so y >= 0
    xi (ε)     : shape parameter, controls tail thickness, requires xi > 0
                 (larger -> heavier tail, extreme losses more likely)
    beta (β)   : scale parameter, controls the width of the distribution / size of
                 extreme losses, requires beta > 0

Constraints:
    beta > 0, xi > 0, y >= 0, and 1 + xi*y/beta > 0 (automatically satisfied when xi>0, y>=0)

Usage:
    python GPD_model.py --xi 0.3 --beta 0.01
    python GPD_model.py --xi 0.1 0.3 0.6 --beta 0.01 0.01 0.01      # compare several parameter sets
    python GPD_model.py --xi 0.3 --beta 0.01 --threshold -0.02 --save gpd.png
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# 1. GPD math functions
# ---------------------------------------------------------------------------
def _check_params(xi: float, beta: float) -> None:
    if not (beta > 0):
        raise ValueError(f"beta must be greater than 0, got beta = {beta}")
    if not (xi > 0):
        raise ValueError(f"xi (epsilon) must be greater than 0, got xi = {xi}")


def gpd_pdf(y, xi: float, beta: float) -> np.ndarray:
    """GPD probability density f(y; xi, beta). Returns 0 where y < 0 (outside the support)."""
    _check_params(xi, beta)
    y = np.asarray(y, dtype=float)
    z = 1.0 + xi * y / beta                      # the term in parentheses, must be > 0
    valid = (y >= 0) & (z > 0)
    out = np.zeros_like(y)
    out[valid] = (1.0 / beta) * z[valid] ** (-1.0 / xi - 1.0)
    return out


def gpd_cdf(y, xi: float, beta: float) -> np.ndarray:
    """Cumulative distribution F(y) = 1 - (1 + xi*y/beta)^(-1/xi)"""
    _check_params(xi, beta)
    y = np.clip(np.asarray(y, dtype=float), 0.0, None)
    return 1.0 - (1.0 + xi * y / beta) ** (-1.0 / xi)


def gpd_survival(y, xi: float, beta: float) -> np.ndarray:
    """Tail probability P(Y > y): conditional probability of losing a further y beyond the threshold"""
    return 1.0 - gpd_cdf(y, xi, beta)


def gpd_quantile(q, xi: float, beta: float) -> np.ndarray:
    """Quantile F^{-1}(q) = beta/xi * ((1-q)^(-xi) - 1)"""
    _check_params(xi, beta)
    q = np.asarray(q, dtype=float)
    return beta / xi * ((1.0 - q) ** (-xi) - 1.0)


def returns_to_exceedances(returns, threshold: float) -> np.ndarray:
    """Convert a return series into GPD inputs y: take all returns left of threshold, y = |r - threshold|"""
    if threshold >= 0:
        raise ValueError("Loss threshold should be negative, e.g. -0.02")
    r = np.asarray(returns, dtype=float)
    tail = r[r <= threshold]
    return np.abs(tail - threshold)


# ---------------------------------------------------------------------------
# 2. Plotting
# ---------------------------------------------------------------------------
COLORS = ["#2a6fdb", "#d9480f", "#2b8a3e", "#862e9c", "#c2255c", "#0b7285"]


def plot_gpd(xis, betas, threshold: float | None = None, save: str | None = None):
    """
    Plot the GPD for one or more (xi, beta) pairs.
    Left: density on a linear y-axis (overall shape/width, mainly driven by beta)
    Right: density on a log y-axis (tail thickness, mainly driven by xi; larger xi -> slower tail decay)
    """
    xis, betas = list(np.atleast_1d(xis)), list(np.atleast_1d(betas))
    if len(betas) == 1 and len(xis) > 1:
        betas *= len(xis)
    if len(xis) == 1 and len(betas) > 1:
        xis *= len(betas)
    if len(xis) != len(betas):
        raise ValueError("xi and beta must have the same length (or one of them a single value)")
    for xi, beta in zip(xis, betas):
        _check_params(xi, beta)

    # x-axis range: max of the 99.5% quantiles across distributions (capped, since heavy tails run long)
    y_max = max(gpd_quantile(0.995, xi, b) for xi, b in zip(xis, betas))
    y_max = min(y_max, 30 * max(betas))
    y = np.linspace(0, y_max, 1000)

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(13, 5))

    for i, (xi, beta) in enumerate(zip(xis, betas)):
        color = COLORS[i % len(COLORS)]
        pdf = gpd_pdf(y, xi, beta)
        label = f"ξ = {xi:g},  β = {beta:g}"
        ax_lin.plot(y, pdf, color=color, lw=2, label=label)
        ax_log.plot(y, pdf, color=color, lw=2, label=label)

        # Mark the 99% quantile (VaR-like: 99% of the time the further loss beyond the threshold stays below this)
        q99 = gpd_quantile(0.99, xi, beta)
        if q99 <= y_max:
            ax_lin.axvline(q99, color=color, ls=":", lw=1)

    ax_lin.set_title("GPD density (linear scale)")
    ax_log.set_title("GPD density (log scale, shows tail thickness)")
    ax_log.set_yscale("log")
    for ax in (ax_lin, ax_log):
        ax.set_xlabel("y = |return − threshold|   (loss distance beyond threshold)")
        ax.set_ylabel("f(y; ξ, β)")
        ax.set_xlim(0, y_max)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

    # If a threshold is given, add a secondary top axis showing the actual return r = threshold - y
    if threshold is not None:
        if threshold >= 0:
            raise ValueError("threshold should be negative")
        for ax in (ax_lin, ax_log):
            sec = ax.secondary_xaxis("top", functions=(lambda v: threshold - v,
                                                       lambda r: threshold - r))
            sec.set_xlabel(f"Corresponding return r (threshold = {threshold:.2%})")
            sec.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))

    fig.suptitle(r"Generalized Pareto Distribution  $f(y;\xi,\beta)=\frac{1}{\beta}"
                 r"\left(1+\frac{\xi y}{\beta}\right)^{-1/\xi-1}$   (dotted line = 99% quantile)")
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=150)
        print(f"Figure saved to {save}")
    else:
        plt.show()
    return fig


def _setup_chinese_font():
    """Font setup for matplotlib on macOS"""
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "Arial Unicode MS",
                                       "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 3. Command-line entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _setup_chinese_font()
    parser = argparse.ArgumentParser(description="Plot the GPD for given (xi, beta)")
    parser.add_argument("--xi", type=float, nargs="+", default=[0.1, 0.3, 0.6],
                        help="shape parameter xi (epsilon), >0, multiple values allowed")
    parser.add_argument("--beta", type=float, nargs="+", default=[0.01],
                        help="scale parameter beta, >0, multiple values allowed (paired with xi)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="loss threshold (negative, e.g. -0.02), shows the corresponding return on the top axis")
    parser.add_argument("--save", type=str, default=None, help="path to save the figure; shows a window if omitted")
    args = parser.parse_args()

    for xi, beta in zip(args.xi if len(args.xi) > 1 else args.xi * len(args.beta),
                        args.beta if len(args.beta) > 1 else args.beta * len(args.xi)):
        print(f"xi={xi:g}, beta={beta:g}:  P(y > 0.05) = {gpd_survival(0.05, xi, beta):.4%},"
              f"  99% quantile = {gpd_quantile(0.99, xi, beta):.4f}")

    plot_gpd(args.xi, args.beta, threshold=args.threshold, save=args.save)
