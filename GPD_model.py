"""
广义帕累托分布 (GPD) 模型 —— 用于建模收益率分布左尾（亏损部分）的超额损失

概率密度函数:
    f(y; xi, beta) = (1/beta) * (1 + xi * y / beta) ** (-1/xi - 1)

变量定义:
    r          : 原始日收益率（负数代表亏损）
    u          : 亏损阈值 threshold（负数，例如 -0.02 表示 -2%）
    y          : 阈值左侧的收益率与阈值的距离  y = |r - u|，仅取 r <= u 的部分，因此 y >= 0
    xi (ε)     : 形状参数，控制尾部厚度，要求 xi > 0（越大尾部越厚，极端亏损越可能发生）
    beta (β)   : 尺度参数，控制分布宽度/极端亏损的规模，要求 beta > 0

约束条件:
    beta > 0, xi > 0, y >= 0, 且 1 + xi*y/beta > 0（在 xi>0, y>=0 时自动满足）

用法:
    python GPD_model.py --xi 0.3 --beta 0.01
    python GPD_model.py --xi 0.1 0.3 0.6 --beta 0.01 0.01 0.01      # 多组参数对比
    python GPD_model.py --xi 0.3 --beta 0.01 --threshold -0.02 --save gpd.png
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# 1. GPD 数学函数
# ---------------------------------------------------------------------------
def _check_params(xi: float, beta: float) -> None:
    if not (beta > 0):
        raise ValueError(f"beta 必须大于 0，当前 beta = {beta}")
    if not (xi > 0):
        raise ValueError(f"xi (epsilon) 必须大于 0，当前 xi = {xi}")


def gpd_pdf(y, xi: float, beta: float) -> np.ndarray:
    """GPD 概率密度 f(y; xi, beta)。y < 0 的位置返回 0（不在定义域内）。"""
    _check_params(xi, beta)
    y = np.asarray(y, dtype=float)
    z = 1.0 + xi * y / beta                      # 括号内的和，必须 > 0
    valid = (y >= 0) & (z > 0)
    out = np.zeros_like(y)
    out[valid] = (1.0 / beta) * z[valid] ** (-1.0 / xi - 1.0)
    return out


def gpd_cdf(y, xi: float, beta: float) -> np.ndarray:
    """累积分布 F(y) = 1 - (1 + xi*y/beta)^(-1/xi)"""
    _check_params(xi, beta)
    y = np.clip(np.asarray(y, dtype=float), 0.0, None)
    return 1.0 - (1.0 + xi * y / beta) ** (-1.0 / xi)


def gpd_survival(y, xi: float, beta: float) -> np.ndarray:
    """尾部概率 P(Y > y)：超过阈值后再多亏 y 的条件概率"""
    return 1.0 - gpd_cdf(y, xi, beta)


def gpd_quantile(q, xi: float, beta: float) -> np.ndarray:
    """分位数 F^{-1}(q) = beta/xi * ((1-q)^(-xi) - 1)"""
    _check_params(xi, beta)
    q = np.asarray(q, dtype=float)
    return beta / xi * ((1.0 - q) ** (-xi) - 1.0)


def returns_to_exceedances(returns, threshold: float) -> np.ndarray:
    """把收益率序列转换成 GPD 的输入 y：取 threshold 左边所有收益率，y = |r - threshold|"""
    if threshold >= 0:
        raise ValueError("亏损阈值 threshold 应为负数，例如 -0.02")
    r = np.asarray(returns, dtype=float)
    tail = r[r <= threshold]
    return np.abs(tail - threshold)


# ---------------------------------------------------------------------------
# 2. 画图
# ---------------------------------------------------------------------------
COLORS = ["#2a6fdb", "#d9480f", "#2b8a3e", "#862e9c", "#c2255c", "#0b7285"]


def plot_gpd(xis, betas, threshold: float | None = None, save: str | None = None):
    """
    根据一组或多组 (xi, beta) 画出 GPD 分布。
    左图: 线性纵轴的密度（看整体形状/宽度，主要受 beta 影响）
    右图: 对数纵轴的密度（看尾部厚度，主要受 xi 影响；xi 越大，尾部下降越慢）
    """
    xis, betas = list(np.atleast_1d(xis)), list(np.atleast_1d(betas))
    if len(betas) == 1 and len(xis) > 1:
        betas *= len(xis)
    if len(xis) == 1 and len(betas) > 1:
        xis *= len(betas)
    if len(xis) != len(betas):
        raise ValueError("xi 和 beta 的个数必须相同（或其中一个只给一个值）")
    for xi, beta in zip(xis, betas):
        _check_params(xi, beta)

    # x 轴范围：取所有分布 99.5% 分位数中的最大值（厚尾分布会很长，避免拉太远做了上限）
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

        # 标出 99% 分位数（类似 VaR：超过阈值后 99% 情况下多亏不超过这个值）
        q99 = gpd_quantile(0.99, xi, beta)
        if q99 <= y_max:
            ax_lin.axvline(q99, color=color, ls=":", lw=1)

    ax_lin.set_title("GPD 概率密度（线性坐标）")
    ax_log.set_title("GPD 概率密度（对数坐标，观察尾部厚度）")
    ax_log.set_yscale("log")
    for ax in (ax_lin, ax_log):
        ax.set_xlabel("y = |收益率 − threshold|   （超出阈值的亏损距离）")
        ax.set_ylabel("f(y; ξ, β)")
        ax.set_xlim(0, y_max)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

    # 如果给了 threshold，在上方加一条副坐标轴，显示对应的真实收益率 r = threshold - y
    if threshold is not None:
        if threshold >= 0:
            raise ValueError("threshold 应为负数")
        for ax in (ax_lin, ax_log):
            sec = ax.secondary_xaxis("top", functions=(lambda v: threshold - v,
                                                       lambda r: threshold - r))
            sec.set_xlabel(f"对应收益率 r（threshold = {threshold:.2%}）")
            sec.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))

    fig.suptitle(r"广义帕累托分布  $f(y;\xi,\beta)=\frac{1}{\beta}"
                 r"\left(1+\frac{\xi y}{\beta}\right)^{-1/\xi-1}$   (点线 = 99% 分位数)")
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=150)
        print(f"图已保存到 {save}")
    else:
        plt.show()
    return fig


def _setup_chinese_font():
    """macOS 下让 matplotlib 正常显示中文"""
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "Arial Unicode MS",
                                       "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 3. 命令行入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _setup_chinese_font()
    parser = argparse.ArgumentParser(description="画出给定 (xi, beta) 的 GPD 分布")
    parser.add_argument("--xi", type=float, nargs="+", default=[0.1, 0.3, 0.6],
                        help="形状参数 xi (epsilon)，>0，可输入多个")
    parser.add_argument("--beta", type=float, nargs="+", default=[0.01],
                        help="尺度参数 beta，>0，可输入多个（与 xi 一一对应）")
    parser.add_argument("--threshold", type=float, default=None,
                        help="亏损阈值（负数，如 -0.02），用于在上方显示对应收益率")
    parser.add_argument("--save", type=str, default=None, help="保存图片路径，不填则弹窗显示")
    args = parser.parse_args()

    for xi, beta in zip(args.xi if len(args.xi) > 1 else args.xi * len(args.beta),
                        args.beta if len(args.beta) > 1 else args.beta * len(args.xi)):
        print(f"xi={xi:g}, beta={beta:g}:  P(y > 0.05) = {gpd_survival(0.05, xi, beta):.4%},"
              f"  99% 分位数 = {gpd_quantile(0.99, xi, beta):.4f}")

    plot_gpd(args.xi, args.beta, threshold=args.threshold, save=args.save)
