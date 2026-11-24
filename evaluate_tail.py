"""
Visual check of the trained gated GPD-mixture model's tail thickness.

The model predicts, for each day, a two-component GPD mixture for the excess loss y = L - u:
    S(y|x) = (1 - pi) S_GPD(y; xi_b, beta_b) + pi S_GPD(y; xi_h, beta_h)
(xi_b, beta_b, pi depend on the factors; xi_h, beta_h are global.)

Part 1 -- one day (method 1)
    Pick a day whose NEXT-day loss exceeded the threshold u. Feed that day's factors to
    the model, draw the two GPD components with GPD_model.plot_gpd plus the mixture, and mark the loss that
    actually happened. S(y) = P(excess > y | loss > u) tells how surprising that loss was
    for the model.

Part 2 -- all validation exceedances (the real test)
    For every validation row with loss > u, compute S(y) with that day's predicted
    mixture parameters. If the model's tail is right, S(y) is uniform on [0, 1]:
    about 5% of days have S < 5%, 1% have S < 1%, ...
        many more small S than expected -> real crashes are worse than predicted -> tail TOO THIN
        many fewer small S than expected -> model expects worse crashes than happen -> tail TOO THICK

Parts 1-2 are CONDITIONAL on the loss exceeding u. Stage 1 of the two-stage model,
p_u(x) = P(L > u | x), turns them into unconditional probabilities:
    P(L > l | x) = p_u(x) * S(l - u | x)

Part 3 -- two-stage model on ALL validation stock-days (artifacts/two_stage_validation.png)
    (a) p_u calibration: mean predicted p_u vs observed exceedance rate per p_u decile
    (b) VaR backtest: breach rate of VaR_alpha vs alpha (99.9 / 99.5 / 99 / 98)
    (c) VaR99 breach rate per p_u decile: two-stage model vs a constant 10% instead of p_u(x)
    (d) the Part 1 stock over the validation period: daily loss vs its VaR99, breaches marked

Model and formulas come from tail_model.py (numpy; no torch needed).

Run:
    .venv/bin/python evaluate_tail.py                         # biggest validation loss
    .venv/bin/python evaluate_tail.py --ticker NVDA --date 2024-08-02
    .venv/bin/python evaluate_tail.py --list                  # show candidate days
    .venv/bin/python evaluate_tail.py --loss 0.26             # day whose loss is closest to 26%
    .venv/bin/python evaluate_tail.py --no-show               # only save the PNGs

`--date` is the date of the factors (day t); the loss happens on the next trading day.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from GPD_model import COLORS, gpd_pdf, gpd_quantile, gpd_survival, plot_gpd
import tail_model

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"
CHECK_LEVELS = [0.01, 0.025, 0.05, 0.10, 0.25, 0.50]
MAE_LEVELS = (0.01, 0.05, 0.10)          # levels used for the calibration MAE

INK = "#1f2328"
MUTED = "#6e7781"
REF_GRAY = "#8c959f"
OBS_COLOR = COLORS[0]
LOSS_COLOR = "#c92a2a"


PARAM_COLS = ["xi_b", "beta_b", "xi_h", "beta_h", "pi"]


# ==========================================
# 0. Two-component GPD mixture (formulas from tail_model.py, the single implementation)
# ==========================================

def _pdf(y, xi, beta):            # one GPD component, vectorised over arrays of parameters
    return np.exp(tail_model.gpd_logpdf(np.asarray(y, dtype=float), xi, beta))


def _sf(y, xi, beta):             # computed directly (not 1 - CDF) so tiny S stays accurate
    return np.exp(tail_model.gpd_logsf(np.asarray(y, dtype=float), xi, beta))


def _stack(xi_b, beta_b, xi_h, beta_h, pi):
    return np.stack(np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (xi_b, beta_b, xi_h, beta_h, pi))), -1)


def mix_pdf(y, xi_b, beta_b, xi_h, beta_h, pi):
    return tail_model.mixture_pdf(_stack(xi_b, beta_b, xi_h, beta_h, pi), y)


def mix_survival(y, xi_b, beta_b, xi_h, beta_h, pi):
    return tail_model.mixture_survival(_stack(xi_b, beta_b, xi_h, beta_h, pi), y)


def mix_quantile(q, xi_b, beta_b, xi_h, beta_h, pi):
    """Excess y with mixture CDF = q (tail_model.mixture_quantile_excess)."""
    return float(tail_model.mixture_quantile_excess(_stack(xi_b, beta_b, xi_h, beta_h, pi), 1.0 - q))


def row_params(row):
    return tuple(float(row[c]) for c in PARAM_COLS)


# ==========================================
# 1. Load model + data, predict mixture parameters
# ==========================================

def load_everything():
    """Model (both stages, tail_model.TwoStageTailModel) + dataset with loss columns."""
    meta_path = ARTIFACT_DIR / "preprocess.json"
    data_path = ARTIFACT_DIR / "dataset.csv"
    for p in (meta_path, data_path, ARTIFACT_DIR / "gpd_model.pt", ARTIFACT_DIR / "pu_model.pt"):
        if not p.exists():
            raise SystemExit(f"Missing {p}. Run `TQT research machine learning part.py` first.")

    meta = json.loads(meta_path.read_text())
    model = tail_model.TwoStageTailModel.from_artifacts(ARTIFACT_DIR)

    df = pd.read_csv(data_path, parse_dates=["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["loss_date"] = df.groupby("ticker")["date"].shift(-1)   # day on which Y is realised

    u = float(meta["threshold_u_loss"])
    df["loss"] = -df["Y"]
    df["excess"] = df["loss"] - u
    df["is_val"] = df["date"] >= pd.Timestamp(meta["train_val_split_date"])
    df["is_exc"] = df["loss"] > u
    # (meta, stage-2 model, stage-1 model, module with the backtest helpers, data)
    return meta, model, model, tail_model, df


def predict_params(model, meta, rows, pu_model=None):
    """Add xi_b, beta_b, xi_h, beta_h, pi, S (and p_u, P_uncond) with tail_model.py."""
    pred = model.predict(rows[meta["feature_names"]])
    out = rows.copy()
    for c in PARAM_COLS:
        out[c] = pred[c].to_numpy(dtype=float)
    out["S"] = tail_model.mixture_survival(out[PARAM_COLS].to_numpy(), np.maximum(out["excess"].values, 0.0))
    if pu_model is not None:                 # stage 1: p_u(x) = P(L > u | x)
        out["p_u"] = pred["p_u"].to_numpy(dtype=float)
        out["P_uncond"] = np.where(out["loss"] > float(meta["threshold_u_loss"]),
                                   out["p_u"] * out["S"], np.nan)
    return out


# ==========================================
# 2. Part 1: one day
# ==========================================

def pick_day(val_exc, ticker, date, loss=None):
    if loss is not None:                     # day whose actual loss is closest to the target
        sel = val_exc[val_exc["ticker"] == ticker.upper()] if ticker else val_exc
        return sel.loc[(sel["loss"] - loss).abs().idxmin()]
    if ticker is None and date is None:
        return val_exc.loc[val_exc["loss"].idxmax()]
    sel = val_exc
    if ticker:
        sel = sel[sel["ticker"] == ticker.upper()]
    if date:
        sel = sel[sel["date"] == pd.Timestamp(date)]
    if sel.empty:
        raise SystemExit("No validation exceedance matches --ticker/--date. "
                         "Use --list to see candidate days.")
    return sel.loc[sel["loss"].idxmax()]


def fmt_prob(p):
    """Percent with 2 decimals, or scientific notation when it would round to 0.00%."""
    return f"{p:.2%}" if p >= 5e-5 else f"{p:.1e}"


def plot_one_day(row, u, save_path, show, meta=None, val_S=None):
    xi_b, beta_b, xi_h, beta_h, pi = params = row_params(row)
    y_real, S = float(row["excess"]), float(row["S"])
    p_u = float(row["p_u"]) if "p_u" in row else None
    q90, q99 = mix_quantile(0.90, *params), mix_quantile(0.99, *params)

    # the two components via GPD_model.plot_gpd (thin), then the mixture on top (bold)
    fig = plot_gpd([xi_b, xi_h], [beta_b, beta_h], threshold=-u, save=None, show=False)
    ax_lin, ax_log = fig.axes[0], fig.axes[1]
    names = [f"base component (weight {1 - pi:.2f}): ξ_b = {xi_b:.3f}, β_b = {beta_b:.4f}",
             f"heavy component (weight {pi:.2f}): ξ_h = {xi_h:.3f}, β_h = {beta_h:.4f}"]
    for ax in (ax_lin, ax_log):
        for line in list(ax.get_lines()):
            if line.get_linestyle() == ":":          # plot_gpd's per-component 99% lines
                line.remove()
        for line, name in zip(ax.get_lines(), names):
            line.set_linewidth(1.1)
            line.set_alpha(0.6)
            line.set_linestyle("--")
            line.set_label(name)

    x_max = max(1.15 * y_real, 1.3 * q99, mix_quantile(0.995, *params))
    grid = np.linspace(0, x_max, 1500)
    pdf = mix_pdf(grid, *params)

    comp = [(xi_b, beta_b), (xi_h, beta_h)]
    for ax in (ax_lin, ax_log):
        for line, (xi_c, beta_c) in zip(ax.get_lines(), comp):   # extend components to full range
            line.set_data(grid, _pdf(grid, xi_c, beta_c))
            line.set_linewidth(1.6)
            line.set_alpha(0.9)
        ax.plot(grid, pdf, color=INK, lw=2.2, label="mixture (what the model predicts)")
        tail = grid >= y_real
        ax.fill_between(grid[tail], pdf[tail], color=OBS_COLOR, alpha=0.18, lw=0)
        ax.axvline(y_real, color=LOSS_COLOR, lw=2)
        ax.axvline(q99, color=MUTED, ls=":", lw=1.2)
        ax.set_xlim(0, x_max)
        ax.set_ylabel("f(y | x)")
        ax.legend(frameon=False, loc="upper center", fontsize=8)
    ax_log.set_ylim(0.3 * pdf[-1], 3 * max(pdf[0], _pdf(0.0, xi_h, beta_h), _pdf(0.0, xi_b, beta_b)))
    ax_lin.set_title("Mixture density (linear scale)")
    ax_log.set_title("Mixture density (log scale, shows tail thickness)")

    ax_lin.annotate(
        f"actual loss {row['loss']:.2%}\nexcess y = {y_real:.2%}\n"
        f"S(y) = P(excess > y) = {fmt_prob(S)}"
        + (f"\np_u(x) = P(L > u) = {p_u:.2%}\nP(L > actual) = p_u·S = {fmt_prob(p_u * S)}"
           if p_u is not None else ""),
        xy=(y_real, ax_lin.get_ylim()[1] * 0.55), xytext=(-8, 0), textcoords="offset points",
        color=INK, fontsize=9, va="center", ha="right")
    ax_lin.text(q99, ax_lin.get_ylim()[1] * 0.78, "  model 99% quantile",
                color=MUTED, fontsize=8, va="top")

    if S < 0.01:
        verdict = "very surprising for the model (S < 1%)"
    elif S < 0.05:
        verdict = "surprising for the model (S < 5%)"
    else:
        verdict = "within what the model expected (S >= 5%)"
    loss_day = row["loss_date"].date() if pd.notna(row["loss_date"]) else "next day"
    if val_S is not None:
        n_days = len(val_S)
        rank = int((np.asarray(val_S) < S).sum()) + 1
    fig.suptitle(
        f"{row['ticker']}: factors on {row['date'].date()} -> loss on {loss_day}   "
        f"π = {pi:.3f}, ξ_b = {xi_b:.3f}, β_b = {beta_b:.4f}, ξ_h = {xi_h:.3f}, β_h = {beta_h:.4f}\n"
        f"Shaded = model probability of a loss at least this deep: {fmt_prob(S)} -- {verdict}.  "
        f"(conditional on loss > u = {u:.2%}; one day is an example, not evidence)"
        + (f"\nStage 1: p_u(x) = P(loss > u) = {p_u:.2%}  ->  unconditional P(loss ≥ actual) = "
           f"p_u·S = {fmt_prob(p_u * S)}" if p_u is not None else "")
        + (f"\nS rank: {rank} of {n_days} validation days (smallest = 1; a correct model's smallest "
           f"S would be about {1 / n_days:.2%})\n{model_label(meta)}" if val_S is not None else ""),
        fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"\n[Part 1] {row['ticker']} factors {row['date'].date()} -> loss on {loss_day}")
    print(f"  predicted pi = {pi:.4f}, xi_b = {xi_b:.4f}, beta_b = {beta_b:.5f}, "
          f"xi_h = {xi_h:.4f}, beta_h = {beta_h:.5f}")
    print(f"  actual loss = {row['loss']:.2%}, excess over u = {y_real:.2%}")
    print(f"  model 90% / 99% quantile of excess = {q90:.2%} / {q99:.2%}")
    print(f"  S(y) = {fmt_prob(S)}  -> {verdict}")
    if p_u is not None:
        print(f"  stage 1 p_u(x) = {p_u:.2%}  -> unconditional P(L >= actual) = {fmt_prob(p_u * S)}")
    if val_S is not None:
        print(f"  S rank {rank} of {n_days} validation days "
              f"(a correct model's smallest S would be about {1 / n_days:.2%})")
    print(f"  figure: {save_path}")
    return fig


# ==========================================
# 3. Part 2: calibration over all validation exceedances
# ==========================================

def calibration_table(S):
    n = len(S)
    rows = []
    for a in CHECK_LEVELS:
        k = int((S < a).sum())
        lo, hi = stats.binom.interval(0.95, n, a)
        p = stats.binomtest(k, n, a).pvalue
        if k > hi:
            verdict = "TOO THIN"
        elif k < lo:
            verdict = "TOO THICK"
        else:
            verdict = "ok"
        rows.append(dict(level=a, expected=a * n, observed=k, obs_frac=k / n,
                         ci_lo=lo / n, ci_hi=hi / n, p_value=p, verdict=verdict))
    return pd.DataFrame(rows)


def model_label(meta):
    """Short description of the trained model for titles and printouts."""
    return (f"two-stage: p_u net + gated GPD mixture {tuple(meta.get('hidden_units', []))}+gate{meta.get('gate_units')}, "
            f"pi_min {meta.get('pi_min')}, {len(meta.get('feature_names', []))} factors, "
            f"L2 {meta.get('l2')}, best epoch {meta.get('best_epoch', '?')}")


def comparison_metrics(rows):
    """Model-comparison metrics: mixture NLL and calibration MAE (pp) at 1/5/10%."""
    P = [rows[c].values for c in PARAM_COLS]
    y = rows["excess"].values
    nll = float(-np.mean(np.log(mix_pdf(y, *P))))
    S = rows["S"].values
    mae = 100.0 * float(np.mean([abs((S < a).mean() - a) for a in MAE_LEVELS]))
    return nll, mae


def print_calibration(name, S):
    tab = calibration_table(S)
    ks = stats.kstest(S, "uniform")
    print(f"\n[Part 2] {name}: {len(S)} exceedance days")
    print(f"  {'level':>6} {'expected':>9} {'observed':>9} {'obs %':>7} "
          f"{'95% range if correct':>21} {'p-value':>8}  verdict")
    for r in tab.itertuples():
        print(f"  {r.level:>6.1%} {r.expected:>9.1f} {r.observed:>9d} {r.obs_frac:>7.2%} "
              f"   {r.ci_lo:>7.2%} - {r.ci_hi:<7.2%}   {r.p_value:>8.3f}  {r.verdict}")
    print(f"  KS test vs uniform: statistic {ks.statistic:.3f}, p-value {ks.pvalue:.3g} "
          f"(small p = S is not uniform = tail shape is off somewhere)")
    return tab, ks


def plot_calibration(val, tab, ks, save_path, meta=None, nll=None, mae=None):
    S = val["S"].values
    n = len(S)
    fig, (ax_h, ax_c) = plt.subplots(1, 2, figsize=(13, 5))

    # (a) histogram of S: should be flat if the tail is right
    bins = np.linspace(0, 1, 21)
    counts, _ = np.histogram(S, bins)
    lo, hi = stats.binom.interval(0.95, n, 1 / 20)
    ax_h.axhspan(lo, hi, color=REF_GRAY, alpha=0.18, lw=0, label="95% range if model is correct")
    ax_h.axhline(n / 20, color=REF_GRAY, ls="--", lw=1.2)
    ax_h.bar(bins[:-1], counts, width=0.05 * 0.92, align="edge", color=OBS_COLOR,
             label="validation days")
    ax_h.set_xlim(0, 1)
    ax_h.set_ylim(0, max(counts.max(), hi) * 1.3)
    ax_h.set_xlabel("S(y) = model probability of a loss at least as deep as the actual one")
    ax_h.set_ylabel("number of days")
    ax_h.set_title("Where actual losses fell in the predicted tail (flat = well calibrated)",
                   fontsize=10)
    ax_h.text(0.01, 0.97, "← pile-up here:\n   tail too thin", transform=ax_h.transAxes,
              va="top", fontsize=8, color=MUTED)
    ax_h.text(0.99, 0.97, "pile-up here: →\ntail too thick", transform=ax_h.transAxes,
              va="top", ha="right", fontsize=8, color=MUTED)
    ax_h.legend(frameon=False, loc="upper center", fontsize=8)

    # (b) expected vs observed share of days with S < level
    lv = tab["level"].values
    ax_c.fill_between(lv, tab["ci_lo"], tab["ci_hi"], color=REF_GRAY, alpha=0.18, lw=0,
                      label="95% range if model is correct")
    ax_c.plot(lv, lv, color=REF_GRAY, ls="--", lw=1.2, label="perfect calibration")
    floor = 0.5 / n          # a share of 0 cannot sit on a log axis: draw it at the floor
    obs_plot = np.maximum(tab["obs_frac"].values, floor)
    ax_c.plot(lv, obs_plot, color=OBS_COLOR, lw=2, marker="o", ms=7,
              markeredgecolor="white", markeredgewidth=1.5, label="observed")
    for r, yv in zip(tab.itertuples(), obs_plot):
        label = f"{r.obs_frac:.1%}" if r.observed else "0 days"
        ax_c.annotate(label, (r.level, yv), xytext=(6, -10),
                      textcoords="offset points", fontsize=8, color=INK)
    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.set_xticks(lv, [f"{x:.1%}" for x in lv])
    ax_c.set_xlabel("level α  (expected share of days with S < α)")
    ax_c.set_ylabel("observed share of days with S < α")
    ax_c.set_title("Above the band = tail too thin, below = too thick", fontsize=10)
    ax_c.legend(frameon=False, fontsize=8, loc="upper left")

    for ax in (ax_h, ax_c):
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    r5 = tab.set_index("level").loc[0.05]
    fig.suptitle(
        f"Validation tail check: {n} days with loss > u.   "
        f"S < 5% on {r5.observed} days ({r5.obs_frac:.1%}; correct model: "
        f"{r5.ci_lo:.1%}–{r5.ci_hi:.1%}) -> {r5.verdict}.   KS p = {ks.pvalue:.3g}\n"
        f"Mixture NLL {nll:.4f}   calibration MAE (1/5/10%) {mae:.2f} pp   |   {model_label(meta)}",
        fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"  figure: {save_path}")
    return fig


# ==========================================
# 4. Part 3: two-stage model on all validation stock-days
# ==========================================

def two_stage_validation(tm, val_all, u, ticker, save_path, meta):
    p_u = val_all["p_u"].values
    params = val_all[PARAM_COLS].values
    L = val_all["loss"].values
    e = (L > u).astype(float)
    n = len(L)

    # (a) p_u calibration per decile
    dec = pd.qcut(p_u, 10, labels=False, duplicates="drop")
    cal = pd.DataFrame({"p": p_u, "e": e, "d": dec}).groupby("d").agg(p=("p", "mean"), e=("e", "mean"),
                                                                       n=("e", "size"))
    auc = tm.auc_score(p_u, e)
    ll = float(-np.mean(e * np.log(np.clip(p_u, 1e-7, 1)) + (1 - e) * np.log(np.clip(1 - p_u, 1e-7, 1))))
    base = 1 - meta["tail_quantile"]
    ll_c = float(-np.mean(e * np.log(base) + (1 - e) * np.log(1 - base)))

    # (b) VaR backtest, (c) VaR99 by p_u decile
    bt = tm.var_backtest(p_u, params, L, u)
    dec99 = pd.DataFrame(tm.var99_by_decile(p_u, params, L, u))

    print(f"\n[Part 3] two-stage model on all {n} validation stock-days")
    print(f"  stage 1: log-loss {ll:.4f} (constant {base:.0%}: {ll_c:.4f}), AUC {auc:.3f}, "
          f"decile calibration error mean {100 * (cal.p - cal.e).abs().mean():.2f} pp")
    for a, r in bt.items():
        print(f"  VaR{100 * (1 - float(a)):g}: breach {r['breach_rate']:.3%} (expected {float(a):.1%}), "
              f"Kupiec p {r['kupiec_p']:.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    lo = stats.binom.ppf(0.025, cal.n, cal.p) / cal.n
    hi = stats.binom.ppf(0.975, cal.n, cal.p) / cal.n
    lim = [0, 1.08 * max(cal.p.max(), cal.e.max())]
    ax_a.plot(lim, lim, color=REF_GRAY, ls="--", lw=1.2, label="perfect calibration")
    ax_a.vlines(cal.p, lo, hi, color=REF_GRAY, lw=5, alpha=0.45,
                label="95% range if p_u is correct")
    ax_a.plot(cal.p, cal.e, color=OBS_COLOR, marker="o", ms=7, lw=2,
              markeredgecolor="white", markeredgewidth=1.5, label="observed rate per decile")
    ax_a.axhline(base, color=MUTED, ls=":", lw=1, label=f"constant {base:.0%}")
    ax_a.set_xlim(lim)
    ax_a.set_ylim(lim)
    ax_a.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_a.set_xlabel("mean predicted p_u(x) in the decile")
    ax_a.set_ylabel("observed share of days with loss > u")
    ax_a.set_title(f"(a) Stage 1 calibration by p_u decile   AUC {auc:.3f}, "
                   f"log-loss {ll:.4f} (const: {ll_c:.4f})", fontsize=10)
    ax_a.legend(frameon=False, fontsize=8, loc="upper left")

    alphas = np.array([float(a) for a in bt])
    obs = np.array([bt[a]["breach_rate"] for a in bt])
    b_lo = stats.binom.ppf(0.025, n, alphas) / n
    b_hi = stats.binom.ppf(0.975, n, alphas) / n
    ax_b.fill_between(alphas, b_lo, b_hi, color=REF_GRAY, alpha=0.2, lw=0,
                      label="95% range if VaR is correct")
    ax_b.plot(alphas, alphas, color=REF_GRAY, ls="--", lw=1.2, label="expected")
    ax_b.plot(alphas, obs, color=OBS_COLOR, marker="o", ms=7, lw=2,
              markeredgecolor="white", markeredgewidth=1.5, label="observed breach rate")
    for a, o in zip(alphas, obs):
        ax_b.annotate(f"{o:.3%}", (a, o), xytext=(6, -12), textcoords="offset points",
                      fontsize=8, color=INK)
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xticks(alphas, [f"VaR{100 * (1 - a):g}\n({a:.1%})" for a in alphas])
    ax_b.minorticks_off()
    ax_b.set_ylabel("share of days with loss > VaR")
    ax_b.set_title(f"(b) Unconditional VaR backtest, {n:,} validation stock-days\n"
                   "below the band = too conservative, above = too risky", fontsize=10)
    ax_b.legend(frameon=False, fontsize=8, loc="upper left")

    x = dec99["decile"].values
    w = 0.38
    ax_c.bar(x - w / 2, dec99["breach_model"], w, color=OBS_COLOR, label="two-stage: p_u(x)·S")
    ax_c.bar(x + w / 2, dec99["breach_constant_p_u"], w, color=REF_GRAY,
             label="constant 10% instead of p_u(x)")
    ax_c.axhline(0.01, color=LOSS_COLOR, ls="--", lw=1.2, label="target 1%")
    ax_c.set_xticks(x, [f"{d}\n{p:.1%}" for d, p in zip(x, dec99["mean_p_u"])])
    ax_c.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
    ax_c.set_xlabel("p_u decile (1 = calmest) and its mean p_u")
    ax_c.set_ylabel("VaR99 breach rate")
    ax_c.set_title("(c) VaR99 breaches by risk decile: p_u(x) keeps them flat at ~1%",
                   fontsize=10)
    ax_c.legend(frameon=False, fontsize=8, loc="upper left")

    one = val_all[val_all["ticker"] == ticker].sort_values("date")
    var99, _ = tm.var_es(one["p_u"].values, one[PARAM_COLS].values, 0.01, u)
    t = one["loss_date"].fillna(one["date"])
    breach = one["loss"].values > np.nan_to_num(var99, nan=u)
    ax_d.bar(t, one["loss"].clip(lower=0), width=1.0, color=REF_GRAY, label="daily loss (gains = 0)")
    ax_d.plot(t, var99, color=OBS_COLOR, lw=1.4, label="VaR99 (two-stage)")
    ax_d.axhline(u, color=MUTED, ls=":", lw=1, label=f"threshold u = {u:.2%}")
    ax_d.scatter(t[breach], one["loss"].values[breach], color=LOSS_COLOR, s=22, zorder=3,
                 label=f"breach ({breach.sum()} of {len(one)} days = {breach.mean():.1%})")
    ax_d.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    import matplotlib.dates as mdates
    ax_d.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    ax_d.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_d.set_ylabel("loss")
    ax_d.set_title(f"(d) {ticker} over the validation period: loss vs its daily VaR99", fontsize=10)
    ax_d.legend(frameon=False, fontsize=8, loc="upper left")

    for ax in axes.ravel():
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Two-stage tail model on ALL validation stock-days:  "
                 "P(L > l | x) = p_u(x) · S(l − u | x)\n" + model_label(meta), fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"  figure: {save_path}")
    return fig


# ==========================================
# 5. Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Tail-thickness check of the dynamic GPD model")
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--loss", type=float, default=None,
                        help="pick the validation day whose actual loss is closest to this (e.g. 0.26)")
    parser.add_argument("--date", type=str, default=None,
                        help="factor date t (YYYY-MM-DD); the loss is on the next trading day")
    parser.add_argument("--list", action="store_true", help="print the 15 largest validation losses")
    parser.add_argument("--no-show", action="store_true", help="save PNGs without opening windows")
    args = parser.parse_args()

    meta, model, pu_model, tm, df = load_everything()
    u = float(meta["threshold_u_loss"])

    val_all = predict_params(model, meta, df[df["is_val"]], pu_model)       # every validation row
    exc = pd.concat([predict_params(model, meta, df[df["is_exc"] & ~df["is_val"]], pu_model),
                     val_all[val_all["is_exc"]]])
    val, train = exc[exc["is_val"]], exc[~exc["is_val"]]
    print(f"Threshold u = {u:.4%} loss, split date {meta['train_val_split_date']}, "
          f"exceedances: train {len(train)}, validation {len(val)}")

    if args.list:
        cols = ["date", "loss_date", "ticker", "loss", "p_u", "pi", "xi_b", "beta_b", "S"]
        top = val.nlargest(15, "loss")[cols].copy()
        top["loss"] = top["loss"].map("{:.2%}".format)
        top["S"] = top["S"].map("{:.2%}".format)
        print(top.to_string(index=False))
        return

    row = pick_day(val, args.ticker, args.date, args.loss)
    day_png = ARTIFACT_DIR / f"tail_day_{row['ticker']}_{row['date'].date()}.png"
    plot_one_day(row, u, day_png, show=False, meta=meta, val_S=val["S"].values)

    print(f"Model: {model_label(meta)}")
    tab, ks = print_calibration("VALIDATION (unseen data)", val["S"].values)
    nll, mae = comparison_metrics(val)
    print(f"  Mixture NLL {nll:.4f}   calibration MAE (1%, 5%, 10%) {mae:.2f} pp")
    print_calibration("training (for reference only; in-sample)", train["S"].values)
    plot_calibration(val, tab, ks, ARTIFACT_DIR / "tail_calibration.png", meta, nll, mae)

    print(f"\n  heavy component (global): xi_h = {val['xi_h'].iloc[0]:.4f}, "
          f"beta_h = {val['beta_h'].iloc[0]:.5f}")
    for c in ("pi", "xi_b", "beta_b"):
        print(f"  predicted {c:<6} on validation exceedances: 5% {val[c].quantile(.05):.4f}, "
              f"median {val[c].median():.4f}, 95% {val[c].quantile(.95):.4f}")

    two_stage_validation(tm, val_all, u, row["ticker"],
                         ARTIFACT_DIR / "two_stage_validation.png", meta)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
