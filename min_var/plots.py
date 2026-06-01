"""
Visualization helpers for min_var experiments.
Consolidates all inline plotting code from log_norm.ipynb Cells 37-44.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as mgs
from matplotlib.patches import Patch

from .config import R_ANN, SYMS, UNIV_A
from .optimizer import build_sigma_mu, trace_frontier, ablation_table


# ── Color palettes ────────────────────────────────────────────────────────────
PORT_COLORS = ["#9aa9c6", "#2563eb", "#16a34a", "#f97316", "#dc2626", "#7c3aed"]
CURVE_COLORS = {
    "Stocks only":       "#9aa9c6",
    "SPY opts + stocks": "#2563eb",
    "All opts + stocks": "#dc2626",
    "ExpC: Mean-Var (SPY opts)":  "#16a34a",
    "ExpD: Mean-Var (All opts)":  "#7c3aed",
}
ABL_MARKERS = {
    "Min-Var":             ("o",  "#1e3a5f", 80),
    "Min-Var + 5% floor":  ("s",  "#16a34a", 80),
    "MV  γ=1":             ("^",  "#f97316", 80),
    "MV  γ=5":             ("^",  "#ea580c", 60),
    "MV  γ=20":            ("^",  "#c2410c", 45),
    "Max-Sharpe*":         ("*",  "#fbbf24", 130),
    "Equal-Weight":        ("D",  "#7c3aed", 70),
}


def plot_trajectories(
    paths: list[pd.Series],
    title: str = "Walk-Forward Portfolio Trajectories",
    colors: list | None = None,
) -> plt.Figure:
    """Cumulative return + rolling drawdown chart."""
    colors = colors or PORT_COLORS

    fig, (ax_traj, ax_dd) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.suptitle(title, fontsize=12, fontweight="bold")

    for path, col in zip(paths, colors):
        reb = path / path.iloc[0]
        ls  = "--" if "benchmark" in (path.name or "").lower() else "-"
        ax_traj.plot(reb.index, reb.values, label=path.name, color=col, lw=1.8, ls=ls)
        dd = (reb / reb.cummax()) - 1
        ax_dd.fill_between(dd.index, dd.values, 0, color=col, alpha=0.25)
        ax_dd.plot(dd.index, dd.values, color=col, lw=0.9)

    ax_traj.axhline(1.0, color="k", lw=0.7, ls="--", alpha=0.5)
    ax_traj.set_ylabel("Cumulative return (rebased to 1.0)")
    ax_traj.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{y:.2f}×"))
    ax_traj.legend(fontsize=9, loc="upper left")
    ax_traj.grid(axis="y", alpha=0.3)

    ax_dd.axhline(0, color="k", lw=0.6, ls="--", alpha=0.5)
    ax_dd.set_ylabel("Drawdown")
    ax_dd.set_xlabel("Date")
    ax_dd.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax_dd.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


def plot_weight_heatmaps(
    exp_results: dict,
    syms: list | None = None,
) -> plt.Figure:
    """One heatmap row per experiment showing weight history over time."""
    syms = syms or SYMS
    exps = [(k, v[0], v[1]) for k, v in exp_results.items()
            if k != "benchmark" and not v[1].empty]
    n_exp = len(exps)
    if n_exp == 0:
        return plt.figure()

    fig, axs = plt.subplots(n_exp, 1, figsize=(13, 3.2 * n_exp), squeeze=False)
    fig.suptitle("Portfolio Weight History at Each Rebalance Date",
                 fontsize=12, fontweight="bold")

    for ax_row, (exp_id, path, wdf) in zip(axs[:, 0], exps):
        plot_cols = list(wdf.columns)
        mat = wdf[plot_cols].fillna(0.0).T
        im  = ax_row.imshow(mat.values, aspect="auto", vmin=0.0, vmax=0.5,
                            cmap="Blues", interpolation="nearest")

        ax_row.set_yticks(range(len(plot_cols)))
        yt_colors = []
        for c in plot_cols:
            if "_call" in c:   yt_colors.append("#16a34a")
            elif "_put" in c:  yt_colors.append("#dc2626")
            else:              yt_colors.append("#1e3a5f")
        ax_row.set_yticklabels(plot_cols, fontsize=7)
        for tick, tc in zip(ax_row.get_yticklabels(), yt_colors):
            tick.set_color(tc)

        step = max(1, len(mat.columns) // 8)
        ax_row.set_xticks(range(0, len(mat.columns), step))
        ax_row.set_xticklabels(
            [str(d)[:10] for d in mat.columns[::step]],
            rotation=28, fontsize=7,
        )
        ax_row.set_title(f"{exp_id}: {path.name}", fontsize=10)
        fig.colorbar(im, ax=ax_row, shrink=0.55, pad=0.01, label="Weight")

    fig.legend(
        handles=[
            Patch(color="#1e3a5f", label="Underlying"),
            Patch(color="#16a34a", label="Call"),
            Patch(color="#dc2626", label="Put"),
        ],
        loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.01),
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    return fig


def plot_vol_return_bars(
    stats_df: pd.DataFrame,
    colors: list | None = None,
) -> plt.Figure:
    """Bar chart: annualised vol (solid) vs annualised return (hatched) per experiment."""
    colors = colors or PORT_COLORS
    labels = list(stats_df.index)
    vols   = [float(str(v).strip("%")) / 100 if isinstance(v, str) else v
              for v in stats_df.get("ann_vol", stats_df.iloc[:, 1])]
    rets   = [float(str(v).strip("%")) / 100 if isinstance(v, str) else v
              for v in stats_df.get("ann_return", stats_df.iloc[:, 0])]

    # Use raw numeric columns if available
    if "ann_vol" in stats_df.columns:
        vols = stats_df["ann_vol"].tolist()
        rets = stats_df["ann_return"].tolist()

    fig, ax = plt.subplots(figsize=(max(10, 2 * len(labels)), 5))
    x   = np.arange(len(labels))
    w_b = 0.35
    ax.bar(x - w_b/2, vols, w_b, color=colors[:len(labels)], alpha=0.85,
           label="Ann Vol", edgecolor="k", lw=0.4)
    ax.bar(x + w_b/2, rets, w_b, color=colors[:len(labels)], alpha=0.45,
           label="Ann Return", edgecolor="k", lw=0.4, hatch="//")

    for i, (v, r) in enumerate(zip(vols, rets)):
        if np.isfinite(v):
            ax.text(x[i] - w_b/2, v + 0.003, f"{v:.1%}", ha="center", va="bottom", fontsize=8)
        if np.isfinite(r):
            ypos = r + 0.003 if r >= 0 else r - 0.018
            ax.text(x[i] + w_b/2, ypos, f"{r:.1%}", ha="center", va="bottom", fontsize=8)

    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, fontsize=8)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title("Annualised Vol (solid) vs Annualised Return (hatched)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


def plot_frontier(
    ef_data: dict,
    ablation: dict | None = None,
    snap_label: str = "",
    rf: float = R_ANN,
) -> plt.Figure:
    """
    Efficient frontier chart.

    ef_data  : {label: (vols_list, rets_list)} — one curve per universe
    ablation : {label: {vol, ret, sharpe}} — optional ablation points on Exp A
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    title = f"Efficient Frontier Ablation{' — ' + snap_label if snap_label else ''}"
    fig.suptitle(title, fontsize=12, fontweight="bold")

    # Left: all frontier curves
    ax = axes[0]
    for uk, (vols, rets) in ef_data.items():
        if not vols:
            continue
        col = CURVE_COLORS.get(uk, "#9aa9c6")
        ax.plot([v * 100 for v in vols], [r * 100 for r in rets],
                lw=2.2, color=col, label=uk)
        ax.scatter([vols[0] * 100], [rets[0] * 100], s=60, color=col, zorder=5, marker="o")

    ax.axhline(rf * 100, color="gray", lw=0.7, ls="--", alpha=0.5,
               label=f"Risk-free ({rf:.1%})")
    ax.set_xlabel("Annualised volatility (%)")
    ax.set_ylabel("Annualised return (%)")
    ax.set_title("Three asset universes", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{y:.0f}%"))

    # Right: Exp A frontier + ablation points
    ax = axes[1]
    exp_a_key = next((k for k in ef_data if "SPY opts" in k), None)
    if exp_a_key:
        vef_a, ref_a = ef_data[exp_a_key]
        ax.plot([v * 100 for v in vef_a], [r * 100 for r in ref_a],
                lw=2.5, color="#2563eb", alpha=0.7, label="Exp A frontier")

    if ablation:
        for lbl, d in ablation.items():
            mk, c, sz = ABL_MARKERS.get(lbl, ("o", "gray", 60))
            sh = d.get("sharpe", np.nan)
            sh_str = f"Sh={sh:.2f}" if np.isfinite(sh) else ""
            ax.scatter([d["vol"] * 100], [d["ret"] * 100],
                       marker=mk, color=c, s=sz, zorder=6,
                       label=f"{lbl}  ({sh_str})")

    ax.axhline(rf * 100, color="gray", lw=0.7, ls="--", alpha=0.5)
    ax.set_xlabel("Annualised volatility (%)")
    ax.set_ylabel("Annualised return (%)")
    ax.set_title("Exp A (SPY opts + stocks): objective ablation", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{y:.0f}%"))

    plt.tight_layout()
    return fig


def plot_accuracy(mu_df: pd.DataFrame) -> plt.Figure:
    """Predicted E[R] vs realized R scatter + distribution histograms."""
    asset_pairs = [
        ("underlying", "mu_pred_und",  "mu_real_und"),
        ("call",       "mu_pred_call", "mu_real_call"),
        ("put",        "mu_pred_put",  "mu_real_put"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Predicted E[R] vs Realized R  ({len(mu_df)} windows)", fontsize=11)

    for ax, (asset, p_col, r_col) in zip(axes, asset_pairs):
        ax.hist(mu_df[r_col], bins=20, alpha=0.6, color="#dc2626", label="Realized", density=True)
        ax.hist(mu_df[p_col], bins=20, alpha=0.5, color="#2563eb", label="Predicted E[R]", density=True)
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.set_title(asset)
        ax.set_xlabel("Return")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    plt.tight_layout()
    return fig


def plot_variance_scatter(mu_df: pd.DataFrame) -> plt.Figure:
    """Scatter of predicted vs realized vol per asset."""
    asset_pairs = [
        ("underlying", "vol_pred_und",  "vol_real_und"),
        ("call",       "vol_pred_call", "vol_real_call"),
        ("put",        "vol_pred_put",  "vol_real_put"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Predicted Vol vs Realized Vol", fontsize=11)

    for ax, (asset, pv, rv) in zip(axes, asset_pairs):
        x, y = mu_df[pv].values, mu_df[rv].values
        ax.scatter(x, y, s=18, alpha=0.5, color="#2563eb")
        lim = max(np.nanpercentile(y, 95), np.nanpercentile(x, 95)) * 1.1
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="Perfect prediction")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("Predicted vol")
        ax.set_ylabel("Realized vol")
        ax.set_title(asset)
        ax.legend(fontsize=9)

    plt.tight_layout()
    return fig


def build_and_plot_frontier(
    rp_idx: dict,
    rets_all: pd.DataFrame,
    univ_stocks: list,
    univ_a: list,
    univ_b: list,
    syms: list,
    snap_date: pd.Timestamp | str = "2023-11-30",
    lookback: int = 63,
    rf: float = R_ANN,
) -> tuple[plt.Figure, dict, dict]:
    """
    High-level helper: build Σ+μ for all three universes, trace frontiers,
    run ablation, and return the frontier figure.

    Returns (fig, ef_data, ablation_dict).
    """
    snap = pd.Timestamp(snap_date)

    ef_data: dict = {}
    for uk, univ in [
        ("Stocks only", univ_stocks),
        ("SPY opts + stocks", univ_a),
        ("All opts + stocks", univ_b),
    ]:
        res = build_sigma_mu(univ, syms, rp_idx, rets_all, snap, lookback=lookback, rf=rf)
        if res is None:
            print(f"  Skipping {uk}")
            continue
        cols, mu, Sig = res
        vols, rets = trace_frontier(cols, mu, Sig)
        if vols:
            ef_data[uk] = (vols, rets)
            print(f"  {uk}: {len(vols)} pts  vol=[{min(vols):.1%}–{max(vols):.1%}]")

    # Ablation on Exp A universe
    abl: dict = {}
    res_a = build_sigma_mu(univ_a, syms, rp_idx, rets_all, snap, lookback=lookback, rf=rf)
    if res_a is not None:
        cols_a, mu_a, sig_a = res_a
        abl = ablation_table(cols_a, mu_a, sig_a, rf=rf)

    fig = plot_frontier(ef_data, ablation=abl, snap_label=str(snap.date()), rf=rf)
    return fig, ef_data, abl
