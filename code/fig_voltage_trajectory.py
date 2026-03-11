"""
Figure 3: Voltage trajectory comparison — Shepherd (proposed) vs baselines.
Three-panel layout:
  (a) Voltage–time curves: Proposed model vs Experimental data
  (b) Prediction error: Proposed vs NBM baseline (mV)
  (c) RMSE bar chart across all 8 batteries, all 5 models

Data sources:
  - voltage_error_timeseries.csv  (Shepherd + NBM predictions)
  - baseline_comparison.csv       (RMSE across 8 batteries × 5 models)
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pub_style import (
    apply_style,
    save_fig,
    label_panel,
    embed_metric,
    MODEL_COLORS,
    DOUBLE_COL_TALL,
    COLORS,
)

apply_style()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "..", "results")

    # ── Load precomputed data ──
    ts = pd.read_csv(os.path.join(results_dir, "voltage_error_timeseries.csv"))
    df_all = pd.read_csv(os.path.join(results_dir, "baseline_comparison.csv"))

    t_h = ts["time_h"].values
    v_exp = ts["voltage_exp"].values
    v_main = ts["voltage_main"].values
    v_nbm = ts["voltage_nbm"].values
    err_main = ts["error_main"].values * 1000  # → mV
    err_nbm = ts["error_nbm"].values * 1000

    # ── Figure layout ──
    fig = plt.figure(figsize=DOUBLE_COL_TALL)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.15, 1], hspace=0.38, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    # ═══════ Panel (a): Voltage trajectories ═══════
    ax_a.plot(
        t_h, v_exp, "-", color=COLORS["exp"], lw=1.6, label="Experimental", zorder=10
    )
    ax_a.plot(
        t_h,
        v_main,
        "-",
        color=COLORS["shepherd"],
        lw=1.2,
        marker="s",
        markevery=(50, 600),
        markersize=3.5,
        label="Shepherd (Proposed)",
        alpha=0.9,
        zorder=8,
    )
    ax_a.plot(
        t_h,
        v_nbm,
        "--",
        color=COLORS["nbm"],
        lw=1.0,
        marker="^",
        markevery=(200, 600),
        markersize=3.5,
        label="NBM",
        alpha=0.8,
        zorder=6,
    )

    ax_a.set_xlabel("Time (h)")
    ax_a.set_ylabel("Terminal voltage (V)")
    ax_a.set_xlim(t_h[0], t_h[-1])
    ax_a.legend(loc="lower left", fontsize=7, ncol=1, framealpha=0.9)
    label_panel(ax_a, "a")

    rmse_shep = np.sqrt(np.mean((v_main - v_exp) ** 2)) * 1000
    rmse_nbm = np.sqrt(np.mean((v_nbm - v_exp) ** 2)) * 1000
    embed_metric(
        ax_a,
        f"Proposed: RMSE = {rmse_shep:.1f} mV\n"
        f"NBM:          RMSE = {rmse_nbm:.1f} mV",
        x=0.97,
        y=0.85,
    )

    # Inset zoom: end-of-discharge region
    # find the last 15% of time
    t_zoom_start = t_h[-1] * 0.80
    mask_zoom = t_h >= t_zoom_start
    axins = ax_a.inset_axes([0.42, 0.12, 0.35, 0.42])
    axins.plot(t_h[mask_zoom], v_exp[mask_zoom], "-", color=COLORS["exp"], lw=1.4)
    axins.plot(
        t_h[mask_zoom],
        v_main[mask_zoom],
        "-",
        color=COLORS["shepherd"],
        lw=1.0,
        marker="s",
        markevery=200,
        markersize=3,
    )
    axins.plot(
        t_h[mask_zoom],
        v_nbm[mask_zoom],
        "--",
        color=COLORS["nbm"],
        lw=0.9,
        marker="^",
        markevery=200,
        markersize=3,
    )
    axins.set_xlabel("")
    axins.set_ylabel("")
    axins.tick_params(labelsize=6)
    axins.set_title("End-of-discharge", fontsize=6, pad=2)
    for spine in axins.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("0.5")
    ax_a.indicate_inset_zoom(axins, edgecolor="0.5", linewidth=0.8)

    # ═══════ Panel (b): Error comparison ═══════
    ax_b.fill_between(
        t_h, -20, 20, color="#43A047", alpha=0.08, zorder=0, label="±20 mV band"
    )
    ax_b.plot(
        t_h,
        err_main,
        "-",
        color=COLORS["shepherd"],
        lw=1.0,
        label=f"Shepherd (RMSE={rmse_shep:.1f} mV)",
        alpha=0.9,
    )
    ax_b.plot(
        t_h,
        err_nbm,
        "--",
        color=COLORS["nbm"],
        lw=0.9,
        label=f"NBM (RMSE={rmse_nbm:.1f} mV)",
        alpha=0.8,
    )
    ax_b.axhline(0, color="k", ls="-", lw=0.4, zorder=1)
    ax_b.set_xlabel("Time (h)")
    ax_b.set_ylabel("Prediction error (mV)")
    ax_b.set_xlim(t_h[0], t_h[-1])
    ax_b.legend(loc="upper left", fontsize=6, framealpha=0.85)
    label_panel(ax_b, "b")

    # ═══════ Panel (c): Multi-battery RMSE bar chart ═══════
    model_order = ["Shepherd (Proposed)", "NBM", "Rint", "Thevenin-1RC", "Thevenin-2RC"]
    model_labels = [
        "Shepherd\n(Proposed)",
        "NBM",
        "Rint",
        "Thévenin\n1-RC",
        "Thévenin\n2-RC",
    ]

    means, stds = [], []
    for m in model_order:
        sub = df_all[df_all["Model"] == m]["RMSE (mV)"]
        means.append(sub.mean())
        stds.append(sub.std())

    x_pos = np.arange(len(model_order))
    colors_bar = [MODEL_COLORS[m] for m in model_order]

    bars = ax_c.bar(
        x_pos,
        means,
        width=0.55,
        color=colors_bar,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.6,
        yerr=stds,
        capsize=3,
        error_kw=dict(lw=0.8, capthick=0.8, color="0.3"),
    )

    for i, (m, s) in enumerate(zip(means, stds)):
        ax_c.text(
            i,
            m + s + 1.2,
            f"{m:.1f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            fontweight="bold",
        )

    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(model_labels, fontsize=6.5)
    ax_c.set_ylabel("RMSE (mV)")
    ax_c.set_ylim(0, max(means) + max(stds) + 12)
    bars[0].set_edgecolor(COLORS["shepherd"])
    bars[0].set_linewidth(1.8)
    label_panel(ax_c, "c")

    # Annotation arrow
    ax_c.annotate(
        f"Best: {means[0]:.1f} mV",
        xy=(0, means[0]),
        xytext=(1.8, means[0] + 20),
        fontsize=6.5,
        fontweight="bold",
        color=COLORS["shepherd"],
        arrowprops=dict(
            arrowstyle="->",
            color=COLORS["shepherd"],
            lw=1.0,
            connectionstyle="arc3,rad=-0.2",
        ),
    )

    fig.align_ylabels([ax_a, ax_b])
    save_fig(fig, "fig_voltage_trajectory", results_dir)
    print("Done: fig_voltage_trajectory")


if __name__ == "__main__":
    main()
