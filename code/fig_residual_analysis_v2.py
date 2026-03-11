"""
Figure 9: Residual analysis — model accuracy across SOC regions + distribution.
Three-panel layout:
  (a) Voltage error vs time (proposed model)
  (b) Error histogram + normal fit
  (c) SOC-segmented RMSE bar chart

Data: voltage_error_timeseries.csv + soc_segmented_error.csv
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pub_style import (
    apply_style,
    save_fig,
    label_panel,
    embed_metric,
    DOUBLE_COL_TALL,
    COLORS,
)

apply_style()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "..", "results")

    ts = pd.read_csv(os.path.join(results_dir, "voltage_error_timeseries.csv"))
    seg = pd.read_csv(os.path.join(results_dir, "soc_segmented_error.csv"))

    err_mv = ts["error_main"].values * 1000
    t_h = ts["time_h"].values
    soc = ts["soc"].values

    fig = plt.figure(figsize=DOUBLE_COL_TALL)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.38, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    # ═══════ Panel (a): Error vs time, coloured by SOC ═══════
    # Create SOC-coloured scatter for richer visual
    sc = ax_a.scatter(
        t_h[::10],
        err_mv[::10],
        c=soc[::10] * 100,
        cmap="RdYlBu",
        s=1.5,
        alpha=0.6,
        zorder=3,
        rasterized=True,
    )
    ax_a.axhline(0, color="k", lw=0.5, zorder=2)
    ax_a.fill_between(t_h, -10, 10, color="#43A047", alpha=0.06, zorder=0)
    ax_a.set_xlabel("Time (h)")
    ax_a.set_ylabel("Voltage error (mV)")
    ax_a.set_xlim(t_h[0], t_h[-1])

    # RMSE annotation
    rmse = np.sqrt(np.mean(err_mv**2))
    mae = np.mean(np.abs(err_mv))
    embed_metric(ax_a, f"RMSE = {rmse:.2f} mV\nMAE  = {mae:.2f} mV", x=0.97, y=0.88)

    cbar = fig.colorbar(sc, ax=ax_a, pad=0.02, shrink=0.8, aspect=25)
    cbar.set_label("SOC (%)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    label_panel(ax_a, "a")

    # ═══════ Panel (b): Error distribution ═══════
    bins = np.linspace(-40, 40, 60)
    ax_b.hist(
        err_mv,
        bins=bins,
        density=True,
        color="#1E88E5",
        alpha=0.6,
        edgecolor="white",
        linewidth=0.3,
        zorder=3,
        label="Observed",
    )

    # Normal fit overlay
    mu, sigma = stats.norm.fit(err_mv)
    x_fit = np.linspace(-40, 40, 200)
    ax_b.plot(
        x_fit,
        stats.norm.pdf(x_fit, mu, sigma),
        "-",
        color="#E53935",
        lw=1.3,
        zorder=4,
        label=f"Normal fit\n$\\mu={mu:.2f}$, $\\sigma={sigma:.2f}$",
    )

    ax_b.set_xlabel("Voltage error (mV)")
    ax_b.set_ylabel("Probability density")
    ax_b.legend(fontsize=6, loc="upper right", framealpha=0.9)

    # Shapiro-Wilk (on subsample)
    _, sw_p = stats.shapiro(err_mv[::50][:5000])
    ax_b.text(
        0.03,
        0.95,
        f"Shapiro–Wilk $p={sw_p:.2e}$",
        transform=ax_b.transAxes,
        fontsize=5.5,
        va="top",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.85),
    )

    label_panel(ax_b, "b")

    # ═══════ Panel (c): SOC-segmented RMSE ═══════
    soc_labels = seg["SOC_range"].values
    rmse_vals = seg["RMSE_mV"].values
    mae_vals = seg["MAE_mV"].values
    bias_vals = seg["Mean_bias_mV"].values

    x = np.arange(len(soc_labels))
    bar_w = 0.30

    bars1 = ax_c.bar(
        x - bar_w / 2,
        rmse_vals,
        width=bar_w,
        color="#1E88E5",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        label="RMSE",
    )
    bars2 = ax_c.bar(
        x + bar_w / 2,
        mae_vals,
        width=bar_w,
        color="#43A047",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        label="MAE",
    )

    # Bias markers
    ax_c2 = ax_c.twinx()
    ax_c2.plot(
        x,
        bias_vals,
        "D-",
        color="#E53935",
        markersize=4,
        lw=1.0,
        label="Mean bias",
        zorder=5,
    )
    ax_c2.axhline(0, color="0.5", ls=":", lw=0.4)
    ax_c2.set_ylabel("Mean bias (mV)", color="#E53935", fontsize=7)
    ax_c2.tick_params(axis="y", labelcolor="#E53935", labelsize=6.5)

    # Value labels
    for i, (r, m) in enumerate(zip(rmse_vals, mae_vals)):
        ax_c.text(
            i - bar_w / 2,
            r + 0.2,
            f"{r:.1f}",
            ha="center",
            va="bottom",
            fontsize=5.5,
            fontweight="bold",
        )

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(soc_labels, fontsize=6, rotation=15, ha="right")
    ax_c.set_ylabel("Error (mV)")
    ax_c.set_xlabel("SOC region")

    # Combined legend
    lines1, labels1 = ax_c.get_legend_handles_labels()
    lines2, labels2 = ax_c2.get_legend_handles_labels()
    ax_c.legend(
        lines1 + lines2,
        labels1 + labels2,
        fontsize=5.5,
        loc="upper right",
        framealpha=0.9,
        ncol=1,
    )

    label_panel(ax_c, "c")

    save_fig(fig, "fig_residual_analysis", results_dir)
    print("Done: fig_residual_analysis")


if __name__ == "__main__":
    main()
