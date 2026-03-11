"""
Figure 6: Parameter sensitivity tornado chart.
Single-panel tornado (horizontal bar) showing ±5% parameter perturbation effects.

Data: parameter_sensitivity_physical.csv
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pub_style import apply_style, save_fig, label_panel, SINGLE_COL_TALL, COLORS

apply_style()

PARAM_LABELS = {
    "E0": "$E_0$ (OCV)",
    "K": "$K$ (polarization)",
    "R0": "$R_0$ (resistance)",
    "A": "$A$ (exp. amplitude)",
    "B": "$B$ (exp. rate)",
    "Q0": "$Q_0$ (capacity)",
    "P_total": "$P_{\\mathrm{total}}$ (power)",
}


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "..", "results")

    df = pd.read_csv(os.path.join(results_dir, "parameter_sensitivity_physical.csv"))

    # Sort by total sensitivity span (|neg| + |pos|)
    df["span"] = df["delta_pct_neg"].abs() + df["delta_pct_pos"].abs()
    df = df.sort_values("span", ascending=True).reset_index(drop=True)

    params = df["parameter"].values
    neg = df["delta_pct_neg"].values
    pos = df["delta_pct_pos"].values

    fig, ax = plt.subplots(figsize=SINGLE_COL_TALL)

    y = np.arange(len(params))
    bar_h = 0.55

    # Negative direction bars (left)
    bars_neg = ax.barh(
        y,
        neg,
        height=bar_h,
        color="#E53935",
        alpha=0.80,
        edgecolor="white",
        linewidth=0.5,
        label="$-5\\%$ perturbation",
    )
    # Positive direction bars (right)
    bars_pos = ax.barh(
        y,
        pos,
        height=bar_h,
        color="#1E88E5",
        alpha=0.80,
        edgecolor="white",
        linewidth=0.5,
        label="$+5\\%$ perturbation",
    )

    # Value labels
    for i, (n, p) in enumerate(zip(neg, pos)):
        if abs(n) > 0.3:
            ax.text(
                n - 0.15,
                i,
                f"{n:+.2f}%",
                ha="right",
                va="center",
                fontsize=5.5,
                fontweight="bold",
                color="white" if abs(n) > 2 else "0.3",
            )
        if abs(p) > 0.3:
            ax.text(
                p + 0.15,
                i,
                f"{p:+.2f}%",
                ha="left",
                va="center",
                fontsize=5.5,
                fontweight="bold",
                color="white" if abs(p) > 2 else "0.3",
            )

    ax.axvline(0, color="0.3", lw=0.6, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([PARAM_LABELS.get(p, p) for p in params], fontsize=7)
    ax.set_xlabel("$\\Delta$TTE (%)")
    ax.set_xlim(min(neg) * 1.3, max(pos) * 1.3)
    ax.legend(fontsize=6.5, loc="lower right", framealpha=0.9)

    # Add baseline TTE reference
    baseline = df["baseline_tte_hours"].iloc[0]
    ax.text(
        0.97,
        0.02,
        f"Baseline TTE = {baseline:.2f} h",
        transform=ax.transAxes,
        fontsize=6.5,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9),
    )

    save_fig(fig, "fig_sensitivity_tornado", results_dir)
    print("Done: fig_sensitivity_tornado")


if __name__ == "__main__":
    main()
