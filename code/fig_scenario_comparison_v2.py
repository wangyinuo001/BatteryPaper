"""
Figure 5: Usage-scenario comparison — TTE and power breakdown.
Two-panel layout:
  (a) Grouped bar: TTE at 0%, 10%, 20% background load for each scenario
  (b) Stacked bar: power-component breakdown per scenario

Data: tte_scenarios.csv (pre-computed), hard-coded power breakdown from paper
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pub_style import (
    apply_style,
    save_fig,
    label_panel,
    embed_metric,
    DOUBLE_COL,
    COMP_COLORS,
    COLORS,
)

apply_style()

SCENARIOS = ["Standby", "Reading", "Navigation", "Video", "Gaming"]

# Power breakdown data (from model computation) — in watts
POWER_DATA = {
    "Standby": {
        "Screen": 0.000,
        "SoC": 0.204,
        "Radio": 0.117,
        "GPS": 0.300,
        "Base": 0.120,
    },
    "Reading": {
        "Screen": 0.461,
        "SoC": 0.612,
        "Radio": 0.235,
        "GPS": 0.300,
        "Base": 0.120,
    },
    "Navigation": {
        "Screen": 0.461,
        "SoC": 0.612,
        "Radio": 0.352,
        "GPS": 0.300,
        "Base": 0.120,
    },
    "Video": {
        "Screen": 0.511,
        "SoC": 0.612,
        "Radio": 1.056,
        "GPS": 0.300,
        "Base": 0.120,
    },
    "Gaming": {
        "Screen": 0.729,
        "SoC": 2.040,
        "Radio": 1.114,
        "GPS": 0.300,
        "Base": 0.120,
    },
}


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "..", "results")

    df = pd.read_csv(os.path.join(results_dir, "tte_scenarios.csv"))

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=DOUBLE_COL, gridspec_kw={"wspace": 0.32}
    )

    # ═══════ Panel (a): TTE grouped bars ═══════
    x = np.arange(len(SCENARIOS))
    bar_w = 0.22
    bg_labels = ["0% bkg", "10% bkg", "20% bkg"]
    bg_colors = ["#1E88E5", "#FFB300", "#E53935"]

    for k, (bg, bg_col) in enumerate(zip([0.0, 0.1, 0.2], bg_colors)):
        ttes = []
        for scen in SCENARIOS:
            row = df[(df["scenario"] == scen) & (df["background_extra"] == bg)]
            ttes.append(row["tte_hours"].values[0] if len(row) > 0 else 0)
        offset = (k - 1) * bar_w
        bars = ax_a.bar(
            x + offset,
            ttes,
            width=bar_w,
            color=bg_col,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
            label=bg_labels[k],
        )
        # Value labels
        for i, v in enumerate(ttes):
            ax_a.text(
                x[i] + offset,
                v + 0.15,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=5.5,
                rotation=90,
            )

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(SCENARIOS, fontsize=7)
    ax_a.set_ylabel("Time to empty (h)")
    ax_a.set_ylim(0, df["tte_hours"].max() * 1.25)
    ax_a.legend(
        fontsize=6.5,
        loc="upper right",
        framealpha=0.9,
        title="Background load",
        title_fontsize=7,
    )
    label_panel(ax_a, "a")

    # ═══════ Panel (b): Power breakdown stacked bar ═══════
    components = ["Screen", "SoC", "Radio", "GPS", "Base"]
    bottoms = np.zeros(len(SCENARIOS))

    for comp in components:
        vals = [POWER_DATA[s][comp] for s in SCENARIOS]
        ax_b.bar(
            x,
            vals,
            width=0.5,
            bottom=bottoms,
            color=COMP_COLORS[comp],
            edgecolor="white",
            linewidth=0.4,
            label=comp,
            alpha=0.88,
        )
        # Label segments > 0.2W
        for i, v in enumerate(vals):
            if v > 0.2:
                ax_b.text(
                    x[i],
                    bottoms[i] + v / 2,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="white",
                    fontweight="bold",
                )
        bottoms += vals

    # Total power labels on top
    totals = [sum(POWER_DATA[s].values()) for s in SCENARIOS]
    for i, t in enumerate(totals):
        ax_b.text(
            x[i],
            t + 0.05,
            f"{t:.2f} W",
            ha="center",
            va="bottom",
            fontsize=6,
            fontweight="bold",
        )

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(SCENARIOS, fontsize=7)
    ax_b.set_ylabel("Power consumption (W)")
    ax_b.set_ylim(0, max(totals) * 1.18)
    ax_b.legend(fontsize=6, loc="upper left", framealpha=0.9, ncol=2)
    label_panel(ax_b, "b")

    save_fig(fig, "fig_scenario_comparison", results_dir)
    print("Done: fig_scenario_comparison")


if __name__ == "__main__":
    main()
