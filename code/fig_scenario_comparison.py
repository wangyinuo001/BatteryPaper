"""
Figure 6: Scenario TTE comparison — Grouped bar chart with power
annotation and component-power stacked inset.
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pub_style import (
    apply_style,
    save_fig,
    COLORS,
    DOUBLE_COL,
    COMP_COLORS,
    PHASE_COLORS,
    label_panel,
)

apply_style()

# Scenario data (from model computation)
SCENARIOS = {
    "Standby": {
        "power": 0.741,
        "tte": 22.64,
        "breakdown": {
            "Screen": 0,
            "SoC": 204.0,
            "Radio": 117.3,
            "GPS": 300,
            "Base": 120,
        },
    },
    "Reading": {
        "power": 1.727,
        "tte": 9.64,
        "breakdown": {
            "Screen": 460.7,
            "SoC": 612.0,
            "Radio": 234.6,
            "GPS": 300,
            "Base": 120,
        },
    },
    "Navigation": {
        "power": 1.845,
        "tte": 9.02,
        "breakdown": {
            "Screen": 460.7,
            "SoC": 612.0,
            "Radio": 351.9,
            "GPS": 300,
            "Base": 120,
        },
    },
    "Video": {
        "power": 2.598,
        "tte": 6.37,
        "breakdown": {
            "Screen": 510.6,
            "SoC": 612.0,
            "Radio": 1055.7,
            "GPS": 300,
            "Base": 120,
        },
    },
    "Gaming": {
        "power": 4.303,
        "tte": 3.79,
        "breakdown": {
            "Screen": 728.6,
            "SoC": 2040.0,
            "Radio": 1114.3,
            "GPS": 300,
            "Base": 120,
        },
    },
}

# Use COMP_COLORS from pub_style for consistency


def main():
    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results"
    )

    names = list(SCENARIOS.keys())
    ttes = [SCENARIOS[n]["tte"] for n in names]
    powers = [SCENARIOS[n]["power"] for n in names]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=DOUBLE_COL, gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.35}
    )

    # ── (a) TTE bar chart ──
    scenario_colors = [PHASE_COLORS.get(n, COLORS["accent"]) for n in names]
    bars = ax1.bar(
        range(len(names)),
        ttes,
        color=scenario_colors,
        edgecolor="white",
        alpha=0.85,
        width=0.6,
    )

    for i, (bar, tte, pwr) in enumerate(zip(bars, ttes, powers)):
        ax1.text(
            i,
            tte + 0.5,
            f"{tte:.1f} h",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
        )
        ax1.text(
            i,
            tte / 2,
            f"{pwr:.2f} W",
            ha="center",
            va="center",
            fontsize=6,
            color="white",
            fontweight="bold",
        )

    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, fontsize=7, rotation=20, ha="right")
    ax1.set_ylabel("Time-to-empty (h)")
    label_panel(ax1, "a")
    ax1.set_ylim(0, max(ttes) * 1.18)

    # ── (b) Stacked power breakdown ──
    comps = ["Screen", "SoC", "Radio", "GPS", "Base"]
    bottom = np.zeros(len(names))
    for comp in comps:
        vals = [SCENARIOS[n]["breakdown"][comp] for n in names]
        ax2.bar(
            range(len(names)),
            vals,
            bottom=bottom,
            width=0.55,
            color=COMP_COLORS[comp],
            edgecolor="white",
            linewidth=0.3,
            label=comp,
            alpha=0.85,
        )
        bottom += np.array(vals)

    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, fontsize=7, rotation=20, ha="right")
    ax2.set_ylabel("Power consumption (mW)")
    label_panel(ax2, "b")
    ax2.legend(fontsize=6.5, loc="upper left", ncol=2)

    save_fig(fig, "fig_scenario_comparison", results_dir)


if __name__ == "__main__":
    main()
