"""
Figure 3: Ablation heatmap — 3×3 grid (Temperature × Aging) for each
removed submodel.  Four panels, one per ablation configuration.
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pub_style import apply_style, save_fig, DOUBLE_COL_TALL, label_panel

apply_style()


def main():
    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results"
    )
    df = pd.read_csv(os.path.join(results_dir, "ablation_study.csv"))

    # Filter to Reading scenario
    df = df[df["Scenario"] == "Reading"].copy()

    configs = [
        "w/o Temperature",
        "w/o Aging",
        "w/o Component Decomp.",
        "w/o Polarization (K=0)",
    ]
    temps = [0, 25, 40]
    agings = ["New (0 cycles)", "Moderate (300 cycles)", "Aged (800 cycles)"]
    aging_labels = ["0 cyc", "300 cyc", "800 cyc"]

    fig, axes = plt.subplots(2, 2, figsize=DOUBLE_COL_TALL)
    axes = axes.flatten()

    titles = [
        "w/o Temperature",
        "w/o Aging",
        "w/o Component Decomp.",
        "w/o Polarization ($K$=0)",
    ]
    panel_letters = ["a", "b", "c", "d"]

    for idx, (config, title) in enumerate(zip(configs, titles)):
        ax = axes[idx]
        label_panel(ax, panel_letters[idx])
        sub = df[df["Configuration"] == config]
        matrix = np.zeros((3, 3))
        for i, t in enumerate(temps):
            for j, a in enumerate(agings):
                row = sub[(sub["Temperature (C)"] == t) & (sub["Aging"] == a)]
                if len(row) > 0:
                    matrix[i, j] = row["Delta TTE (%)"].values[0]

        vmax = max(abs(matrix.min()), abs(matrix.max()), 1)
        norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
        im = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")

        for i in range(3):
            for j in range(3):
                val = matrix[i, j]
                color = "white" if abs(val) > vmax * 0.55 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:+.1f}%",
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                    color=color,
                )

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(aging_labels, fontsize=6.5)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["0°C", "25°C", "40°C"], fontsize=6.5)
        ax.set_title(title, fontsize=8, fontweight="bold", pad=4)
        if idx >= 2:
            ax.set_xlabel("Aging level")
        if idx % 2 == 0:
            ax.set_ylabel("Temperature")

        plt.colorbar(im, ax=ax, shrink=0.7, label="ΔTTE (%)")

    save_fig(fig, "fig_ablation_heatmap", results_dir)


if __name__ == "__main__":
    main()
