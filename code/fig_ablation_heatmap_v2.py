"""
Figure 2: Ablation study — TTE impact of removing each model component.
Two-panel layout:
  (a) Heatmap: |ΔTTE%| per ablated component × scenario (at 25°C, new battery)
  (b) Grouped bar: component importance across temperature × aging conditions

Data: ablation_study.csv
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

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

ABLATION_LABELS = {
    "w/o Temperature": "w/o Temperature\ncorrection",
    "w/o Aging": "w/o Aging\ndegradation",
    "w/o Component Decomp.": "w/o Component\ndecomposition",
    "w/o Polarization (K=0)": "w/o Polarization\n($K=0$)",
}

SCENARIO_ORDER = ["Standby", "Reading", "Navigation", "Video", "Gaming"]
COMPONENT_ORDER = [
    "w/o Temperature",
    "w/o Aging",
    "w/o Component Decomp.",
    "w/o Polarization (K=0)",
]


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "..", "results")

    df = pd.read_csv(os.path.join(results_dir, "ablation_study.csv"))

    # ── Build heatmap data: 25°C, New battery ──
    ref = df[(df["Temperature (C)"] == 25) & (df["Aging"] == "New (0 cycles)")]
    ref = ref[ref["Configuration"] != "Full Model"]

    heat = np.zeros((len(COMPONENT_ORDER), len(SCENARIO_ORDER)))
    for i, comp in enumerate(COMPONENT_ORDER):
        for j, scen in enumerate(SCENARIO_ORDER):
            row = ref[(ref["Configuration"] == comp) & (ref["Scenario"] == scen)]
            if len(row) > 0:
                heat[i, j] = abs(row["Delta TTE (%)"].values[0])

    # ── Build grouped bar data: average across scenarios ──
    temps = [0, 25, 40]
    agings = ["New (0 cycles)", "Moderate (300 cycles)", "Aged (800 cycles)"]
    aging_labels = ["New", "300 cyc.", "800 cyc."]

    bar_data = {}  # component → list of mean |ΔTTE%| for each temp×aging combo
    conditions = []
    for comp in COMPONENT_ORDER:
        vals = []
        for t in temps:
            for a in agings:
                sub = df[
                    (df["Temperature (C)"] == t)
                    & (df["Aging"] == a)
                    & (df["Configuration"] == comp)
                ]
                vals.append(sub["Delta TTE (%)"].abs().mean())
        bar_data[comp] = vals
    for t in temps:
        for a_lab in aging_labels:
            conditions.append(f"{t}°C\n{a_lab}")

    # ── Figure ──
    fig = plt.figure(figsize=DOUBLE_COL_TALL)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.3], wspace=0.35)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    # ═══════ Panel (a): Heatmap ═══════
    cmap = LinearSegmentedColormap.from_list(
        "impact", ["#F5F5F5", "#FFCDD2", "#E53935", "#B71C1C"], N=256
    )
    im = ax_a.imshow(heat, cmap=cmap, aspect="auto", vmin=0, vmax=75)

    # Annotate cells
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat[i, j]
            color = "white" if val > 40 else "black"
            ax_a.text(
                j,
                i,
                f"{val:.1f}%",
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color=color,
            )

    ax_a.set_xticks(range(len(SCENARIO_ORDER)))
    ax_a.set_xticklabels(SCENARIO_ORDER, fontsize=7, rotation=30, ha="right")
    ax_a.set_yticks(range(len(COMPONENT_ORDER)))
    ax_a.set_yticklabels([ABLATION_LABELS[c] for c in COMPONENT_ORDER], fontsize=6.5)
    ax_a.set_title(
        "$|\\Delta$TTE$|$ (%) — Reference: 25°C, new battery", fontsize=8, pad=8
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax_a, fraction=0.046, pad=0.06, shrink=0.9)
    cbar.set_label("$|\\Delta$TTE$|$ (%)", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)

    label_panel(ax_a, "a", x=-0.22)

    # ═══════ Panel (b): Grouped bar ═══════
    n_cond = len(conditions)
    n_comp = len(COMPONENT_ORDER)
    x_base = np.arange(n_cond)
    bar_w = 0.18

    comp_colors = ["#E53935", "#1E88E5", "#43A047", "#FFB300"]
    for k, comp in enumerate(COMPONENT_ORDER):
        offset = (k - n_comp / 2 + 0.5) * bar_w
        ax_b.bar(
            x_base + offset,
            bar_data[comp],
            width=bar_w,
            color=comp_colors[k],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.4,
            label=ABLATION_LABELS[comp].replace("\n", " "),
        )

    ax_b.set_xticks(x_base)
    ax_b.set_xticklabels(conditions, fontsize=6, rotation=0)
    ax_b.set_ylabel("Mean $|\\Delta$TTE$|$ (%)")
    ax_b.set_title("Component importance across conditions", fontsize=8, pad=8)
    ax_b.legend(fontsize=5.5, loc="upper left", ncol=1, framealpha=0.9)

    # Add vertical separators between temperature groups
    for pos in [2.5, 5.5]:
        ax_b.axvline(pos, color="0.7", ls="--", lw=0.5, zorder=0)

    # Temperature labels at top
    for i, t_label in enumerate(["0°C", "25°C", "40°C"]):
        ax_b.text(
            i * 3 + 1,
            ax_b.get_ylim()[1] * 0.95,
            t_label,
            ha="center",
            fontsize=7,
            fontweight="bold",
            color="0.4",
        )

    label_panel(ax_b, "b", x=-0.10)

    save_fig(fig, "fig_ablation_heatmap", results_dir)
    print("Done: fig_ablation_heatmap")


if __name__ == "__main__":
    main()
