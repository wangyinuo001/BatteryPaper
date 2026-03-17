"""
Figure 5: Parameter sensitivity — Tornado chart.
Shows ±5 % perturbation effect on TTE (Video Streaming).
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pub_style import apply_style, save_fig, COLORS, SINGLE_COL_TALL

apply_style()


def main():
    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results"
    )
    df = pd.read_csv(os.path.join(results_dir, "parameter_sensitivity_physical.csv"))

    # Rename for display
    display = {
        "P_total": "$P_{\\mathrm{total}}$",
        "Q0": "$Q_0$",
        "E0": "$E_0$",
        "K": "$K$",
        "R0": "$R_0$",
        "A": "$A$",
        "B": "$B$",
    }
    df["label"] = df["parameter"].map(display)

    # Sort by max absolute sensitivity
    df["max_sens"] = df[["delta_pct_neg", "delta_pct_pos"]].abs().max(axis=1)
    df = df.sort_values("max_sens", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(3.54, 2.8))

    y = np.arange(len(df))
    neg = df["delta_pct_neg"].values
    pos = df["delta_pct_pos"].values

    ax.barh(
        y,
        neg,
        height=0.55,
        color=COLORS["accent"],
        alpha=0.75,
        label="−5 % perturbation",
    )
    ax.barh(
        y,
        pos,
        height=0.55,
        color=COLORS["shepherd"],
        alpha=0.75,
        label="+5 % perturbation",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(df["label"].values)
    ax.set_xlabel("ΔTTE (%)")
    ax.set_ylim(-0.5, len(df) - 0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.legend(fontsize=6.5, loc="lower right")
    fig.subplots_adjust(left=0.18, bottom=0.15)

    # Annotate values — labels placed outside bar ends, aligned by value sign
    all_vis = np.concatenate([neg, pos])
    all_vis = all_vis[np.abs(all_vis) > 0.5]
    left_x = all_vis[all_vis < 0].min() - 0.15 if np.any(all_vis < 0) else 0
    right_x = all_vis[all_vis > 0].max() + 0.15 if np.any(all_vis > 0) else 0

    for i, (n, p) in enumerate(zip(neg, pos)):
        for v in [n, p]:
            if abs(v) > 0.5:
                if v > 0:
                    ax.text(
                        right_x, i, f"{v:+.1f}%", va="center", ha="left", fontsize=6
                    )
                else:
                    ax.text(
                        left_x, i, f"{v:+.1f}%", va="center", ha="right", fontsize=6
                    )

    ax.set_xlim(left_x - 2.0, right_x + 2.0)

    save_fig(fig, "fig_sensitivity_tornado", results_dir)


if __name__ == "__main__":
    main()
