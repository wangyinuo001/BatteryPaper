"""
Figure 2: Cross-batch validation — per-cell RMSE distributions for
Batch-1 (same-batch) and Batch-2 (cross-batch, no re-fitting).
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
    df = pd.read_csv(os.path.join(results_dir, "cross_batch_validation.csv"))

    same = df[df["Batch"].str.contains("Same", case=False, na=False)][
        "RMSE (mV)"
    ].values
    cross = df[
        df["Batch"].str.contains("Cross", case=False, na=False)
        | df["Batch"].str.contains("Batch-2", case=False, na=False)
    ]["RMSE (mV)"].values

    if len(same) == 0:
        same = df[df["Batch"].str.contains("Batch-1", case=False, na=False)][
            "RMSE (mV)"
        ].values
    if len(cross) == 0:
        cross = df[~df["Batch"].str.contains("Batch-1", case=False, na=False)][
            "RMSE (mV)"
        ].values

    fig, ax = plt.subplots(figsize=SINGLE_COL_TALL)

    parts = ax.violinplot(
        [same, cross], positions=[1, 2], showmeans=True, showmedians=True, widths=0.55
    )

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor([COLORS["shepherd"], COLORS["accent"]][i])
        pc.set_alpha(0.4)
    parts["cmeans"].set_color("k")
    parts["cmedians"].set_color(COLORS["gray"])
    parts["cmins"].set_color(COLORS["gray"])
    parts["cmaxes"].set_color(COLORS["gray"])
    parts["cbars"].set_color(COLORS["gray"])

    # Overlay strip-points
    np.random.seed(42)
    for i, (vals, pos) in enumerate([(same, 1), (cross, 2)]):
        jitter = np.random.uniform(-0.08, 0.08, len(vals))
        ax.scatter(
            pos + jitter,
            vals,
            s=14,
            alpha=0.6,
            color=[COLORS["shepherd"], COLORS["accent"]][i],
            edgecolors="white",
            linewidths=0.3,
            zorder=5,
        )

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Same-batch\n(Batch-1, 2C)", "Cross-batch\n(Batch-2, 3C)"])
    ax.set_ylabel("RMSE (mV)")

    # Annotate means
    for vals, pos, clr in [(same, 1, COLORS["shepherd"]), (cross, 2, COLORS["accent"])]:
        m, s = np.mean(vals), np.std(vals)
        ax.text(
            pos,
            m + 2.5,
            f"{m:.1f}±{s:.1f} mV",
            ha="center",
            fontsize=6.5,
            fontweight="bold",
            color=clr,
        )

    save_fig(fig, "fig_cross_batch", results_dir)


if __name__ == "__main__":
    main()
