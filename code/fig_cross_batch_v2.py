"""
Figure 4: Cross-batch generalisation — Batch-1 (training) vs Batch-2 (unseen).
Two-panel layout:
  (a) Box plot: RMSE distribution for same-batch vs cross-batch
  (b) Per-battery RMSE scatter with individual points + summary stats

Data: cross_batch_validation.csv
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
    SINGLE_COL_TALL,
    COLORS,
)

apply_style()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "..", "results")

    df = pd.read_csv(os.path.join(results_dir, "cross_batch_validation.csv"))

    b1 = df[df["Batch"] == "Batch-1 (Same)"]["RMSE (mV)"].values
    b2 = df[df["Batch"] == "Batch-2 (Cross)"]["RMSE (mV)"].values

    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=SINGLE_COL_TALL, gridspec_kw={"hspace": 0.45}
    )

    # ═══════ Panel (a): Box + strip plot ═══════
    bp = ax_a.boxplot(
        [b1, b2],
        positions=[1, 2],
        widths=0.45,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="white", lw=1.5),
        boxprops=dict(lw=0.8),
        whiskerprops=dict(lw=0.8),
        capprops=dict(lw=0.8),
    )

    box_colors = ["#1E88E5", "#E53935"]
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    # Overlay individual data points (jittered)
    np.random.seed(42)
    jitter1 = 1 + np.random.normal(0, 0.04, len(b1))
    jitter2 = 2 + np.random.normal(0, 0.04, len(b2))
    ax_a.scatter(
        jitter1,
        b1,
        c="#1E88E5",
        s=18,
        alpha=0.7,
        edgecolors="white",
        linewidths=0.4,
        zorder=5,
        marker="o",
    )
    ax_a.scatter(
        jitter2,
        b2,
        c="#E53935",
        s=18,
        alpha=0.7,
        edgecolors="white",
        linewidths=0.4,
        zorder=5,
        marker="D",
    )

    ax_a.set_xticks([1, 2])
    ax_a.set_xticklabels(
        ["Batch-1 (Same)\n$n=8$, 2C rate", "Batch-2 (Cross)\n$n=15$, 3C rate"],
        fontsize=7,
    )
    ax_a.set_ylabel("RMSE (mV)")

    # Mann-Whitney p-value annotation
    from scipy.stats import mannwhitneyu

    stat, p = mannwhitneyu(b1, b2, alternative="two-sided")
    sig = "n.s." if p > 0.05 else f"$p={p:.3f}$"

    y_max = max(b1.max(), b2.max())
    ax_a.plot(
        [1, 1, 2, 2], [y_max + 1, y_max + 2, y_max + 2, y_max + 1], lw=0.8, color="0.3"
    )
    ax_a.text(1.5, y_max + 2.3, sig, ha="center", va="bottom", fontsize=7, color="0.3")

    embed_metric(
        ax_a,
        f"Same:  {b1.mean():.1f} ± {b1.std():.1f} mV\n"
        f"Cross: {b2.mean():.1f} ± {b2.std():.1f} mV",
        x=0.97,
        y=0.70,
    )
    label_panel(ax_a, "a")

    # ═══════ Panel (b): Per-battery scatter ═══════
    all_labels = list(df["Battery"])
    all_rmse = df["RMSE (mV)"].values
    all_batch = df["Batch"].values

    x_pos = np.arange(len(all_labels))
    colors = ["#1E88E5" if "Batch-1" in b else "#E53935" for b in all_batch]
    markers = ["o" if "Batch-1" in b else "D" for b in all_batch]

    for i, (x, y, c, m) in enumerate(zip(x_pos, all_rmse, colors, markers)):
        ax_b.scatter(
            x, y, c=c, s=25, marker=m, edgecolors="white", linewidths=0.4, zorder=5
        )

    # Reference line: Batch-1 mean
    ax_b.axhline(
        b1.mean(),
        color="#1E88E5",
        ls="--",
        lw=0.8,
        alpha=0.6,
        label=f"B1 mean ({b1.mean():.1f} mV)",
    )
    ax_b.axhline(
        b2.mean(),
        color="#E53935",
        ls="--",
        lw=0.8,
        alpha=0.6,
        label=f"B2 mean ({b2.mean():.1f} mV)",
    )

    # Shade the region between means
    ax_b.axhspan(
        min(b1.mean(), b2.mean()),
        max(b1.mean(), b2.mean()),
        color="0.9",
        alpha=0.5,
        zorder=0,
    )

    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(all_labels, fontsize=5, rotation=45, ha="right")
    ax_b.set_ylabel("RMSE (mV)")
    ax_b.set_xlabel("Battery cell")
    ax_b.legend(fontsize=6, loc="upper right", framealpha=0.9)
    label_panel(ax_b, "b")

    save_fig(fig, "fig_cross_batch", results_dir)
    print("Done: fig_cross_batch")


if __name__ == "__main__":
    main()
