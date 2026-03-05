import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


def main() -> None:
    data_path = os.path.join("results", "parameter_sensitivity_physical.csv")
    df = pd.read_csv(data_path)

    preferred_order = ["P_total", "E0", "K", "R0", "A", "B", "Q0"]
    order = [p for p in preferred_order if p in df["parameter"].values]
    df = df.set_index("parameter").loc[order].reset_index()

    params = df["parameter"].tolist()
    neg = df["delta_pct_neg"].values
    pos = df["delta_pct_pos"].values

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), sharey=True)

    neg_order = np.argsort(-neg)
    pos_order = np.argsort(-pos)

    neg_sorted = neg[neg_order]
    pos_sorted = pos[pos_order]
    params_neg = [params[i] for i in neg_order]
    params_pos = [params[i] for i in pos_order]

    palette = [
        "#9EB8D8",
        "#BBD3E6",
        "#D6E6F2",
        "#EEF4E9",
        "#FFF3C4",
        "#F0D59A",
        "#D88964",
    ]
    neg_colors = palette[: len(neg_sorted)]
    pos_colors = palette[: len(pos_sorted)]

    x_neg = np.arange(len(params_neg))
    x_pos = np.arange(len(params_pos))

    axes[0].set_facecolor("#fbfbfb")
    axes[1].set_facecolor("#fbfbfb")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#333333")
        ax.spines["bottom"].set_color("#333333")
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)

    bars_neg = axes[0].bar(x_neg, neg_sorted, color=neg_colors, width=0.7, edgecolor="#222222", linewidth=1.5)
    axes[0].axhline(0, color="#1e1e1e", linewidth=1.3, alpha=0.85)
    axes[0].set_title("-5% change", fontsize=12, pad=8)
    axes[0].set_ylabel("TTE change (%)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)

    bars_pos = axes[1].bar(x_pos, pos_sorted, color=pos_colors, width=0.7, edgecolor="#222222", linewidth=1.5)
    axes[1].axhline(0, color="#1e1e1e", linewidth=1.3, alpha=0.85)
    axes[1].set_title("+5% change", fontsize=12, pad=8)
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)

    for bars in (bars_neg, bars_pos):
        for bar in bars:
            bar.set_path_effects([
                pe.SimplePatchShadow(offset=(1.1, -1.1), shadow_rgbFace=(0, 0, 0, 0.18)),
                pe.Normal(),
            ])

    axes[0].set_xticks(x_neg)
    axes[1].set_xticks(x_pos)
    axes[0].set_xticklabels(params_neg, fontsize=10, fontweight="bold", rotation=20, ha="right")
    axes[1].set_xticklabels(params_pos, fontsize=10, fontweight="bold", rotation=20, ha="right")

    for i, n in enumerate(neg_sorted):
        axes[0].text(
            i,
            n + (0.1 if n >= 0 else -0.1),
            f"{n:.2f}%",
            ha="center",
            va="bottom" if n >= 0 else "top",
            fontsize=9,
            fontweight="bold",
            color="#1e1e1e",
        )

    for i, p in enumerate(pos_sorted):
        axes[1].text(
            i,
            p + (0.1 if p >= 0 else -0.1),
            f"{p:.2f}%",
            ha="center",
            va="bottom" if p >= 0 else "top",
            fontsize=9,
            fontweight="bold",
            color="#1e1e1e",
        )

    fig.suptitle("Parameter Sensitivity (TTE change for ±5%)", fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()

    out_path = os.path.join("results", "fig_parameter_sensitivity.png")
    plt.savefig(out_path, dpi=320, bbox_inches="tight", facecolor="white")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
