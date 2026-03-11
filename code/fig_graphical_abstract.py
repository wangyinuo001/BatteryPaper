"""
Graphical Abstract – High-quality visual summary for ECM submission
===================================================================
A professional 1-page figure summarising the paper:
  Left:   Input (Smartphone components + Cell data)
  Centre: Unified Model (4 coupled submodels)
  Right:  Output (TTE prediction + Energy management)
  Bottom: Key metrics bar

Output: graphical_abstract.pdf (placed in results/)
"""

import sys, os

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pub_style import apply_style, COLORS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def draw_rounded_box(
    ax,
    x,
    y,
    w,
    h,
    text,
    facecolor,
    edgecolor="#333333",
    fontsize=8,
    fontweight="bold",
    text_color="white",
    alpha=0.95,
    subtext=None,
    subsize=6.5,
):
    """Draw a rounded rectangle with centred text."""
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.2,
        alpha=alpha,
        zorder=3,
    )
    ax.add_patch(box)
    if subtext:
        ax.text(
            x + w / 2,
            y + h * 0.62,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight=fontweight,
            color=text_color,
            zorder=4,
        )
        ax.text(
            x + w / 2,
            y + h * 0.32,
            subtext,
            ha="center",
            va="center",
            fontsize=subsize,
            color=text_color,
            alpha=0.9,
            zorder=4,
            style="italic",
        )
    else:
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight=fontweight,
            color=text_color,
            zorder=4,
        )


def draw_arrow(ax, x1, y1, x2, y2, color="#555555"):
    """Draw a curved arrow between two points."""
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        color=color,
        linewidth=1.5,
        connectionstyle="arc3,rad=0.0",
        mutation_scale=12,
        zorder=2,
    )
    ax.add_patch(arrow)


def generate():
    apply_style()
    # Elsevier GA size: ~531 × 295 pixels at 96 dpi = ~5.5 × 3.1 inches
    fig, ax = plt.subplots(figsize=(7.48, 3.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ─── Colour definitions ───
    c_input = "#1565C0"  # blue
    c_model = "#C62828"  # red
    c_output = "#2E7D32"  # green
    c_metric = "#00838F"  # teal
    c_data = "#6A1B9A"  # purple

    # ─── Title ───
    ax.text(
        0.50,
        0.96,
        "A Unified Continuous-Time Electrochemical–Power Coupled Framework\n"
        "for Smartphone Battery Time-to-Empty Prediction",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#1a1a1a",
    )

    # ─── Row 1: Main flow (3 boxes + arrows) ───
    bw, bh = 0.22, 0.20
    y_top = 0.58

    # INPUT box
    draw_rounded_box(
        ax,
        0.04,
        y_top,
        bw,
        bh,
        "INPUT",
        c_input,
        subtext="Screen · SoC · Radio\nGPS · Baseline\n(5 components)",
        fontsize=9,
    )

    # MODEL box
    draw_rounded_box(
        ax,
        0.39,
        y_top,
        bw,
        bh,
        "UNIFIED MODEL",
        c_model,
        subtext="Shepherd voltage\n+ Temperature\n+ Aging + Power",
        fontsize=9,
    )

    # OUTPUT box
    draw_rounded_box(
        ax,
        0.74,
        y_top,
        bw,
        bh,
        "OUTPUT",
        c_output,
        subtext="TTE Prediction\n+ Energy Mgmt\n(+42% runtime)",
        fontsize=9,
    )

    # Arrows: Input → Model → Output
    draw_arrow(ax, 0.04 + bw + 0.01, y_top + bh / 2, 0.39 - 0.01, y_top + bh / 2)
    draw_arrow(ax, 0.39 + bw + 0.01, y_top + bh / 2, 0.74 - 0.01, y_top + bh / 2)

    # ─── Row 2: Submodel boxes (4 small boxes under MODEL) ───
    sub_w, sub_h = 0.14, 0.10
    sub_y = 0.42
    sub_colors = ["#EF5350", "#FF7043", "#FFA726", "#FFCA28"]  # warm gradient
    sub_labels = [
        "Shepherd\nVoltage",
        "Power\nDecomp.",
        "Temp.\nCoupling",
        "Aging\nDynamics",
    ]
    sub_xs = [0.11, 0.28, 0.58, 0.75]

    for i, (sx, label, sc) in enumerate(zip(sub_xs, sub_labels, sub_colors)):
        draw_rounded_box(
            ax,
            sx,
            sub_y,
            sub_w,
            sub_h,
            label,
            sc,
            fontsize=7,
            fontweight="normal",
            text_color="#1a1a1a",
            alpha=0.85,
        )

    # Arrows from submodels up to the model box
    for sx in sub_xs:
        mid_x = sx + sub_w / 2
        draw_arrow(
            ax,
            mid_x,
            sub_y + sub_h + 0.005,
            0.39 + bw / 2,
            y_top - 0.005,
            color="#999999",
        )

    # ─── Row 2 side: Data box ───
    draw_rounded_box(
        ax,
        0.04,
        sub_y - 0.01,
        0.06,
        sub_h + 0.02,
        "XJTU\nData",
        c_data,
        fontsize=6.5,
        fontweight="bold",
        text_color="white",
    )
    draw_arrow(
        ax,
        0.10 + 0.005,
        sub_y + sub_h / 2,
        sub_xs[0] - 0.005,
        sub_y + sub_h / 2,
        color="#999999",
    )

    # ─── Row 3: Key metrics bar ───
    bar_y = 0.08
    bar_h = 0.18
    # Background bar
    bar_bg = FancyBboxPatch(
        (0.04, bar_y),
        0.92,
        bar_h,
        boxstyle="round,pad=0.005,rounding_size=0.01",
        facecolor="#F5F5F5",
        edgecolor="#CCCCCC",
        linewidth=0.8,
        zorder=1,
    )
    ax.add_patch(bar_bg)

    # 4 metric blocks
    metrics = [
        ("RMSE", "17.85 mV", "75% better\nthan Nernst"),
        ("Speed", "0.009 ms/sim-s", "9000× faster\nthan P2D"),
        ("Cross-batch", "20.36 mV", "No re-fitting\nB1→B2"),
        ("Energy Mgmt", "+42% runtime", "Adaptive\nthrottling"),
    ]
    metric_colors = [c_model, c_metric, c_input, c_output]
    n = len(metrics)
    gap = 0.92 / n
    for i, (title, val, desc) in enumerate(metrics):
        mx = 0.04 + gap * i + gap / 2
        ax.text(
            mx,
            bar_y + bar_h * 0.78,
            title,
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            color=metric_colors[i],
        )
        ax.text(
            mx,
            bar_y + bar_h * 0.50,
            val,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#1a1a1a",
        )
        ax.text(
            mx,
            bar_y + bar_h * 0.20,
            desc,
            ha="center",
            va="center",
            fontsize=5.5,
            color="#666666",
            style="italic",
        )

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for fmt in ("pdf", "png"):
        path = os.path.join(RESULTS_DIR, f"graphical_abstract.{fmt}")
        fig.savefig(path, format=fmt, dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"  Saved: graphical_abstract.pdf/png")


main = generate

if __name__ == "__main__":
    generate()
