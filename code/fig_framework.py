"""
Figure 1: Model framework diagram.
Professional schematic showing the 4-submodel architecture:
  Shepherd voltage → Temperature correction → Component decomposition
  → Aging degradation → TTE integration

Single-column figure using matplotlib patches + text (no external images).
"""

import os, sys
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.patheffects as pe

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pub_style import apply_style, save_fig, SINGLE_COL_TALL

apply_style()


def draw_box(ax, cx, cy, w, h, text, fc, ec, fontsize=6, bold=True):
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        boxstyle="round,pad=0.08",
        fc=fc,
        ec=ec,
        lw=1.0,
        zorder=3,
    )
    ax.add_patch(box)
    fw = "bold" if bold else "normal"
    ax.text(
        cx,
        cy,
        text,
        fontsize=fontsize,
        ha="center",
        va="center",
        zorder=4,
        color=ec,
        fontweight=fw,
        linespacing=1.3,
    )


def draw_arrow(ax, x1, y1, x2, y2, color="0.4"):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0, mutation_scale=10),
    )


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "..", "results")

    fig, ax = plt.subplots(figsize=(3.54, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    # ── Title ──
    ax.text(
        5,
        13.5,
        "Proposed Battery Model Framework",
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="top",
        color="#37474F",
    )

    # ── Input block ──
    draw_box(
        ax,
        5,
        12.2,
        6,
        1.0,
        "Inputs: Usage scenario, Temperature,\nBattery age (cycle count)",
        "#ECEFF1",
        "#455A64",
        fontsize=6,
    )

    # ── Submodel 1: Shepherd Voltage ──
    draw_box(
        ax,
        5,
        10.3,
        7.5,
        1.2,
        "Submodel 1: Shepherd Voltage\n"
        "$V = E_0 - R(T)\\cdot I - K\\frac{I}{SOC} + A e^{-B(1-SOC)}$",
        "#E3F2FD",
        "#1565C0",
        fontsize=5.5,
    )

    # ── Submodel 2: Temperature ──
    draw_box(
        ax,
        2.7,
        8.0,
        4.0,
        1.2,
        "Submodel 2:\nTemperature\n$R(T)$: Arrhenius\n$Q_0(T)$: Logistic",
        "#FFF3E0",
        "#E65100",
        fontsize=5,
    )

    # ── Submodel 3: Component Decomposition ──
    draw_box(
        ax,
        7.3,
        8.0,
        4.0,
        1.2,
        "Submodel 3:\nComponent Decomp.\n$P = \\sum_i P_i(\\xi_i)$",
        "#E8F5E9",
        "#2E7D32",
        fontsize=5,
    )

    # ── Submodel 4: Aging ──
    draw_box(
        ax,
        5,
        5.8,
        7.5,
        1.2,
        "Submodel 4: Aging Degradation\n"
        "$Q(n) = Q_0 \\cdot (1 - \\alpha \\cdot n^\\beta)$",
        "#FCE4EC",
        "#C62828",
        fontsize=5.5,
    )

    # ── Integration block ──
    draw_box(
        ax,
        5,
        3.8,
        7.5,
        1.0,
        "ODE Integration: $\\frac{dSOC}{dt} = -\\frac{I(t)}{Q_0(T,n)}$\n"
        "Coupling: $I = P_{\\mathrm{total}} / V(SOC, I, T)$",
        "#F3E5F5",
        "#6A1B9A",
        fontsize=5.5,
    )

    # ── Output block ──
    draw_box(
        ax,
        5,
        2.2,
        6,
        1.0,
        "Outputs: TTE, SOC(t), V(t),\nEnergy consumption profile",
        "#C8E6C9",
        "#2E7D32",
        fontsize=6,
    )

    # ── Validation box ──
    draw_box(
        ax,
        5,
        0.7,
        6,
        0.8,
        "Validation: XJTU battery dataset (8 cells × 2 batches)",
        "#FFF9C4",
        "#F57F17",
        fontsize=5.5,
        bold=False,
    )

    # ── Arrows ──
    draw_arrow(ax, 5, 11.7, 5, 10.9)
    draw_arrow(ax, 3.5, 9.7, 2.7, 8.6)
    draw_arrow(ax, 6.5, 9.7, 7.3, 8.6)
    draw_arrow(ax, 2.7, 7.4, 3.5, 6.4)
    draw_arrow(ax, 7.3, 7.4, 6.5, 6.4)
    draw_arrow(ax, 5, 5.2, 5, 4.3)
    draw_arrow(ax, 5, 3.3, 5, 2.7)
    draw_arrow(ax, 5, 1.7, 5, 1.1)

    # ── Side annotations ──
    ax.text(
        0.3,
        10.3,
        "Physics\nLayer",
        fontsize=5,
        ha="center",
        color="#1565C0",
        style="italic",
        rotation=90,
        va="center",
    )
    ax.text(
        0.3,
        6.9,
        "Correction\nLayer",
        fontsize=5,
        ha="center",
        color="#E65100",
        style="italic",
        rotation=90,
        va="center",
    )
    ax.text(
        0.3,
        3.8,
        "Numerical\nLayer",
        fontsize=5,
        ha="center",
        color="#6A1B9A",
        style="italic",
        rotation=90,
        va="center",
    )

    save_fig(fig, "fig_framework", results_dir)
    print("Done: fig_framework")


if __name__ == "__main__":
    main()
