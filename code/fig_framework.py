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

    fig, ax = plt.subplots(figsize=(3.54, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis("off")

    # ── Box height convention ──
    # 2-line boxes:  h = 1.1
    # 3-line boxes:  h = 1.4
    # 2-line + tall math: h = 1.2

    # ── Title ──
    ax.text(
        5,
        15.5,
        "Proposed Battery Model Framework",
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="top",
        color="#666666",
    )

    # ── Input block  (2 lines, h=1.1) ──
    draw_box(
        ax,
        5,
        14.0,
        7.0,
        1.1,
        "Inputs: Usage scenario, Temperature,\nBattery age (cycle count)",
        "#E8E6E4",
        "#9E9895",
        fontsize=6,
    )

    # ── Submodel 1: Shepherd Voltage  (3 lines, h=1.4) ──
    draw_box(
        ax,
        5,
        12.15,
        8.2,
        1.4,
        "Submodel 1: Shepherd Voltage\n"
        "$V = E_0 - R\\!\\cdot\\! I - K\\frac{I}{SOC}$\n"
        "$\\quad + A\\,\\exp(-B(1-SOC))$",
        "#D6E4F0",
        "#4E79A7",
        fontsize=6,
    )

    # ── Submodel 2: Temperature  (3 lines, h=1.4) ──
    draw_box(
        ax,
        2.7,
        9.65,
        4.3,
        1.4,
        "Submodel 2: Temperature\n$R(T)$: Arrhenius\n$Q_0(T)$: Logistic",
        "#FDEBD0",
        "#F28E2B",
        fontsize=6,
    )

    # ── Submodel 3: Component Decomposition  (3 lines, h=1.4) ──
    draw_box(
        ax,
        7.3,
        9.65,
        4.3,
        1.4,
        "Submodel 3:\nComponent Decomp.\n$P = \\sum_i P_i(\\xi_i)$",
        "#DFF0D8",
        "#59A14F",
        fontsize=6,
    )

    # ── Submodel 4: Aging  (2 lines, h=1.1) ──
    draw_box(
        ax,
        5,
        7.45,
        8.2,
        1.1,
        "Submodel 4: Aging Degradation\n" "$Q(n) = Q_0(1 - \\alpha \\cdot n^\\beta)$",
        "#F5D5D5",
        "#E15759",
        fontsize=6,
    )

    # ── Integration block  (2 lines + tall math, h=1.2) ──
    draw_box(
        ax,
        5,
        5.45,
        8.2,
        1.2,
        "ODE Integration\n"
        "$\\frac{dSOC}{dt} = -\\frac{I(t)}{Q_0(T,n)}$,  "
        "$I = \\frac{P_{\\mathrm{total}}}{V}$",
        "#E8D8E8",
        "#B07AA1",
        fontsize=6.5,
    )

    # ── Output block  (2 lines, h=1.1) ──
    draw_box(
        ax,
        5,
        3.55,
        7.0,
        1.1,
        "Outputs: TTE, SOC(t), V(t),\nEnergy consumption profile",
        "#DFF0D8",
        "#59A14F",
        fontsize=6,
    )

    # ── Validation box  (2 lines, h=1.1) ──
    draw_box(
        ax,
        5,
        1.65,
        8.2,
        1.1,
        "Validation: XJTU CC (8 cells × 2 batches)\n"
        "+ NASA random-walk dynamic (4 cells × 200 cycles)",
        "#FDEBD0",
        "#F28E2B",
        fontsize=5.5,
        bold=False,
    )

    # ── Arrows (start/end at box edges) ──
    draw_arrow(ax, 5, 13.45, 5, 12.85)  # Input → Shepherd
    draw_arrow(ax, 3.5, 11.45, 2.7, 10.35)  # Shepherd → Temp
    draw_arrow(ax, 6.5, 11.45, 7.3, 10.35)  # Shepherd → Component
    draw_arrow(ax, 2.7, 8.95, 3.5, 8.0)  # Temp → Aging
    draw_arrow(ax, 7.3, 8.95, 6.5, 8.0)  # Component → Aging
    draw_arrow(ax, 5, 6.9, 5, 6.05)  # Aging → Integration
    draw_arrow(ax, 5, 4.85, 5, 4.1)  # Integration → Output
    draw_arrow(ax, 5, 3.0, 5, 2.2)  # Output → Validation

    # ── Side annotations ──
    ax.text(
        0.2,
        12.15,
        "Physics\nLayer",
        fontsize=6,
        ha="center",
        color="#4E79A7",
        style="italic",
        rotation=90,
        va="center",
    )
    ax.text(
        0.2,
        8.55,
        "Correction\nLayer",
        fontsize=6,
        ha="center",
        color="#F28E2B",
        style="italic",
        rotation=90,
        va="center",
    )
    ax.text(
        0.2,
        5.45,
        "Numerical\nLayer",
        fontsize=6,
        ha="center",
        color="#B07AA1",
        style="italic",
        rotation=90,
        va="center",
    )

    save_fig(fig, "fig_framework", results_dir)
    print("Done: fig_framework")


if __name__ == "__main__":
    main()
