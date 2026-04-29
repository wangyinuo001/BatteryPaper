"""
Figure 1: Model framework diagram.
Layered-band layout, three computational tiers (Voltage core / Coupled
corrections / Numerical integration) plus an I/O strip on top and a
validation strip on the bottom.

Single-column figure built from matplotlib patches + text.
"""

import os, sys
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pub_style import apply_style, save_fig

apply_style()


# ── Palette tuned to the rest of the paper ──────────────────────────
BAND = {
    "io": {"bg": "#F2F2F2", "stripe": "#9E9895", "fg": "#333333"},
    "core": {"bg": "#EEF3F8", "stripe": "#4E79A7", "fg": "#1F3B5C"},
    "corr": {"bg": "#FBF1E4", "stripe": "#F28E2B", "fg": "#7A4413"},
    "num": {"bg": "#F0E6F0", "stripe": "#B07AA1", "fg": "#5C2D55"},
    "val": {"bg": "#E9F1E4", "stripe": "#59A14F", "fg": "#2A4F26"},
    "aging": {"bg": "#FAE5E5", "stripe": "#E15759", "fg": "#7A2A2B"},
}


def draw_band(ax, y_lo, y_hi, key, label):
    bg = BAND[key]
    ax.add_patch(
        Rectangle((0.30, y_lo), 9.40, y_hi - y_lo, fc=bg["bg"], ec="none", zorder=0)
    )
    # left coloured stripe
    ax.add_patch(
        Rectangle((0.30, y_lo), 0.18, y_hi - y_lo, fc=bg["stripe"], ec="none", zorder=1)
    )
    ax.text(
        0.10,
        (y_lo + y_hi) / 2,
        label,
        fontsize=6.2,
        ha="center",
        va="center",
        rotation=90,
        color=bg["stripe"],
        fontweight="bold",
    )


def draw_card(
    ax, cx, cy, w, h, title, body, key, fontsize_title=6.6, fontsize_body=6.0
):
    pal = BAND[key]
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        fc="white",
        ec=pal["stripe"],
        lw=0.9,
        zorder=3,
    )
    ax.add_patch(box)
    # accent strip on top of card
    ax.add_patch(
        Rectangle(
            (cx - w / 2 + 0.06, cy + h / 2 - 0.30),
            w - 0.12,
            0.04,
            fc=pal["stripe"],
            ec="none",
            alpha=0.95,
            zorder=4,
        )
    )
    ax.text(
        cx,
        cy + h / 2 - 0.18,
        title,
        fontsize=fontsize_title,
        ha="center",
        va="center",
        color=pal["fg"],
        fontweight="bold",
        zorder=5,
    )
    if body:
        ax.text(
            cx,
            cy - 0.05,
            body,
            fontsize=fontsize_body,
            ha="center",
            va="center",
            color="#222222",
            zorder=5,
            linespacing=1.35,
        )


def arrow(ax, x1, y1, x2, y2, color="#666666", lw=0.9, head=8):
    a = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle=f"-|>,head_length={head/12:.2f}," f"head_width={head/16:.2f}",
        color=color,
        lw=lw,
        mutation_scale=head,
        zorder=2,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(a)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "..", "results")

    fig, ax = plt.subplots(figsize=(3.54, 5.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16.0)
    ax.axis("off")

    # ── Background bands (bottom-up) ──────────────────────────────
    draw_band(ax, 0.30, 2.40, "val", "Validation")
    draw_band(ax, 2.55, 6.55, "num", "Numerical")
    draw_band(ax, 6.70, 11.10, "corr", "Coupled corrections")
    draw_band(ax, 11.25, 13.85, "core", "Voltage core")
    draw_band(ax, 14.00, 15.70, "io", "Inputs")

    # ── Cards (top to bottom) ─────────────────────────────────────
    # Inputs
    draw_card(
        ax,
        5.00,
        14.85,
        8.6,
        1.30,
        "Inputs",
        r"Usage scenario  $\mid$  Temperature $T$  $\mid$  Cycle count $N$",
        "io",
        fontsize_title=6.8,
        fontsize_body=6.0,
    )

    # Voltage core (Submodel 1)
    draw_card(
        ax,
        5.00,
        12.55,
        8.6,
        1.90,
        r"Submodel 1  $\cdot$  Modified Shepherd voltage",
        r"$V = E_0 - R_0 I - K\,I/\mathrm{SOC}" r" + A\exp(-B(1-\mathrm{SOC}))$",
        "core",
        fontsize_title=6.8,
        fontsize_body=6.4,
    )

    # Corrections — two side-by-side cards on one row + Aging below
    draw_card(
        ax,
        2.85,
        9.85,
        3.7,
        1.55,
        r"Submodel 2 $\cdot$ Temperature",
        r"$Q_0(T)$: logistic" "\n" r"$R_0(T)$: Arrhenius",
        "corr",
        fontsize_body=5.8,
    )
    draw_card(
        ax,
        7.05,
        9.85,
        3.7,
        1.55,
        r"Submodel 3 $\cdot$ Components",
        r"$P_{\mathrm{tot}} = \sum_i P_i(\xi_i)$",
        "corr",
        fontsize_body=5.9,
    )
    draw_card(
        ax,
        5.00,
        7.65,
        8.6,
        1.20,
        r"Submodel 4 $\cdot$ Aging",
        r"$Q_0(N)=Q_0(0)(1-\alpha N^{\beta})$,  " r"$R_0(N)=R_0(0)(1+\gamma N)$",
        "aging",
        fontsize_body=5.9,
    )

    # Numerical integration
    draw_card(
        ax,
        5.00,
        5.55,
        8.6,
        1.80,
        r"Coupled DAE  $\cdot$  ODE integration",
        r"$\dfrac{d\,\mathrm{SOC}}{dt}=-\dfrac{I(t)}{Q_0(T,N)}$,"
        r"  $I = P_{\mathrm{tot}}/V$",
        "num",
        fontsize_body=6.5,
    )

    # Outputs (still inside numerical band, lower half)
    draw_card(
        ax,
        5.00,
        3.30,
        8.6,
        1.20,
        r"Outputs",
        r"TTE  $\mid$  SOC$(t)$  $\mid$  $V(t)$  $\mid$  energy profile",
        "num",
        fontsize_body=6.0,
    )

    # Validation strip (XJTU + NASA)
    draw_card(
        ax,
        5.00,
        1.45,
        8.6,
        1.20,
        r"Validation",
        r"XJTU CC (8 cells $\times$ 2 batches)  $\mid$  "
        r"NASA RW (4 cells $\times$ 200 cycles)",
        "val",
        fontsize_body=5.9,
    )

    # ── Arrows (vertical flow + corrections converging into Aging) ─
    arrow(ax, 5.00, 14.20, 5.00, 13.55)  # Inputs -> Voltage core
    arrow(ax, 5.00, 11.55, 5.00, 10.65)  # Voltage core -> corrections row
    arrow(ax, 2.85, 9.05, 4.10, 8.30)
    arrow(ax, 7.05, 9.05, 5.90, 8.30)
    arrow(ax, 5.00, 7.00, 5.00, 6.50)  # aging -> DAE
    arrow(ax, 5.00, 4.60, 5.00, 3.95)  # DAE -> Outputs
    arrow(ax, 5.00, 2.65, 5.00, 2.10)  # Outputs -> Validation

    save_fig(fig, "fig_framework", results_dir)
    print("Done: fig_framework")


if __name__ == "__main__":
    main()
