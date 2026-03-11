"""
Graphical Abstract — ECM-quality visual summary.
Single wide figure with embedded mini-results:
  Left: Model framework schematic (text + arrows)
  Center: Key validation result (voltage fit)
  Right: Key finding (TTE comparison)

Data: voltage_error_timeseries.csv, tte_scenarios.csv
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pub_style import apply_style, save_fig, COLORS, COMP_COLORS, PHASE_COLORS

apply_style()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "..", "results")

    ts = pd.read_csv(os.path.join(results_dir, "voltage_error_timeseries.csv"))

    fig = plt.figure(figsize=(7.48, 3.8))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1, 1], wspace=0.30)

    ax_left = fig.add_subplot(gs[0])
    ax_mid = fig.add_subplot(gs[1])
    ax_right = fig.add_subplot(gs[2])

    # ═══════ Left panel: Model framework ═══════
    ax_left.set_xlim(0, 10)
    ax_left.set_ylim(0, 10)
    ax_left.axis("off")

    # Title
    ax_left.text(
        5,
        9.5,
        "Continuous-Time Battery Model",
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="top",
    )

    # Boxes
    boxes = [
        (
            5,
            7.8,
            "Shepherd Voltage\n$V = E_0 - R(T)I - K\\frac{I}{SOC} + Ae^{-B(1-SOC)}$",
            "#E3F2FD",
            "#1565C0",
        ),
        (2.5, 5.8, "Temperature\nCorrection\n$R(T), Q_0(T)$", "#FFF3E0", "#E65100"),
        (
            7.5,
            5.8,
            "Component\nDecomposition\n$P = \\sum P_i(\\xi_i)$",
            "#E8F5E9",
            "#2E7D32",
        ),
        (
            2.5,
            3.5,
            "Aging\nDegradation\n$Q(n) = Q_0(1-\\alpha n^\\beta)$",
            "#FCE4EC",
            "#C62828",
        ),
        (7.5, 3.5, "Power–Current\nCoupling\n$I = P/V$", "#F3E5F5", "#6A1B9A"),
        (
            5,
            1.5,
            "Time to Empty\n$t_{\\mathrm{empty}} = \\int_0^{t_c} dt$",
            "#ECEFF1",
            "#37474F",
        ),
    ]

    for cx, cy, text, fc, ec in boxes:
        box = FancyBboxPatch(
            (cx - 1.8, cy - 0.9),
            3.6,
            1.8,
            boxstyle="round,pad=0.15",
            fc=fc,
            ec=ec,
            lw=1.0,
            zorder=3,
        )
        ax_left.add_patch(box)
        ax_left.text(
            cx,
            cy,
            text,
            fontsize=5.5,
            ha="center",
            va="center",
            zorder=4,
            color=ec,
            fontweight="bold",
        )

    # Arrows
    arrow_kw = dict(arrowstyle="-|>", color="0.4", lw=1.2, mutation_scale=12, zorder=5)
    arrows = [
        ((5, 6.9), (5, 6.1)),  # shepherd → temp/comp (split)
        ((3.5, 6.1), (2.5, 6.1)),  # to temp
        ((6.5, 6.1), (7.5, 6.1)),  # to comp
        ((2.5, 4.9), (2.5, 4.4)),  # temp → aging
        ((7.5, 4.9), (7.5, 4.4)),  # comp → power-current
        ((2.5, 2.6), (4.0, 1.8)),  # aging → TTE
        ((7.5, 2.6), (6.0, 1.8)),  # power → TTE
    ]
    for start, end in arrows:
        ax_left.annotate("", xy=end, xytext=start, arrowprops=arrow_kw)

    # ═══════ Center panel: Validation result ═══════
    t_h = ts["time_h"].values
    v_exp = ts["voltage_exp"].values
    v_main = ts["voltage_main"].values

    ax_mid.plot(t_h, v_exp, "-", color=COLORS["exp"], lw=1.4, label="Experimental")
    ax_mid.plot(
        t_h,
        v_main,
        "-",
        color=COLORS["shepherd"],
        lw=1.0,
        label="Proposed model",
        alpha=0.9,
    )
    ax_mid.set_xlabel("Time (h)", fontsize=7)
    ax_mid.set_ylabel("Voltage (V)", fontsize=7)
    ax_mid.set_xlim(t_h[0], t_h[-1])
    ax_mid.legend(fontsize=5.5, loc="lower left", framealpha=0.9)
    ax_mid.set_title("Model Validation", fontsize=8, fontweight="bold", pad=6)

    rmse = np.sqrt(np.mean((v_main - v_exp) ** 2)) * 1000
    ax_mid.text(
        0.97,
        0.92,
        f"RMSE = {rmse:.1f} mV",
        transform=ax_mid.transAxes,
        fontsize=7,
        ha="right",
        va="top",
        fontweight="bold",
        color=COLORS["shepherd"],
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9),
    )

    # ═══════ Right panel: Key TTE result ═══════
    scenarios = ["Standby", "Reading", "Navigation", "Video", "Gaming"]
    ttes = [19.67, 9.64, 9.02, 6.37, 3.79]  # from tte_scenarios.csv at bg=0

    x = np.arange(len(scenarios))
    colors = [PHASE_COLORS.get(s, "#999") for s in scenarios]
    bars = ax_right.barh(
        x, ttes, height=0.55, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5
    )

    for i, v in enumerate(ttes):
        ax_right.text(
            v + 0.2, i, f"{v:.1f} h", va="center", fontsize=6.5, fontweight="bold"
        )

    ax_right.set_yticks(x)
    ax_right.set_yticklabels(scenarios, fontsize=7)
    ax_right.set_xlabel("Time to empty (h)", fontsize=7)
    ax_right.set_title("Usage Scenario TTE", fontsize=8, fontweight="bold", pad=6)
    ax_right.set_xlim(0, max(ttes) * 1.25)
    ax_right.invert_yaxis()

    save_fig(fig, "graphical_abstract", results_dir)
    print("Done: graphical_abstract")


if __name__ == "__main__":
    main()
