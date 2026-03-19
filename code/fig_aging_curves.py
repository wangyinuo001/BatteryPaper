"""
Figure 7: Aging degradation curves — Capacity fade + Resistance growth
vs. equivalent full cycles, with TTE impact overlay.
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pub_style import apply_style, save_fig, COLORS, DOUBLE_COL, label_panel
from main_model import MainBatteryModel

apply_style()

# Aging parameters (from paper Eqs. 11–12)
ALPHA_Q = 2.56e-4
BETA_R = 1.229e-3
GAMMA = 1.085
Q0_NOM = 5.0
R0_NOM = 0.035


def q_aging(N):
    return Q0_NOM * (1 - ALPHA_Q * N**GAMMA)


def r_aging(N):
    return R0_NOM * (1 + BETA_R * N**GAMMA)


def main():
    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results"
    )
    model = MainBatteryModel(Q0=Q0_NOM)

    N_vals = np.linspace(0, 1000, 200)
    Q_vals = q_aging(N_vals)
    R_vals = r_aging(N_vals)

    # TTE at milestones
    milestones = [0, 100, 200, 300, 500, 800, 1000]
    P_video = 2.598  # W
    ttes = []
    for n in milestones:
        m = MainBatteryModel(Q0=q_aging(n), R0=r_aging(n))
        res = m.predict_discharge(P_video, temp_k=298.15)
        ttes.append(res["discharge_time"])

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=DOUBLE_COL, gridspec_kw={"wspace": 0.38}
    )

    # ── (a) Capacity fade & Resistance growth ──
    ax1.plot(
        N_vals, Q_vals, "-", color=COLORS["shepherd"], lw=1.3, label="$Q_0(N)$ (Ah)"
    )
    ax1_twin = ax1.twinx()
    ax1_twin.plot(
        N_vals,
        R_vals * 1000,
        "--",
        color=COLORS["accent"],
        lw=1.3,
        label="$R_0(N)$ (mΩ)",
    )

    ax1.set_xlabel("Equivalent full cycles $N$")
    ax1.set_ylabel("Effective capacity $Q_0$ (Ah)", color=COLORS["shepherd"])
    ax1_twin.set_ylabel("Internal resistance $R_0$ (mΩ)", color=COLORS["accent"])
    ax1.tick_params(axis="y", labelcolor=COLORS["shepherd"])
    ax1_twin.tick_params(axis="y", labelcolor=COLORS["accent"])
    label_panel(ax1, "a")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=6.5, loc="center right")

    # ── (b) TTE vs cycle count ──
    ax2.fill_between(milestones, ttes, alpha=0.12, color=COLORS["shepherd"])
    ax2.plot(
        milestones,
        ttes,
        "o-",
        color=COLORS["shepherd"],
        lw=1.3,
        ms=5,
        markeredgecolor="white",
        markeredgewidth=0.5,
    )

    for n, tte in zip(milestones, ttes):
        pct = (tte / ttes[0] - 1) * 100
        if n > 0:
            ax2.annotate(
                f"{pct:+.0f}%",
                xy=(n, tte),
                xytext=(0, -14),
                textcoords="offset points",
                fontsize=7,
                ha="center",
                color=COLORS["gray"],
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.0),
            )

    ax2.set_xlabel("Equivalent full cycles $N$")
    ax2.set_ylabel("Time-to-empty (h)")
    label_panel(ax2, "b")

    save_fig(fig, "fig_aging_curves", results_dir)


if __name__ == "__main__":
    main()
