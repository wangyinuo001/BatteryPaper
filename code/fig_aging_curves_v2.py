"""
Figure 10: Battery aging degradation curves.
Two-panel layout:
  (a) Capacity vs cycle count for 3 usage scenarios
  (b) TTE vs cycle count with capacity overlay (dual-y)

Data: Simulated via MainBatteryModel + aging model
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pub_style import (
    apply_style,
    save_fig,
    label_panel,
    embed_metric,
    DOUBLE_COL,
    PHASE_COLORS,
    COLORS,
)
from main_model import MainBatteryModel

apply_style()

# Aging model: Q(n) = Q0 * (1 - alpha * n^beta)
AGING_ALPHA = 0.0005
AGING_BETA = 0.75

SCENARIOS = {
    "Standby": 0.741,
    "Reading": 1.727,
    "Gaming": 4.303,
}

SCENARIO_MARKERS = {"Standby": "o", "Reading": "s", "Gaming": "v"}


def capacity_at_cycle(Q0, n):
    return Q0 * (1 - AGING_ALPHA * n**AGING_BETA)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "..", "results")

    model = MainBatteryModel(Q0=5.0)
    cycles = np.arange(0, 1001, 10)

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=DOUBLE_COL, gridspec_kw={"wspace": 0.40}
    )

    # ═══════ Panel (a): Capacity degradation ═══════
    Q_vals = capacity_at_cycle(5.0, cycles)
    Q_pct = Q_vals / 5.0 * 100

    ax_a.plot(cycles, Q_pct, "-", color=COLORS["shepherd"], lw=1.5, zorder=3)
    ax_a.fill_between(cycles, Q_pct, 100, color=COLORS["shepherd"], alpha=0.06)

    # EOL threshold
    ax_a.axhline(80, color="#E53935", ls="--", lw=0.8, zorder=2)
    ax_a.text(50, 79, "80% EOL threshold", fontsize=6, color="#E53935", va="top")

    # Find EOL cycle
    eol_idx = np.argmin(np.abs(Q_pct - 80))
    eol_cycle = cycles[eol_idx]
    ax_a.axvline(eol_cycle, color="#E53935", ls=":", lw=0.6, alpha=0.5)
    ax_a.annotate(
        f"EOL ≈ {eol_cycle} cycles",
        xy=(eol_cycle, 80),
        xytext=(eol_cycle - 200, 85),
        fontsize=6.5,
        fontweight="bold",
        color="#E53935",
        arrowprops=dict(arrowstyle="->", color="#E53935", lw=0.8),
    )

    ax_a.set_xlabel("Cycle number")
    ax_a.set_ylabel("Remaining capacity (%)")
    ax_a.set_xlim(0, 1000)
    ax_a.set_ylim(70, 102)

    embed_metric(
        ax_a,
        f"$Q(n) = Q_0(1 - \\alpha n^\\beta)$\n"
        f"$\\alpha={AGING_ALPHA}$, $\\beta={AGING_BETA}$",
        x=0.97,
        y=0.55,
        fontsize=6,
    )

    label_panel(ax_a, "a")

    # ═══════ Panel (b): TTE degradation per scenario ═══════
    for scen, power in SCENARIOS.items():
        ttes = []
        for n in cycles:
            Q_aged = capacity_at_cycle(5.0, n)
            model_aged = MainBatteryModel(Q0=Q_aged)
            res = model_aged.predict_discharge(power, temp_k=298.15, dt=1.0)
            ttes.append(res["discharge_time"])
        color = PHASE_COLORS.get(scen, COLORS["gray"])
        mk = SCENARIO_MARKERS[scen]
        ax_b.plot(
            cycles,
            ttes,
            "-",
            color=color,
            lw=1.2,
            marker=mk,
            markevery=10,
            markersize=3.5,
            label=f"{scen} ({power:.1f} W)",
            alpha=0.9,
        )

    ax_b.set_xlabel("Cycle number")
    ax_b.set_ylabel("Time to empty (h)")
    ax_b.set_xlim(0, 1000)
    ax_b.legend(
        fontsize=6,
        loc="upper right",
        framealpha=0.9,
        title="Scenario",
        title_fontsize=6.5,
    )

    # Capacity overlay (secondary y-axis)
    ax_b2 = ax_b.twinx()
    ax_b2.plot(cycles, Q_pct, ":", color="0.5", lw=0.8, alpha=0.5)
    ax_b2.set_ylabel("Capacity (%)", color="0.5", fontsize=7)
    ax_b2.tick_params(axis="y", labelcolor="0.5", labelsize=6.5)
    ax_b2.set_ylim(70, 102)

    label_panel(ax_b, "b")

    save_fig(fig, "fig_aging_curves", results_dir)
    print("Done: fig_aging_curves")


if __name__ == "__main__":
    main()
