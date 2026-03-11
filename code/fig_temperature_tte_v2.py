"""
Figure 7: Temperature effect on TTE — model prediction across temperature range.
Two-panel layout:
  (a) TTE vs temperature for 5 scenarios (curve + markers)
  (b) Capacity / resistance ratio vs temperature (physical mechanism)

Data: Model-computed (MainBatteryModel)
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
    SINGLE_COL_TALL,
    COLORS,
    PHASE_COLORS,
)
from main_model import MainBatteryModel

apply_style()

SCENARIOS = {
    "Standby": 0.741,
    "Reading": 1.727,
    "Navigation": 1.845,
    "Video": 2.598,
    "Gaming": 4.303,
}

SCENARIO_MARKERS = {
    "Standby": "o",
    "Reading": "s",
    "Navigation": "D",
    "Video": "^",
    "Gaming": "v",
}


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "..", "results")

    model = MainBatteryModel(Q0=5.0)
    temps_c = np.arange(-10, 51, 2)
    temps_k = temps_c + 273.15

    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=SINGLE_COL_TALL, gridspec_kw={"hspace": 0.40}
    )

    # ═══════ Panel (a): TTE vs Temperature ═══════
    for scen, power in SCENARIOS.items():
        ttes = []
        for tk in temps_k:
            res = model.predict_discharge(power, temp_k=tk, dt=1.0)
            ttes.append(res["discharge_time"])
        color = PHASE_COLORS.get(scen, COLORS["gray"])
        mk = SCENARIO_MARKERS[scen]
        ax_a.plot(
            temps_c,
            ttes,
            "-",
            color=color,
            lw=1.2,
            marker=mk,
            markevery=5,
            markersize=4,
            label=f"{scen} ({power:.1f} W)",
            alpha=0.9,
        )

    ax_a.set_xlabel("Temperature (°C)")
    ax_a.set_ylabel("Time to empty (h)")
    ax_a.set_xlim(-10, 50)
    ax_a.legend(
        fontsize=5.5,
        loc="upper left",
        framealpha=0.9,
        title="Scenario (power)",
        title_fontsize=6,
    )

    # Highlight reference temperature
    ax_a.axvline(25, color="0.6", ls=":", lw=0.6, zorder=0)
    ax_a.text(
        26, ax_a.get_ylim()[1] * 0.95, "25°C ref.", fontsize=6, color="0.5", va="top"
    )

    label_panel(ax_a, "a")

    # ═══════ Panel (b): Physical mechanism ═══════
    Q_ratio = np.array([model.get_capacity_at_temp(tk) / model.Q0 for tk in temps_k])
    R_ratio = np.array([model.get_resistance_at_temp(tk) / model.R0 for tk in temps_k])

    color_q = "#1E88E5"
    color_r = "#E53935"

    ax_b.plot(
        temps_c,
        Q_ratio * 100,
        "-",
        color=color_q,
        lw=1.3,
        marker="o",
        markevery=5,
        markersize=3.5,
        label="$Q_0(T)/Q_{0,\\mathrm{ref}}$",
    )
    ax_b.set_xlabel("Temperature (°C)")
    ax_b.set_ylabel("Effective capacity (%)", color=color_q)
    ax_b.tick_params(axis="y", labelcolor=color_q)
    ax_b.set_xlim(-10, 50)
    ax_b.set_ylim(0, 120)

    ax_r = ax_b.twinx()
    ax_r.plot(
        temps_c,
        R_ratio,
        "-",
        color=color_r,
        lw=1.3,
        marker="s",
        markevery=5,
        markersize=3.5,
        label="$R(T)/R_{\\mathrm{ref}}$",
    )
    ax_r.set_ylabel("Resistance ratio", color=color_r)
    ax_r.tick_params(axis="y", labelcolor=color_r)

    # Highlight extreme zones
    ax_b.axvspan(-10, 0, color="#E3F2FD", alpha=0.3, zorder=0)
    ax_b.axvspan(40, 50, color="#FFEBEE", alpha=0.3, zorder=0)
    ax_b.text(-5, 105, "Cold", fontsize=6, ha="center", color="#1565C0")
    ax_b.text(45, 105, "Hot", fontsize=6, ha="center", color="#C62828")

    ax_b.axvline(25, color="0.6", ls=":", lw=0.6, zorder=0)

    # Combined legend
    lines1, labels1 = ax_b.get_legend_handles_labels()
    lines2, labels2 = ax_r.get_legend_handles_labels()
    ax_b.legend(
        lines1 + lines2,
        labels1 + labels2,
        fontsize=6,
        loc="center right",
        framealpha=0.9,
    )

    label_panel(ax_b, "b")

    save_fig(fig, "fig_temperature_tte", results_dir)
    print("Done: fig_temperature_tte")


if __name__ == "__main__":
    main()
