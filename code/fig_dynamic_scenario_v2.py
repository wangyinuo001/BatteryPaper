"""
Figure 8: Dynamic mixed-usage scenario — sequential phase simulation.
Three-panel stacked layout:
  (a) Power profile with phase shading
  (b) SOC trajectory with phase transitions
  (c) Voltage trajectory with cutoff

Data: Simulated via MainBatteryModel
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pub_style import (
    apply_style,
    save_fig,
    label_panel,
    embed_metric,
    DOUBLE_COL_TALL,
    PHASE_COLORS,
    COLORS,
)
from main_model import MainBatteryModel

apply_style()

PHASES = [
    {"name": "Gaming", "power": 4.303, "duration": 30 * 60},
    {"name": "Video Streaming", "power": 2.598, "duration": 60 * 60},
    {"name": "Reading", "power": 1.727, "duration": 90 * 60},
    {"name": "Standby", "power": 0.741, "duration": None},  # until empty
]


def simulate_mixed_usage(model, temp_k=298.15, dt=1.0, soc_init=1.0):
    Q_eff = model.get_capacity_at_temp(temp_k)
    time_pts, soc_pts, volt_pts, power_pts, phase_ids = [0], [soc_init], [], [0], [0]
    volt_pts.append(model.terminal_voltage(soc_init, 0.0, temp_k))

    soc, t, I_avg = soc_init, 0.0, 0.4

    for pi, phase in enumerate(PHASES):
        P, dur = phase["power"], phase["duration"]
        t0 = t
        while soc > 0.05:
            V = model.terminal_voltage(soc, I_avg, temp_k)
            if V < model.V_cutoff:
                break
            I_new = P / V if V > 0 else I_avg
            I_avg = 0.9 * I_avg + 0.1 * I_new
            soc += -I_avg * dt / (Q_eff * 3600)
            t += dt
            time_pts.append(t / 3600)
            soc_pts.append(soc)
            volt_pts.append(V)
            power_pts.append(P)
            phase_ids.append(pi)
            if dur is not None and (t - t0) >= dur:
                break
        if soc <= 0.05 or V < model.V_cutoff:
            break

    return {
        k: np.array(v)
        for k, v in [
            ("time", time_pts),
            ("soc", soc_pts),
            ("voltage", volt_pts[: len(time_pts)]),
            ("power", power_pts),
            ("phase", phase_ids),
        ]
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "..", "results")

    model = MainBatteryModel(Q0=5.0)
    sim = simulate_mixed_usage(model)

    fig = plt.figure(figsize=DOUBLE_COL_TALL)
    gs = gridspec.GridSpec(3, 1, height_ratios=[0.8, 1, 1], hspace=0.12)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1], sharex=ax_a)
    ax_c = fig.add_subplot(gs[2], sharex=ax_a)

    t = sim["time"]

    # Phase boundaries and shading
    phase_boundaries = []
    prev = sim["phase"][0]
    start = t[0]
    for i in range(1, len(sim["phase"])):
        if sim["phase"][i] != prev or i == len(sim["phase"]) - 1:
            phase_boundaries.append((start, t[i], int(prev)))
            start = t[i]
            prev = sim["phase"][i]

    for ax in [ax_a, ax_b, ax_c]:
        for t_start, t_end, pi in phase_boundaries:
            name = PHASES[pi]["name"]
            color = PHASE_COLORS.get(name, "#CCCCCC")
            ax.axvspan(t_start, t_end, color=color, alpha=0.12, zorder=0)

    # ═══════ Panel (a): Power profile ═══════
    ax_a.fill_between(
        t, 0, sim["power"], step="post", color="#1E88E5", alpha=0.3, zorder=2
    )
    ax_a.step(t, sim["power"], where="post", color="#1E88E5", lw=1.0, zorder=3)
    ax_a.set_ylabel("Power (W)")
    ax_a.set_ylim(0, 5.5)
    plt.setp(ax_a.get_xticklabels(), visible=False)
    label_panel(ax_a, "a")

    # Phase labels at top
    for t_start, t_end, pi in phase_boundaries:
        mid = (t_start + t_end) / 2
        name = PHASES[pi]["name"]
        ax_a.text(
            mid,
            5.0,
            name,
            ha="center",
            va="bottom",
            fontsize=6,
            fontweight="bold",
            color=PHASE_COLORS.get(name, "0.3"),
        )

    # ═══════ Panel (b): SOC ═══════
    ax_b.plot(t, sim["soc"] * 100, "-", color=COLORS["shepherd"], lw=1.3, zorder=3)
    ax_b.axhline(5, color="0.5", ls="--", lw=0.6, zorder=1)
    ax_b.text(t[-1] * 0.98, 7, "5% cutoff", fontsize=6, ha="right", color="0.5")
    ax_b.set_ylabel("SOC (%)")
    ax_b.set_ylim(0, 105)
    plt.setp(ax_b.get_xticklabels(), visible=False)
    label_panel(ax_b, "b")

    # Annotate phase transitions
    for t_start, t_end, pi in phase_boundaries:
        soc_start = np.interp(t_start, t, sim["soc"]) * 100
        soc_end = np.interp(t_end, t, sim["soc"]) * 100
        if pi < len(PHASES):
            delta = soc_start - soc_end
            if delta > 2:
                mid_t = (t_start + t_end) / 2
                mid_s = (soc_start + soc_end) / 2
                ax_b.annotate(
                    f"$\\Delta$SOC={delta:.0f}%",
                    xy=(mid_t, mid_s),
                    fontsize=5.5,
                    ha="center",
                    color="0.4",
                    bbox=dict(
                        boxstyle="round,pad=0.15", fc="white", ec="0.7", alpha=0.8
                    ),
                )

    # ═══════ Panel (c): Voltage ═══════
    ax_c.plot(t, sim["voltage"], "-", color=COLORS["exp"], lw=1.2, zorder=3)
    ax_c.axhline(model.V_cutoff, color="#E53935", ls="--", lw=0.8, zorder=1)
    ax_c.text(
        t[-1] * 0.98,
        model.V_cutoff + 0.03,
        "$V_{\\mathrm{cutoff}}$",
        fontsize=6,
        ha="right",
        color="#E53935",
    )
    ax_c.set_xlabel("Time (h)")
    ax_c.set_ylabel("Voltage (V)")
    ax_c.set_xlim(t[0], t[-1] * 1.02)
    label_panel(ax_c, "c")

    # Total TTE annotation
    embed_metric(ax_c, f"Total TTE = {t[-1]:.2f} h", x=0.97, y=0.85)

    # Legend for phases
    legend_patches = [
        Patch(facecolor=PHASE_COLORS.get(p["name"], "#CCC"), alpha=0.5, label=p["name"])
        for p in PHASES
    ]
    ax_a.legend(
        handles=legend_patches, fontsize=5.5, loc="upper right", ncol=2, framealpha=0.9
    )

    save_fig(fig, "fig_dynamic_scenario", results_dir)
    print("Done: fig_dynamic_scenario")


if __name__ == "__main__":
    main()
