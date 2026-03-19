"""
NEW EXPERIMENT: Dynamic mixed-usage scenario.
Simulates a realistic usage pattern:
  Phase 1: 30 min Gaming (4.303 W)
  Phase 2: 60 min Video Streaming (2.598 W)
  Phase 3: 90 min Reading (1.727 W)
  Phase 4: Standby until empty (0.741 W)

Also generates Figure 8 for the paper.
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pub_style import (
    apply_style,
    save_fig,
    COLORS,
    DOUBLE_COL_TALL,
    PHASE_COLORS,
    label_panel,
)
from main_model import MainBatteryModel

apply_style()

# Phase definitions
PHASES = [
    {
        "name": "Gaming",
        "power": 4.303,
        "duration": 30 * 60,
        "color": PHASE_COLORS["Gaming"],
    },
    {
        "name": "Video Streaming",
        "power": 2.598,
        "duration": 60 * 60,
        "color": PHASE_COLORS["Video Streaming"],
    },
    {
        "name": "Reading",
        "power": 1.727,
        "duration": 90 * 60,
        "color": PHASE_COLORS["Reading"],
    },
    {
        "name": "Standby",
        "power": 0.741,
        "duration": None,
        "color": PHASE_COLORS["Standby"],
    },  # until empty
]


def simulate_mixed_usage(model, temp_k=298.15, dt=1.0, soc_init=1.0):
    """Simulate sequential usage phases."""
    Q_eff = model.get_capacity_at_temp(temp_k)

    time_pts = [0.0]
    soc_pts = [soc_init]
    volt_pts = [model.terminal_voltage(soc_init, 0.0, temp_k)]
    curr_pts = [0.0]
    power_pts = [0.0]
    phase_ids = [0]

    soc = soc_init
    t = 0.0
    I_avg = 0.4  # initial guess

    for phase_idx, phase in enumerate(PHASES):
        P = phase["power"]
        dur = phase["duration"]
        phase_t0 = t

        # Reset current estimate at phase transition to avoid voltage discontinuity
        V_est = model.terminal_voltage(soc, I_avg, temp_k)
        if V_est > 0:
            I_avg = P / V_est

        while soc > 0.05:
            V = model.terminal_voltage(soc, I_avg, temp_k)
            if V < model.V_cutoff:
                break

            I_new = P / V if V > 0 else I_avg
            I_avg = 0.9 * I_avg + 0.1 * I_new

            dsoc = -I_avg * dt / (Q_eff * 3600)
            soc += dsoc
            t += dt

            time_pts.append(t / 3600)
            soc_pts.append(soc)
            volt_pts.append(V)
            curr_pts.append(I_avg)
            power_pts.append(P)
            phase_ids.append(phase_idx)

            if dur is not None and (t - phase_t0) >= dur:
                break

        if soc <= 0.05 or V < model.V_cutoff:
            break

    return {
        "time": np.array(time_pts),
        "soc": np.array(soc_pts),
        "voltage": np.array(volt_pts),
        "current": np.array(curr_pts),
        "power": np.array(power_pts),
        "phase": np.array(phase_ids),
        "tte": time_pts[-1],
    }


def main():
    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results"
    )
    model = MainBatteryModel(Q0=5.0)

    # ── Run simulation ──
    result = simulate_mixed_usage(model, temp_k=298.15)
    print(f"Total TTE: {result['tte']:.2f} h")

    # Also run constant-power baselines for comparison
    const_results = {}
    for phase in PHASES:
        res = model.predict_discharge(phase["power"], temp_k=298.15)
        const_results[phase["name"]] = res["discharge_time"]
        print(f"  Constant {phase['name']}: {res['discharge_time']:.2f} h")

    # ── Figure ──
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=DOUBLE_COL_TALL,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.08},
        sharex=True,
    )

    t = result["time"]
    phase_arr = result["phase"]

    # Draw phase background shading
    for ax in (ax1, ax2):
        prev_phase = phase_arr[0]
        start_t = t[0]
        for i in range(1, len(t)):
            if phase_arr[i] != prev_phase or i == len(t) - 1:
                end_t = t[i]
                ax.axvspan(
                    start_t, end_t, alpha=0.08, color=PHASES[prev_phase]["color"]
                )
                prev_phase = phase_arr[i]
                start_t = end_t

    # Phase labels (top panel) — stagger vertically to avoid overlap
    v_top = max(result["voltage"])
    prev_phase = phase_arr[0]
    start_t = t[0]
    label_idx = 0
    for i in range(1, len(t)):
        if phase_arr[i] != prev_phase or i == len(t) - 1:
            mid = (start_t + t[i]) / 2
            disp_name = PHASES[prev_phase]["name"].replace(" Streaming", "")
            y_off = 0.14 if label_idx % 2 == 0 else 0.05
            ax1.text(
                mid,
                v_top + y_off,
                disp_name,
                ha="center",
                va="bottom",
                fontsize=6,
                fontweight="bold",
                color=PHASES[prev_phase]["color"],
            )
            label_idx += 1
            prev_phase = phase_arr[i]
            start_t = t[i]

    # Panel (a): Voltage
    ax1.plot(t, result["voltage"], "-", color=COLORS["shepherd"], lw=1.0)
    ax1.axhline(
        2.5, color=COLORS["gray"], ls="--", lw=0.5, label="$V_{\\mathrm{cutoff}}$"
    )
    ax1.set_ylabel("Terminal voltage (V)")
    label_panel(ax1, "a")
    ax1.legend(fontsize=6.5, loc="lower left")
    ax1.set_ylim(2.35, v_top + 0.38)

    # Panel (b): SOC
    ax2.plot(t, result["soc"] * 100, "-", color=COLORS["accent"], lw=1.0)
    ax2.axhline(
        5, color=COLORS["gray"], ls="--", lw=0.5, label="SOC$_{\\mathrm{min}}$ = 5%"
    )
    ax2.set_xlabel("Time (h)")
    ax2.set_ylabel("State of charge (%)")
    label_panel(ax2, "b")
    ax2.legend(fontsize=6.5, loc="upper right")

    # Mark phase transitions
    transitions = []
    prev_p = phase_arr[0]
    for i in range(1, len(t)):
        if phase_arr[i] != prev_p:
            transitions.append(t[i])
            prev_p = phase_arr[i]

    for tr in transitions:
        for ax in (ax1, ax2):
            ax.axvline(tr, color=COLORS["gray"], ls=":", lw=0.5, alpha=0.6)

    # TTE annotation
    ax2.annotate(
        f"TTE = {result['tte']:.1f} h",
        xy=(result["tte"], 5),
        xytext=(-1.5, 25),
        textcoords="offset points",
        fontsize=7,
        fontweight="bold",
        color=COLORS["shepherd"],
        arrowprops=dict(arrowstyle="->", color=COLORS["shepherd"], lw=0.7),
    )

    save_fig(fig, "fig_dynamic_scenario", results_dir)

    # ── Save summary CSV ──
    import csv

    csv_path = os.path.join(results_dir, "dynamic_scenario.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Phase",
                "Power_W",
                "Duration_min",
                "SOC_start",
                "SOC_end",
                "Time_start_h",
                "Time_end_h",
            ]
        )
        prev_p = phase_arr[0]
        start_idx = 0
        for i in range(1, len(t)):
            if phase_arr[i] != prev_p or i == len(t) - 1:
                p = PHASES[prev_p]
                w.writerow(
                    [
                        p["name"],
                        p["power"],
                        (t[i] - t[start_idx]) * 60,
                        f"{result['soc'][start_idx]*100:.1f}",
                        f"{result['soc'][i]*100:.1f}",
                        f"{t[start_idx]:.3f}",
                        f"{t[i]:.3f}",
                    ]
                )
                prev_p = phase_arr[i]
                start_idx = i

    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
