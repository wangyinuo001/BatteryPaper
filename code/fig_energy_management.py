"""
Fig 10 – Adaptive Energy Management Case Study
================================================
Demonstrates how the proposed model enables proactive power management:
- Baseline: Continue gaming until battery dies
- Strategy: When SOC drops below 20%, adaptively throttle to extend runtime

Two strategies compared:
A) No management: constant Gaming (4.303 W) until depletion
B) Model-aware adaptive: Gaming → auto-throttle at SOC=20%
   (reduce screen 90%→50%, SoC 95%→40%, radio 95%→30%, refresh 120→60Hz)
   Effective power drops from 4.303 W to ~1.55 W

Output: 2-panel figure showing SOC + power profiles, with TTE comparison.
"""

import sys, os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pub_style import apply_style, save_fig, COLORS, DOUBLE_COL_TALL, label_panel
from main_model import MainBatteryModel

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def compute_scenario_power(screen_brightness, screen_eta, soc_xi, radio_xi, gps_xi):
    """Compute total power from component intensities."""
    # Screen power model (from paper)
    gamma_s = 0.0012  # W/(cd/m²·m²) simplified
    lambda_s = 0.35  # W base
    L = screen_brightness / 100.0 * 500  # brightness % → cd/m²
    A_s = 0.0045  # ~6.7" screen area m²
    P_screen = (
        screen_eta * (gamma_s * A_s * L + lambda_s * A_s) if screen_eta > 0 else 0
    )

    # Other components
    P_soc_max = 2.04
    P_radio_max = 1.173
    P_gps_max = 0.30
    P_base = 0.12

    P_soc = soc_xi * P_soc_max + 0.10 * P_soc_max  # +10% background
    P_radio = radio_xi * P_radio_max
    P_gps = gps_xi * P_gps_max

    return P_screen + P_soc + P_radio + P_gps + P_base


# Gaming full power
P_GAMING = 4.303  # W (from paper Table 2)

# Throttled power: reduce screen to 50% at 60Hz, SoC to 40%, radio to 30%
P_THROTTLED = compute_scenario_power(
    screen_brightness=50,
    screen_eta=1.0,  # 60Hz, 50% brightness
    soc_xi=0.40,
    radio_xi=0.30,
    gps_xi=0.10,
)

# Low-power mode: further reduce to reading-like
P_LOWPOWER = compute_scenario_power(
    screen_brightness=40, screen_eta=1.0, soc_xi=0.15, radio_xi=0.10, gps_xi=0.10
)

SOC_THROTTLE_1 = 0.20  # First throttle threshold
SOC_THROTTLE_2 = 0.10  # Second throttle threshold (aggressive)


def simulate_adaptive(model, temp_k=298.15, dt=1.0):
    """Simulate adaptive energy management strategy."""
    Q_eff = model.get_capacity_at_temp(temp_k)
    soc = 1.0
    t = 0
    t_max = 40 * 3600  # 40h max

    time_pts, soc_pts, voltage_pts, power_pts, phase_pts = [], [], [], [], []
    I_avg = P_GAMING / 3.7

    while soc > 0.05 and t < t_max:
        # Determine power level based on SOC
        if soc > SOC_THROTTLE_1:
            P_load = P_GAMING
            phase = 0  # Full gaming
        elif soc > SOC_THROTTLE_2:
            P_load = P_THROTTLED
            phase = 1  # Throttled
        else:
            P_load = P_LOWPOWER
            phase = 2  # Low-power

        V_term = model.terminal_voltage(soc, I_avg, temp_k)
        if V_term < model.V_cutoff:
            break

        I_new = P_load / V_term if V_term > 0 else I_avg
        I_avg = 0.9 * I_avg + 0.1 * I_new

        dsoc = -I_avg * dt / (Q_eff * 3600)
        soc += dsoc
        t += dt

        if int(t) % 30 == 0:  # Record every 30s
            time_pts.append(t / 3600)
            soc_pts.append(soc)
            voltage_pts.append(V_term)
            power_pts.append(P_load)
            phase_pts.append(phase)

    return {
        "time": np.array(time_pts),
        "soc": np.array(soc_pts),
        "voltage": np.array(voltage_pts),
        "power": np.array(power_pts),
        "phase": np.array(phase_pts),
        "tte": time_pts[-1] if time_pts else 0,
    }


def simulate_baseline(model, temp_k=298.15, dt=1.0):
    """Simulate constant gaming (no management)."""
    result = model.predict_discharge(P_GAMING, temp_k=temp_k, dt=dt)
    return {
        "time": result["time"],
        "soc": result["soc"],
        "voltage": result["voltage"],
        "power": np.full_like(result["time"], P_GAMING),
        "tte": result["discharge_time"],
    }


def generate(fig=None, axes=None):
    """Generate adaptive energy management comparison figure."""
    apply_style()
    model = MainBatteryModel(Q0=5.0)

    print("  Simulating baseline (constant gaming)...")
    baseline = simulate_baseline(model)
    print(f"    Baseline TTE: {baseline['tte']:.2f} h")

    print("  Simulating adaptive management...")
    adaptive = simulate_adaptive(model)
    print(f"    Adaptive TTE: {adaptive['tte']:.2f} h")

    extension = (adaptive["tte"] - baseline["tte"]) / baseline["tte"] * 100
    print(f"    Extension: +{extension:.1f}%")

    # ── Create figure ──
    if fig is None:
        fig, axes = plt.subplots(
            2, 1, figsize=DOUBLE_COL_TALL, height_ratios=[1.2, 0.8], sharex=True
        )

    ax_soc, ax_power = axes

    # ── Panel (a): SOC trajectory ──
    ax_soc.plot(
        baseline["time"],
        baseline["soc"] * 100,
        color=COLORS["gray"],
        lw=1.2,
        ls="--",
        label=f'No management (TTE = {baseline["tte"]:.1f} h)',
    )

    # Adaptive: color by phase
    for phase_id, (clr, lbl) in enumerate(
        [
            (COLORS["shepherd"], "Full gaming"),
            (COLORS["accent"], "Throttled (−64% power)"),
            (COLORS["nbm"], "Low-power (−81% power)"),
        ]
    ):
        mask = adaptive["phase"] == phase_id
        if mask.any():
            # Find contiguous segments
            t_seg = adaptive["time"][mask]
            s_seg = adaptive["soc"][mask] * 100
            ax_soc.plot(t_seg, s_seg, color=clr, lw=1.3, label=lbl)

    # Threshold lines
    ax_soc.axhline(SOC_THROTTLE_1 * 100, color="gray", ls=":", lw=0.5, alpha=0.7)
    ax_soc.axhline(SOC_THROTTLE_2 * 100, color="gray", ls=":", lw=0.5, alpha=0.7)
    ax_soc.text(
        0.1, SOC_THROTTLE_1 * 100 + 1.5, "SOC = 20%", fontsize=6.5, color="gray"
    )
    ax_soc.text(
        0.1, SOC_THROTTLE_2 * 100 + 1.5, "SOC = 10%", fontsize=6.5, color="gray"
    )

    # Annotation: TTE extension
    ax_soc.annotate(
        f"+{extension:.0f}% runtime",
        xy=(adaptive["tte"], 5),
        xytext=(adaptive["tte"] * 0.7, 40),
        fontsize=8,
        fontweight="bold",
        color=COLORS["shepherd"],
        arrowprops=dict(arrowstyle="->", color=COLORS["shepherd"], lw=1.0),
    )

    ax_soc.set_ylabel("SOC (%)")
    ax_soc.set_ylim(0, 105)
    ax_soc.legend(loc="upper right", fontsize=6.5, framealpha=0.9)
    label_panel(ax_soc, "a")

    # ── Panel (b): Power profile ──
    # Baseline: constant
    ax_power.fill_between(baseline["time"], P_GAMING, alpha=0.15, color=COLORS["gray"])
    ax_power.plot(
        baseline["time"],
        baseline["power"],
        color=COLORS["gray"],
        lw=1.0,
        ls="--",
        label="No management",
    )

    # Adaptive: step function
    ax_power.plot(
        adaptive["time"],
        adaptive["power"],
        color=COLORS["shepherd"],
        lw=1.2,
        label="Adaptive management",
    )
    ax_power.fill_between(
        adaptive["time"], adaptive["power"], alpha=0.15, color=COLORS["shepherd"]
    )

    # Energy saved annotation
    E_baseline = baseline["tte"] * P_GAMING  # Wh
    E_adaptive = np.trapezoid(adaptive["power"], adaptive["time"])  # Wh
    ax_power.text(
        0.55,
        0.85,
        f"Total energy: {E_adaptive:.1f} Wh (adapt.) vs {E_baseline:.1f} Wh (base)",
        transform=ax_power.transAxes,
        fontsize=6.5,
        color="gray",
        ha="center",
    )

    ax_power.set_xlabel("Time (h)")
    ax_power.set_ylabel("Power (W)")
    ax_power.set_ylim(0, 5.5)
    ax_power.legend(loc="upper right", fontsize=6.5)
    label_panel(ax_power, "b")

    fig.align_ylabels(axes)
    save_fig(fig, "fig_energy_management", RESULTS_DIR)
    print(f"  ✓ Energy management figure saved")

    # Save summary data
    import csv

    csv_path = os.path.join(RESULTS_DIR, "energy_management_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Strategy", "TTE_h", "Total_Energy_Wh", "Extension_%"])
        w.writerow(
            ["No management", f'{baseline["tte"]:.2f}', f"{E_baseline:.2f}", "0.0"]
        )
        w.writerow(
            [
                "Adaptive",
                f'{adaptive["tte"]:.2f}',
                f"{E_adaptive:.2f}",
                f"{extension:.1f}",
            ]
        )
    print(f"  ✓ Summary CSV saved")

    return fig


if __name__ == "__main__":
    generate()

# Alias for generate_all_figures.py
main = generate
