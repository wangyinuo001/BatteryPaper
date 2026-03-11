"""
Figure 11: Adaptive energy management — comparison of full-power vs adaptive.
Three-panel stacked layout:
  (a) Power profile: raw vs adaptive
  (b) SOC trajectory: both strategies side-by-side
  (c) Voltage trajectory with extension annotation

Data: Simulated via MainBatteryModel with adaptive thresholds
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
    COLORS,
)
from main_model import MainBatteryModel

apply_style()

P_GAMING = 4.303
P_THROTTLED = 1.727  # reading-level throttle
P_LOWPOWER = 0.741  # standby-level low power

SOC_THRESH_1 = 0.20  # first throttle at 20%
SOC_THRESH_2 = 0.10  # aggressive throttle at 10%


def simulate_strategy(model, adaptive=False, temp_k=298.15, dt=1.0):
    Q_eff = model.get_capacity_at_temp(temp_k)
    soc, t, I_avg = 1.0, 0.0, P_GAMING / 3.7

    time_pts, soc_pts, volt_pts, power_pts, phase_pts = [], [], [], [], []
    t_max = 20 * 3600

    while soc > 0.05 and t < t_max:
        if adaptive:
            if soc > SOC_THRESH_1:
                P = P_GAMING
                phase = 0
            elif soc > SOC_THRESH_2:
                P = P_THROTTLED
                phase = 1
            else:
                P = P_LOWPOWER
                phase = 2
        else:
            P = P_GAMING
            phase = 0

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
        phase_pts.append(phase)

    return {
        k: np.array(v)
        for k, v in [
            ("time", time_pts),
            ("soc", soc_pts),
            ("voltage", volt_pts),
            ("power", power_pts),
            ("phase", phase_pts),
        ]
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "..", "results")

    model = MainBatteryModel(Q0=5.0)
    sim_raw = simulate_strategy(model, adaptive=False)
    sim_adp = simulate_strategy(model, adaptive=True)

    fig = plt.figure(figsize=DOUBLE_COL_TALL)
    gs = gridspec.GridSpec(3, 1, height_ratios=[0.7, 1, 1], hspace=0.12)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1], sharex=ax_a)
    ax_c = fig.add_subplot(gs[2], sharex=ax_a)

    c_raw = "#E53935"
    c_adp = "#1E88E5"

    t_max_plot = max(sim_raw["time"][-1], sim_adp["time"][-1]) * 1.05

    # ═══════ Panel (a): Power profiles ═══════
    ax_a.step(
        sim_raw["time"],
        sim_raw["power"],
        where="post",
        color=c_raw,
        lw=1.0,
        label="No management",
        alpha=0.8,
    )
    ax_a.step(
        sim_adp["time"],
        sim_adp["power"],
        where="post",
        color=c_adp,
        lw=1.2,
        label="Adaptive",
        alpha=0.9,
    )

    # Threshold annotations
    for thresh, label_text in [
        (SOC_THRESH_1, "20% SOC\nthrottle"),
        (SOC_THRESH_2, "10% SOC\nlow-power"),
    ]:
        # find time when adaptive hits this threshold
        idx = np.argmin(np.abs(sim_adp["soc"] - thresh))
        t_thresh = sim_adp["time"][idx]
        ax_a.axvline(t_thresh, color=c_adp, ls=":", lw=0.5, alpha=0.5)

    ax_a.set_ylabel("Power (W)")
    ax_a.set_ylim(0, 5.5)
    ax_a.legend(fontsize=6.5, loc="upper right", framealpha=0.9, ncol=2)
    plt.setp(ax_a.get_xticklabels(), visible=False)
    label_panel(ax_a, "a")

    # ═══════ Panel (b): SOC ═══════
    ax_b.plot(
        sim_raw["time"],
        sim_raw["soc"] * 100,
        "-",
        color=c_raw,
        lw=1.2,
        label="No management",
    )
    ax_b.plot(
        sim_adp["time"],
        sim_adp["soc"] * 100,
        "-",
        color=c_adp,
        lw=1.3,
        label="Adaptive",
    )

    ax_b.axhline(SOC_THRESH_1 * 100, color="0.6", ls="--", lw=0.5)
    ax_b.axhline(SOC_THRESH_2 * 100, color="0.6", ls="--", lw=0.5)
    ax_b.text(
        0.05,
        SOC_THRESH_1 * 100 + 1,
        f"{SOC_THRESH_1*100:.0f}%",
        fontsize=6,
        color="0.5",
    )
    ax_b.text(
        0.05,
        SOC_THRESH_2 * 100 + 1,
        f"{SOC_THRESH_2*100:.0f}%",
        fontsize=6,
        color="0.5",
    )

    ax_b.axhline(5, color="0.5", ls=":", lw=0.4)
    ax_b.set_ylabel("SOC (%)")
    ax_b.set_ylim(0, 105)
    ax_b.legend(fontsize=6.5, loc="upper right", framealpha=0.9)
    plt.setp(ax_b.get_xticklabels(), visible=False)
    label_panel(ax_b, "b")

    # ═══════ Panel (c): Voltage ═══════
    ax_c.plot(
        sim_raw["time"],
        sim_raw["voltage"],
        "-",
        color=c_raw,
        lw=1.0,
        label="No management",
    )
    ax_c.plot(
        sim_adp["time"], sim_adp["voltage"], "-", color=c_adp, lw=1.2, label="Adaptive"
    )

    ax_c.axhline(model.V_cutoff, color="0.5", ls="--", lw=0.6)
    ax_c.text(
        0.1, model.V_cutoff + 0.02, "$V_{\\mathrm{cutoff}}$", fontsize=6, color="0.5"
    )

    # Extension annotation
    tte_raw = sim_raw["time"][-1]
    tte_adp = sim_adp["time"][-1]
    ext = (tte_adp - tte_raw) / tte_raw * 100
    ax_c.annotate(
        "",
        xy=(tte_adp, model.V_cutoff + 0.15),
        xytext=(tte_raw, model.V_cutoff + 0.15),
        arrowprops=dict(arrowstyle="<->", color="#43A047", lw=1.2),
    )
    ax_c.text(
        (tte_raw + tte_adp) / 2,
        model.V_cutoff + 0.20,
        f"+{ext:.0f}% extension\n({tte_adp - tte_raw:.1f} h)",
        ha="center",
        fontsize=6.5,
        fontweight="bold",
        color="#43A047",
    )

    ax_c.set_xlabel("Time (h)")
    ax_c.set_ylabel("Voltage (V)")
    ax_c.set_xlim(0, t_max_plot)
    ax_c.legend(fontsize=6.5, loc="lower left", framealpha=0.9)
    label_panel(ax_c, "c")

    embed_metric(
        ax_c,
        f"Raw: TTE = {tte_raw:.2f} h\n" f"Adaptive: TTE = {tte_adp:.2f} h",
        x=0.97,
        y=0.65,
    )

    save_fig(fig, "fig_energy_management", results_dir)
    print("Done: fig_energy_management")


if __name__ == "__main__":
    main()
