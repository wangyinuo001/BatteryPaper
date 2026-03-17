"""
Figure 4: Temperature–TTE curve.
Simulates Video Streaming TTE at temperatures from −10 to 50 °C.
Shows the unimodal pattern arising from capacity vs. leakage competition.
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pub_style import apply_style, save_fig, COLORS, SINGLE_COL_TALL
from main_model import MainBatteryModel

apply_style()

# SoC leakage model (from paper Eq. 15)
# Note: c1 in A/K^2 (converted from mA/K^2 literature values)
C1_LEAK = 0.49328
C2_LEAK = -3740.0
V_NOM = 3.7


def soc_leakage_power(temp_k):
    return V_NOM * C1_LEAK * temp_k**2 * np.exp(C2_LEAK / temp_k)


def main():
    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results"
    )
    model = MainBatteryModel(Q0=5.0)

    # Video Streaming power breakdown
    P_screen = 510.6e-3
    P_soc_dyn = 612.0e-3
    P_radio = 1055.7e-3
    P_gps = 300.0e-3
    P_base = 120.0e-3
    P_base_total = P_screen + P_soc_dyn + P_radio + P_gps + P_base  # ~2.598 W

    # Reference leakage at 25°C (already included in the component measurements)
    T_ref = 298.15  # 25°C
    P_leak_ref = soc_leakage_power(T_ref)

    temps_c = np.arange(-10, 51, 2)
    ttes = []

    for tc in temps_c:
        tk = tc + 273.15
        # Incremental leakage: only the CHANGE from 25°C reference
        P_leak = soc_leakage_power(tk)
        dP_leak = P_leak - P_leak_ref
        P_total = P_base_total + dP_leak
        result = model.predict_discharge(P_total, temp_k=tk)
        ttes.append(result["discharge_time"])

    ttes = np.array(ttes)

    # Find peak
    peak_idx = np.argmax(ttes)
    peak_t = temps_c[peak_idx]
    peak_tte = ttes[peak_idx]

    fig, ax = plt.subplots(figsize=SINGLE_COL_TALL)

    ax.fill_between(temps_c, ttes, alpha=0.12, color=COLORS["shepherd"])
    ax.plot(temps_c, ttes, "-", color=COLORS["shepherd"], lw=1.5, zorder=4)
    ax.plot(
        peak_t,
        peak_tte,
        "o",
        color=COLORS["shepherd"],
        ms=6,
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=5,
    )

    # Annotate peak — place text to the left to avoid overlap with right-side label
    ax.annotate(
        f"Peak: {peak_tte:.1f} h\nat {peak_t} °C",
        xy=(peak_t, peak_tte),
        xytext=(peak_t - 22, peak_tte + 0.25),
        fontsize=7,
        fontweight="bold",
        color=COLORS["shepherd"],
        arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=0.6),
    )

    # Region labels — position below the curve to avoid clashing with peak annotation
    ax.text(
        -5,
        ttes[0] - 0.40,
        "Capacity loss\n(low $T$)",
        fontsize=7,
        ha="center",
        va="top",
        style="italic",
        color=COLORS["accent"],
    )
    ax.text(
        47,
        ttes[-1] - 0.40,
        "Leakage rise\n(high $T$)",
        fontsize=7,
        ha="center",
        va="top",
        style="italic",
        color=COLORS["accent"],
    )

    ax.set_xlabel("Ambient temperature (°C)")
    ax.set_ylabel("Time-to-empty (h)")
    ax.set_xlim(-12, 52)
    ax.set_ylim(ttes.min() - 1.0, peak_tte + 0.8)

    save_fig(fig, "fig_temperature_tte", results_dir)


if __name__ == "__main__":
    main()
