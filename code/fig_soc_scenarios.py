"""
Figure 9: SOC discharge trajectories for all five usage scenarios.
Single panel showing how SOC evolves over time under each workload.
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pub_style import apply_style, save_fig, COLORS, SINGLE_COL_TALL
from main_model import MainBatteryModel

apply_style()

SCENARIOS = [
    {"name": "Standby", "power": 0.741, "color": "#FFB300", "ls": "-"},
    {"name": "Reading", "power": 1.727, "color": "#43A047", "ls": "--"},
    {"name": "Navigation", "power": 1.845, "color": "#1E88E5", "ls": "-."},
    {"name": "Video", "power": 2.598, "color": "#8E24AA", "ls": ":"},
    {"name": "Gaming", "power": 4.303, "color": "#E53935", "ls": "-"},
]


def main():
    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results"
    )
    model = MainBatteryModel(Q0=5.0)

    fig, ax = plt.subplots(figsize=SINGLE_COL_TALL)

    for sc in SCENARIOS:
        res = model.predict_discharge(sc["power"], temp_k=298.15)
        ax.plot(
            res["time"],
            res["soc"] * 100,
            ls=sc["ls"],
            color=sc["color"],
            lw=1.0,
            label=f'{sc["name"]} ({sc["power"]:.2f} W)',
        )

    ax.axhline(5, color=COLORS["gray"], ls="--", lw=0.5, alpha=0.6)
    ax.text(0.3, 7, "SOC$_{\\mathrm{min}}$ = 5%", fontsize=6, color=COLORS["gray"])

    ax.set_xlabel("Time (h)")
    ax.set_ylabel("State of charge (%)")
    ax.set_title(
        "SOC discharge under five scenarios\n(5000 mAh, 25°C, new battery)",
        fontweight="bold",
        pad=6,
    )
    ax.legend(fontsize=6, loc="upper right")
    ax.set_ylim(-2, 105)
    ax.set_xlim(0)

    save_fig(fig, "fig_soc_scenarios", results_dir)


if __name__ == "__main__":
    main()
