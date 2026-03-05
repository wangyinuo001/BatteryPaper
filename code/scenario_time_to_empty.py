"""
Scenario-based time-to-empty predictions and initial SOC analysis.

*** IMPORTANT: TTE IS NOT COMPUTED BY SIMPLE FORMULA ***

TTE is calculated by numerically integrating the discharge ODE system:
    dSOC/dt = -I(t) / Q₀(T)
    V(t) = f_Shepherd(SOC, I, T)
    I(t) = P(t) / V(t)

The commonly cited formula "TTE = Q·SOC/P" is INCORRECT because it assumes:
1. Constant voltage (wrong: V drops with SOC and current)
2. Constant current (wrong: I = P/V varies as V changes)
3. Linear SOC-voltage relationship (wrong: Shepherd model is nonlinear)

This script performs proper integration via MainBatteryModel.predict_discharge().

Outputs:
- results/fig_tte_by_soc.png
- results/tte_by_soc.csv
- results/tte_scenarios.csv
- results/tte_scenarios_table.tex
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from aging_model import BatteryModel
from main_model import MainBatteryModel


SCENARIOS = {
    "Standby": {"screen": (0, 0.0), "soc": 10, "radio": 10, "gps": 10},
    "Navigation": {"screen": (70, 1.0), "soc": 30, "radio": 30, "gps": 100},
    "Gaming": {"screen": (90, 1.3), "soc": 100, "radio": 95, "gps": 10},
    "Video": {"screen": (80, 1.0), "soc": 30, "radio": 90, "gps": 10},
    "Reading": {"screen": (70, 1.0), "soc": 30, "radio": 20, "gps": 10},
}


def build_inputs(scenario, background_extra=0.0, temperature_c=25):
    screen_percent, eta = scenario["screen"]
    screen_brightness = (screen_percent / 100.0) * eta

    cpu_load = (scenario["soc"] / 100.0) + background_extra
    cpu_load = min(cpu_load, 1.0)

    radio_activity = scenario["radio"] / 100.0
    gps_on = scenario["gps"] > 0

    return {
        "brightness": screen_brightness,
        "cpu_load": cpu_load,
        "network_mode": "4g",
        "network_activity": radio_activity,
        "gps": gps_on,
        "temperature": temperature_c + 273.15,
    }


def compute_power_w(power_model, inputs):
    p_mw = power_model.get_power_consumption(0.0, inputs)
    return p_mw / 1000.0


def simulate_tte(main_model, power_w, temp_k, soc_init=1.0):
    """
    Compute TTE by numerically integrating the discharge ODE system.

    This calls MainBatteryModel.predict_discharge() which integrates:
        dSOC/dt = -I(t) / Q₀(T)

    until SOC cutoff or voltage cutoff is reached.

    Returns: (tte_hours, time_array, soc_array)
    """
    out = main_model.predict_discharge(
        P_load=power_w, temp_k=temp_k, soc_initial=soc_init
    )
    return out["discharge_time"], out["time"], out["soc"]


def run_scenarios():
    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results"
    )
    os.makedirs(results_dir, exist_ok=True)

    power_model = BatteryModel(cycle_count=0, calendar_time_hours=0)
    main_model = MainBatteryModel(Q0=5.0)

    rows = []
    background_levels = [0.0, 0.1, 0.2]

    for name, scenario in SCENARIOS.items():
        for bg in background_levels:
            inputs = build_inputs(scenario, background_extra=bg)
            power_w = compute_power_w(power_model, inputs)
            tte, _, _ = simulate_tte(
                main_model, power_w, inputs["temperature"], soc_init=1.0
            )
            rows.append(
                {
                    "scenario": name,
                    "background_extra": bg,
                    "tte_hours": tte,
                }
            )

    df = pd.DataFrame(rows)
    scenario_csv = os.path.join(results_dir, "tte_scenarios.csv")
    df.to_csv(scenario_csv, index=False)

    tex_path = os.path.join(results_dir, "tte_scenarios_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Scenario time-to-empty table\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Time-to-empty under usage scenarios and background load}\n")
        f.write("\\label{tab:tte_scenarios}\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\hline\n")
        f.write("Scenario & Background +0\% & +10\% & +20\% \\\\\n")
        f.write("\\hline\n")
        for name in SCENARIOS.keys():
            row = df[df["scenario"] == name].sort_values("background_extra")
            vals = [f"{v:.2f}" for v in row["tte_hours"].values]
            f.write(f"{name} & {vals[0]} & {vals[1]} & {vals[2]} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"Saved: {scenario_csv}")
    print(f"Saved: {tex_path}")


def run_tte_by_soc():
    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results"
    )
    os.makedirs(results_dir, exist_ok=True)

    power_model = BatteryModel(cycle_count=0, calendar_time_hours=0)
    main_model = MainBatteryModel(Q0=5.0)
    soc_levels = np.arange(1.0, 0.0, -0.1)

    baseline = SCENARIOS["Video"]
    background_levels = [0.0, 0.1, 0.2]

    rows = []
    tte_curves = {bg: [] for bg in background_levels}

    for soc0 in soc_levels:
        for bg in background_levels:
            inputs = build_inputs(baseline, background_extra=bg)
            power_w = compute_power_w(power_model, inputs)
            tte, _, _ = simulate_tte(
                main_model, power_w, inputs["temperature"], soc_init=soc0
            )
            tte_curves[bg].append(tte)
        rows.append(
            {
                "soc0": soc0,
                "tte_bg0": tte_curves[0.0][-1],
                "tte_bg10": tte_curves[0.1][-1],
                "tte_bg20": tte_curves[0.2][-1],
            }
        )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(results_dir, "tte_by_soc.csv")
    df.to_csv(csv_path, index=False)

    soc_percent = soc_levels * 100
    y_mid = np.array(tte_curves[0.1])
    y_low = np.minimum(np.array(tte_curves[0.0]), np.array(tte_curves[0.2]))
    y_high = np.maximum(np.array(tte_curves[0.0]), np.array(tte_curves[0.2]))

    plt.figure(figsize=(7.5, 5))
    plt.plot(soc_percent, y_mid, "o-", label="Background +10% (median)")
    plt.fill_between(
        soc_percent, y_low, y_high, alpha=0.2, label="Background 0-20% range"
    )
    plt.gca().invert_xaxis()
    plt.xlabel("Initial SOC (%)")
    plt.ylabel("Time-to-Empty (hours)")
    plt.title("Time-to-Empty vs Initial SOC (Video Scenario)")
    plt.grid(alpha=0.3)
    plt.legend()

    fig_path = os.path.join(results_dir, "fig_tte_by_soc.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"Saved: {csv_path}")
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    run_scenarios()
    run_tte_by_soc()
