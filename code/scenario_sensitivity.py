"""
Scenario-level sensitivity analysis: perturb component utilization levels
and measure impact on TTE.

This analyzes sensitivity to:
- Screen brightness percentage
- CPU load (SoC) percentage
- Radio activity percentage
- GPS on/off state

Each component is perturbed by ±5% of its utilization level in the baseline scenario.
TTE is recomputed via numerical integration of the discharge ODE system.

Outputs:
- results/scenario_sensitivity.csv
- results/scenario_sensitivity_table.tex
"""

import os
import copy
import numpy as np
import pandas as pd

from aging_model import BatteryModel
from main_model import MainBatteryModel


# Baseline scenario: Video Streaming
BASELINE_SCENARIO = {
    "screen": (80, 1.0),  # (brightness %, refresh multiplier)
    "soc": 20,  # CPU load %
    "radio": 90,  # Radio activity %
    "gps": 10,  # GPS usage %
}


def build_inputs(scenario, temperature_c=25):
    """Convert scenario definition to model inputs."""
    screen_percent, eta = scenario["screen"]
    screen_brightness = (screen_percent / 100.0) * eta

    cpu_load = scenario["soc"] / 100.0
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


def compute_tte(scenario, power_model, main_model, temp_c=25):
    """
    Compute TTE by numerical integration of discharge ODE system.

    Returns: (tte_hours, power_watts)
    """
    inputs = build_inputs(scenario, temperature_c=temp_c)
    power_w = power_model.get_power_consumption(0.0, inputs) / 1000.0

    result = main_model.predict_discharge(
        P_load=power_w, temp_k=inputs["temperature"], soc_initial=1.0
    )

    return result["discharge_time"], power_w


def run_scenario_sensitivity(results_dir):
    """
    Perturb each scenario component by ±5% and measure TTE change.

    Method:
    1. Compute baseline TTE via ODE integration
    2. For each component (screen, cpu, radio, gps):
       - Perturb by ±5% (or ±5 percentage points for absolute values)
       - Re-integrate to get new TTE
       - Compute relative change: Δt/t = (t' - t₀)/t₀
    """
    power_model = BatteryModel(cycle_count=0, calendar_time_hours=0)
    main_model = MainBatteryModel(Q0=1.991)

    # Baseline
    tte_baseline, power_baseline = compute_tte(
        BASELINE_SCENARIO, power_model, main_model
    )

    print(f"Baseline scenario:")
    print(f"  Screen: {BASELINE_SCENARIO['screen'][0]}%")
    print(f"  CPU (SoC): {BASELINE_SCENARIO['soc']}%")
    print(f"  Radio: {BASELINE_SCENARIO['radio']}%")
    print(f"  GPS: {BASELINE_SCENARIO['gps']}%")
    print(f"  Power: {power_baseline:.3f}W")
    print(f"  TTE: {tte_baseline:.3f}h")
    print()

    # Sensitivity analysis
    rows = []

    # 1. Screen brightness
    for delta in [-5, +5]:
        scenario = copy.deepcopy(BASELINE_SCENARIO)
        scenario["screen"] = (scenario["screen"][0] + delta, scenario["screen"][1])
        tte, power = compute_tte(scenario, power_model, main_model)
        delta_pct = (tte - tte_baseline) / tte_baseline * 100
        rows.append(
            {
                "component": "Screen Brightness",
                "perturbation": f"{delta:+d}%",
                "new_value": f"{scenario['screen'][0]}%",
                "power_w": power,
                "tte_hours": tte,
                "delta_tte_pct": delta_pct,
            }
        )

    # 2. CPU load (SoC)
    for delta in [-5, +5]:
        scenario = copy.deepcopy(BASELINE_SCENARIO)
        scenario["soc"] = scenario["soc"] + delta
        tte, power = compute_tte(scenario, power_model, main_model)
        delta_pct = (tte - tte_baseline) / tte_baseline * 100
        rows.append(
            {
                "component": "CPU Load (SoC)",
                "perturbation": f"{delta:+d}%",
                "new_value": f"{scenario['soc']}%",
                "power_w": power,
                "tte_hours": tte,
                "delta_tte_pct": delta_pct,
            }
        )

    # 3. Radio activity
    for delta in [-5, +5]:
        scenario = copy.deepcopy(BASELINE_SCENARIO)
        scenario["radio"] = scenario["radio"] + delta
        tte, power = compute_tte(scenario, power_model, main_model)
        delta_pct = (tte - tte_baseline) / tte_baseline * 100
        rows.append(
            {
                "component": "Radio Activity",
                "perturbation": f"{delta:+d}%",
                "new_value": f"{scenario['radio']}%",
                "power_w": power,
                "tte_hours": tte,
                "delta_tte_pct": delta_pct,
            }
        )

    # 4. GPS (binary: 10% -> 5% or 15%)
    for delta in [-5, +5]:
        scenario = copy.deepcopy(BASELINE_SCENARIO)
        scenario["gps"] = max(0, scenario["gps"] + delta)
        tte, power = compute_tte(scenario, power_model, main_model)
        delta_pct = (tte - tte_baseline) / tte_baseline * 100
        rows.append(
            {
                "component": "GPS Usage",
                "perturbation": f"{delta:+d}%",
                "new_value": f"{scenario['gps']}%",
                "power_w": power,
                "tte_hours": tte,
                "delta_tte_pct": delta_pct,
            }
        )

    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = os.path.join(results_dir, "scenario_sensitivity.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Generate LaTeX table
    tex_path = os.path.join(results_dir, "scenario_sensitivity_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Scenario component sensitivity\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Sensitivity of TTE to scenario component perturbations}\n")
        f.write("\\label{tab:scenario_sensitivity}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\hline\n")
        f.write(
            "Component & Perturbation & New Value & Power (W) & $\\Delta$TTE (\\%) \\\\\n"
        )
        f.write("\\hline\n")

        for _, row in df.iterrows():
            f.write(
                f"{row['component']} & {row['perturbation']} & {row['new_value']} & "
                f"{row['power_w']:.3f} & {row['delta_tte_pct']:+.2f} \\\\\n"
            )

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"Saved: {tex_path}")

    # Print summary
    print("\nSensitivity Summary (ordered by impact):")
    df_sorted = df.sort_values("delta_tte_pct", key=abs, ascending=False)
    for _, row in df_sorted.iterrows():
        print(
            f"  {row['component']:20s} {row['perturbation']:>5s} → {row['delta_tte_pct']:+6.2f}% TTE change"
        )

    return df, csv_path, tex_path


def main():
    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results"
    )
    os.makedirs(results_dir, exist_ok=True)

    run_scenario_sensitivity(results_dir)


if __name__ == "__main__":
    main()
