"""
Sensitivity analysis for Main Model predictions.
Exports CSV/TeX tables for parameter and usage pattern perturbations.
"""

import os
import numpy as np
import pandas as pd
from main_model import MainBatteryModel


def run_sensitivity():
    model = MainBatteryModel(
        Q0=5.0
    )  # 5000mAh smartphone battery, consistent with scenarios
    results = []

    # Baseline scenario
    baseline_power = 5.0  # W
    baseline_temp_c = 25
    baseline_temp_k = baseline_temp_c + 273.15

    base = model.predict_discharge(P_load=baseline_power, temp_k=baseline_temp_k)
    results.append(
        {
            "case": "baseline",
            "category": "baseline",
            "power_w": baseline_power,
            "temp_c": baseline_temp_c,
            "q0_ah": model.Q0,
            "discharge_time_h": base["discharge_time"],
        }
    )

    # Temperature sensitivity
    for temp_c in [-10, 0, 10, 25, 40, 50]:
        temp_k = temp_c + 273.15
        out = model.predict_discharge(P_load=baseline_power, temp_k=temp_k)
        results.append(
            {
                "case": f"temp_{temp_c}C",
                "category": "temperature",
                "power_w": baseline_power,
                "temp_c": temp_c,
                "q0_ah": model.Q0,
                "discharge_time_h": out["discharge_time"],
            }
        )

    # Capacity assumption sensitivity (Q0 ±10%)
    for factor in [0.9, 1.0, 1.1]:
        temp_k = baseline_temp_k
        q0 = model.Q0 * factor
        local = MainBatteryModel(Q0=q0)
        out = local.predict_discharge(P_load=baseline_power, temp_k=temp_k)
        results.append(
            {
                "case": f"q0_{int(factor*100)}pct",
                "category": "capacity",
                "power_w": baseline_power,
                "temp_c": baseline_temp_c,
                "q0_ah": q0,
                "discharge_time_h": out["discharge_time"],
            }
        )

    # Usage pattern fluctuations: power +/- 20%
    for factor in [0.8, 1.0, 1.2]:
        power = baseline_power * factor
        out = model.predict_discharge(P_load=power, temp_k=baseline_temp_k)
        results.append(
            {
                "case": f"power_{int(factor*100)}pct",
                "category": "power",
                "power_w": power,
                "temp_c": baseline_temp_c,
                "q0_ah": model.Q0,
                "discharge_time_h": out["discharge_time"],
            }
        )

    df = pd.DataFrame(results)

    # Add relative change vs baseline
    baseline_time = df.loc[df["case"] == "baseline", "discharge_time_h"].iloc[0]
    df["delta_vs_baseline_pct"] = (df["discharge_time_h"] / baseline_time - 1) * 100

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results")
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, "sensitivity_analysis.csv")
    df.to_csv(csv_path, index=False)

    # Export a compact TeX table
    tex_path = os.path.join(results_dir, "sensitivity_analysis_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Sensitivity Analysis Table\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write(
            "\\caption{Sensitivity of discharge time to temperature, capacity, and power load}\n"
        )
        f.write("\\label{tab:sensitivity}\n")
        f.write("\\begin{tabular}{llrrrr}\n")
        f.write("\\hline\n")
        f.write(
            "Case & Category & Power (W) & Temp ($^\\circ$C) & $Q_0$ (Ah) & $\\Delta t$ (\\%)\\\\\n"
        )
        f.write("\\hline\n")
        for _, row in df.iterrows():
            f.write(
                f"{row['case']} & {row['category']} & {row['power_w']:.2f} & {row['temp_c']:.0f} & {row['q0_ah']:.3f} & {row['delta_vs_baseline_pct']:.1f}\\\\\n"
            )
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"Saved: {csv_path}")
    print(f"Saved: {tex_path}")


if __name__ == "__main__":
    run_sensitivity()
