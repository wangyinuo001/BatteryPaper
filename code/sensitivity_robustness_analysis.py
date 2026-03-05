"""
Sensitivity and robustness analysis for assumptions, parameters, and usage patterns.

Outputs:
- results/aging_assumption_impact.csv
- results/aging_assumption_impact_table.tex
- results/parameter_sensitivity_physical.csv
- results/parameter_sensitivity_physical_table.tex
- results/usage_pattern_order.csv
- results/usage_pattern_background.csv
"""

import os
import numpy as np
import pandas as pd
import main_model as mm
from main_model import MainBatteryModel
from aging_model import BatteryModel, R_INTERNAL_NOMINAL


TEMP_C = 25
TEMP_K = TEMP_C + 273.15
BASE_Q0_AH = 5.0
HOURS_PER_CYCLE = 2.0

GAMING_SCENARIO = {"screen": (90, 1.3), "soc": 95, "radio": 95, "gps": 10}
READING_SCENARIO = {"screen": (70, 1.0), "soc": 20, "radio": 20, "gps": 10}
STREAMING_SCENARIO = {"screen": (80, 1.0), "soc": 30, "radio": 90, "gps": 10}


def build_inputs(scenario, temperature_c=25):
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


def power_from_scenario(power_model, scenario, temperature_c=25):
    inputs = build_inputs(scenario, temperature_c=temperature_c)
    p_mw = power_model.get_power_consumption(0.0, inputs)
    return p_mw / 1000.0


def simulate_to_soc(
    main_model,
    power_w,
    temp_k,
    soc_initial,
    soc_target,
    dt=1.0,
):
    Q_eff = main_model.get_capacity_at_temp(temp_k)
    soc0 = float(np.clip(soc_initial, 0.01, 1.0))
    soc_target = float(np.clip(soc_target, 0.01, 1.0))
    t_max = 30.0

    time_points = [0.0]
    soc_points = [soc0]
    voltage_points = [main_model.terminal_voltage(soc0, 0, temp_k)]
    current_points = [0.0]

    t = 0.0
    soc = soc0
    I_avg = power_w / 3.7

    while soc > soc_target and soc > 0.05 and t < t_max * 3600:
        V_term = main_model.terminal_voltage(soc, I_avg, temp_k)

        if V_term < main_model.V_cutoff:
            break

        I_new = power_w / V_term if V_term > 0 else I_avg
        I_avg = 0.9 * I_avg + 0.1 * I_new

        dsoc = -I_avg * dt / (Q_eff * 3600)
        soc += dsoc
        t += dt

        time_points.append(t / 3600.0)
        soc_points.append(soc)
        voltage_points.append(V_term)
        current_points.append(I_avg)

    return {
        "time": np.array(time_points),
        "soc": np.array(soc_points),
        "voltage": np.array(voltage_points),
        "current": np.array(current_points),
        "discharge_time": time_points[-1],
        "temp_k": temp_k,
        "capacity": Q_eff,
    }


def run_parameter_sensitivity(results_dir, base_power_w):
    base_r0 = mm.SHEPHERD_R0
    base_params = {
        "E0": mm.SHEPHERD_E0,
        "K": mm.SHEPHERD_K,
        "A": mm.SHEPHERD_A,
        "B": mm.SHEPHERD_B,
        "R0": base_r0,
    }

    base_model = MainBatteryModel(
        Q0=BASE_Q0_AH,
        E0=base_params["E0"],
        K=base_params["K"],
        A=base_params["A"],
        B=base_params["B"],
        R0=base_params["R0"],
    )
    base_tte = base_model.predict_discharge(base_power_w, temp_k=TEMP_K)[
        "discharge_time"
    ]

    rows = []
    for param in ["E0", "K", "R0", "A", "B", "Q0", "P_total"]:
        for factor in [0.95, 1.05]:
            q0 = BASE_Q0_AH
            power_w = base_power_w
            params = dict(base_params)

            if param == "Q0":
                q0 = BASE_Q0_AH * factor
            elif param == "P_total":
                power_w = base_power_w * factor
            else:
                params[param] = base_params[param] * factor

            model = MainBatteryModel(
                Q0=q0,
                E0=params["E0"],
                K=params["K"],
                A=params["A"],
                B=params["B"],
                R0=params["R0"],
            )
            tte = model.predict_discharge(power_w, temp_k=TEMP_K)["discharge_time"]
            delta = (tte / base_tte - 1.0) * 100

            rows.append(
                {
                    "parameter": param,
                    "factor": factor,
                    "tte_hours": tte,
                    "delta_pct": delta,
                    "baseline_tte_hours": base_tte,
                }
            )

    df = pd.DataFrame(rows)
    wide_rows = []
    for param in ["E0", "K", "R0", "A", "B", "Q0", "P_total"]:
        sub = df[df["parameter"] == param].sort_values("factor")
        neg = sub.iloc[0]["delta_pct"]
        pos = sub.iloc[1]["delta_pct"]
        tte_neg = sub.iloc[0]["tte_hours"]
        tte_pos = sub.iloc[1]["tte_hours"]
        wide_rows.append(
            {
                "parameter": param,
                "tte_neg_hours": tte_neg,
                "tte_pos_hours": tte_pos,
                "delta_pct_neg": neg,
                "delta_pct_pos": pos,
                "sensitivity_neg": neg / 5.0,
                "sensitivity_pos": pos / 5.0,
                "baseline_tte_hours": base_tte,
            }
        )

    wide_df = pd.DataFrame(wide_rows)
    csv_path = os.path.join(results_dir, "parameter_sensitivity_physical.csv")
    wide_df.to_csv(csv_path, index=False)

    tex_path = os.path.join(results_dir, "parameter_sensitivity_physical_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Physical parameter sensitivity table\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write(
            "\\caption{Sensitivity of TTE to $\\pm5\\%$ perturbations of physical parameters}\n"
        )
        f.write("\\label{tab:physical_sensitivity}\n")
        f.write("\\begin{tabular}{lrrrrrr}\n")
        f.write("\\hline\n")
        f.write(
            "Parameter & $t_{-5\\%}$ (h) & $t_{+5\\%}$ (h) & -5\\% $\\Delta t$ (\\%) & +5\\% $\\Delta t$ (\\%) & $t_0$ (h)\\\\\\n"
        )
        f.write("\\hline\n")

        for param in ["E0", "K", "R0", "A", "B", "Q0", "P_total"]:
            row = wide_df[wide_df["parameter"] == param].iloc[0]
            f.write(
                f"{param} & {row['tte_neg_hours']:.3f} & {row['tte_pos_hours']:.3f} "
                f"& {row['delta_pct_neg']:.2f} & {row['delta_pct_pos']:.2f} & {row['baseline_tte_hours']:.3f}\\\\\\n"
            )

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    return df, csv_path, tex_path, base_tte


def run_aging_assumption(results_dir, base_power_w):
    cycles = [0, 200, 500, 1000]
    rows = []

    for cyc in cycles:
        calendar_hours = cyc * HOURS_PER_CYCLE
        aging = BatteryModel(
            q_nom_mah=BASE_Q0_AH * 1000.0,
            cycle_count=cyc,
            calendar_time_hours=calendar_hours,
        )
        q_eff_mah, r_eff = aging.get_effective_parameters(TEMP_K)
        model = MainBatteryModel(
            Q0=q_eff_mah / 1000.0,
            E0=mm.SHEPHERD_E0,
            K=mm.SHEPHERD_K,
            A=mm.SHEPHERD_A,
            B=mm.SHEPHERD_B,
            R0=r_eff,
        )
        tte = model.predict_discharge(base_power_w, temp_k=TEMP_K)["discharge_time"]
        rows.append(
            {
                "cycle_count": cyc,
                "calendar_hours": calendar_hours,
                "q0_ah": q_eff_mah / 1000.0,
                "r0_ohm": r_eff,
                "tte_hours": tte,
            }
        )

    df = pd.DataFrame(rows)
    base_tte = df.loc[df["cycle_count"] == 0, "tte_hours"].iloc[0]
    df["delta_vs_fresh_pct"] = (df["tte_hours"] / base_tte - 1.0) * 100

    csv_path = os.path.join(results_dir, "aging_assumption_impact.csv")
    df.to_csv(csv_path, index=False)

    tex_path = os.path.join(results_dir, "aging_assumption_impact_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Aging assumption impact table\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Impact of aging (cycle count) on $Q_0$, $R_0$, and TTE}\n")
        f.write("\\label{tab:aging_impact}\n")
        f.write("\\begin{tabular}{rrrrr}\n")
        f.write("\\hline\n")
        f.write("Cycles & Calendar (h) & $Q_0$ (Ah) & $R_0$ ($\\Omega$) & $\\Delta t$ (\\%)\\\\\n")
        f.write("\\hline\n")

        for _, row in df.iterrows():
            f.write(
                f"{int(row['cycle_count'])} & {row['calendar_hours']:.0f} & "
                f"{row['q0_ah']:.3f} & {row['r0_ohm']:.4f} & {row['delta_vs_fresh_pct']:.2f}\\\\\n"
            )

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    return df, csv_path, tex_path


def run_usage_pattern(results_dir):
    power_model = BatteryModel(
        q_nom_mah=BASE_Q0_AH * 1000.0,
        cycle_count=0,
        calendar_time_hours=0,
    )
    gaming_power = power_from_scenario(power_model, GAMING_SCENARIO, TEMP_C)
    reading_power = power_from_scenario(power_model, READING_SCENARIO, TEMP_C)

    base_model = MainBatteryModel(
        Q0=BASE_Q0_AH,
        E0=mm.SHEPHERD_E0,
        K=mm.SHEPHERD_K,
        A=mm.SHEPHERD_A,
        B=mm.SHEPHERD_B,
        R0=mm.SHEPHERD_R0,
    )

    soc_mid = 0.525
    soc_end = 0.05

    gaming_first = simulate_to_soc(
        base_model,
        gaming_power,
        TEMP_K,
        soc_initial=1.0,
        soc_target=soc_mid,
    )
    reading_second = simulate_to_soc(
        base_model,
        reading_power,
        TEMP_K,
        soc_initial=gaming_first["soc"][-1],
        soc_target=soc_end,
    )
    tte_gaming_reading = (
        gaming_first["discharge_time"] + reading_second["discharge_time"]
    )

    reading_first = simulate_to_soc(
        base_model,
        reading_power,
        TEMP_K,
        soc_initial=1.0,
        soc_target=soc_mid,
    )
    gaming_second = simulate_to_soc(
        base_model,
        gaming_power,
        TEMP_K,
        soc_initial=reading_first["soc"][-1],
        soc_target=soc_end,
    )
    tte_reading_gaming = (
        reading_first["discharge_time"] + gaming_second["discharge_time"]
    )

    gaming_only = simulate_to_soc(
        base_model,
        gaming_power,
        TEMP_K,
        soc_initial=1.0,
        soc_target=soc_end,
    )
    reading_only = simulate_to_soc(
        base_model,
        reading_power,
        TEMP_K,
        soc_initial=1.0,
        soc_target=soc_end,
    )

    df_order = pd.DataFrame(
        [
            {
                "pattern": "gaming_only",
                "soc_split": "100%-5%",
                "gaming_power_w": gaming_power,
                "reading_power_w": reading_power,
                "tte_hours": gaming_only["discharge_time"],
            },
            {
                "pattern": "reading_only",
                "soc_split": "100%-5%",
                "gaming_power_w": gaming_power,
                "reading_power_w": reading_power,
                "tte_hours": reading_only["discharge_time"],
            },
            {
                "pattern": "gaming_then_reading",
                "soc_split": "100%-52.5%-5%",
                "gaming_power_w": gaming_power,
                "reading_power_w": reading_power,
                "tte_hours": tte_gaming_reading,
            },
            {
                "pattern": "reading_then_gaming",
                "soc_split": "100%-52.5%-5%",
                "gaming_power_w": gaming_power,
                "reading_power_w": reading_power,
                "tte_hours": tte_reading_gaming,
            },
        ]
    )
    base_ref = df_order.loc[
        df_order["pattern"] == "gaming_then_reading", "tte_hours"
    ].iloc[0]
    df_order["delta_vs_gaming_first_pct"] = (
        df_order["tte_hours"] / base_ref - 1.0
    ) * 100

    order_csv = os.path.join(results_dir, "usage_pattern_order.csv")
    df_order.to_csv(order_csv, index=False)

    streaming_power = power_from_scenario(power_model, STREAMING_SCENARIO, TEMP_C)
    background_factor = 1.2
    background_power = streaming_power * background_factor

    base_tte = base_model.predict_discharge(streaming_power, temp_k=TEMP_K)[
        "discharge_time"
    ]
    bg_tte = base_model.predict_discharge(background_power, temp_k=TEMP_K)[
        "discharge_time"
    ]

    df_background = pd.DataFrame(
        [
            {
                "pattern": "baseline",
                "power_w": streaming_power,
                "background_factor": 1.0,
                "tte_hours": base_tte,
            },
            {
                "pattern": "background_apps",
                "power_w": background_power,
                "background_factor": background_factor,
                "tte_hours": bg_tte,
            },
        ]
    )
    df_background["delta_vs_baseline_pct"] = (
        df_background["tte_hours"] / df_background.iloc[0]["tte_hours"] - 1.0
    ) * 100

    background_csv = os.path.join(results_dir, "usage_pattern_background.csv")
    df_background.to_csv(background_csv, index=False)

    return df_order, order_csv, df_background, background_csv


def main():
    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results"
    )
    os.makedirs(results_dir, exist_ok=True)

    power_model = BatteryModel(
        q_nom_mah=BASE_Q0_AH * 1000.0,
        cycle_count=0,
        calendar_time_hours=0,
    )
    base_power = power_from_scenario(power_model, STREAMING_SCENARIO, TEMP_C)

    _, aging_csv, aging_tex = run_aging_assumption(results_dir, base_power)
    print(f"Saved: {aging_csv}")
    print(f"Saved: {aging_tex}")

    _, param_csv, param_tex, _ = run_parameter_sensitivity(results_dir, base_power)
    print(f"Saved: {param_csv}")
    print(f"Saved: {param_tex}")

    _, pattern_csv, _, background_csv = run_usage_pattern(results_dir)
    print(f"Saved: {pattern_csv}")
    print(f"Saved: {background_csv}")


if __name__ == "__main__":
    main()
