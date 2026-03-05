"""
Assumption and sensitivity analysis for the continuous-time battery model.

*** TIME TO EMPTY (TTE) CALCULATION METHOD ***
TTE is computed by numerically integrating the battery discharge ODE system:

    dSOC/dt = -I(t) / Q₀(T, N)                    [State equation]
    V(t) = f(SOC(t), I(t), T, N)                  [Shepherd voltage model]
    I(t) = P(t) / V(t)                            [Power constraint]

where:
    - Q₀(T, N): effective capacity (temperature & aging dependent)
    - V(t): terminal voltage (function of SOC, current, temperature, aging)
    - P(t): power load (constant or time-varying)
    - N: cycle count (aging state)

Integration stops when:
    1. SOC ≤ 5% (state-based cutoff)
    2. V(t) < V_cutoff (voltage-based cutoff)
    3. t > t_max (safety limit)

The discharge time is:
    t_empty = ∫₀^{t_cutoff} dt

Sensitivity analysis perturbs model parameters and recomputes TTE via the same
numerical integration, yielding ΔT_empty / T_empty for each parameter.

Outputs:
- results/fig_aging_grid.png
- results/fig_usage_patterns.png
- results/assumption_sensitivity.csv
- results/assumption_sensitivity_table.tex
- results/usage_pattern_comparison.csv
"""

import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import aging_model as am
from aging_model import BatteryModel
from main_model import MainBatteryModel


BASELINE_SCENARIO = {
    "screen": (80, 1.0),
    "soc": 20,
    "radio": 90,
    "gps": 10,
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
    Simulate discharge and return TTE via numerical integration.

    This function calls MainBatteryModel.predict_discharge(), which integrates:
        dSOC/dt = -I(t) / Q₀(T)

    using Euler's method with adaptive current smoothing. The integration
    continues until SOC reaches cutoff or voltage drops below V_cutoff.

    Returns:
        (discharge_time, time_array, soc_array)
    """
    out = main_model.predict_discharge(
        P_load=power_w, temp_k=temp_k, soc_initial=soc_init
    )
    return out["discharge_time"], out["time"], out["soc"]


def run_aging_grid(results_dir):
    inputs = build_inputs(BASELINE_SCENARIO)
    power_model = BatteryModel(cycle_count=0, calendar_time_hours=0)
    main_model = MainBatteryModel(Q0=1.991)
    cycles = [0, 200, 500]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)

    rows = []
    for idx, cyc in enumerate(cycles):
        model = BatteryModel(cycle_count=cyc, calendar_time_hours=0)
        power_w = compute_power_w(model, inputs)
        tte, t, soc = simulate_tte(
            main_model, power_w, inputs["temperature"], soc_init=1.0
        )
        axes[idx].plot(t, soc, linewidth=2)
        axes[idx].set_title(f"Cycles={cyc}")
        axes[idx].set_xlabel("Time (h)")
        axes[idx].grid(alpha=0.3)
        rows.append({"cycle_count": cyc, "tte_hours": tte})

    axes[0].set_ylabel("SOC")
    fig.suptitle("SOC Trajectories Under Aging Levels", fontsize=12)
    fig.tight_layout()

    fig_path = os.path.join(results_dir, "fig_aging_grid.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(results_dir, "aging_levels.csv")
    df.to_csv(csv_path, index=False)

    return rows, fig_path, csv_path


def run_parameter_sensitivity(results_dir):
    """
    Conduct parameter sensitivity analysis by perturbing key model parameters
    by ±5% and measuring the relative change in TTE.

    For each parameter θ:
        1. Compute baseline TTE: t₀ = ∫ dt (with θ = θ₀)
        2. Perturb: θ' = θ₀ × (1 ± 0.05)
        3. Re-integrate to get t': t' = ∫ dt (with θ = θ')
        4. Compute sensitivity: Δt/t = (t' - t₀) / t₀

    This reveals which parameters have the strongest influence on discharge time.
    """
    base_inputs = build_inputs(BASELINE_SCENARIO)

    power_model = BatteryModel(cycle_count=0, calendar_time_hours=0)
    main_model = MainBatteryModel(Q0=1.991)

    base_power = compute_power_w(power_model, base_inputs)
    base_tte, _, _ = simulate_tte(main_model, base_power, base_inputs["temperature"])

    params = {
        "K_SCREEN_BRIGHTNESS": "screen_brightness",
        "K_CPU_LOAD": "cpu_load",
        "P_4G_ACTIVE": "radio",
        "P_GPS_ON": "gps",
        "P_BASE_IDLE": "baseline",
    }

    rows = []

    for param_name in params.keys():
        for factor in [0.95, 1.05]:
            # Patch aging_model constants
            original_value = getattr(am, param_name)
            setattr(am, param_name, original_value * factor)

            power_model = BatteryModel(cycle_count=0, calendar_time_hours=0)
            power_w = compute_power_w(power_model, base_inputs)
            tte, _, _ = simulate_tte(main_model, power_w, base_inputs["temperature"])
            delta = (tte / base_tte - 1.0) * 100

            rows.append(
                {
                    "parameter": param_name,
                    "factor": factor,
                    "tte_hours": tte,
                    "delta_pct": delta,
                }
            )

            setattr(am, param_name, original_value)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(results_dir, "assumption_sensitivity.csv")
    df.to_csv(csv_path, index=False)

    tex_path = os.path.join(results_dir, "assumption_sensitivity_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Parameter sensitivity table\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Parameter sensitivity under $\\pm5\\%$ perturbations}\n")
        f.write("\\label{tab:param_sensitivity}\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\hline\n")
        f.write(
            "Parameter & -5\% $\\Delta t$ (\%) & +5\% $\\Delta t$ (\%) & Baseline $t_{\\mathrm{empty}}$ (h)\\\\\n"
        )
        f.write("\\hline\n")

        for param in params.keys():
            sub = df[df["parameter"] == param].sort_values("factor")
            neg = sub.iloc[0]["delta_pct"]
            pos = sub.iloc[1]["delta_pct"]
            f.write(f"{param} & {neg:.2f} & {pos:.2f} & {base_tte:.2f}\\\\\n")

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    return df, csv_path, tex_path


def make_switching_inputs(schedule, temperature_c=25):
    def make_func(key):
        def fn(t):
            elapsed = 0.0
            for duration, scenario in schedule:
                if t < elapsed + duration:
                    inputs = build_inputs(
                        scenario, background_extra=0.0, temperature_c=temperature_c
                    )
                    if key == "brightness":
                        return inputs["brightness"]
                    if key == "cpu_load":
                        return inputs["cpu_load"]
                    if key == "network_activity":
                        return inputs["network_activity"]
                    if key == "gps":
                        return inputs["gps"]
                elapsed += duration
            inputs = build_inputs(
                schedule[-1][1], background_extra=0.0, temperature_c=temperature_c
            )
            if key == "brightness":
                return inputs["brightness"]
            if key == "cpu_load":
                return inputs["cpu_load"]
            if key == "network_activity":
                return inputs["network_activity"]
            if key == "gps":
                return inputs["gps"]
            return 0.0

        return fn

    return {
        "brightness": make_func("brightness"),
        "cpu_load": make_func("cpu_load"),
        "network_mode": "4g",
        "network_activity": make_func("network_activity"),
        "gps": make_func("gps"),
        "temperature": temperature_c + 273.15,
    }


def run_usage_pattern(results_dir):
    power_model = BatteryModel(cycle_count=0, calendar_time_hours=0)
    main_model = MainBatteryModel(Q0=1.991)

    steady_inputs = build_inputs(BASELINE_SCENARIO)
    steady_power = compute_power_w(power_model, steady_inputs)
    steady_tte, steady_t, steady_soc = simulate_tte(
        main_model, steady_power, steady_inputs["temperature"]
    )

    switching_schedule = [
        (0.5, BASELINE_SCENARIO),
        (0.25, {"screen": (70, 1.0), "soc": 20, "radio": 20, "gps": 10}),
    ]
    switching_inputs = make_switching_inputs(switching_schedule)
    switching_power = compute_power_w(power_model, switching_inputs)
    switching_tte, switching_t, switching_soc = simulate_tte(
        main_model, switching_power, switching_inputs["temperature"]
    )

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(steady_t, steady_soc, label="Steady (Video)")
    ax.plot(switching_t, switching_soc, label="Switching (Video/Reading)")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("SOC")
    ax.set_title("Usage Pattern Effects on SOC")
    ax.grid(alpha=0.3)
    ax.legend()

    fig_path = os.path.join(results_dir, "fig_usage_patterns.png")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    df = pd.DataFrame(
        [
            {"pattern": "steady_video", "tte_hours": steady_tte},
            {"pattern": "switching_video_reading", "tte_hours": switching_tte},
        ]
    )

    csv_path = os.path.join(results_dir, "usage_pattern_comparison.csv")
    df.to_csv(csv_path, index=False)

    return df, csv_path, fig_path


def run_tte_by_soc(results_dir):
    power_model = BatteryModel(cycle_count=0, calendar_time_hours=0)
    main_model = MainBatteryModel(Q0=1.991)

    soc_levels = np.linspace(1.0, 0.1, 100)
    background_levels = [0.0, 0.2, 0.4]

    tte_matrix = {}
    for bg in background_levels:
        ttes = []
        inputs = build_inputs(BASELINE_SCENARIO, background_extra=0.0)
        base_power_w = compute_power_w(power_model, inputs)
        power_w = base_power_w * (1.0 + bg)
        for soc0 in soc_levels:
            tte, _, _ = simulate_tte(
                main_model, power_w, inputs["temperature"], soc_init=soc0
            )
            ttes.append(tte)
        tte_matrix[f"tte_bg{int(bg * 100)}"] = ttes

    df = pd.DataFrame({"soc0": soc_levels, **tte_matrix})
    csv_path = os.path.join(results_dir, "tte_by_soc.csv")
    df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    soc_percent = soc_levels * 100
    for key, label, style in [
        ("tte_bg0", "Background +0%", "-"),
        ("tte_bg20", "Background +20%", "--"),
        ("tte_bg40", "Background +40%", ":"),
    ]:
        ax.plot(soc_percent, df[key].values, linestyle=style, linewidth=2, label=label)

    drops = df["tte_bg0"].values[:-1] - df["tte_bg0"].values[1:]
    fastest_idx = int(np.argmax(drops))
    soc_hi = soc_percent[fastest_idx]
    soc_lo = soc_percent[fastest_idx + 1]
    drop_val = drops[fastest_idx]

    ax.axvspan(soc_lo, soc_hi, color="orange", alpha=0.15)
    ax.text(
        (soc_hi + soc_lo) / 2.0,
        df["tte_bg0"].max() * 0.9,
        f"Fastest 10% drop: {soc_hi:.0f}%→{soc_lo:.0f}%\nΔt={drop_val:.2f} h",
        ha="center",
        va="top",
        fontsize=9,
        color="darkorange",
    )

    ax.set_xlabel("Initial SOC (%)")
    ax.set_ylabel("Remaining Usage Time (h)")
    ax.set_title("Remaining Time vs. Initial SOC")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_xlim(100, 10)
    ax.set_xticks([100, 80, 60, 40, 20, 10])
    ax.set_xticklabels(["100%", "80%", "60%", "40%", "20%", "10%"])

    fig_path = os.path.join(results_dir, "fig_tte_by_soc.png")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    return df, csv_path, fig_path


def main():
    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results"
    )
    os.makedirs(results_dir, exist_ok=True)

    _, fig_aging, csv_aging = run_aging_grid(results_dir)
    print(f"Saved: {fig_aging}")
    print(f"Saved: {csv_aging}")

    _, csv_param, tex_param = run_parameter_sensitivity(results_dir)
    print(f"Saved: {csv_param}")
    print(f"Saved: {tex_param}")

    _, csv_pattern, fig_pattern = run_usage_pattern(results_dir)
    print(f"Saved: {csv_pattern}")
    print(f"Saved: {fig_pattern}")

    _, csv_tte, fig_tte = run_tte_by_soc(results_dir)
    print(f"Saved: {csv_tte}")
    print(f"Saved: {fig_tte}")


if __name__ == "__main__":
    main()
