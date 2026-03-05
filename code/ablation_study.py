"""
Ablation Study
==============
Systematically remove each module from the full model to quantify contribution:
  1. FULL: Shepherd + Temperature + Aging + Component Decomposition
  2. w/o Temperature Coupling (Arrhenius disabled, fixed T=25°C)
  3. w/o Aging (cycle_count=0 always)
  4. w/o Component Decomposition (single lumped power)
  5. w/o Shepherd Polarization (K=0, remove i/SOC term)

Metric: TTE prediction difference for 5 usage scenarios × 3 aging levels

Output: results/ablation_study.csv, results/ablation_study_table.tex
"""

import os
import sys
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main_model import MainBatteryModel
from aging_model import BatteryModel

# ==================== Scenarios (same as scenario_time_to_empty.py) ====================
SCENARIOS = {
    "Standby": {"screen": (0, 0.0), "soc": 10, "radio": 10, "gps": 10},
    "Navigation": {"screen": (70, 1.0), "soc": 30, "radio": 30, "gps": 100},
    "Gaming": {"screen": (90, 1.3), "soc": 100, "radio": 95, "gps": 10},
    "Video": {"screen": (80, 1.0), "soc": 30, "radio": 90, "gps": 10},
    "Reading": {"screen": (70, 1.0), "soc": 30, "radio": 20, "gps": 10},
}

AGING_LEVELS = [
    {"label": "New (0 cycles)", "cycle_count": 0, "calendar_hours": 0},
    {"label": "Moderate (300 cycles)", "cycle_count": 300, "calendar_hours": 4380},
    {"label": "Aged (800 cycles)", "cycle_count": 800, "calendar_hours": 11680},
]


def build_inputs(scenario, temperature_c=25):
    screen_percent, eta = scenario["screen"]
    screen_brightness = (screen_percent / 100.0) * eta
    cpu_load = min(scenario["soc"] / 100.0, 1.0)
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


def get_power_w(power_model, inputs):
    return power_model.get_power_consumption(0.0, inputs) / 1000.0


def compute_tte(main_model, power_w, temp_k=298.15, soc_initial=1.0):
    """Compute TTE via numerical integration."""
    out = main_model.predict_discharge(
        P_load=power_w, temp_k=temp_k, soc_initial=soc_initial
    )
    return out["discharge_time"]


def get_lumped_power(scenario_inputs):
    """Compute lumped power without component decomposition (simple empirical)."""
    # Sum up a generic current drain without decomposition
    # Just use average device power = 1.5W baseline (heuristic, no component model)
    brightness = scenario_inputs.get("brightness", 0.5)
    cpu_load = scenario_inputs.get("cpu_load", 0.3)
    net_activity = scenario_inputs.get("network_activity", 0.3)
    gps_on = int(scenario_inputs.get("gps", False))

    # Simple linear model: P = a0 + a1*brightness + a2*cpu + a3*net + a4*gps
    P_base_mw = 150   # Base standby
    P_screen = brightness * 800
    P_cpu = cpu_load * 1200
    P_net = net_activity * 500
    P_gps = gps_on * 300
    P_total_mw = P_base_mw + P_screen + P_cpu + P_net + P_gps
    return P_total_mw / 1000.0


# ==================== Ablation Configurations ====================
def run_full_model(scenario_name, scenario, aging_level, temperature_c=25):
    """FULL model: Shepherd + Temp + Aging + Component Decomposition."""
    power_model = BatteryModel(
        cycle_count=aging_level["cycle_count"],
        calendar_time_hours=aging_level["calendar_hours"]
    )
    aging_factors = power_model.get_aging_factors()

    main_model = MainBatteryModel(Q0=5.0)
    # Apply aging
    main_model.Q0 *= aging_factors['capacity_retention']
    main_model.R0 *= aging_factors['resistance_factor']

    inputs = build_inputs(scenario, temperature_c)
    power_w = get_power_w(power_model, inputs)
    temp_k = temperature_c + 273.15
    tte = compute_tte(main_model, power_w, temp_k)
    return tte, power_w


def run_no_temperature(scenario_name, scenario, aging_level, temperature_c=25):
    """Without temperature coupling: fix T=25°C in voltage model."""
    power_model = BatteryModel(
        cycle_count=aging_level["cycle_count"],
        calendar_time_hours=aging_level["calendar_hours"]
    )
    aging_factors = power_model.get_aging_factors()

    main_model = MainBatteryModel(Q0=5.0)
    main_model.Q0 *= aging_factors['capacity_retention']
    main_model.R0 *= aging_factors['resistance_factor']

    # Override temperature methods to always return 25°C values
    main_model.get_capacity_at_temp = lambda t: main_model.Q0
    main_model.get_resistance_at_temp = lambda t: main_model.R0

    inputs = build_inputs(scenario, temperature_c)
    power_w = get_power_w(power_model, inputs)
    tte = compute_tte(main_model, power_w, 298.15)  # Force 25°C
    return tte, power_w


def run_no_aging(scenario_name, scenario, aging_level, temperature_c=25):
    """Without aging: always fresh battery regardless of cycle count."""
    power_model = BatteryModel(cycle_count=0, calendar_time_hours=0)  # No aging
    main_model = MainBatteryModel(Q0=5.0)  # Fresh capacity

    inputs = build_inputs(scenario, temperature_c)
    power_w = get_power_w(power_model, inputs)
    temp_k = temperature_c + 273.15
    tte = compute_tte(main_model, power_w, temp_k)
    return tte, power_w


def run_no_decomposition(scenario_name, scenario, aging_level, temperature_c=25):
    """Without component decomposition: use lumped power model."""
    power_model = BatteryModel(
        cycle_count=aging_level["cycle_count"],
        calendar_time_hours=aging_level["calendar_hours"]
    )
    aging_factors = power_model.get_aging_factors()

    main_model = MainBatteryModel(Q0=5.0)
    main_model.Q0 *= aging_factors['capacity_retention']
    main_model.R0 *= aging_factors['resistance_factor']

    inputs = build_inputs(scenario, temperature_c)
    power_w = get_lumped_power(inputs)
    temp_k = temperature_c + 273.15
    tte = compute_tte(main_model, power_w, temp_k)
    return tte, power_w


def run_no_polarization(scenario_name, scenario, aging_level, temperature_c=25):
    """Without Shepherd K term: remove i/SOC polarization."""
    power_model = BatteryModel(
        cycle_count=aging_level["cycle_count"],
        calendar_time_hours=aging_level["calendar_hours"]
    )
    aging_factors = power_model.get_aging_factors()

    main_model = MainBatteryModel(Q0=5.0)
    main_model.Q0 *= aging_factors['capacity_retention']
    main_model.R0 *= aging_factors['resistance_factor']
    main_model.K = 0.0  # Remove polarization term

    inputs = build_inputs(scenario, temperature_c)
    power_w = get_power_w(power_model, inputs)
    temp_k = temperature_c + 273.15
    tte = compute_tte(main_model, power_w, temp_k)
    return tte, power_w


ABLATION_CONFIGS = {
    "Full Model": run_full_model,
    "w/o Temperature": run_no_temperature,
    "w/o Aging": run_no_aging,
    "w/o Component Decomp.": run_no_decomposition,
    "w/o Polarization (K=0)": run_no_polarization,
}


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    rows = []
    temperature_c = 25

    print("=" * 85)
    print("ABLATION STUDY: Module Contribution Analysis")
    print("=" * 85)

    for aging in AGING_LEVELS:
        print(f"\n--- {aging['label']} ---")
        for scenario_name, scenario in SCENARIOS.items():
            # Get full model TTE as reference
            tte_full, p_full = run_full_model(
                scenario_name, scenario, aging, temperature_c
            )

            for config_name, config_func in ABLATION_CONFIGS.items():
                tte, power_w = config_func(
                    scenario_name, scenario, aging, temperature_c
                )
                delta = tte - tte_full
                delta_pct = (delta / tte_full * 100) if tte_full > 0 else 0

                rows.append({
                    'Aging': aging['label'],
                    'Scenario': scenario_name,
                    'Configuration': config_name,
                    'TTE (h)': round(tte, 3),
                    'Power (W)': round(power_w, 4),
                    'Delta TTE (h)': round(delta, 3),
                    'Delta TTE (%)': round(delta_pct, 2),
                })

            # Print summary for this scenario
            full_row = [r for r in rows if r['Scenario'] == scenario_name
                        and r['Aging'] == aging['label']
                        and r['Configuration'] == 'Full Model'][-1]
            print(f"  {scenario_name:12s} Full={full_row['TTE (h)']:6.2f}h | ", end='')
            for config_name in list(ABLATION_CONFIGS.keys())[1:]:
                r = [r for r in rows if r['Scenario'] == scenario_name
                     and r['Aging'] == aging['label']
                     and r['Configuration'] == config_name][-1]
                print(f"{config_name.split('(')[0].strip():18s}={r['Delta TTE (%)']:+6.1f}%  ", end='')
            print()

    df = pd.DataFrame(rows)

    # Save full results
    csv_path = os.path.join(results_dir, 'ablation_study.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # ====== Create summary pivot table ======
    # Average delta across scenarios per configuration and aging level
    summary = df[df['Configuration'] != 'Full Model'].groupby(
        ['Configuration', 'Aging']
    ).agg({
        'Delta TTE (%)': 'mean',
        'Delta TTE (h)': 'mean',
    }).round(2).reset_index()

    summary_csv = os.path.join(results_dir, 'ablation_study_summary.csv')
    summary.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")

    # ====== LaTeX table ======
    tex_path = os.path.join(results_dir, 'ablation_study_table.tex')
    # Create a compact table: rows = configurations, columns = scenarios (for New battery)
    new_data = df[(df['Aging'] == 'New (0 cycles)')]

    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write('% Ablation Study Table\n')
        f.write('\\begin{table}[htbp]\n')
        f.write('\\centering\n')
        f.write('\\caption{Ablation study: TTE (hours) for new battery under different '
                'model configurations. Delta shows deviation from Full Model.}\n')
        f.write('\\label{tab:ablation_study}\n')
        scen_names = list(SCENARIOS.keys())
        cols = 'l' + 'r' * len(scen_names)
        f.write(f'\\begin{{tabular}}{{{cols}}}\n')
        f.write('\\hline\n')
        f.write('Configuration & ' + ' & '.join(scen_names) + ' \\\\\n')
        f.write('\\hline\n')

        for config_name in ABLATION_CONFIGS.keys():
            values = []
            for sn in scen_names:
                row = new_data[(new_data['Configuration'] == config_name) &
                               (new_data['Scenario'] == sn)].iloc[0]
                if config_name == 'Full Model':
                    values.append(f"{row['TTE (h)']:.2f}")
                else:
                    values.append(f"{row['TTE (h)']:.2f} ({row['Delta TTE (%)']:+.1f}\\%)")
            f.write(f"{config_name} & " + " & ".join(values) + " \\\\\n")

        f.write('\\hline\n')
        f.write('\\end{tabular}\n')
        f.write('\\end{table}\n')
    print(f"Saved: {tex_path}")

    # ====== Visualization ======
    fig, ax = plt.subplots(figsize=(12, 6))

    configs_no_full = [c for c in ABLATION_CONFIGS.keys() if c != 'Full Model']
    x = np.arange(len(SCENARIOS))
    width = 0.18
    colors = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981']

    for i, config in enumerate(configs_no_full):
        deltas = []
        for sn in SCENARIOS.keys():
            row = new_data[(new_data['Configuration'] == config) &
                           (new_data['Scenario'] == sn)].iloc[0]
            deltas.append(row['Delta TTE (%)'])
        ax.bar(x + i * width, deltas, width, label=config, color=colors[i], alpha=0.85)

    ax.set_xlabel('Usage Scenario')
    ax.set_ylabel('TTE Change (%)')
    ax.set_title('Ablation Study: Module Contribution (New Battery)')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(SCENARIOS.keys())
    ax.legend(fontsize=8, loc='best')
    ax.axhline(0, color='k', ls='--', lw=0.8)
    ax.grid(axis='y', alpha=0.3)

    fig_path = os.path.join(results_dir, 'fig_ablation_study.png')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    plt.close()


if __name__ == '__main__':
    main()
