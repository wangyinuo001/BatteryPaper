"""
Master script: Generate ALL publication-quality figures.
Run from the code/ directory.
"""

import os, sys, importlib, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SCRIPTS = [
    ("fig_voltage_trajectory", "Fig 1: Voltage trajectory comparison"),
    ("fig_cross_batch", "Fig 2: Cross-batch generalisation"),
    ("fig_ablation_heatmap", "Fig 3: Ablation heatmap"),
    ("fig_temperature_tte", "Fig 4: Temperature–TTE curve"),
    ("fig_sensitivity_tornado", "Fig 5: Sensitivity tornado"),
    ("fig_scenario_comparison", "Fig 6: Scenario TTE + power"),
    ("fig_aging_curves", "Fig 7: Aging degradation"),
    ("fig_dynamic_scenario", "Fig 8: Dynamic mixed-usage"),
    ("fig_residual_analysis", "Fig 9: Residual analysis + SOC-segmented error"),
    ("fig_soc_scenarios", "Fig 10: SOC discharge trajectories"),
    ("fig_energy_management_v2", "Fig 11: Adaptive energy management"),
]


def main():
    t0 = time.time()
    for mod_name, desc in SCRIPTS:
        print(f"\n{'='*60}")
        print(f"  {desc}")
        print(f"{'='*60}")
        try:
            mod = importlib.import_module(mod_name)
            mod.main()
            print(f"  ✓ {desc} completed.")
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback

            traceback.print_exc()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  All figures generated in {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
