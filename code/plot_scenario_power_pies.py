"""
Generate individual power breakdown pie charts for ALL scenarios.
Shows the actual power distribution for each usage scenario.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from aging_model import BatteryModel

# All scenarios from Table 6
SCENARIOS = {
    "Standby": {"screen": (0, 0), "soc": 0, "radio": 10, "gps": 10},
    "Navigation": {"screen": (70, 1.0), "soc": 20, "radio": 30, "gps": 100},
    "Gaming": {"screen": (90, 1.3), "soc": 95, "radio": 95, "gps": 10},
    "Video": {"screen": (80, 1.0), "soc": 20, "radio": 90, "gps": 10},
    "Reading": {"screen": (70, 1.0), "soc": 20, "radio": 20, "gps": 10},
}


def build_inputs(scenario, background_extra=0.1, temperature_c=25):
    """Convert scenario to model inputs."""
    screen_percent, eta = scenario["screen"]
    screen_brightness_B = screen_percent / 100.0
    screen_eta = eta
    cpu_load = min((scenario["soc"] / 100.0) + background_extra, 1.0)
    radio_activity = scenario["radio"] / 100.0
    gps_on = scenario["gps"] > 0
    
    return {
        "brightness_B": screen_brightness_B,
        "brightness_eta": screen_eta,
        "cpu_load": cpu_load,
        "network_mode": "4g",
        "network_activity": radio_activity,
        "gps": gps_on,
        "temperature": temperature_c + 273.15,
    }


def get_component_powers(scenario_def):
    """Get individual component powers for a scenario."""
    inputs = build_inputs(scenario_def)
    power_model = BatteryModel(cycle_count=0, calendar_time_hours=0)
    
    # Calculate individual components
    screen_percent, eta = scenario_def["screen"]
    B = screen_percent / 100.0
    p_screen = eta * (0.4991 * B + 0.1113) * 1000  # mW
    
    cpu_load = inputs["cpu_load"]
    p_soc = cpu_load * 2040  # mW
    
    radio_activity = inputs["network_activity"]
    p_radio = radio_activity * 1173  # mW
    
    p_gps = 300 if inputs["gps"] else 0  # mW
    
    p_baseline = 120  # mW
    
    p_total = p_screen + p_soc + p_radio + p_gps + p_baseline
    
    return {
        "Screen": p_screen,
        "SoC": p_soc,
        "Radio": p_radio,
        "GPS": p_gps,
        "Baseline": p_baseline,
        "Total": p_total
    }


def create_scenario_pie(scenario_name, scenario_def, ax):
    """Create a pie chart for one scenario."""
    powers = get_component_powers(scenario_def)
    
    # Prepare data (exclude zero values)
    components = []
    values = []
    colors_map = {
        "Screen": "#FF6B6B",
        "SoC": "#4ECDC4",
        "Radio": "#45B7D1",
        "GPS": "#FFA07A",
        "Baseline": "#95E1D3"
    }
    colors = []
    
    for comp in ["Screen", "SoC", "Radio", "GPS", "Baseline"]:
        if powers[comp] > 0:
            components.append(comp)
            values.append(powers[comp])
            colors.append(colors_map[comp])
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        values,
        labels=components,
        colors=colors,
        autopct=lambda pct: f'{pct:.1f}%\n({pct*powers["Total"]/100:.0f}mW)' if pct > 3 else '',
        startangle=90,
        textprops={'fontsize': 10, 'weight': 'bold'},
        pctdistance=0.75
    )
    
    # Enhance text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(9)
        autotext.set_weight('bold')
    
    # Title with scenario info
    screen_percent, eta = scenario_def["screen"]
    title = f"{scenario_name}\n"
    title += f"Screen: {screen_percent}%, η={eta} | SoC: {scenario_def['soc']}%\n"
    title += f"Radio: {scenario_def['radio']}% | GPS: {scenario_def['gps']}%\n"
    title += f"Total: {powers['Total']:.1f}mW = {powers['Total']/1000:.2f}W"
    
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    
    return powers


def main():
    """Generate individual pie charts for all scenarios."""
    
    # Create 2x3 subplot grid (5 scenarios)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    print("=" * 70)
    print("POWER BREAKDOWN PIE CHARTS - ALL SCENARIOS")
    print("=" * 70)
    
    for idx, (scenario_name, scenario_def) in enumerate(SCENARIOS.items()):
        ax = axes[idx]
        powers = create_scenario_pie(scenario_name, scenario_def, ax)
        
        print(f"\n{scenario_name}:")
        for comp in ["Screen", "SoC", "Radio", "GPS", "Baseline"]:
            if powers[comp] > 0:
                pct = powers[comp] / powers["Total"] * 100
                print(f"  {comp:10s}: {powers[comp]:7.1f}mW ({pct:5.1f}%)")
        print(f"  {'Total':10s}: {powers['Total']:7.1f}mW")
    
    # Hide the last subplot (we only have 5 scenarios)
    axes[5].axis('off')
    
    # Overall title
    fig.suptitle(
        'Power Consumption Breakdown by Usage Scenario\n'
        'Battery: 5000mAh (Samsung Galaxy S25 Ultra)',
        fontsize=16,
        fontweight='bold',
        y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    output_path = os.path.join(results_dir, "fig_all_scenarios_power_pies.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")
    
    # Also create individual files for each scenario
    print("\n" + "=" * 70)
    print("GENERATING INDIVIDUAL SCENARIO PIE CHARTS")
    print("=" * 70)
    
    for scenario_name, scenario_def in SCENARIOS.items():
        fig_single, ax_single = plt.subplots(figsize=(8, 8))
        create_scenario_pie(scenario_name, scenario_def, ax_single)
        
        single_path = os.path.join(results_dir, f"fig_power_pie_{scenario_name.lower()}.png")
        plt.savefig(single_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {single_path}")
        plt.close()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("Generated files:")
    print(f"  1. Combined view: fig_all_scenarios_power_pies.png")
    print(f"  2. Individual files:")
    for scenario_name in SCENARIOS.keys():
        print(f"     - fig_power_pie_{scenario_name.lower()}.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
