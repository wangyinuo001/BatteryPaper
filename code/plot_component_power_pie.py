"""
Create pie charts showing component power consumption breakdown.
Shows the power contribution of each component in Video Streaming scenario.

Video Streaming Scenario Parameters:
- Screen: 80% brightness, η=1.0 (normal refresh)
- SoC: 20% load + 10% background = 30%
- Radio: 90% activity (high data streaming)
- GPS: 10% usage
- Baseline: Always on (100%)
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def get_component_power_breakdown():
    """
    Calculate power consumption of each component in Video Streaming scenario.
    
    Video Streaming Parameters:
    - Screen: B=0.8 (80%), η=1.0 → P = 1.0*(0.4991*0.8 + 0.1113) = 0.5106 W = 510.6 mW
    - SoC: 30% load → P = 0.3 * 2040 = 612.0 mW
    - Radio: 90% activity → P = 0.9 * 1173 = 1055.7 mW
    - GPS: 10% usage → P = 300 mW
    - Baseline: Always on → P = 120 mW
    
    Total: 2598.3 mW = 2.598 W
    """

    # Video Streaming scenario parameters
    screen_B = 0.8  # 80% brightness
    screen_eta = 1.0  # Normal refresh rate
    soc_load = 0.3  # 30% (20% + 10% background)
    radio_activity = 0.9  # 90%
    gps_on = True  # GPS active
    
    # Calculate individual component powers
    baseline_power = 120.0  # mW (always on)
    
    # Screen: P_screen = η(0.4991*B + 0.1113) W
    screen_power = screen_eta * (0.4991 * screen_B + 0.1113) * 1000  # Convert W to mW
    
    # SoC: proportional to load
    cpu_power = soc_load * 2040.0  # mW
    
    # Radio: proportional to activity
    radio_power = radio_activity * 1173.0  # mW
    
    # GPS: on/off
    gps_power = 300.0 if gps_on else 0.0  # mW
    
    components = {
        "Baseline": baseline_power,
        "Screen (80%)": screen_power,
        "SoC (30%)": cpu_power,
        "Radio (90%)": radio_power,
        "GPS": gps_power,
    }
    
    total_power = sum(components.values())
    
    # Return both components and the individual values for printing
    return (
        components,
        total_power,
        {
            "screen_B": screen_B,
            "screen_eta": screen_eta,
            "soc_load": soc_load,
            "radio_activity": radio_activity,
            "screen_power": screen_power,
            "cpu_power": cpu_power,
            "radio_power": radio_power,
            "gps_power": gps_power,
            "baseline_power": baseline_power,
        },
    )


def create_pie_charts():
    """Create pie charts for power breakdown."""
    components, total_power, params = get_component_power_breakdown()

    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results"
    )
    os.makedirs(results_dir, exist_ok=True)

    # Print breakdown
    print("=" * 70)
    print("COMPONENT POWER BREAKDOWN - VIDEO STREAMING SCENARIO")
    print("=" * 70)
    print("Scenario parameters:")
    print(f"  Screen: {int(params['screen_B']*100)}% brightness, η={params['screen_eta']} → {params['screen_power']:.1f}mW")
    print(f"  SoC: {int(params['soc_load']*100)}% load → {params['cpu_power']:.1f}mW")
    print(f"  Radio: {int(params['radio_activity']*100)}% activity → {params['radio_power']:.1f}mW")
    print(f"  GPS: Active → {params['gps_power']:.1f}mW")
    print(f"  Baseline: Always on → {params['baseline_power']:.1f}mW")
    print("-" * 70)
    for name, power in components.items():
        percentage = (power / total_power) * 100
        print(f"{name:25s}: {power:7.1f} mW  ({percentage:5.1f}%)")
    print("-" * 70)
    print(
        f"{'TOTAL Power':<25s}: {total_power:7.1f} mW  ({total_power/1000:.3f} W) (100.0%)"
    )
    print(f"{'Predicted TTE':<25s}: 6.42 hours (5000mAh battery)")
    print("=" * 70)

    # Create figure with two pie charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Prepare data
    labels = list(components.keys())
    sizes = list(components.values())
    percentages = [(s / total_power) * 100 for s in sizes]

    # Color scheme (professional palette)
    colors = ["#95E1D3", "#4ECDC4", "#FF6B6B", "#45B7D1", "#FFA07A"]
    explode = (0.02, 0.02, 0.02, 0.02, 0.02)  # Subtle separation

    # ========== Left: Simple percentage view ==========
    wedges1, texts1, autotexts1 = ax1.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 11, "fontweight": "bold"},
    )

    # Make percentage text more visible
    for autotext in autotexts1:
        autotext.set_color("white")
        autotext.set_fontsize(10)
        autotext.set_weight("bold")

    ax1.set_title(
        "Video Streaming\nPower Distribution",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # ========== Right: Power in Watts with values ==========
    # Create custom labels with both name and absolute power
    labels_with_power = [
        f"{name}\n{size:.0f} mW\n({pct:.1f}%)"
        for name, size, pct in zip(labels, sizes, percentages)
    ]

    wedges2, texts2 = ax2.pie(
        sizes,
        explode=explode,
        labels=labels_with_power,
        colors=colors,
        startangle=90,
        textprops={"fontsize": 11, "fontweight": "bold"},
    )

    ax2.set_title(
        "Video Streaming\nPower with Values\nTotal: {:.2f}W".format(
            total_power / 1000
        ),
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()

    # Save figure
    fig_path = os.path.join(results_dir, "fig_component_power_breakdown.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {fig_path}")

    # ========== Create single detailed pie chart ==========
    fig2, ax = plt.subplots(figsize=(12, 9))

    # Sort by size for better visualization
    sorted_data = sorted(
        zip(labels, sizes, percentages), key=lambda x: x[1], reverse=True
    )
    sorted_labels, sorted_sizes, sorted_percentages = zip(*sorted_data)

    # Assign colors based on component type
    color_map = {
        "SoC (30%)": "#ff6b6b",  # Red
        "Radio (90%)": "#4ecdc4",  # Teal - highest in Video
        "Screen (80%)": "#45b7d1",  # Blue
        "GPS": "#f7b731",  # Yellow
        "Baseline": "#a8e6cf",  # Light green
    }
    sorted_colors = [color_map.get(label, "#cccccc") for label in sorted_labels]

    # Create pie with custom formatting
    wedges, texts, autotexts = ax.pie(
        sorted_sizes,
        labels=sorted_labels,
        colors=sorted_colors,
        autopct=lambda pct: f"{pct:.1f}%\n({pct*total_power/100:.0f} mW)",
        startangle=140,
        shadow=True,
        explode=[0.08 if i == 0 else 0.03 for i in range(len(sorted_labels))],
        textprops={"fontsize": 11},
    )

    # Enhance text visibility
    for text in texts:
        text.set_fontsize(12)
        text.set_fontweight("bold")

    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(10)
        autotext.set_fontweight("bold")

    # Add title with summary
    ax.set_title(
        "Video Streaming Power Breakdown\n"
        f"Total: {total_power/1000:.2f}W | TTE: 6.42h (5000mAh)\n"
        f"Screen 80% | SoC 30% | Radio 90% | GPS ON",
        fontsize=14,
        fontweight="bold",
        pad=25,
    )

    # Add legend with detailed info
    legend_labels = [
        f"{label}: {size:.0f} mW ({pct:.1f}%)"
        for label, size, pct in zip(sorted_labels, sorted_sizes, sorted_percentages)
    ]
    ax.legend(
        legend_labels,
        title="Component Power Details",
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        fontsize=10,
        title_fontsize=12,
        framealpha=0.9,
    )

    # Add annotation for key insight
    key_insight = (
        f"Key Insights:\n"
        f"• Radio dominates at {sorted_percentages[0]:.1f}% of max power\n"
        f"• Top 2 components = {sorted_percentages[0]+sorted_percentages[1]:.1f}%\n"
        f"• Baseline idle = {color_map}"
    )

    plt.tight_layout()

    fig_path2 = os.path.join(results_dir, "fig_component_power_breakdown_detailed.png")
    plt.savefig(fig_path2, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {fig_path2}")

    plt.show()

    # ========== Save data to CSV ==========
    import pandas as pd

    df = pd.DataFrame(
        {
            "Component": labels,
            "Power_mW": sizes,
            "Power_W": [s / 1000 for s in sizes],
            "Percentage": percentages,
        }
    )
    df = df.sort_values("Power_mW", ascending=False)

    csv_path = os.path.join(results_dir, "component_power_breakdown.csv")
    df.to_csv(csv_path, index=False, float_format="%.3f")
    print(f"Saved: {csv_path}")

    # ========== Generate LaTeX table ==========
    tex_path = os.path.join(results_dir, "component_power_breakdown_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Component Power Breakdown Table\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write(
            "\\caption{Power Consumption of Individual Components at 100\\% Utilization}\n"
        )
        f.write("\\label{tab:component_power}\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\hline\n")
        f.write("Component & Power (mW) & Power (W) & Percentage (\\%) \\\\\n")
        f.write("\\hline\n")
        for _, row in df.iterrows():
            f.write(
                f"{row['Component']} & {row['Power_mW']:.2f} & {row['Power_W']:.3f} & {row['Percentage']:.1f} \\\\\n"
            )
        f.write("\\hline\n")
        f.write(
            f"\\textbf{{Total}} & \\textbf{{{total_power:.2f}}} & \\textbf{{{total_power/1000:.3f}}} & \\textbf{{100.0}} \\\\\n"
        )
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"Saved: {tex_path}")

    print("\n" + "=" * 70)
    print("FILES GENERATED:")
    print("=" * 70)
    print(f"1. {fig_path}")
    print(f"2. {fig_path2}")
    print(f"3. {csv_path}")
    print(f"4. {tex_path}")
    print("=" * 70)

    return components, total_power


if __name__ == "__main__":
    create_pie_charts()
