"""
Generate TTE loss analysis for ALL scenarios.
Creates:
1. A comprehensive table comparing all scenarios
2. Individual detailed analysis for Video Streaming (typical example)
3. Comparison figure showing all scenarios
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from aging_model import BatteryModel
from main_model import MainBatteryModel

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
    screen_brightness_B = screen_percent / 100.0  # B ∈ [0,1]
    screen_eta = eta  # η refresh multiplier
    cpu_load = min((scenario["soc"] / 100.0) + background_extra, 1.0)
    radio_activity = scenario["radio"] / 100.0
    gps_on = scenario["gps"] > 0

    return {
        "brightness_B": screen_brightness_B,  # Separate B
        "brightness_eta": screen_eta,  # Separate η
        "cpu_load": cpu_load,
        "network_mode": "4g",
        "network_activity": radio_activity,
        "gps": gps_on,
        "temperature": temperature_c + 273.15,
    }


def compute_tte_by_soc(scenario_name, scenario_def):
    """Compute TTE for different initial SOC levels."""
    power_model = BatteryModel(cycle_count=0, calendar_time_hours=0)
    main_model = MainBatteryModel(Q0=5.0)

    # SOC levels from 100% to 10% in 10% steps (key points only)
    soc_levels = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    inputs = build_inputs(scenario_def)
    power_w = power_model.get_power_consumption(0.0, inputs) / 1000.0

    tte_values = []
    print(f"  Computing {scenario_name}...", end="", flush=True)
    for i, soc0 in enumerate(soc_levels):
        print(f" {int(soc0*100)}%", end="", flush=True)
        result = main_model.predict_discharge(
            P_load=power_w,
            temp_k=inputs["temperature"],
            soc_initial=soc0,
            dt=2.0,  # Larger time step for speed
        )
        tte_values.append(result["discharge_time"])
    print(" Done!")

    # Calculate losses per 10% interval
    tte_losses = []
    intervals = []
    for i in range(len(tte_values) - 1):
        loss = tte_values[i] - tte_values[i + 1]
        tte_losses.append(loss)
        soc_high = int(soc_levels[i] * 100)
        soc_low = int(soc_levels[i + 1] * 100)
        intervals.append(f"{soc_high}%→{soc_low}%")

    return {
        "scenario": scenario_name,
        "intervals": intervals,
        "tte_losses": tte_losses,
        "total_tte": tte_values[0],
        "power_w": power_w,
        "scenario_params": scenario_def,
    }


def main():
    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results"
    )
    os.makedirs(results_dir, exist_ok=True)

    # Compute for all scenarios
    all_results = {}
    for name, scenario in SCENARIOS.items():
        print(f"Computing TTE loss analysis for {name}...")
        all_results[name] = compute_tte_by_soc(name, scenario)

    # ========== Create comprehensive comparison table ==========
    table_rows = []
    for name in ["Standby", "Navigation", "Gaming", "Video", "Reading"]:
        result = all_results[name]
        for interval, loss in zip(result["intervals"], result["tte_losses"]):
            table_rows.append(
                {
                    "Scenario": name,
                    "SOC_Interval": interval,
                    "TTE_Loss_hours": loss,
                    "Total_TTE_hours": result["total_tte"],
                    "Power_W": result["power_w"],
                }
            )

    df_all = pd.DataFrame(table_rows)
    csv_path = os.path.join(results_dir, "tte_loss_all_scenarios.csv")
    df_all.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # ========== Create summary statistics table ==========
    summary_rows = []
    for name, result in all_results.items():
        losses = result["tte_losses"]
        summary_rows.append(
            {
                "Scenario": name,
                "Total_TTE (h)": f"{result['total_tte']:.2f}",
                "Power (W)": f"{result['power_w']:.3f}",
                "Max_Loss (h)": f"{max(losses):.2f}",
                "Min_Loss (h)": f"{min(losses):.2f}",
                "Avg_Loss (h)": f"{np.mean(losses):.2f}",
                "Loss_Range (h)": f"{max(losses)-min(losses):.2f}",
                "CV (%)": f"{(np.std(losses)/np.mean(losses)*100):.1f}",
            }
        )

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(results_dir, "tte_loss_summary.csv")
    df_summary.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")

    # Print summary table
    print("\n" + "=" * 90)
    print("TTE LOSS ANALYSIS SUMMARY - ALL SCENARIOS")
    print("=" * 90)
    print(df_summary.to_string(index=False))
    print("=" * 90)

    # ========== Video Streaming detailed table (typical example) ==========
    video_result = all_results["Video"]
    video_rows = []
    cumulative = 0
    total_loss = sum(video_result["tte_losses"])

    for interval, loss in zip(video_result["intervals"], video_result["tte_losses"]):
        cumulative += loss
        pct = (loss / total_loss) * 100
        video_rows.append(
            {
                "SOC_Interval": interval,
                "TTE_Loss (h)": f"{loss:.2f}",
                "Percentage (%)": f"{pct:.1f}",
                "Cumulative_Loss (h)": f"{cumulative:.2f}",
            }
        )

    df_video = pd.DataFrame(video_rows)
    video_csv = os.path.join(results_dir, "tte_loss_video_streaming_detailed.csv")
    df_video.to_csv(video_csv, index=False)
    print(f"\nSaved: {video_csv}")

    print("\n" + "=" * 70)
    print("TYPICAL EXAMPLE: VIDEO STREAMING SCENARIO")
    print("=" * 70)
    print(f"Scenario Parameters:")
    print(f"  - Screen: 80% brightness, 1.0× refresh")
    print(f"  - SoC: 20% load")
    print(f"  - Radio: 90% activity (high data transfer)")
    print(f"  - GPS: 10% usage")
    print(f"  - Background: +10% SoC")
    print(f"Total Power: {video_result['power_w']:.3f}W")
    print(f"Total TTE (100% SOC): {video_result['total_tte']:.2f}h")
    print(f"\nTTE Loss per 10% SOC Interval:")
    print("-" * 70)
    print(df_video.to_string(index=False))
    print("=" * 70)

    # ========== Create comparison visualization ==========
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, 9))

    for idx, (name, result) in enumerate(all_results.items()):
        ax = axes[idx]
        losses = result["tte_losses"]
        intervals = result["intervals"]

        bars = ax.bar(
            range(len(intervals)),
            losses,
            color=colors,
            edgecolor="black",
            linewidth=1.5,
            alpha=0.85,
        )

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, losses)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_title(
            f'{name}\n(Total TTE: {result["total_tte"]:.2f}h, Power: {result["power_w"]:.3f}W)',
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("SOC Interval", fontsize=10)
        ax.set_ylabel("TTE Loss (h)", fontsize=10)
        ax.set_xticks(range(len(intervals)))
        ax.set_xticklabels(intervals, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.axhline(
            y=np.mean(losses), color="red", linestyle="--", linewidth=1.5, alpha=0.7
        )

        # Highlight Video Streaming
        if name == "Video":
            for spine in ax.spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(3)
            ax.set_title(
                f'{name} (TYPICAL EXAMPLE)\n(Total TTE: {result["total_tte"]:.2f}h, Power: {result["power_w"]:.3f}W)',
                fontsize=11,
                fontweight="bold",
                color="red",
            )

    # Remove extra subplot
    fig.delaxes(axes[5])

    fig.suptitle(
        "TTE Loss Analysis Across All Usage Scenarios\n"
        + "Red border indicates typical example (Video Streaming) for detailed analysis",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    fig_path = os.path.join(results_dir, "fig_tte_loss_all_scenarios_comparison.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {fig_path}")

    plt.show()

    # ========== Generate LaTeX table ==========
    tex_path = os.path.join(results_dir, "tte_loss_all_scenarios_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% TTE Loss Summary - All Scenarios\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{TTE Loss Statistics Across Usage Scenarios}\n")
        f.write("\\label{tab:tte_loss_summary}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\hline\n")
        f.write(
            "Scenario & Total TTE (h) & Power (W) & Max Loss (h) & Min Loss (h) & Avg Loss (h) & CV (\\%) \\\\\n"
        )
        f.write("\\hline\n")
        for row in summary_rows:
            f.write(
                f"{row['Scenario']} & {row['Total_TTE (h)']} & {row['Power (W)']} & "
                f"{row['Max_Loss (h)']} & {row['Min_Loss (h)']} & {row['Avg_Loss (h)']} & "
                f"{row['CV (%)']} \\\\\n"
            )
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"Saved: {tex_path}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("Generated files:")
    print(f"  1. {csv_path}")
    print(f"  2. {summary_csv}")
    print(f"  3. {video_csv} (TYPICAL EXAMPLE for paper)")
    print(f"  4. {fig_path}")
    print(f"  5. {tex_path}")
    print("\nUse the Video Streaming data as typical example in Section 6.1")
    print("Reference other scenarios for comparison in Section 6.2-6.4")
    print("=" * 70)


if __name__ == "__main__":
    main()
