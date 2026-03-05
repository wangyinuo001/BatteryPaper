"""
Section 5.1: Dataset Overview and Characteristics
=================================================
Visualize two datasets used for validation:
1. XJTU: 390 cycles, NCM chemistry, room temperature
2. Kaggle: B5/B6/B7, NASA-inspired, long-term degradation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pandas as pd
import os


def load_xjtu_dataset(batch_name, file_name):
    """Load XJTU dataset for a specific batch/file"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    battery_file = os.path.join(
        script_dir, f"../data/XJTU battery dataset/{batch_name}/{file_name}"
    )

    mat_data = loadmat(battery_file)
    data = mat_data["data"]
    n_cycles = data.shape[1]

    cycles = []
    capacities = []
    voltages_avg = []
    currents_avg = []

    for i in range(n_cycles):
        cycle_data = data[0, i]
        v = cycle_data["voltage_V"].flatten()
        c = cycle_data["current_A"].flatten()
        t = cycle_data["relative_time_min"].flatten()

        # Discharge phase
        discharge_mask = c < 0
        if not np.any(discharge_mask):
            continue

        v_d = v[discharge_mask]
        c_d = -c[discharge_mask]
        t_d = t[discharge_mask]

        dt = np.diff(t_d, prepend=t_d[0]) * 60
        capacity = np.sum(c_d * dt) / 3600

        cycles.append(i + 1)
        capacities.append(capacity)
        voltages_avg.append(np.mean(v_d))
        currents_avg.append(np.mean(c_d))

    return {
        "cycle": np.array(cycles),
        "capacity": np.array(capacities),
        "voltage_avg": np.array(voltages_avg),
        "current_avg": np.array(currents_avg),
    }


def apply_kaggle_filters(df):
    """Apply Kaggle-only filters: remove calendar 1–5h and resistance < 0.07."""
    df = df.copy()
    discharge_current = df["disI"].abs().replace(0, np.nan)
    discharge_hours = df["BCt"] / discharge_current
    resistance_est = (4.2 - df["disV"]) / discharge_current
    df = df.loc[~discharge_hours.between(1.0, 5.0, inclusive="both")]
    df = df.loc[resistance_est >= 0.07]
    return df


def load_kaggle_dataset():
    """Load Kaggle NASA-inspired dataset with filtering"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, "../data/Battery_dataset.csv")

    df = pd.read_csv(csv_file)

    # Split by battery
    batteries = {}
    for battery_id in ["B5", "B6", "B7"]:
        df_battery = df[df["battery_id"] == battery_id].copy()
        df_battery = apply_kaggle_filters(df_battery)
        batteries[battery_id] = {
            "cycle": df_battery["cycle"].values,
            "capacity": df_battery["BCt"].values,
            "soh": df_battery["SOH"].values,
            "rul": df_battery["RUL"].values,
            "voltage_charge": df_battery["chV"].values,
            "voltage_discharge": df_battery["disV"].values,
            "temp_charge": df_battery["chT"].values,
            "temp_discharge": df_battery["disT"].values,
        }

    return batteries


def create_dataset_overview():
    """Create comprehensive dataset overview figure"""

    print("\n" + "=" * 80)
    print("Section 5.1: Dataset Overview")
    print("=" * 80)

    # Load data
    xjtu_b1 = load_xjtu_dataset("Batch-1", "2C_battery-1.mat")
    xjtu_b2 = load_xjtu_dataset("Batch-2", "3C_battery-1.mat")
    kaggle = load_kaggle_dataset()

    print(f"\nXJTU Dataset (Batch-1/2):")
    print(f"  Batch-1 cycles: {len(xjtu_b1['cycle'])}")
    print(
        f"  Batch-1 capacity: {xjtu_b1['capacity'].min():.3f} - {xjtu_b1['capacity'].max():.3f} Ah"
    )
    print(f"  Batch-2 cycles: {len(xjtu_b2['cycle'])}")
    print(
        f"  Batch-2 capacity: {xjtu_b2['capacity'].min():.3f} - {xjtu_b2['capacity'].max():.3f} Ah"
    )
    print(f"  Chemistry: LiNi0.5Co0.2Mn0.3O2 (NCM)")

    print(f"\nKaggle Dataset:")
    for bid in ["B5", "B6", "B7"]:
        b = kaggle[bid]
        print(
            f"  {bid}: {len(b['cycle'])} cycles, "
            f"Capacity {b['capacity'].min():.3f}-{b['capacity'].max():.3f} Ah"
        )

    # Create figure with 3 rows x 3 columns
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # ========== Row 1: Capacity Degradation ==========
    # XJTU
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(
        xjtu_b1["cycle"],
        xjtu_b1["capacity"],
        "o-",
        markersize=3,
        linewidth=1.5,
        color="blue",
        label="Batch-1 (2C)",
    )
    ax1.plot(
        xjtu_b2["cycle"],
        xjtu_b2["capacity"],
        "o-",
        markersize=3,
        linewidth=1.5,
        color="teal",
        label="Batch-2 (3C)",
    )
    ax1.set_xlabel("Cycle Number", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Capacity (Ah)", fontsize=11, fontweight="bold")
    ax1.set_title(
        "XJTU: Capacity Fade\n(Batch-1/2)", fontsize=12, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3)
    ax1.axhline(
        y=xjtu_b1["capacity"][0] * 0.8,
        color="r",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
    )
    ax1.legend(fontsize=9)

    # Kaggle B5
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(
        kaggle["B5"]["cycle"],
        kaggle["B5"]["capacity"],
        "o-",
        markersize=3,
        linewidth=1.5,
        color="green",
        label="B5",
    )
    ax2.set_xlabel("Cycle Number", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Capacity (Ah)", fontsize=11, fontweight="bold")
    ax2.set_title(
        "Kaggle B5: Capacity\n(NASA-inspired)", fontsize=12, fontweight="bold"
    )
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=2.0 * 0.8, color="r", linestyle="--", linewidth=1, alpha=0.5)

    # Kaggle B6+B7
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(
        kaggle["B6"]["cycle"],
        kaggle["B6"]["capacity"],
        "o-",
        markersize=3,
        linewidth=1.5,
        color="orange",
        label="B6",
    )
    ax3.plot(
        kaggle["B7"]["cycle"],
        kaggle["B7"]["capacity"],
        "s-",
        markersize=3,
        linewidth=1.5,
        color="purple",
        label="B7",
    )
    ax3.set_xlabel("Cycle Number", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Capacity (Ah)", fontsize=11, fontweight="bold")
    ax3.set_title(
        "Kaggle B6/B7: Capacity\n(NASA-inspired)", fontsize=12, fontweight="bold"
    )
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=2.0 * 0.8, color="r", linestyle="--", linewidth=1, alpha=0.5)

    # ========== Row 2: Voltage Trends ==========
    # XJTU
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(
        xjtu_b1["cycle"],
        xjtu_b1["voltage_avg"],
        "o-",
        markersize=3,
        linewidth=1.5,
        color="blue",
        label="Batch-1 (2C)",
    )
    ax4.plot(
        xjtu_b2["cycle"],
        xjtu_b2["voltage_avg"],
        "o-",
        markersize=3,
        linewidth=1.5,
        color="teal",
        label="Batch-2 (3C)",
    )
    ax4.set_xlabel("Cycle Number", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Avg Discharge Voltage (V)", fontsize=11, fontweight="bold")
    ax4.set_title("XJTU: Voltage Evolution", fontsize=12, fontweight="bold")
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9)

    # Kaggle B5 discharge voltage
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(
        kaggle["B5"]["cycle"],
        kaggle["B5"]["voltage_discharge"],
        "o-",
        markersize=3,
        linewidth=1.5,
        color="green",
    )
    ax5.set_xlabel("Cycle Number", fontsize=11, fontweight="bold")
    ax5.set_ylabel("Avg Discharge Voltage (V)", fontsize=11, fontweight="bold")
    ax5.set_title("Kaggle B5: Voltage", fontsize=12, fontweight="bold")
    ax5.grid(True, alpha=0.3)

    # Kaggle B6+B7 voltage
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(
        kaggle["B6"]["cycle"],
        kaggle["B6"]["voltage_discharge"],
        "o-",
        markersize=3,
        linewidth=1.5,
        color="orange",
        label="B6",
    )
    ax6.plot(
        kaggle["B7"]["cycle"],
        kaggle["B7"]["voltage_discharge"],
        "s-",
        markersize=3,
        linewidth=1.5,
        color="purple",
        label="B7",
    )
    ax6.set_xlabel("Cycle Number", fontsize=11, fontweight="bold")
    ax6.set_ylabel("Avg Discharge Voltage (V)", fontsize=11, fontweight="bold")
    ax6.set_title("Kaggle B6/B7: Voltage", fontsize=12, fontweight="bold")
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)

    # ========== Row 3: SOH and Temperature ==========
    # XJTU SOH
    ax7 = fig.add_subplot(gs[2, 0])
    soh_xjtu_b1 = 100 * xjtu_b1["capacity"] / xjtu_b1["capacity"][0]
    soh_xjtu_b2 = 100 * xjtu_b2["capacity"] / xjtu_b2["capacity"][0]
    ax7.plot(
        xjtu_b1["cycle"],
        soh_xjtu_b1,
        "o-",
        markersize=3,
        linewidth=1.5,
        color="blue",
        label="Batch-1 (2C)",
    )
    ax7.plot(
        xjtu_b2["cycle"],
        soh_xjtu_b2,
        "o-",
        markersize=3,
        linewidth=1.5,
        color="teal",
        label="Batch-2 (3C)",
    )
    ax7.axhline(y=80, color="r", linestyle="--", linewidth=2, label="EOL (80%)")
    ax7.set_xlabel("Cycle Number", fontsize=11, fontweight="bold")
    ax7.set_ylabel("State of Health (%)", fontsize=11, fontweight="bold")
    ax7.set_title("XJTU: SOH Degradation", fontsize=12, fontweight="bold")
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3)

    # Kaggle SOH comparison
    ax8 = fig.add_subplot(gs[2, 1])
    for bid, color, marker in [
        ("B5", "green", "o"),
        ("B6", "orange", "s"),
        ("B7", "purple", "^"),
    ]:
        ax8.plot(
            kaggle[bid]["cycle"],
            kaggle[bid]["soh"],
            marker=marker,
            markersize=3,
            linewidth=1.5,
            color=color,
            label=bid,
        )
    ax8.axhline(y=80, color="r", linestyle="--", linewidth=2, label="EOL")
    ax8.set_xlabel("Cycle Number", fontsize=11, fontweight="bold")
    ax8.set_ylabel("State of Health (%)", fontsize=11, fontweight="bold")
    ax8.set_title("Kaggle: SOH Comparison", fontsize=12, fontweight="bold")
    ax8.legend(fontsize=10)
    ax8.grid(True, alpha=0.3)

    # Kaggle Temperature
    ax9 = fig.add_subplot(gs[2, 2])
    for bid, color, marker in [
        ("B5", "green", "o"),
        ("B6", "orange", "s"),
        ("B7", "purple", "^"),
    ]:
        ax9.plot(
            kaggle[bid]["cycle"],
            kaggle[bid]["temp_discharge"],
            marker=marker,
            markersize=3,
            linewidth=1.5,
            color=color,
            label=bid,
        )
    ax9.set_xlabel("Cycle Number", fontsize=11, fontweight="bold")
    ax9.set_ylabel("Discharge Temp (°C)", fontsize=11, fontweight="bold")
    ax9.set_title("Kaggle: Temperature", fontsize=12, fontweight="bold")
    ax9.legend(fontsize=10)
    ax9.grid(True, alpha=0.3)

    # Save
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results")
    output_file = os.path.join(output_dir, "fig_dataset_overview.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")

    print("\n" + "=" * 80)
    print(f"Saved: {output_file}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    create_dataset_overview()
