"""
XJTU Batch-2 Constant-Current Overview
=====================================
Generate a 2x2 overview figure for XJTU Batch-2 (3C constant-current) cells.
Outputs: ../results/fig_xjtu_batch2_overview_2x2.png
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


def load_batch_metrics(batch_dir):
    mat_files = sorted(glob.glob(os.path.join(batch_dir, "*.mat")))
    batteries = []

    for mat_file in mat_files:
        try:
            mat_data = loadmat(mat_file)
            data = mat_data["data"]
            n_cycles = data.shape[1]

            cycles = []
            capacities = []
            voltages_avg = []
            currents_avg = []
            temps_avg = []
            powers_avg = []
            resistances_est = []
            discharge_times = []
            soh = []

            for i in range(n_cycles):
                cycle_data = data[0, i]
                voltage = cycle_data["voltage_V"].flatten()
                current = cycle_data["current_A"].flatten()
                time_min = cycle_data["relative_time_min"].flatten()
                temp = cycle_data["temperature_C"].flatten()

                discharge_mask = current < 0
                if not np.any(discharge_mask):
                    continue

                v_d = voltage[discharge_mask]
                c_d = -current[discharge_mask]
                t_d = time_min[discharge_mask]
                temp_d = temp[discharge_mask]

                dt = np.diff(t_d, prepend=t_d[0]) * 60.0
                capacity = np.sum(c_d * dt) / 3600.0
                discharge_time_h = (t_d[-1] - t_d[0]) / 60.0

                mid_idx = len(v_d) // 2
                window = min(100, len(v_d) - mid_idx)
                if window > 5:
                    v_mid = np.mean(v_d[mid_idx : mid_idx + window])
                    i_mid = np.mean(c_d[mid_idx : mid_idx + window])
                    ocv_est = 3.7
                    r_est = max((ocv_est - v_mid) / max(i_mid, 1e-6), 0.0)
                else:
                    r_est = np.nan

                cycles.append(i + 1)
                capacities.append(capacity)
                voltages_avg.append(np.mean(v_d))
                currents_avg.append(np.mean(c_d))
                temps_avg.append(np.mean(temp_d))
                powers_avg.append(np.mean(v_d * c_d))
                resistances_est.append(r_est)
                discharge_times.append(discharge_time_h)
                soh.append(capacity / capacities[0] * 100.0)

            batteries.append(
                {
                    "file": os.path.basename(mat_file),
                    "cycle": np.array(cycles),
                    "capacity": np.array(capacities),
                    "voltage_avg": np.array(voltages_avg),
                    "current_avg": np.array(currents_avg),
                    "temp_avg": np.array(temps_avg),
                    "power_avg": np.array(powers_avg),
                    "resistance": np.array(resistances_est),
                    "discharge_time_h": np.array(discharge_times),
                    "soh": np.array(soh),
                }
            )
        except Exception as exc:
            print(f"Error loading {mat_file}: {exc}")

    return batteries


def summarize_batches(batteries):
    all_cap = np.concatenate([b["capacity"] for b in batteries])
    all_curr = np.concatenate([b["current_avg"] for b in batteries])
    all_volt = np.concatenate([b["voltage_avg"] for b in batteries])
    all_temp = np.concatenate([b["temp_avg"] for b in batteries])
    all_power = np.concatenate([b["power_avg"] for b in batteries])
    all_resistance = np.concatenate([b["resistance"] for b in batteries])
    all_discharge = np.concatenate([b["discharge_time_h"] for b in batteries])
    all_soh = np.concatenate([b["soh"] for b in batteries])
    all_cycles = np.array([len(b["cycle"]) for b in batteries])

    return {
        "battery_count": len(batteries),
        "cycle_min": int(np.min(all_cycles)),
        "cycle_max": int(np.max(all_cycles)),
        "capacity_min": float(np.min(all_cap)),
        "capacity_max": float(np.max(all_cap)),
        "current_mean": float(np.mean(all_curr)),
        "voltage_mean": float(np.mean(all_volt)),
        "temp_mean": float(np.mean(all_temp)),
        "power_mean": float(np.mean(all_power)),
        "resistance_mean": float(np.nanmean(all_resistance)),
        "discharge_mean": float(np.mean(all_discharge)),
        "soh_min": float(np.min(all_soh)),
        "soh_max": float(np.max(all_soh)),
    }


def summarize_distribution(values):
    values = values[~np.isnan(values)]
    return {
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }


def create_overview_figure(batteries, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        "XJTU Batch-2 Overview",
        fontsize=14,
        fontweight="bold",
    )

    ax1, ax2, ax3, ax4 = axes.flatten()

    # Capacity fade
    for b in batteries:
        ax1.plot(b["cycle"], b["capacity"], alpha=0.7, linewidth=1)
    ax1.set_title("Capacity Fade (Ah)")
    ax1.set_xlabel("Cycle Number")
    ax1.set_ylabel("Capacity (Ah)")
    ax1.grid(True, alpha=0.3)

    # Voltage evolution
    for b in batteries:
        ax2.plot(b["cycle"], b["voltage_avg"], alpha=0.7, linewidth=1)
    ax2.set_title("Avg Discharge Voltage")
    ax2.set_xlabel("Cycle Number")
    ax2.set_ylabel("Voltage (V)")
    ax2.grid(True, alpha=0.3)

    # SOH degradation
    for b in batteries:
        ax3.plot(b["cycle"], b["soh"], alpha=0.7, linewidth=1)
    ax3.set_title("SOH Degradation")
    ax3.set_xlabel("Cycle Number")
    ax3.set_ylabel("SOH (%)")
    ax3.axhline(80, color="red", linestyle="--", linewidth=1, alpha=0.6)
    ax3.grid(True, alpha=0.3)

    # Temperature trend
    for b in batteries:
        ax4.plot(b["cycle"], b["temp_avg"], alpha=0.7, linewidth=1)
    ax4.set_title("Avg Discharge Temperature")
    ax4.set_xlabel("Cycle Number")
    ax4.set_ylabel("Temperature (°C)")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    base_dir = r"d:\WYN_COLLEGE\s21\!MCM_ICM\batteries_new\data\XJTU battery dataset"
    batch_dir = os.path.join(base_dir, "Batch-2")

    batteries = load_batch_metrics(batch_dir)
    if not batteries:
        raise RuntimeError("No Batch-2 files found.")

    stats = summarize_batches(batteries)
    print("XJTU Batch-2 Overview Stats")
    print("=" * 40)
    print(f"Batteries: {stats['battery_count']}")
    print(f"Cycles per battery: {stats['cycle_min']} - {stats['cycle_max']}")
    print(f"Capacity range: {stats['capacity_min']:.3f} - {stats['capacity_max']:.3f} Ah")
    print(f"Avg discharge current: {stats['current_mean']:.3f} A")
    print(f"Avg discharge voltage: {stats['voltage_mean']:.3f} V")
    print(f"Avg temperature: {stats['temp_mean']:.2f} °C")

    all_capacity = np.concatenate([b["capacity"] for b in batteries])
    all_voltage = np.concatenate([b["voltage_avg"] for b in batteries])
    all_temp = np.concatenate([b["temp_avg"] for b in batteries])
    all_soh = np.concatenate([b["soh"] for b in batteries])

    cap_stats = summarize_distribution(all_capacity)
    volt_stats = summarize_distribution(all_voltage)
    temp_stats = summarize_distribution(all_temp)
    soh_stats = summarize_distribution(all_soh)

    print("\nDistribution Summary (Batch-2)")
    print("-" * 40)
    print(
        "Capacity (Ah): min={min:.3f}, p10={p10:.3f}, median={median:.3f}, mean={mean:.3f}, p90={p90:.3f}, max={max:.3f}, std={std:.3f}".format(
            **cap_stats
        )
    )
    print(
        "Voltage (V): min={min:.3f}, p10={p10:.3f}, median={median:.3f}, mean={mean:.3f}, p90={p90:.3f}, max={max:.3f}, std={std:.3f}".format(
            **volt_stats
        )
    )
    print(
        "SOH (%): min={min:.2f}, p10={p10:.2f}, median={median:.2f}, mean={mean:.2f}, p90={p90:.2f}, max={max:.2f}, std={std:.2f}".format(
            **soh_stats
        )
    )
    print(
        "Temp (°C): min={min:.2f}, p10={p10:.2f}, median={median:.2f}, mean={mean:.2f}, p90={p90:.2f}, max={max:.2f}, std={std:.2f}".format(
            **temp_stats
        )
    )

    output_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "results",
        "fig_xjtu_batch2_overview_2x2.png",
    )
    create_overview_figure(batteries, output_path)
