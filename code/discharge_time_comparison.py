"""
Model Comparison: Discharge Time Prediction Task
=================================================
Compare three models on the REAL task:
Input: Power load P, battery parameters (V, R, Q)
Output: Discharge time (Time to Empty)

Models:
1. Experimental Data
2. NBM (Nernst-Baseline Model)
3. Main Model (Shepherd + Arrhenius)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main_model import MainBatteryModel


def load_xjtu_discharge():
    """Load XJTU discharge data"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    battery_file = os.path.join(
        script_dir, "../data/XJTU battery dataset/Batch-1/2C_battery-1.mat"
    )

    mat_data = loadmat(battery_file)
    cycle0 = mat_data["data"][0, 0]

    voltage = cycle0["voltage_V"].flatten()
    current = cycle0["current_A"].flatten()
    time_min = cycle0["relative_time_min"].flatten()

    discharge_mask = current < 0
    time_h = time_min[discharge_mask] / 60
    voltage_d = voltage[discharge_mask]
    current_d = -current[discharge_mask]

    dt = np.diff(time_h, prepend=time_h[0]) * 3600
    capacity = np.cumsum(current_d * dt) / 3600
    Q_total = capacity[-1]
    dod = capacity / Q_total
    soc = 1 - dod

    # Calculate power
    power = voltage_d * current_d

    return {
        "time": time_h,
        "voltage": voltage_d,
        "current": current_d,
        "power": power,
        "dod": dod,
        "soc": soc,
        "Q_total": Q_total,
        "discharge_time": time_h[-1],  # Actual discharge time
    }


def nbm_discharge_time(
    P_load, Q0=1.991, E0=4.5000, K=-1.1718, alpha=0.8022, R0=0.0000, V_cutoff=2.5
):
    """
    NBM prediction: Calculate discharge time for given power load
    Fitted parameters from XJTU Batch-1 Battery-1 Cycle 0
    """
    soc = 1.0
    t = 0
    dt = 1.0  # 1 second time step

    time_points = []
    soc_points = []
    voltage_points = []

    while soc > 0.05 and t < 20 * 3600:  # Max 20 hours
        # NBM voltage
        soc_clip = np.clip(soc, 0.01, 0.99)
        RT_nF = 0.026
        V_oc = (
            E0
            + RT_nF * np.log(soc_clip / (1 - soc_clip))
            + K * np.exp(-alpha * soc_clip)
        )

        # Solve for current: P = (V_oc - I*R0) * I
        if R0 < 1e-6:
            # If R0 ≈ 0, direct solution: I = P / V_oc
            I = P_load / V_oc if V_oc > 0 else 0
            V_term = V_oc
        else:
            # Quadratic: I^2 * R0 - I * V_oc + P = 0
            a = R0
            b = -V_oc
            c = P_load
            discriminant = b**2 - 4 * a * c

            if discriminant < 0:
                break

            I = (-b - np.sqrt(discriminant)) / (2 * a)
            V_term = V_oc - I * R0

        if V_term < V_cutoff:
            break

        # Update SOC
        dsoc = -I * dt / (Q0 * 3600)
        soc += dsoc
        t += dt

        time_points.append(t / 3600)
        soc_points.append(soc)
        voltage_points.append(V_term)

    return {
        "discharge_time": t / 3600,
        "time": np.array(time_points),
        "soc": np.array(soc_points),
        "voltage": np.array(voltage_points),
    }


def main_model_discharge_time(P_load, Q0=1.991, temp_k=298.15, V_cutoff=2.5):
    """
    Main Model prediction: Shepherd + Arrhenius temperature
    """
    model = MainBatteryModel(Q0=Q0, V_cutoff=V_cutoff)
    result = model.predict_discharge(P_load, temp_k=temp_k, dt=1.0)

    return {
        "time": result["time"],
        "soc": result["soc"],
        "voltage": result["voltage"],
        "current": result["current"],
        "discharge_time": result["discharge_time"],
    }


def create_discharge_time_comparison():
    """Compare discharge time predictions"""

    print("\n" + "=" * 80)
    print("Discharge Time Prediction: Experimental vs NBM vs Main Model")
    print("=" * 80)

    # Load experimental data
    exp_data = load_xjtu_discharge()
    P_avg = np.mean(exp_data["power"])
    Q0 = exp_data["Q_total"]

    print(f"\nInput Parameters:")
    print(f"  Power Load: {P_avg:.3f} W")
    print(f"  Battery Capacity: {Q0:.3f} Ah")
    print(f"  Temperature: 25°C")

    print(f"\nExperimental Output:")
    print(f"  Discharge Time: {exp_data['discharge_time']:.3f} hours")

    # Model predictions
    nbm_result = nbm_discharge_time(P_avg, Q0=Q0)
    main_result = main_model_discharge_time(P_avg, Q0=Q0)

    print(f"\nNBM Prediction:")
    print(f"  Discharge Time: {nbm_result['discharge_time']:.3f} hours")
    print(
        f"  Error: {abs(nbm_result['discharge_time'] - exp_data['discharge_time']):.3f} h ({abs(nbm_result['discharge_time'] - exp_data['discharge_time'])/exp_data['discharge_time']*100:.1f}%)"
    )

    print(f"\nMain Model Prediction:")
    print(f"  Discharge Time: {main_result['discharge_time']:.3f} hours")
    print(
        f"  Error: {abs(main_result['discharge_time'] - exp_data['discharge_time']):.3f} h ({abs(main_result['discharge_time'] - exp_data['discharge_time'])/exp_data['discharge_time']*100:.1f}%)"
    )

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {"exp": "#2E3440", "nbm": "#5E81AC", "main": "#BF616A"}

    # ========== Panel A: Discharge Time Bar Chart ==========
    ax1 = axes[0]

    models = ["Experiment", "NBM", "Main Model"]
    times = [
        exp_data["discharge_time"],
        nbm_result["discharge_time"],
        main_result["discharge_time"],
    ]
    bar_colors = [colors["exp"], colors["nbm"], colors["main"]]

    bars = ax1.bar(
        models, times, color=bar_colors, alpha=0.8, edgecolor="black", linewidth=1.5
    )

    # Add value labels
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.05,
            f"{time:.2f}h",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Add error annotations
    nbm_error = (
        abs(nbm_result["discharge_time"] - exp_data["discharge_time"])
        / exp_data["discharge_time"]
        * 100
    )
    main_error = (
        abs(main_result["discharge_time"] - exp_data["discharge_time"])
        / exp_data["discharge_time"]
        * 100
    )

    ax1.text(
        1,
        times[1] * 0.5,
        f"Error:\n{nbm_error:.1f}%",
        ha="center",
        fontsize=10,
        bbox=dict(
            boxstyle="round", facecolor="white", alpha=0.8, edgecolor=colors["nbm"]
        ),
    )
    ax1.text(
        2,
        times[2] * 0.5,
        f"Error:\n{main_error:.1f}%",
        ha="center",
        fontsize=10,
        bbox=dict(
            boxstyle="round", facecolor="white", alpha=0.8, edgecolor=colors["main"]
        ),
    )

    ax1.set_ylabel("Discharge Time (hours)", fontsize=12, fontweight="bold")
    ax1.set_title(
        "(A) Discharge Time Prediction", fontsize=13, fontweight="bold", pad=10
    )
    ax1.set_ylim([0, max(times) * 1.15])
    ax1.grid(True, alpha=0.2, axis="y")

    # ========== Panel B: SOC vs Time ==========
    ax2 = axes[1]

    ax2.plot(
        exp_data["time"],
        exp_data["soc"] * 100,
        color=colors["exp"],
        linewidth=3,
        marker="o",
        markersize=3,
        markevery=20,
        alpha=0.8,
        label="Experiment",
    )
    ax2.plot(
        nbm_result["time"],
        nbm_result["soc"] * 100,
        color=colors["nbm"],
        linewidth=2.5,
        linestyle="--",
        alpha=0.85,
        label="NBM",
    )
    ax2.plot(
        main_result["time"],
        main_result["soc"] * 100,
        color=colors["main"],
        linewidth=2.5,
        linestyle="-",
        alpha=0.85,
        label="Main Model",
    )

    ax2.set_xlabel("Time (hours)", fontsize=12)
    ax2.set_ylabel("State of Charge (%)", fontsize=12)
    ax2.set_title(
        "(B) SOC Depletion Trajectory", fontsize=13, fontweight="bold", pad=10
    )
    ax2.legend(fontsize=11, loc="upper right", framealpha=0.95, edgecolor="gray")
    ax2.grid(True, alpha=0.2)
    ax2.set_ylim([0, 105])

    # ========== Panel C: Error Metrics Comparison ==========
    ax3 = axes[2]

    metrics = ["Time Error (h)", "Error (%)", "Accuracy (%)"]
    nbm_vals = [
        abs(nbm_result["discharge_time"] - exp_data["discharge_time"]),
        nbm_error,
        100 - nbm_error,
    ]
    main_vals = [
        abs(main_result["discharge_time"] - exp_data["discharge_time"]),
        abs(main_error),
        100 - abs(main_error),
    ]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax3.bar(
        x - width / 2,
        nbm_vals,
        width,
        label="NBM",
        color=colors["nbm"],
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )
    bars2 = ax3.bar(
        x + width / 2,
        main_vals,
        width,
        label="Main Model",
        color=colors["main"],
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )

    ax3.set_ylabel("Value", fontsize=12, fontweight="bold")
    ax3.set_title(
        "(C) Prediction Error Metrics", fontsize=13, fontweight="bold", pad=10
    )
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics, fontsize=10)
    ax3.legend(fontsize=11, loc="upper left")
    ax3.grid(True, alpha=0.2, axis="y")

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()

    # Save
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "../results/fig_discharge_time_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")

    print(f"\n{'='*80}")
    print(f"Figure saved: {os.path.abspath(save_path)}")
    print(f"{'='*80}\n")

    return {
        "exp_time": exp_data["discharge_time"],
        "nbm_time": nbm_result["discharge_time"],
        "main_time": main_result["discharge_time"],
        "nbm_error": nbm_error,
        "main_error": main_error,
    }


if __name__ == "__main__":
    results = create_discharge_time_comparison()
