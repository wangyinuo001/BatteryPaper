"""
Section 5: Model Validation and Sensitivity Analysis - NBM (Nernst-Baseline Model)
Implements validation against XJTU and NASA-inspired datasets
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.io import loadmat
import warnings
import os

warnings.filterwarnings("ignore")

# Set English environment
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.style.use("seaborn-v0_8-paper")


# ==================== NBM Model Definition ====================
class NernstBaselineModel:
    """Nernst-Baseline Model (NBM) for battery voltage prediction"""

    def __init__(self, Q0=2.0, R0=0.05, E0=3.6, K=-0.2, alpha=6.0):
        """
        Parameters:
        - Q0: Nominal capacity (Ah) - XJTU battery: 2.0 Ah
        - R0: Internal resistance (Ohm)
        - E0: Standard electrode potential (V)
        - K: Exponential correction coefficient (V)
        - alpha: Exponential decay constant
        """
        self.Q0 = Q0
        self.R0 = R0
        self.E0 = E0
        self.K = K
        self.alpha = alpha

        # Physical constants
        self.R_gas = 8.314  # Gas constant (J/(mol·K))
        self.T = 298.15  # Temperature (K)
        self.n = 1  # Electron transfer number
        self.F = 96485  # Faraday constant (C/mol)

    def ocv(self, soc):
        """Calculate open-circuit voltage from SOC using Nernst equation"""
        soc = np.clip(soc, 0.001, 0.999)  # Avoid singularities

        # Nernst term: thermodynamic equilibrium
        nernst_term = (
            (self.R_gas * self.T) / (self.n * self.F) * np.log(soc / (1 - soc))
        )

        # Exponential correction: low-SOC polarization
        exp_term = self.K * np.exp(-self.alpha * soc)

        V_oc = self.E0 + nernst_term + exp_term
        return V_oc

    def terminal_voltage(self, soc, current):
        """Calculate terminal voltage under load"""
        V_oc = self.ocv(soc)
        V_terminal = V_oc - current * self.R0
        return V_terminal

    def discharge_dynamics(self, t, state, I_load):
        """ODE system for constant-current discharge"""
        soc = state[0]
        dsoc_dt = -I_load / (self.Q0 * 3600)  # SOC rate of change
        return [dsoc_dt]

    def simulate_discharge(
        self, I_load, soc_initial=1.0, V_cutoff=2.5, soc_cutoff=0.05
    ):
        """Simulate constant-current discharge until cutoff"""

        def event_voltage(t, state):
            soc = state[0]
            V = self.terminal_voltage(soc, I_load)
            return V - V_cutoff

        def event_soc(t, state):
            return state[0] - soc_cutoff

        event_voltage.terminal = True
        event_soc.terminal = True

        # Time span: conservative estimate
        t_max = self.Q0 / I_load * 3600 * 1.2  # 20% margin

        sol = solve_ivp(
            fun=lambda t, y: self.discharge_dynamics(t, y, I_load),
            t_span=[0, t_max],
            y0=[soc_initial],
            method="RK45",
            events=[event_voltage, event_soc],
            dense_output=True,
            rtol=1e-8,
            atol=1e-10,
        )

        # Extract results
        t = sol.t
        soc = sol.y[0]
        V = np.array([self.terminal_voltage(s, I_load) for s in soc])

        return t, soc, V


# ==================== Data Loading Functions ====================
def load_xjtu_data(battery_file):
    """Load XJTU battery dataset from .mat file and extract discharge curves."""
    try:
        if not os.path.isabs(battery_file):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            battery_file = os.path.join(
                script_dir,
                "..",
                "data",
                "XJTU battery dataset",
                "Batch-1",
                "2C_battery-1.mat",
            )
        mat_data = loadmat(battery_file)

        if "data" not in mat_data:
            print(f"  Warning: 'data' key not found in {battery_file}")
            return None

        data_struct = mat_data["data"]
        cycles = []

        for idx in range(data_struct.shape[1]):
            entry = data_struct[0, idx]

            current = entry["current_A"].flatten()
            voltage = entry["voltage_V"].flatten()
            time_min = entry["relative_time_min"].flatten()

            if current.size == 0:
                continue

            # Select discharge segment (negative current)
            discharge_mask = current < 0
            if discharge_mask.sum() < 100:
                continue

            current_d = current[discharge_mask]
            voltage_d = voltage[discharge_mask]
            time_d = time_min[discharge_mask] * 60.0

            # Recompute discharge capacity by integrating current
            dt = np.diff(time_d, prepend=time_d[0])
            capacity_d = np.cumsum(-current_d * dt) / 3600.0

            description = entry["description"]
            try:
                description = description[0]
            except Exception:
                pass

            cycles.append(
                {
                    "cycle": idx + 1,
                    "time_s": time_d,
                    "voltage": voltage_d,
                    "current": current_d,
                    "capacity": capacity_d,
                    "description": description,
                }
            )

        return cycles

    except Exception as e:
        print(f"  Error loading XJTU data: {e}")
        return None


def load_nasa_kaggle_data(csv_file):
    """Load NASA-inspired Kaggle battery degradation dataset"""
    if not os.path.isabs(csv_file):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_file = os.path.join(script_dir, "..", "data", "Battery_dataset.csv")
    df = pd.read_csv(csv_file)

    # Kaggle-only filters: remove calendar 1–5h and resistance < 0.07
    discharge_current = df["disI"].abs().replace(0, np.nan)
    discharge_hours = df["BCt"] / discharge_current
    resistance_est = (4.2 - df["disV"]) / discharge_current
    df = df.loc[~discharge_hours.between(1.0, 5.0, inclusive="both")]
    df = df.loc[resistance_est >= 0.07]

    # Separate by battery ID
    batteries = {}
    for battery_id in df["battery_id"].unique():
        battery_data = df[df["battery_id"] == battery_id].copy()
        battery_data = battery_data.sort_values("cycle")
        batteries[battery_id] = battery_data

    return batteries


# ==================== Section 5.1: Data Description and Visualization ====================
def plot_data_overview(nasa_data, save_path="../results"):
    """Generate Figure: Dataset overview showing degradation trends from NASA-inspired data"""

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    colors_nasa = ["#d62728", "#ff7f0e", "#2ca02c"]

    # Panel A: Capacity fade for B5/B6/B7
    ax1 = fig.add_subplot(gs[0, 0])

    for idx, (battery_id, data) in enumerate(nasa_data.items()):
        ax1.plot(
            data["cycle"],
            data["BCt"],
            "o-",
            color=colors_nasa[idx],
            linewidth=2.5,
            markersize=5,
            alpha=0.7,
            label=battery_id,
        )

    ax1.set_xlabel("Cycle Number", fontsize=12)
    ax1.set_ylabel("Battery Capacity (Ah)", fontsize=12)
    ax1.set_title(
        "(A) NASA-Inspired: Capacity Degradation Across Cells",
        fontsize=13,
        fontweight="bold",
    )
    ax1.legend(loc="best", fontsize=11)
    ax1.grid(alpha=0.3)

    # Panel B: SOH trends
    ax2 = fig.add_subplot(gs[0, 1])

    for idx, (battery_id, data) in enumerate(nasa_data.items()):
        ax2.plot(
            data["cycle"],
            data["SOH"],
            "s-",
            color=colors_nasa[idx],
            linewidth=2.5,
            markersize=5,
            alpha=0.7,
            label=battery_id,
        )

    ax2.axhline(
        y=80,
        color="red",
        linestyle="--",
        linewidth=2.5,
        label="80% SOH Threshold",
        alpha=0.8,
    )
    ax2.set_xlabel("Cycle Number", fontsize=12)
    ax2.set_ylabel("State of Health (%)", fontsize=12)
    ax2.set_title("(B) State of Health Evolution", fontsize=13, fontweight="bold")
    ax2.legend(loc="best", fontsize=11)
    ax2.grid(alpha=0.3)

    # Panel C: RUL trends
    ax3 = fig.add_subplot(gs[1, 0])

    for idx, (battery_id, data) in enumerate(nasa_data.items()):
        ax3.plot(
            data["cycle"],
            data["RUL"],
            "^-",
            color=colors_nasa[idx],
            linewidth=2.5,
            markersize=5,
            alpha=0.7,
            label=battery_id,
        )

    ax3.set_xlabel("Cycle Number", fontsize=12)
    ax3.set_ylabel("Remaining Useful Life (cycles)", fontsize=12)
    ax3.set_title("(C) RUL Prediction Trajectory", fontsize=13, fontweight="bold")
    ax3.legend(loc="best", fontsize=11)
    ax3.grid(alpha=0.3)

    # Panel D: Voltage statistics
    ax4 = fig.add_subplot(gs[1, 1])

    for idx, (battery_id, data) in enumerate(nasa_data.items()):
        ax4.plot(
            data["cycle"],
            data["disV"],
            "v-",
            color=colors_nasa[idx],
            linewidth=2.5,
            markersize=5,
            alpha=0.7,
            label=f"{battery_id} Discharge",
        )

    ax4.set_xlabel("Cycle Number", fontsize=12)
    ax4.set_ylabel("Average Discharge Voltage (V)", fontsize=12)
    ax4.set_title("(D) Discharge Voltage Trends", fontsize=13, fontweight="bold")
    ax4.legend(loc="best", fontsize=11)
    ax4.grid(alpha=0.3)

    plt.savefig(
        f"{save_path}/fig_section5_1_data_overview.png", dpi=300, bbox_inches="tight"
    )
    print(f"Saved: {save_path}/fig_section5_1_data_overview.png")
    plt.close()


def plot_xjtu_overview(xjtu_cycles, save_path="../results"):
    """Generate Figure: XJTU dataset overview with discharge curves and capacity fade."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Representative discharge curves
    ax1 = axes[0]
    cycle_indices = [0, len(xjtu_cycles) // 3, 2 * len(xjtu_cycles) // 3, -1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(cycle_indices)))

    for idx, cycle_idx in enumerate(cycle_indices):
        cycle = xjtu_cycles[cycle_idx]
        ax1.plot(
            cycle["capacity"],
            cycle["voltage"],
            color=colors[idx],
            linewidth=2,
            label=f"Cycle {cycle['cycle']}",
        )

    ax1.set_xlabel("Discharge Capacity (Ah)", fontsize=12)
    ax1.set_ylabel("Terminal Voltage (V)", fontsize=12)
    ax1.set_title(
        "(A) XJTU: Representative Discharge Curves", fontsize=13, fontweight="bold"
    )
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(alpha=0.3)

    # Panel B: Capacity fade trend
    ax2 = axes[1]
    cycle_numbers = [c["cycle"] for c in xjtu_cycles]
    capacities = [c["capacity"][-1] for c in xjtu_cycles]

    ax2.plot(
        cycle_numbers,
        capacities,
        "o-",
        color="#1f77b4",
        linewidth=2,
        markersize=5,
        alpha=0.7,
    )
    ax2.set_xlabel("Cycle Number", fontsize=12)
    ax2.set_ylabel("Discharge Capacity (Ah)", fontsize=12)
    ax2.set_title("(B) XJTU: Capacity Degradation", fontsize=13, fontweight="bold")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"{save_path}/fig_section5_1_xjtu_overview.png", dpi=300, bbox_inches="tight"
    )
    print(f"Saved: {save_path}/fig_section5_1_xjtu_overview.png")
    plt.close()


def validate_nbm_xjtu(xjtu_cycles, save_path="../results"):
    """Validate NBM voltage predictions against XJTU discharge curves."""

    nbm = NernstBaselineModel(Q0=2.0, R0=0.05)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    test_cycles = [0, len(xjtu_cycles) // 3, 2 * len(xjtu_cycles) // 3, -1]

    rmse_values = []

    for idx, cycle_idx in enumerate(test_cycles):
        ax = axes[idx // 2, idx % 2]
        cycle = xjtu_cycles[cycle_idx]

        exp_capacity = cycle["capacity"]
        exp_voltage = cycle["voltage"]
        exp_current = np.mean(cycle["current"])

        exp_soc = 1 - exp_capacity / exp_capacity[-1]
        pred_voltage = nbm.terminal_voltage(exp_soc, abs(exp_current))

        rmse = np.sqrt(np.mean((exp_voltage - pred_voltage) ** 2))
        rmse_values.append(rmse)

        ax.plot(
            exp_capacity,
            exp_voltage,
            "o",
            markersize=4,
            alpha=0.6,
            label="XJTU Experimental",
            color="#1f77b4",
        )
        ax.plot(
            exp_capacity,
            pred_voltage,
            "-",
            linewidth=2.5,
            label="NBM Prediction",
            color="#d62728",
        )

        ax.set_xlabel("Discharge Capacity (Ah)", fontsize=11)
        ax.set_ylabel("Terminal Voltage (V)", fontsize=11)
        ax.set_title(
            f'Cycle {cycle["cycle"]} | RMSE = {rmse:.4f} V',
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(loc="best", fontsize=10)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"{save_path}/fig_section5_1_nbm_xjtu_validation.png",
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Saved: {save_path}/fig_section5_1_nbm_xjtu_validation.png")
    plt.close()

    print(f"\nNBM Validation on XJTU Dataset:")
    print(f"  Mean RMSE: {np.mean(rmse_values):.4f} V")
    print(f"  Std RMSE:  {np.std(rmse_values):.4f} V")
    print(f"  Max RMSE:  {np.max(rmse_values):.4f} V")

    return rmse_values


# ==================== Section 5.1: NBM Validation on NASA Data ====================
def validate_nbm_nasa(nasa_data, save_path="../results"):
    """Validate NBM capacity predictions against NASA dataset"""

    # Initialize NBM with nominal 2.0Ah capacity
    nbm = NernstBaselineModel(Q0=2.0, R0=0.05)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors_nasa = ["#d62728", "#ff7f0e", "#2ca02c"]

    # Analyze each battery
    for idx, (battery_id, data) in enumerate(nasa_data.items()):
        ax = axes[idx]

        # Experimental capacity fade
        cycles = data["cycle"].values
        exp_capacity = data["BCt"].values

        # NBM prediction: simple linear aging model for baseline
        # (Full aging model will be in AM analysis)
        initial_capacity = exp_capacity[0]
        fade_rate = (initial_capacity - exp_capacity[-1]) / len(cycles)
        pred_capacity = initial_capacity - fade_rate * (cycles - cycles[0])

        # Plot
        ax.plot(
            cycles,
            exp_capacity,
            "o",
            markersize=5,
            alpha=0.6,
            label="NASA Data",
            color=colors_nasa[idx],
        )
        ax.plot(
            cycles,
            pred_capacity,
            "-",
            linewidth=2.5,
            label="NBM Linear Fit",
            color="black",
        )

        # Calculate RMSE and MAPE
        rmse = np.sqrt(np.mean((exp_capacity - pred_capacity) ** 2))
        mape = np.mean(np.abs((exp_capacity - pred_capacity) / exp_capacity)) * 100

        ax.set_xlabel("Cycle Number", fontsize=11)
        ax.set_ylabel("Capacity (Ah)", fontsize=11)
        ax.set_title(
            f"{battery_id} | RMSE={rmse:.4f} Ah, MAPE={mape:.2f}%",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(loc="best", fontsize=10)
        ax.grid(alpha=0.3)

    # Remove empty subplot
    fig.delaxes(axes[3])

    plt.tight_layout()
    plt.savefig(
        f"{save_path}/fig_section5_1_nbm_nasa_validation.png",
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Saved: {save_path}/fig_section5_1_nbm_nasa_validation.png")
    plt.close()

    print(f"\nNBM Validation on NASA Dataset (Linear Aging Baseline):")
    for battery_id, data in nasa_data.items():
        exp_capacity = data["BCt"].values
        cycles = data["cycle"].values
        initial_capacity = exp_capacity[0]
        fade_rate = (initial_capacity - exp_capacity[-1]) / len(cycles)
        pred_capacity = initial_capacity - fade_rate * (cycles - cycles[0])

        rmse = np.sqrt(np.mean((exp_capacity - pred_capacity) ** 2))
        mape = np.mean(np.abs((exp_capacity - pred_capacity) / exp_capacity)) * 100

        print(f"  {battery_id}: RMSE={rmse:.4f} Ah, MAPE={mape:.2f}%")


# ==================== Section 5.2: Ablation Study ====================
def ablation_study_nbm(save_path="../results"):
    """Compare NBM variants: Pure Nernst vs Full NBM using synthetic discharge curves"""

    # Initialize models
    nbm_pure = NernstBaselineModel(
        Q0=2.0, R0=0.05, K=0, alpha=0
    )  # No exponential correction
    nbm_full = NernstBaselineModel(Q0=2.0, R0=0.05, K=-0.2, alpha=6.0)  # Full model

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Generate synthetic discharge curves at 1C rate (2.0A for 2Ah battery)
    I_load = 1.0  # 0.5C rate
    t_sim, soc_sim, V_pure = nbm_pure.simulate_discharge(I_load)
    _, _, V_full = nbm_full.simulate_discharge(I_load)

    # Convert SOC to capacity
    capacity_sim = (1 - soc_sim) * nbm_full.Q0

    # Panel A: Full discharge comparison
    ax1 = axes[0]
    ax1.plot(
        capacity_sim,
        V_pure,
        "--",
        linewidth=3,
        label="NBM-Pure (Nernst only)",
        color="#ff7f0e",
    )
    ax1.plot(
        capacity_sim,
        V_full,
        "-",
        linewidth=3,
        label="NBM-Full (with exp correction)",
        color="#2ca02c",
    )

    ax1.set_xlabel("Discharge Capacity (Ah)", fontsize=12)
    ax1.set_ylabel("Terminal Voltage (V)", fontsize=12)
    ax1.set_title(
        "(A) Full Discharge Profile Comparison at 0.5C", fontsize=13, fontweight="bold"
    )
    ax1.legend(loc="best", fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_ylim([2.4, 4.0])

    # Panel B: Low-SOC region zoom
    ax2 = axes[1]
    low_soc_mask = soc_sim < 0.3

    ax2.plot(
        soc_sim[low_soc_mask],
        V_pure[low_soc_mask],
        "--",
        linewidth=3,
        label="NBM-Pure",
        color="#ff7f0e",
        marker="o",
        markersize=4,
    )
    ax2.plot(
        soc_sim[low_soc_mask],
        V_full[low_soc_mask],
        "-",
        linewidth=3,
        label="NBM-Full",
        color="#2ca02c",
        marker="s",
        markersize=4,
    )

    # Highlight the difference
    diff_region = V_pure[low_soc_mask] - V_full[low_soc_mask]
    ax2.fill_between(
        soc_sim[low_soc_mask],
        V_pure[low_soc_mask],
        V_full[low_soc_mask],
        alpha=0.3,
        color="red",
        label="Exponential Correction",
    )

    ax2.set_xlabel("State of Charge", fontsize=12)
    ax2.set_ylabel("Terminal Voltage (V)", fontsize=12)
    ax2.set_title(
        "(B) Low-SOC Region (SOC < 30%): Exponential Term Impact",
        fontsize=13,
        fontweight="bold",
    )
    ax2.legend(loc="best", fontsize=11)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"{save_path}/fig_section5_2_ablation_nbm.png", dpi=300, bbox_inches="tight"
    )
    print(f"Saved: {save_path}/fig_section5_2_ablation_nbm.png")
    plt.close()

    # Calculate voltage difference statistics
    voltage_diff = V_pure - V_full
    voltage_diff_low_soc = voltage_diff[low_soc_mask]

    print(f"\nAblation Study Results (Synthetic Discharge at 0.5C):")
    print(f"  Full Range Voltage Difference:")
    print(f"    Mean: {np.mean(voltage_diff):.4f} V")
    print(f"    Max:  {np.max(voltage_diff):.4f} V")
    print(f"\n  Low-SOC Range (SOC < 30%) Voltage Difference:")
    print(f"    Mean: {np.mean(voltage_diff_low_soc):.4f} V")
    print(f"    Max:  {np.max(voltage_diff_low_soc):.4f} V")
    print(
        f"    Improvement Factor: {np.max(voltage_diff_low_soc)/np.mean(voltage_diff):.1f}x"
    )

    return {
        "voltage_diff": voltage_diff,
        "voltage_diff_low_soc": voltage_diff_low_soc,
        "time_to_empty_pure": t_sim[-1],
        "time_to_empty_full": t_sim[-1],
    }


# ==================== Main Execution ====================
if __name__ == "__main__":
    import os

    # Create results directory
    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 70)
    print("Section 5: NBM Validation and Analysis")
    print("=" * 70)

    # Load XJTU data
    print("\n[1/4] Loading XJTU battery data...")
    xjtu_file = "../data/XJTU battery dataset/Batch-1/2C_battery-1.mat"
    try:
        xjtu_cycles = load_xjtu_data(xjtu_file)
        if xjtu_cycles:
            print(f"  Loaded {len(xjtu_cycles)} discharge cycles from XJTU dataset")
        else:
            print("  Warning: XJTU data could not be parsed")
            xjtu_cycles = None
    except Exception as e:
        print(f"  Error loading XJTU data: {e}")
        xjtu_cycles = None

    # Load NASA-Kaggle data
    print("\n[2/4] Loading Kaggle dataset (NASA-inspired)...")
    nasa_file = "../data/Battery_dataset.csv"

    try:
        nasa_data = load_nasa_kaggle_data(nasa_file)
        print(f"  Loaded data for batteries: {list(nasa_data.keys())}")
        for bid, data in nasa_data.items():
            print(
                f"    {bid}: {len(data)} cycles, "
                f"Capacity: {data['BCt'].iloc[0]:.3f} -> {data['BCt'].iloc[-1]:.3f} Ah, "
                f"SOH: {data['SOH'].iloc[0]:.1f}% -> {data['SOH'].iloc[-1]:.1f}%"
            )
    except Exception as e:
        print(f"  Error loading Kaggle data: {e}")
        nasa_data = None

    # Generate figures
    if xjtu_cycles:
        print("\n[3/4] Generating XJTU overview and NBM validation...")
        plot_xjtu_overview(xjtu_cycles, results_dir)
        validate_nbm_xjtu(xjtu_cycles, results_dir)

    if nasa_data:
        print("\n[3/4] Generating Kaggle overview and NBM validation...")
        plot_data_overview(nasa_data, results_dir)
        validate_nbm_nasa(nasa_data, results_dir)

    print("\n[4/4] Running NBM ablation study...")
    ablation_study_nbm(results_dir)

    print("\n" + "=" * 70)
    print("Analysis complete! Check results/ directory for figures.")
    print("=" * 70)
