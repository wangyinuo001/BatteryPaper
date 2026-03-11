"""
Baseline Comparison Experiment
===============================
Compare ALL voltage models on XJTU Batch-1 discharge data:
  1. NBM (Nernst-Based Model) - your baseline
  2. Shepherd-Based Model (Proposed) - your main model
  3. Rint (simplest ECM) - new baseline
  4. 1st-order Thevenin ECM - new baseline
  5. 2nd-order Thevenin ECM - new baseline

Metrics: RMSE, MAE, MaxError, MeanRelError
Output: results/baseline_comparison.csv, results/baseline_comparison_table.tex
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from thevenin_ecm import (
    RintModel,
    TheveninFirstOrder,
    TheveninSecondOrder,
    identify_thevenin_params,
    fit_ocv_soc,
    ocv_polynomial,
)

# ==================== Shepherd & NBM from existing code ====================
# NBM Parameters (from voltage_error_analysis.py)
NBM_E0 = 4.5000
NBM_K = -1.1718
NBM_ALPHA = 0.8022
NBM_R0 = 0.0000
NBM_RT_NF = 0.0257

# Shepherd Parameters (from main_model.py)
SHEP_E0 = 3.3843
SHEP_K = 0.0175
SHEP_A = 0.8096
SHEP_B = 1.0062
SHEP_R0 = 0.0000  # R0 set to 0 for point-wise comparison (no dynamic state)


def nbm_voltage(soc, current):
    soc = np.clip(soc, 0.01, 0.99)
    nernst = NBM_RT_NF * np.log(soc / (1 - soc))
    exponential = NBM_K * np.exp(-NBM_ALPHA * soc)
    return NBM_E0 + nernst + exponential - current * NBM_R0


def shepherd_voltage(soc, current, Q0):
    soc = np.clip(soc, 0.01, 1.0)
    polarization = -SHEP_K * (current / soc)
    exponential = SHEP_A * np.exp(-SHEP_B * Q0 * (1 - soc))
    return SHEP_E0 + polarization + exponential - SHEP_R0 * current


# ==================== Data Loading ====================
def load_discharge_data(mat_path, cycle_idx=0):
    """Load a single discharge cycle from XJTU .mat file."""
    mat_data = loadmat(mat_path)
    data = mat_data["data"]
    cycle = data[0, cycle_idx]

    voltage = cycle["voltage_V"].flatten()
    current = cycle["current_A"].flatten()
    time_min = cycle["relative_time_min"].flatten()

    # Select discharge (current < 0)
    mask = current < 0
    if mask.sum() < 100:
        return None

    time_s = time_min[mask] * 60.0
    voltage_d = voltage[mask]
    current_d = -current[mask]  # Make positive

    # Compute SOC
    dt = np.diff(time_s, prepend=time_s[0])
    capacity = np.cumsum(current_d * dt) / 3600.0
    Q_total = capacity[-1]
    soc = 1 - capacity / Q_total

    return {
        "time_s": time_s,
        "voltage": voltage_d,
        "current": current_d,
        "soc": soc,
        "Q_total": Q_total,
    }


def compute_metrics(v_pred, v_true):
    """Compute voltage prediction error metrics."""
    error = v_pred - v_true
    return {
        "RMSE (mV)": np.sqrt(np.mean(error**2)) * 1000,
        "MAE (mV)": np.mean(np.abs(error)) * 1000,
        "MaxError (mV)": np.max(np.abs(error)) * 1000,
        "MeanRelErr (%)": np.mean(np.abs(error / v_true)) * 100,
        "StdErr (mV)": np.std(error) * 1000,
    }


def run_pointwise_comparison(data, ocv_coeffs, thev1_params, thev2_params):
    """Run point-wise voltage prediction for all models."""
    soc = data["soc"]
    current = data["current"]
    voltage = data["voltage"]
    Q0 = data["Q_total"]
    time_s = data["time_s"]

    results = {}

    # 1. NBM
    v_nbm = nbm_voltage(soc, current)
    results["NBM"] = compute_metrics(v_nbm, voltage)

    # 2. Shepherd (Proposed)
    v_shep = shepherd_voltage(soc, current, Q0)
    results["Shepherd (Proposed)"] = compute_metrics(v_shep, voltage)

    # 3. Rint
    R_int = thev1_params["R0"]  # Use same R0
    v_rint = ocv_polynomial(soc, ocv_coeffs) - R_int * current
    results["Rint"] = compute_metrics(v_rint, voltage)

    # 4. 1st-order Thevenin (simulate with actual current)
    R0 = thev1_params["R0"]
    R1, C1 = thev1_params["R1"], thev1_params["C1"]
    v_rc1 = 0.0
    v_thev1 = []
    for i in range(len(time_s)):
        I = current[i]
        ocv = ocv_polynomial(soc[i], ocv_coeffs)
        V = ocv - R0 * I - v_rc1
        v_thev1.append(V)
        if i < len(time_s) - 1:
            dt_i = time_s[i + 1] - time_s[i]
            v_rc1 += (I / C1 - v_rc1 / (R1 * C1)) * dt_i
    v_thev1 = np.array(v_thev1)
    results["Thevenin-1RC"] = compute_metrics(v_thev1, voltage)

    # 5. 2nd-order Thevenin
    R0 = thev2_params["R0"]
    R1, C1 = thev2_params["R1"], thev2_params["C1"]
    R2, C2 = thev2_params["R2"], thev2_params["C2"]
    v_rc1, v_rc2 = 0.0, 0.0
    v_thev2 = []
    for i in range(len(time_s)):
        I = current[i]
        ocv = ocv_polynomial(soc[i], ocv_coeffs)
        V = ocv - R0 * I - v_rc1 - v_rc2
        v_thev2.append(V)
        if i < len(time_s) - 1:
            dt_i = time_s[i + 1] - time_s[i]
            v_rc1 += (I / C1 - v_rc1 / (R1 * C1)) * dt_i
            v_rc2 += (I / C2 - v_rc2 / (R2 * C2)) * dt_i
    v_thev2 = np.array(v_thev2)
    results["Thevenin-2RC"] = compute_metrics(v_thev2, voltage)

    predictions = {
        "NBM": v_nbm,
        "Shepherd (Proposed)": v_shep,
        "Rint": v_rint,
        "Thevenin-1RC": v_thev1,
        "Thevenin-2RC": v_thev2,
    }

    return results, predictions


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "..", "results")
    data_dir = os.path.join(script_dir, "..", "data", "XJTU battery dataset")
    os.makedirs(results_dir, exist_ok=True)

    # ====== Load multiple batteries for robust comparison ======
    batteries = []
    for i in range(1, 9):
        mat_path = os.path.join(data_dir, "Batch-1", f"2C_battery-{i}.mat")
        if os.path.exists(mat_path):
            data = load_discharge_data(mat_path, cycle_idx=0)
            if data is not None:
                data["name"] = f"Bat-{i}"
                batteries.append(data)
                print(
                    f"Loaded Batch-1/2C_battery-{i}: Q={data['Q_total']:.3f}Ah, "
                    f"{len(data['voltage'])} points"
                )

    if not batteries:
        print("ERROR: No battery data found!")
        return

    # ====== Fit Thevenin parameters on first battery ======
    ref = batteries[0]
    print(f"\n--- Fitting Thevenin ECM parameters on {ref['name']} ---")

    ocv_coeffs = fit_ocv_soc(ref["soc"], ref["voltage"], degree=6)
    print(f"OCV polynomial coefficients: {[f'{c:.4f}' for c in ocv_coeffs]}")

    print("Fitting 1st-order Thevenin...")
    thev1_params = identify_thevenin_params(
        ref["time_s"], ref["voltage"], -ref["current"], ref["Q_total"], order=1
    )
    print(
        f"  R0={thev1_params['R0']:.4f}, R1={thev1_params['R1']:.4f}, "
        f"C1={thev1_params['C1']:.1f}, RMSE_fit={thev1_params['rmse_fit']*1000:.2f}mV"
    )

    print("Fitting 2nd-order Thevenin...")
    thev2_params = identify_thevenin_params(
        ref["time_s"], ref["voltage"], -ref["current"], ref["Q_total"], order=2
    )
    print(
        f"  R0={thev2_params['R0']:.4f}, R1={thev2_params['R1']:.4f}, "
        f"C1={thev2_params['C1']:.1f}, R2={thev2_params['R2']:.4f}, "
        f"C2={thev2_params['C2']:.1f}, RMSE_fit={thev2_params['rmse_fit']*1000:.2f}mV"
    )

    # ====== Run comparison on ALL batteries ======
    all_rows = []
    for bat in batteries:
        print(f"\n--- Evaluating on {bat['name']} ---")
        metrics, preds = run_pointwise_comparison(
            bat, ocv_coeffs, thev1_params, thev2_params
        )
        for model_name, m in metrics.items():
            row = {"Battery": bat["name"], "Model": model_name}
            row.update(m)
            all_rows.append(row)
            print(
                f"  {model_name:20s}: RMSE={m['RMSE (mV)']:7.2f}mV  "
                f"MAE={m['MAE (mV)']:7.2f}mV  MaxE={m['MaxError (mV)']:7.1f}mV"
            )

    df = pd.DataFrame(all_rows)

    # ====== Aggregate: mean ± std across batteries ======
    agg = (
        df.groupby("Model")
        .agg(
            {
                "RMSE (mV)": ["mean", "std"],
                "MAE (mV)": ["mean", "std"],
                "MaxError (mV)": ["mean", "std"],
                "MeanRelErr (%)": ["mean", "std"],
            }
        )
        .round(2)
    )

    print("\n" + "=" * 90)
    print("BASELINE COMPARISON RESULTS (averaged over 8 batteries)")
    print("=" * 90)

    summary_rows = []
    for model in ["NBM", "Rint", "Thevenin-1RC", "Thevenin-2RC", "Shepherd (Proposed)"]:
        if model in agg.index:
            rmse_m = agg.loc[model, ("RMSE (mV)", "mean")]
            rmse_s = agg.loc[model, ("RMSE (mV)", "std")]
            mae_m = agg.loc[model, ("MAE (mV)", "mean")]
            mae_s = agg.loc[model, ("MAE (mV)", "std")]
            maxe_m = agg.loc[model, ("MaxError (mV)", "mean")]
            rel_m = agg.loc[model, ("MeanRelErr (%)", "mean")]
            print(
                f"  {model:20s}: RMSE={rmse_m:6.2f}±{rmse_s:.2f}  "
                f"MAE={mae_m:6.2f}±{mae_s:.2f}  MaxE={maxe_m:6.1f}  RelErr={rel_m:.3f}%"
            )
            summary_rows.append(
                {
                    "Model": model,
                    "RMSE (mV)": f"{rmse_m:.2f} ± {rmse_s:.2f}",
                    "MAE (mV)": f"{mae_m:.2f} ± {mae_s:.2f}",
                    "MaxError (mV)": f"{maxe_m:.1f}",
                    "MeanRelErr (%)": f"{rel_m:.3f}",
                }
            )

    # ====== Save results ======
    csv_path = os.path.join(results_dir, "baseline_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved per-battery results: {csv_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(results_dir, "baseline_comparison_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary: {summary_csv}")

    # ====== LaTeX table ======
    tex_path = os.path.join(results_dir, "baseline_comparison_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Baseline Comparison Table\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write(
            "\\caption{Voltage prediction accuracy comparison across baseline models "
            "(averaged over 8 XJTU Batch-1 batteries, Cycle 0)}\n"
        )
        f.write("\\label{tab:baseline_comparison}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\hline\n")
        f.write(
            "Model & RMSE (mV) & MAE (mV) & Max Error (mV) & Rel. Error (\\%) \\\\\n"
        )
        f.write("\\hline\n")
        for _, row in summary_df.iterrows():
            f.write(
                f"{row['Model']} & {row['RMSE (mV)']} & {row['MAE (mV)']} & "
                f"{row['MaxError (mV)']} & {row['MeanRelErr (%)']} \\\\\n"
            )
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"Saved LaTeX: {tex_path}")

    # ====== Visualization ======
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Use first battery for visualization
    bat = batteries[0]
    _, preds = run_pointwise_comparison(bat, ocv_coeffs, thev1_params, thev2_params)

    colors = {
        "NBM": "#3b82f6",
        "Rint": "#f59e0b",
        "Thevenin-1RC": "#10b981",
        "Thevenin-2RC": "#8b5cf6",
        "Shepherd (Proposed)": "#ef4444",
    }

    ax1 = axes[0]
    ax1.plot(
        bat["time_s"] / 3600,
        bat["voltage"],
        "k-",
        lw=2,
        label="Experimental",
        alpha=0.8,
    )
    for name, v in preds.items():
        ax1.plot(bat["time_s"] / 3600, v, "--", lw=1.2, color=colors[name], label=name)
    ax1.set_xlabel("Time (h)")
    ax1.set_ylabel("Voltage (V)")
    ax1.set_title("Voltage Predictions")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2 = axes[1]
    for name, v in preds.items():
        err = (v - bat["voltage"]) * 1000
        ax2.plot(bat["time_s"] / 3600, err, "-", lw=1.2, color=colors[name], label=name)
    ax2.axhline(0, color="k", ls="--", lw=0.8)
    ax2.set_xlabel("Time (h)")
    ax2.set_ylabel("Error (mV)")
    ax2.set_title("Prediction Errors")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.suptitle(
        "Baseline Comparison: Voltage Accuracy on XJTU Batch-1", fontweight="bold"
    )
    plt.tight_layout()
    fig_path = os.path.join(results_dir, "fig_baseline_comparison.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {fig_path}")
    plt.close()


if __name__ == "__main__":
    main()
