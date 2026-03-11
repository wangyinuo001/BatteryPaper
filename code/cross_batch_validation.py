"""
Cross-Batch Validation
======================
Goal: Demonstrate model generalizability by fitting parameters on Batch-1 (2C),
      then validating on Batch-2 (3C) data.

This tests whether the Shepherd model fitted on one charge rate can
predict discharge behavior at a different charge rate—a crucial test
for ECM publication quality.

Procedure:
  1. Fit Shepherd parameters on XJTU Batch-1 (8 batteries, 2C rate)
     - Already done: E0=3.3843, K=0.0175, A=0.8096, B=1.0062
  2. Load Batch-2 (15 batteries, 3C rate) discharge data
  3. Compute voltage prediction error (RMSE, MAE, MaxE) for each
  4. Compare with same-batch error as control

Output: results/cross_batch_validation.csv, results/cross_batch_table.tex
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize, differential_evolution

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Shepherd parameters fitted on Batch-1
SHEP_BATCH1 = {
    "E0": 3.3843,
    "K": 0.0175,
    "A": 0.8096,
    "B": 1.0062,
    "R0": 0.035,
}


def shepherd_voltage(soc, current, Q0, params):
    """Shepherd model: V = E0 - R0*I - K*(I/SOC) + A*exp(-B*Q0*(1-SOC))"""
    soc = np.clip(soc, 0.01, 1.0)
    V = (
        params["E0"]
        - params["R0"] * current
        - params["K"] * (current / soc)
        + params["A"] * np.exp(-params["B"] * Q0 * (1 - soc))
    )
    return V


def load_discharge_cycle(mat_path, cycle_idx=0):
    """Load a single discharge cycle from XJTU .mat file."""
    try:
        mat_data = loadmat(mat_path)
        data = mat_data["data"]
        if cycle_idx >= data.shape[1]:
            return None
        cycle = data[0, cycle_idx]

        voltage = cycle["voltage_V"].flatten()
        current = cycle["current_A"].flatten()
        time_min = cycle["relative_time_min"].flatten()

        # Select discharge segments
        mask = current < -0.1
        if mask.sum() < 50:
            return None

        time_s = time_min[mask] * 60.0
        time_s -= time_s[0]  # Normalize to start at 0
        voltage_d = voltage[mask]
        current_d = -current[mask]  # Make positive

        # Compute SOC
        dt = np.diff(time_s, prepend=time_s[0])
        dt[0] = dt[1] if len(dt) > 1 else 1.0
        capacity = np.cumsum(current_d * dt) / 3600.0
        Q_total = capacity[-1]
        soc = np.clip(1 - capacity / Q_total, 0.01, 1.0)

        return {
            "time_s": time_s,
            "voltage": voltage_d,
            "current": current_d,
            "soc": soc,
            "Q_total": Q_total,
        }
    except Exception as e:
        print(f"  Error loading {mat_path}: {e}")
        return None


def compute_metrics(v_pred, v_true):
    """Compute prediction error metrics."""
    err = v_pred - v_true
    return {
        "RMSE (mV)": np.sqrt(np.mean(err**2)) * 1000,
        "MAE (mV)": np.mean(np.abs(err)) * 1000,
        "MaxError (mV)": np.max(np.abs(err)) * 1000,
        "MeanRelErr (%)": np.mean(np.abs(err / v_true)) * 100,
    }


def fit_shepherd_to_data(data):
    """Re-fit Shepherd parameters to a given discharge dataset using optimization."""
    soc = data["soc"]
    current = data["current"]
    voltage = data["voltage"]
    Q0 = data["Q_total"]

    def objective(x):
        E0, K, A, B, R0 = x
        params = {"E0": E0, "K": K, "A": A, "B": B, "R0": R0}
        v_pred = shepherd_voltage(soc, current, Q0, params)
        return np.sqrt(np.mean((v_pred - voltage) ** 2))

    bounds = [(3.0, 4.5), (0.001, 0.1), (0.1, 2.0), (0.1, 5.0), (0.0, 0.3)]
    x0 = [3.3843, 0.0175, 0.8096, 1.0062, 0.035]

    result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B")

    if result.success or result.fun < 0.1:
        E0, K, A, B, R0 = result.x
        return {"E0": E0, "K": K, "A": A, "B": B, "R0": R0, "RMSE_fit": result.fun}
    else:
        # Fallback: differential_evolution for global search
        result_de = differential_evolution(objective, bounds, seed=42, maxiter=200)
        E0, K, A, B, R0 = result_de.x
        return {"E0": E0, "K": K, "A": A, "B": B, "R0": R0, "RMSE_fit": result_de.fun}


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "..", "results")
    data_dir = os.path.join(script_dir, "..", "data", "XJTU battery dataset")
    os.makedirs(results_dir, exist_ok=True)

    # ====== Part A: Load Batch-1 data (2C) - same batch as fitting ======
    print("=" * 80)
    print("CROSS-BATCH VALIDATION")
    print("=" * 80)
    print("\n--- Loading Batch-1 (2C, training set) ---")

    batch1_data = []
    batch1_dir = os.path.join(data_dir, "Batch-1")
    for i in range(1, 9):
        mat_path = os.path.join(batch1_dir, f"2C_battery-{i}.mat")
        d = load_discharge_cycle(mat_path, cycle_idx=0)
        if d is not None:
            d["name"] = f"B1-Bat{i}"
            batch1_data.append(d)
            print(f"  {d['name']}: Q={d['Q_total']:.3f}Ah, {len(d['voltage'])} pts")

    # ====== Part B: Load Batch-2 data (3C) - cross-domain test ======
    print("\n--- Loading Batch-2 (3C, test set) ---")

    batch2_data = []
    batch2_dir = os.path.join(data_dir, "Batch-2")
    for i in range(1, 16):
        mat_path = os.path.join(batch2_dir, f"3C_battery-{i}.mat")
        if os.path.exists(mat_path):
            d = load_discharge_cycle(mat_path, cycle_idx=0)
            if d is not None:
                d["name"] = f"B2-Bat{i}"
                batch2_data.append(d)
                print(f"  {d['name']}: Q={d['Q_total']:.3f}Ah, {len(d['voltage'])} pts")

    # ====== Part C: Evaluate Batch-1 parameters on both batches ======
    print("\n--- Evaluating Batch-1 fitted parameters ---")
    rows = []

    # Same-batch (Batch-1 → Batch-1) as control
    for d in batch1_data:
        v_pred = shepherd_voltage(d["soc"], d["current"], d["Q_total"], SHEP_BATCH1)
        m = compute_metrics(v_pred, d["voltage"])
        row = {"Battery": d["name"], "Batch": "Batch-1 (Same)", "Condition": "2C"}
        row.update(m)
        rows.append(row)
        print(f"  {d['name']:10s} (same): RMSE={m['RMSE (mV)']:7.2f}mV")

    # Cross-batch (Batch-1 → Batch-2) transfer
    for d in batch2_data:
        v_pred = shepherd_voltage(d["soc"], d["current"], d["Q_total"], SHEP_BATCH1)
        m = compute_metrics(v_pred, d["voltage"])
        row = {"Battery": d["name"], "Batch": "Batch-2 (Cross)", "Condition": "3C"}
        row.update(m)
        rows.append(row)
        print(f"  {d['name']:10s} (cross): RMSE={m['RMSE (mV)']:7.2f}mV")

    df = pd.DataFrame(rows)

    # ====== Part D: Re-fit on Batch-2 for upper-bound comparison ======
    print("\n--- Re-fitting Shepherd on Batch-2 (3C) for reference ---")
    if batch2_data:
        ref_b2 = batch2_data[0]
        b2_params = fit_shepherd_to_data(ref_b2)
        print(
            f"  Batch-2 fitted params: E0={b2_params['E0']:.4f}, K={b2_params['K']:.4f}, "
            f"A={b2_params['A']:.4f}, B={b2_params['B']:.4f}, R0={b2_params['R0']:.4f}"
        )
        print(f"  Fit RMSE: {b2_params['RMSE_fit']*1000:.2f}mV")

        # Evaluate re-fitted on all Batch-2
        refit_rows = []
        for d in batch2_data:
            v_pred = shepherd_voltage(d["soc"], d["current"], d["Q_total"], b2_params)
            m = compute_metrics(v_pred, d["voltage"])
            refit_rows.append(m)
            print(f"  {d['name']:10s} (refit): RMSE={m['RMSE (mV)']:7.2f}mV")

        # Average refit RMSE
        avg_refit_rmse = np.mean([r["RMSE (mV)"] for r in refit_rows])
    else:
        avg_refit_rmse = None

    # ====== Summary Statistics ======
    print("\n" + "=" * 80)
    print("CROSS-BATCH VALIDATION SUMMARY")
    print("=" * 80)

    b1_metrics = df[df["Batch"] == "Batch-1 (Same)"]
    b2_metrics = df[df["Batch"] == "Batch-2 (Cross)"]

    b1_rmse_mean = b1_metrics["RMSE (mV)"].mean()
    b1_rmse_std = b1_metrics["RMSE (mV)"].std()
    b2_rmse_mean = b2_metrics["RMSE (mV)"].mean()
    b2_rmse_std = b2_metrics["RMSE (mV)"].std()

    print(
        f"\nSame-batch (B1→B1):  RMSE = {b1_rmse_mean:.2f} ± {b1_rmse_std:.2f} mV "
        f"(n={len(b1_metrics)})"
    )
    print(
        f"Cross-batch (B1→B2): RMSE = {b2_rmse_mean:.2f} ± {b2_rmse_std:.2f} mV "
        f"(n={len(b2_metrics)})"
    )
    if avg_refit_rmse is not None:
        print(f"Refit on B2 (B2→B2): RMSE = {avg_refit_rmse:.2f} mV (reference)")

    degradation = (b2_rmse_mean - b1_rmse_mean) / b1_rmse_mean * 100
    print(f"\nCross-domain RMSE increase: {degradation:+.1f}%")

    # ====== Save results ======
    csv_path = os.path.join(results_dir, "cross_batch_validation.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    summary_rows = [
        {
            "Evaluation": "Same-batch (B1→B1)",
            "N": len(b1_metrics),
            "RMSE (mV)": f"{b1_rmse_mean:.2f} ± {b1_rmse_std:.2f}",
            "MAE (mV)": f"{b1_metrics['MAE (mV)'].mean():.2f} ± {b1_metrics['MAE (mV)'].std():.2f}",
            "MaxError (mV)": f"{b1_metrics['MaxError (mV)'].mean():.1f}",
        },
        {
            "Evaluation": "Cross-batch (B1→B2)",
            "N": len(b2_metrics),
            "RMSE (mV)": f"{b2_rmse_mean:.2f} ± {b2_rmse_std:.2f}",
            "MAE (mV)": f"{b2_metrics['MAE (mV)'].mean():.2f} ± {b2_metrics['MAE (mV)'].std():.2f}",
            "MaxError (mV)": f"{b2_metrics['MaxError (mV)'].mean():.1f}",
        },
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(results_dir, "cross_batch_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")

    # ====== LaTeX table ======
    tex_path = os.path.join(results_dir, "cross_batch_table.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Cross-Batch Validation Table\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write(
            "\\caption{Cross-batch validation: Shepherd model fitted on XJTU Batch-1 "
            "(2C rate) evaluated on Batch-2 (3C rate)}\n"
        )
        f.write("\\label{tab:cross_batch}\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\hline\n")
        f.write("Evaluation & N & RMSE (mV) & MAE (mV) \\\\\n")
        f.write("\\hline\n")
        for _, row in summary_df.iterrows():
            f.write(
                f"{row['Evaluation']} & {row['N']} & {row['RMSE (mV)']} & "
                f"{row['MAE (mV)']} \\\\\n"
            )
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"Saved: {tex_path}")

    # ====== Visualization ======
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Batch-1 representative voltage curve
    if batch1_data:
        ax = axes[0, 0]
        d = batch1_data[0]
        v_pred = shepherd_voltage(d["soc"], d["current"], d["Q_total"], SHEP_BATCH1)
        ax.plot(
            d["time_s"] / 60, d["voltage"], "k-", lw=2, label="Experimental", alpha=0.8
        )
        ax.plot(d["time_s"] / 60, v_pred, "r--", lw=1.5, label="Shepherd (Batch-1 fit)")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Voltage (V)")
        ax.set_title(f'Batch-1 (Same-batch): {d["name"]}')
        ax.legend()
        ax.grid(alpha=0.3)

    # Panel B: Batch-2 representative voltage curve
    if batch2_data:
        ax = axes[0, 1]
        d = batch2_data[0]
        v_pred = shepherd_voltage(d["soc"], d["current"], d["Q_total"], SHEP_BATCH1)
        ax.plot(
            d["time_s"] / 60, d["voltage"], "k-", lw=2, label="Experimental", alpha=0.8
        )
        ax.plot(d["time_s"] / 60, v_pred, "r--", lw=1.5, label="Shepherd (Batch-1 fit)")
        if batch2_data and "b2_params" in dir():
            v_refit = shepherd_voltage(d["soc"], d["current"], d["Q_total"], b2_params)
            ax.plot(
                d["time_s"] / 60,
                v_refit,
                "b:",
                lw=1.5,
                label="Shepherd (Batch-2 refit)",
            )
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Voltage (V)")
        ax.set_title(f'Batch-2 (Cross-batch): {d["name"]}')
        ax.legend()
        ax.grid(alpha=0.3)

    # Panel C: RMSE box plot comparison
    ax = axes[1, 0]
    box_data = [b1_metrics["RMSE (mV)"].values, b2_metrics["RMSE (mV)"].values]
    bp = ax.boxplot(
        box_data,
        labels=["Same-batch\n(B1→B1)", "Cross-batch\n(B1→B2)"],
        patch_artist=True,
    )
    bp["boxes"][0].set_facecolor("#3b82f6")
    bp["boxes"][1].set_facecolor("#ef4444")
    for box in bp["boxes"]:
        box.set_alpha(0.7)
    ax.set_ylabel("RMSE (mV)")
    ax.set_title("Voltage Prediction Error Distribution")
    ax.grid(axis="y", alpha=0.3)

    # Panel D: per-battery RMSE bar chart
    ax = axes[1, 1]
    all_names = list(df["Battery"])
    all_rmse = list(df["RMSE (mV)"])
    colors = ["#3b82f6" if "B1" in n else "#ef4444" for n in all_names]
    ax.barh(range(len(all_names)), all_rmse, color=colors, alpha=0.8)
    ax.set_yticks(range(len(all_names)))
    ax.set_yticklabels(all_names, fontsize=7)
    ax.set_xlabel("RMSE (mV)")
    ax.set_title("Per-Battery RMSE")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    plt.suptitle(
        "Cross-Batch Validation: Shepherd Model Generalizability",
        fontweight="bold",
        fontsize=13,
    )
    plt.tight_layout()
    fig_path = os.path.join(results_dir, "fig_cross_batch_validation.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {fig_path}")
    plt.close()


if __name__ == "__main__":
    main()
