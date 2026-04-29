"""
NASA Randomized Battery Usage – Dynamic Load Validation
========================================================
Validates the Shepherd voltage model on NASA PCoE Dataset #11,
which cycles 18650 Li-ion cells with randomly generated current
profiles (0.5–4 A, 5-min segments) at room temperature.

Strategy
--------
1. Load reference CC discharge (2 A → 3.2 V) to fit Shepherd parameters
   for the NASA cell chemistry.
2. Predict voltage under random-walk (RW) discharge profiles.
3. Compute RMSE / MAE between measured and predicted voltage.
4. Generate publication-quality figure.

Citation
--------
B. Bole, C. Kulkarni, M. Daigle, "Adaptation of an Electrochemistry-
based Li-Ion Battery Model to Account for Deterioration Observed Under
Randomized Use", Annual Conf. PHM Society, 2014.
"""

import sys, os, pathlib
import numpy as np
from scipy.optimize import minimize
import scipy.io as sio
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pub_style import (
    apply_style,
    save_fig,
    label_panel,
    embed_metric,
    COLORS,
    DOUBLE_COL_TALL,
)

_PALETTE = [
    COLORS["exp"],
    COLORS["shepherd"],
    COLORS["nbm"],
    COLORS["rint"],
    COLORS["thev1"],
    COLORS["thev2"],
]

# ── paths ──────────────────────────────────────────────────────────
BASE = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = (
    BASE
    / "data"
    / "11. Randomized Battery Usage Data Set"
    / "RW_RoomTemp"
    / "Battery_Uniform_Distribution_Discharge_Room_Temp_DataSet_2Post"
    / "data"
    / "Matlab"
)
RESULTS_DIR = BASE / "results"
FIG_DIR = RESULTS_DIR
FIG_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════
#  Helper: load a .mat battery file
# ══════════════════════════════════════════════════════════════════
def load_battery(name="RW3"):
    """Return the step array from a NASA .mat file."""
    path = DATA_DIR / f"{name}.mat"
    if not path.exists():
        raise FileNotFoundError(path)
    mat = sio.loadmat(str(path))
    return mat["data"]["step"][0, 0]


def extract_steps(steps, comment_filter):
    """Yield (voltage, current, temperature, relativeTime) for matching steps."""
    for i in range(steps.shape[1]):
        s = steps[0, i]
        c = str(s["comment"][0]) if s["comment"].size else ""
        if c == comment_filter:
            V = s["voltage"][0].flatten()
            I = s["current"][0].flatten()
            T = s["temperature"][0].flatten()
            t = s["relativeTime"][0].flatten()
            yield V, I, T, t


def build_rw_cycles(steps, max_cycles=None):
    """
    Build full random-walk discharge cycles.
    Each cycle = all discharge segments between consecutive charges.
    Returns list of dicts {V, I, T, t} with continuous time.
    """
    # Find indices of 'charge (after random walk discharge)'
    charge_idx = []
    for i in range(steps.shape[1]):
        c = str(steps[0, i]["comment"][0]) if steps[0, i]["comment"].size else ""
        if c == "charge (after random walk discharge)":
            charge_idx.append(i)

    cycles = []
    n = min(len(charge_idx) - 1, max_cycles) if max_cycles else len(charge_idx) - 1
    for ci in range(n):
        start = charge_idx[ci] + 1
        end = charge_idx[ci + 1]
        all_V, all_I, all_T, all_t = [], [], [], []
        t_off = 0.0
        for j in range(start, end):
            s = steps[0, j]
            stype = str(s["type"][0]) if s["type"].size else ""
            c = str(s["comment"][0]) if s["comment"].size else ""
            V = s["voltage"][0].flatten()
            I = s["current"][0].flatten()
            T = s["temperature"][0].flatten()
            rt = s["relativeTime"][0].flatten()
            if c == "discharge (random walk)":
                # Only include active discharge segments
                all_V.extend(V)
                all_I.extend(I)
                all_T.extend(T)
                all_t.extend(rt + t_off)
            # Still advance time offset for SOC accounting
            t_off += rt[-1] if len(rt) else 0.0
        if len(all_V) > 10:
            cycles.append(
                {
                    "V": np.array(all_V),
                    "I": np.array(all_I),
                    "T": np.array(all_T),
                    "t": np.array(all_t),
                }
            )
    return cycles


# ══════════════════════════════════════════════════════════════════
#  Shepherd model (same formulation as paper, re-parameterised)
# ══════════════════════════════════════════════════════════════════
def shepherd_voltage(soc, I, E0, K, A, B, R0):
    """V = E0 - R0*I - K*(I/SOC) + A*exp(-B*(1-SOC))"""
    soc = np.clip(soc, 0.01, 1.0)
    return E0 - R0 * I - K * (I / soc) + A * np.exp(-B * (1 - soc))


def simulate_discharge(I_profile, dt_profile, Q_cap, E0, K, A, B, R0, soc0=1.0):
    """
    Euler-integrate the Shepherd model given a measured current profile.

    Parameters
    ----------
    I_profile : 1-D array  – measured current (A) at each sample
    dt_profile : 1-D array – time step (s) between consecutive samples
    Q_cap : float           – effective capacity (Ah)
    soc0 : float            – initial SOC

    Returns
    -------
    V_pred : array of predicted terminal voltages
    soc    : array of SOC values
    """
    n = len(I_profile)
    V_pred = np.zeros(n)
    soc_arr = np.zeros(n)
    soc = soc0
    for k in range(n):
        soc_arr[k] = soc
        V_pred[k] = shepherd_voltage(soc, I_profile[k], E0, K, A, B, R0)
        if k < n - 1:
            dsoc = -I_profile[k] * dt_profile[k] / (Q_cap * 3600.0)
            soc = np.clip(soc + dsoc, 0.001, 1.0)
    return V_pred, soc_arr


# ══════════════════════════════════════════════════════════════════
#  Step 1: Fit Shepherd parameters on reference CC discharge
# ══════════════════════════════════════════════════════════════════
def fit_shepherd_on_ref(steps):
    """
    Fit (E0, K, A, B, R0) on multiple reference discharges (CC @ ~1 A).
    Uses up to 5 reference cycles for robust fitting.
    Returns fitted parameters and the effective capacity Q_cap.
    """
    ref_data = []
    for V_meas, I_meas, T_meas, t_meas in extract_steps(steps, "reference discharge"):
        ref_data.append((V_meas, I_meas, T_meas, t_meas))
        if len(ref_data) >= 5:
            break

    # Use first reference to estimate capacity
    V0, I0, T0, t0 = ref_data[0]
    I_const = np.median(I0)
    Q_cap = I_const * t0[-1] / 3600.0
    print(
        f"  Reference discharges used: {len(ref_data)}, "
        f"I={I_const:.2f} A, Q_cap={Q_cap:.3f} Ah"
    )

    # Build SOC trajectories for all reference cycles
    ref_socs = []
    for V_m, I_m, T_m, t_m in ref_data:
        dt_m = np.diff(t_m)
        soc_m = np.zeros(len(t_m))
        soc_m[0] = 1.0
        for k in range(1, len(t_m)):
            dsoc = -I_m[k - 1] * dt_m[k - 1] / (Q_cap * 3600.0)
            soc_m[k] = soc_m[k - 1] + dsoc
        ref_socs.append(soc_m)

    def cost(params):
        E0, K, A, B, R0 = params
        if R0 < 0 or K < 0 or E0 < 2 or E0 > 5:
            return 1e6
        total_err = 0.0
        total_pts = 0
        for idx_r, (V_m, I_m, T_m, t_m) in enumerate(ref_data):
            I_c = np.median(I_m)
            V_pred = shepherd_voltage(ref_socs[idx_r], I_c, E0, K, A, B, R0)
            total_err += np.sum((V_pred - V_m) ** 2)
            total_pts += len(V_m)
        return total_err / total_pts

    x0 = [3.38, 0.015, 0.80, 1.0, 0.06]
    bounds = [(2.5, 4.5), (0.001, 0.2), (0.01, 2.0), (0.1, 50.0), (0.005, 0.5)]
    res = minimize(cost, x0, method="L-BFGS-B", bounds=bounds)
    E0, K, A, B, R0 = res.x
    V_fit = shepherd_voltage(ref_socs[0], I_const, E0, K, A, B, R0)
    rmse_fit = np.sqrt(np.mean((V_fit - ref_data[0][0]) ** 2)) * 1000
    print(f"  Fitted: E0={E0:.4f}, K={K:.4f}, A={A:.4f}, " f"B={B:.4f}, R0={R0:.4f}")
    print(f"  CC fit RMSE = {rmse_fit:.2f} mV")
    return {
        "E0": E0,
        "K": K,
        "A": A,
        "B": B,
        "R0": R0,
        "Q_cap": Q_cap,
        "rmse_fit_mV": rmse_fit,
    }


# ══════════════════════════════════════════════════════════════════
#  Step 2: Evaluate on random-walk cycles
# ══════════════════════════════════════════════════════════════════
def evaluate_rw_cycles(steps, params, n_cycles=50):
    """Predict voltage for the first *n_cycles* RW discharge cycles."""
    cycles = build_rw_cycles(steps, max_cycles=n_cycles)
    p = params
    results = []
    for idx, cyc in enumerate(cycles):
        I = cyc["I"]
        V_meas = cyc["V"]
        t = cyc["t"]
        dt = np.diff(t)
        dt = np.append(dt, dt[-1])  # pad last

        V_pred, soc = simulate_discharge(
            I, dt, p["Q_cap"], p["E0"], p["K"], p["A"], p["B"], p["R0"]
        )

        err = V_pred - V_meas
        rmse = np.sqrt(np.mean(err**2)) * 1000  # mV
        mae = np.mean(np.abs(err)) * 1000
        results.append(
            {
                "cycle": idx,
                "rmse_mV": rmse,
                "mae_mV": mae,
                "n_pts": len(V_meas),
                "t": t,
                "V_meas": V_meas,
                "V_pred": V_pred,
                "I": I,
                "soc": soc,
                "err": err,
            }
        )
    return results


# ══════════════════════════════════════════════════════════════════
#  Step 3: Publication figure
# ══════════════════════════════════════════════════════════════════
def make_figure(results, params, battery_name, save=True):
    """Three-panel figure: (a) voltage overlay, (b) current profile, (c) error."""
    apply_style()

    # Pick a representative mid-life cycle for the figure
    mid = min(10, len(results) - 1)
    r = results[mid]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=DOUBLE_COL_TALL,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.6, 1.6], "hspace": 0.12},
    )

    t_min = r["t"] / 60.0  # seconds → minutes

    # (a) Voltage
    ax = axes[0]
    ax.plot(
        t_min, r["V_meas"], "-", color=_PALETTE[0], lw=1.0, label="Measured", alpha=0.85
    )
    ax.plot(
        t_min, r["V_pred"], "--", color=_PALETTE[1], lw=1.1, label="Shepherd prediction"
    )
    ax.set_ylabel("Terminal voltage (V)")
    ax.legend(loc="lower left", fontsize=7, ncol=2)
    embed_metric(
        ax,
        f"RMSE = {r['rmse_mV']:.1f} mV   MAE = {r['mae_mV']:.1f} mV",
        x=0.97,
        y=0.05,
    )
    ax.set_title(
        f"NASA {battery_name} \u2014 random-walk cycle {r['cycle']+1}",
        fontsize=8.5,
    )
    label_panel(ax, "a")

    # (b) Current
    ax = axes[1]
    ax.plot(t_min, r["I"], "-", color=_PALETTE[2], lw=0.7)
    ax.set_ylabel("Current (A)")
    ax.axhline(0, color="grey", lw=0.4, ls=":")
    label_panel(ax, "b")

    # (c) Error
    ax = axes[2]
    ax.plot(t_min, r["err"] * 1000, "-", color=_PALETTE[3], lw=0.6)
    ax.axhline(0, color="grey", lw=0.4, ls=":")
    ax.fill_between(
        t_min, r["err"] * 1000, 0, color=_PALETTE[3], alpha=0.15, linewidth=0
    )
    ax.set_ylabel("Voltage error (mV)")
    ax.set_xlabel("Time (min)")
    label_panel(ax, "c")

    fig.align_ylabels(axes)

    if save:
        save_fig(fig, "fig_nasa_dynamic_validation", str(FIG_DIR))
    return fig


def make_summary_figure(all_battery_results, save=True):
    """Box-plot of RMSE across batteries and cycles."""
    apply_style()

    fig, ax = plt.subplots(figsize=(3.54, 3.0))
    data_plot = []
    labels = []
    for bname, res_list in all_battery_results.items():
        rmses = [r["rmse_mV"] for r in res_list]
        data_plot.append(rmses)
        labels.append(bname)

    bp = ax.boxplot(
        data_plot, labels=labels, patch_artist=True, widths=0.5, showfliers=True
    )
    for patch, color in zip(bp["boxes"], _PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Voltage RMSE (mV)")
    ax.set_xlabel("Battery cell")
    ax.set_title("Dynamic-load voltage prediction accuracy", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, lw=0.5)
    plt.tight_layout()

    if save:
        out = FIG_DIR / "fig_nasa_dynamic_boxplot.pdf"
        fig.savefig(str(out), dpi=300, bbox_inches="tight")
        print(f"  Summary figure saved → {out}")
    return fig


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("NASA Randomized Battery Usage — Dynamic Load Validation")
    print("=" * 65)

    batteries = ["RW3", "RW4", "RW5", "RW6"]
    all_results = {}
    all_params = {}

    for bname in batteries:
        print(f"\n{'─'*50}")
        print(f"Battery {bname}")
        print(f"{'─'*50}")
        steps = load_battery(bname)
        params = fit_shepherd_on_ref(steps)
        all_params[bname] = params
        rw_results = evaluate_rw_cycles(steps, params, n_cycles=50)
        all_results[bname] = rw_results

        rmses = [r["rmse_mV"] for r in rw_results]
        maes = [r["mae_mV"] for r in rw_results]
        print(f"  RW cycles evaluated: {len(rw_results)}")
        print(
            f"  RMSE: {np.mean(rmses):.2f} ± {np.std(rmses):.2f} mV  "
            f"(range {np.min(rmses):.1f}–{np.max(rmses):.1f})"
        )
        print(f"  MAE:  {np.mean(maes):.2f} ± {np.std(maes):.2f} mV")

    # ── aggregate statistics ──
    print(f"\n{'='*65}")
    print("Aggregate results across all batteries")
    print(f"{'='*65}")
    all_rmse = [r["rmse_mV"] for res in all_results.values() for r in res]
    all_mae = [r["mae_mV"] for res in all_results.values() for r in res]
    print(f"  Total cycles: {len(all_rmse)}")
    print(f"  Overall RMSE: {np.mean(all_rmse):.2f} ± {np.std(all_rmse):.2f} mV")
    print(f"  Overall MAE:  {np.mean(all_mae):.2f} ± {np.std(all_mae):.2f} mV")

    # ── save CSV ──
    import csv

    csv_path = RESULTS_DIR / "nasa_dynamic_validation.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["battery", "cycle", "rmse_mV", "mae_mV", "n_points"])
        for bname, res_list in all_results.items():
            for r in res_list:
                w.writerow(
                    [
                        bname,
                        r["cycle"],
                        f"{r['rmse_mV']:.2f}",
                        f"{r['mae_mV']:.2f}",
                        r["n_pts"],
                    ]
                )
    print(f"  CSV saved → {csv_path}")

    # ── save LaTeX table ──
    tex_path = RESULTS_DIR / "nasa_dynamic_validation_table.tex"
    with open(tex_path, "w") as f:
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("Cell & CC fit RMSE (mV) & RW RMSE (mV) & RW MAE (mV) & Cycles \\\\\n")
        f.write("\\midrule\n")
        for bname in batteries:
            p = all_params[bname]
            rmses = [r["rmse_mV"] for r in all_results[bname]]
            maes = [r["mae_mV"] for r in all_results[bname]]
            f.write(
                f"{bname} & {p['rmse_fit_mV']:.1f} & "
                f"${np.mean(rmses):.1f} \\pm {np.std(rmses):.1f}$ & "
                f"${np.mean(maes):.1f} \\pm {np.std(maes):.1f}$ & "
                f"{len(rmses)} \\\\\n"
            )
        f.write("\\midrule\n")
        f.write(
            f"All & --- & "
            f"${np.mean(all_rmse):.1f} \\pm {np.std(all_rmse):.1f}$ & "
            f"${np.mean(all_mae):.1f} \\pm {np.std(all_mae):.1f}$ & "
            f"{len(all_rmse)} \\\\\n"
        )
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"  LaTeX table saved → {tex_path}")

    # ── figures ──
    make_figure(all_results["RW3"], all_params["RW3"], "RW3")
    make_summary_figure(all_results)

    print(f"\n{'='*65}")
    print("Dynamic validation complete.")
    print(f"{'='*65}")
    return all_results, all_params


if __name__ == "__main__":
    main()
