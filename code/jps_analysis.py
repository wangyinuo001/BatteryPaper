"""
JPS revision — three analyses for stronger reviewer defence
============================================================
A. Thévenin-1RC dynamic validation on NASA RW3–RW6 (CC-fit -> RW test).
B. Generalised Shepherd: V = E0 - K * I / SOC^p - R*I + A*exp(-B*Q0*(1-SOC))
   refit on XJTU Batch-1 (8 cells); compare RMSE with the p=1 baseline.
C. Practical identifiability via Hessian condition number of the Shepherd
   cost function around the optimum (5-parameter set).

Outputs
-------
results/jps_nasa_ecm_dynamic.csv
results/jps_generalised_shepherd.csv
results/jps_identifiability.csv
"""

import sys, pathlib, csv, json
import numpy as np
from scipy.optimize import minimize, differential_evolution
import scipy.io as sio

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nasa_dynamic_validation import (
    load_battery,
    extract_steps,
    build_rw_cycles,
)

BASE = HERE.parent
RESULTS_DIR = BASE / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────
#  Common helpers
# ──────────────────────────────────────────────────────────────────
def coulomb_soc(I, t, Q_cap_Ah):
    dt = np.diff(t, prepend=t[0])
    cap = np.cumsum(np.abs(I) * dt) / 3600.0
    return np.clip(1.0 - cap / Q_cap_Ah, 0.001, 1.0)


# ══════════════════════════════════════════════════════════════════
#  A.  Thévenin-1RC dynamic validation on NASA
# ══════════════════════════════════════════════════════════════════
def fit_thev1_on_nasa_ref(steps, max_refs=5):
    """Fit OCV(SOC) polynomial + (R0,R1,C1) from reference CC discharges."""
    ref_data = []
    for V, I, T, t in extract_steps(steps, "reference discharge"):
        ref_data.append((V, I, T, t))
        if len(ref_data) >= max_refs:
            break

    V0, I0, T0, t0 = ref_data[0]
    I_const = float(np.median(I0))
    Q_cap = I_const * t0[-1] / 3600.0

    soc0 = coulomb_soc(I0, t0, Q_cap)
    mask = (soc0 > 0.05) & (soc0 < 0.99)
    ocv_coef = np.polyfit(soc0[mask], V0[mask], 6)

    # downsample for fit speed
    n_target = 600
    step_ds = max(1, len(t0) // n_target)
    t_ds = t0[::step_ds]
    V_ds = V0[::step_ds]
    I_ds = I0[::step_ds]
    soc_ds = coulomb_soc(I_ds, t_ds, Q_cap)

    def cost(params):
        R0, R1, C1 = params
        v_rc = 0.0
        sse = 0.0
        for k in range(len(t_ds)):
            I_k = I_ds[k]
            ocv = np.polyval(ocv_coef, soc_ds[k])
            V_pred = ocv - R0 * I_k - v_rc
            err = V_pred - V_ds[k]
            if not np.isfinite(err):
                return 1e10
            sse += err * err
            if k < len(t_ds) - 1:
                dt = t_ds[k + 1] - t_ds[k]
                v_rc += (I_k / C1 - v_rc / (R1 * C1)) * dt
                v_rc = float(np.clip(v_rc, -2.0, 2.0))
        return sse / len(t_ds)

    bounds = [(0.005, 0.3), (0.001, 0.2), (50.0, 50000.0)]
    res = differential_evolution(cost, bounds, maxiter=150, seed=42, tol=1e-7)
    R0, R1, C1 = res.x
    rmse_fit_mV = float(np.sqrt(res.fun) * 1000)
    return {
        "ocv_coef": ocv_coef,
        "R0": float(R0),
        "R1": float(R1),
        "C1": float(C1),
        "Q_cap": float(Q_cap),
        "rmse_fit_mV": rmse_fit_mV,
    }


def evaluate_thev1_on_rw(steps, params, n_cycles=50):
    cycles = build_rw_cycles(steps, max_cycles=n_cycles)
    R0, R1, C1 = params["R0"], params["R1"], params["C1"]
    ocv_coef = params["ocv_coef"]
    Q_cap = params["Q_cap"]
    out = []
    for idx, cyc in enumerate(cycles):
        I = cyc["I"]
        V_meas = cyc["V"]
        t = cyc["t"]
        dt = np.diff(t, prepend=t[0])
        soc = 1.0
        v_rc = 0.0
        V_pred = np.zeros(len(t))
        for k in range(len(t)):
            ocv = np.polyval(ocv_coef, np.clip(soc, 0.005, 0.995))
            V_pred[k] = ocv - R0 * I[k] - v_rc
            if k < len(t) - 1:
                # update RC and SOC
                dt_k = max(dt[k + 1], 1e-3)
                v_rc += (I[k] / C1 - v_rc / (R1 * C1)) * dt_k
                v_rc = float(np.clip(v_rc, -2.0, 2.0))
                soc -= I[k] * dt_k / (Q_cap * 3600.0)
                soc = float(np.clip(soc, 0.001, 1.0))
        err = V_pred - V_meas
        rmse = float(np.sqrt(np.mean(err * err)) * 1000)
        mae = float(np.mean(np.abs(err)) * 1000)
        out.append({"cycle": idx, "rmse_mV": rmse, "mae_mV": mae, "n_pts": len(V_meas)})
    return out


def part_A_nasa_ecm():
    print("\n=== A. Thévenin-1RC on NASA random-walk ===")
    rows = []
    summary = []
    for bname in ["RW3", "RW4", "RW5", "RW6"]:
        steps = load_battery(bname)
        params = fit_thev1_on_nasa_ref(steps)
        print(
            f"  {bname}: CC fit RMSE = {params['rmse_fit_mV']:.2f} mV  "
            f"(R0={params['R0']:.4f}, R1={params['R1']:.4f}, C1={params['C1']:.0f})"
        )
        per_cycle = evaluate_thev1_on_rw(steps, params, n_cycles=50)
        rmses = np.array([r["rmse_mV"] for r in per_cycle])
        maes = np.array([r["mae_mV"] for r in per_cycle])
        for r in per_cycle:
            rows.append({"cell": bname, **r})
        summary.append(
            {
                "cell": bname,
                "cc_fit_rmse_mV": round(params["rmse_fit_mV"], 2),
                "rw_rmse_mean_mV": round(float(rmses.mean()), 2),
                "rw_rmse_std_mV": round(float(rmses.std()), 2),
                "rw_mae_mean_mV": round(float(maes.mean()), 2),
                "rw_mae_std_mV": round(float(maes.std()), 2),
                "n_cycles": len(per_cycle),
            }
        )
        print(
            f"    RW: RMSE = {rmses.mean():.2f} ± {rmses.std():.2f} mV "
            f"(MAE {maes.mean():.2f} ± {maes.std():.2f}) over {len(per_cycle)} cycles"
        )
    all_rmse = np.array([r["rmse_mV"] for r in rows])
    all_mae = np.array([r["mae_mV"] for r in rows])
    summary.append(
        {
            "cell": "All",
            "cc_fit_rmse_mV": "",
            "rw_rmse_mean_mV": round(float(all_rmse.mean()), 2),
            "rw_rmse_std_mV": round(float(all_rmse.std()), 2),
            "rw_mae_mean_mV": round(float(all_mae.mean()), 2),
            "rw_mae_std_mV": round(float(all_mae.std()), 2),
            "n_cycles": len(rows),
        }
    )

    csv_path = RESULTS_DIR / "jps_nasa_ecm_dynamic.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        for s in summary:
            w.writerow(s)
    print(f"  → {csv_path}")
    print(
        f"  AGGREGATE: Thévenin-1RC RW RMSE = {all_rmse.mean():.2f} ± "
        f"{all_rmse.std():.2f} mV across {len(rows)} cycles"
    )
    return summary


# ══════════════════════════════════════════════════════════════════
#  B.  Generalised Shepherd  K / SOC^p  on XJTU Batch-1
# ══════════════════════════════════════════════════════════════════
def load_xjtu_cycle(mat_path, cycle_idx=0):
    mat = sio.loadmat(str(mat_path))
    data = mat["data"]
    cyc = data[0, cycle_idx]
    V = cyc["voltage_V"].flatten()
    I = cyc["current_A"].flatten()
    t_min = cyc["relative_time_min"].flatten()
    mask = I < 0
    if mask.sum() < 100:
        return None
    t_s = t_min[mask] * 60.0
    V_d = V[mask]
    I_d = -I[mask]
    Q_total = float(np.cumsum(I_d * np.diff(t_s, prepend=t_s[0]))[-1] / 3600.0)
    soc = 1.0 - np.cumsum(I_d * np.diff(t_s, prepend=t_s[0])) / 3600.0 / Q_total
    return {"t": t_s, "V": V_d, "I": I_d, "soc": soc, "Q": Q_total}


def shepherd_general(soc, I, theta, Q0):
    """V = E0 - K*I/SOC^p - R*I + A*exp(-B*Q0*(1-SOC))"""
    E0, K, A, B, R0, p = theta
    soc_c = np.clip(soc, 0.01, 1.0)
    return (
        E0 - K * I / np.power(soc_c, p) - R0 * I + A * np.exp(-B * Q0 * (1.0 - soc_c))
    )


def shepherd_classic(soc, I, theta, Q0):
    """p fixed at 1.0 (classic Shepherd)."""
    return shepherd_general(soc, I, list(theta) + [1.0], Q0)


def fit_shepherd(cycles, generalised=False):
    """Fit Shepherd parameters to multiple discharge cycles via DE."""

    def cost(theta_local):
        sse = 0.0
        n = 0
        for c in cycles:
            if generalised:
                V_pred = shepherd_general(c["soc"], c["I"], theta_local, c["Q"])
            else:
                V_pred = shepherd_classic(c["soc"], c["I"], theta_local, c["Q"])
            d = V_pred - c["V"]
            if not np.all(np.isfinite(d)):
                return 1e10
            sse += float(np.sum(d * d))
            n += len(d)
        return sse / n

    if generalised:
        bounds = [
            (3.0, 4.5),
            (1e-4, 0.2),
            (0.0, 2.0),
            (0.05, 50.0),
            (1e-4, 0.3),
            (0.3, 2.5),
        ]
    else:
        bounds = [(3.0, 4.5), (1e-4, 0.2), (0.0, 2.0), (0.05, 50.0), (1e-4, 0.3)]
    res = differential_evolution(
        cost, bounds, maxiter=400, seed=42, popsize=25, tol=1e-8, polish=True
    )
    return res.x, float(np.sqrt(res.fun))


def part_B_generalised_shepherd():
    print("\n=== B. Generalised Shepherd K/SOC^p on XJTU Batch-1 ===")
    data_dir = BASE / "data" / "XJTU battery dataset" / "Batch-1"
    cells = []
    for i in range(1, 9):
        p = data_dir / f"2C_battery-{i}.mat"
        if not p.exists():
            continue
        d = load_xjtu_cycle(p, 0)
        if d is not None:
            cells.append(d)
            print(f"  Loaded {p.name}: Q={d['Q']:.3f} Ah, n={len(d['V'])}")
    if not cells:
        print("  No XJTU cells found; skipping part B.")
        return None

    # Refit classic and generalised on the same data
    print("  Fitting classic Shepherd (p=1) ...")
    theta_c, rmse_c = fit_shepherd(cells, generalised=False)
    print(f"    θ_c = {[round(x,4) for x in theta_c]},  RMSE = {rmse_c*1000:.2f} mV")
    print("  Fitting generalised Shepherd (p free) ...")
    theta_g, rmse_g = fit_shepherd(cells, generalised=True)
    print(f"    θ_g = {[round(x,4) for x in theta_g]},  RMSE = {rmse_g*1000:.2f} mV")

    # Per-cell RMSE for both models
    per_cell = []
    for c in cells:
        V_c = shepherd_classic(c["soc"], c["I"], theta_c, c["Q"])
        V_g = shepherd_general(c["soc"], c["I"], theta_g, c["Q"])
        per_cell.append(
            {
                "n_pts": len(c["V"]),
                "rmse_classic_mV": float(np.sqrt(np.mean((V_c - c["V"]) ** 2)) * 1000),
                "rmse_general_mV": float(np.sqrt(np.mean((V_g - c["V"]) ** 2)) * 1000),
            }
        )
    rc = np.array([r["rmse_classic_mV"] for r in per_cell])
    rg = np.array([r["rmse_general_mV"] for r in per_cell])
    print(
        f"  Per-cell mean RMSE: classic = {rc.mean():.2f} ± {rc.std():.2f} mV  ;  "
        f"generalised = {rg.mean():.2f} ± {rg.std():.2f} mV"
    )
    print(
        f"  Δ improvement = {(rc.mean()-rg.mean()):+.2f} mV "
        f"({100*(rc.mean()-rg.mean())/rc.mean():+.1f}%)  with p* = {theta_g[5]:.3f}"
    )

    out_path = RESULTS_DIR / "jps_generalised_shepherd.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "E0",
                "K",
                "A",
                "B",
                "R0",
                "p",
                "rmse_pooled_mV",
                "rmse_percell_mean_mV",
                "rmse_percell_std_mV",
            ]
        )
        w.writerow(
            [
                "classic",
                *[f"{x:.5f}" for x in theta_c],
                1.0,
                f"{rmse_c*1000:.3f}",
                f"{rc.mean():.3f}",
                f"{rc.std():.3f}",
            ]
        )
        w.writerow(
            [
                "generalised",
                *[f"{x:.5f}" for x in theta_g],
                f"{rmse_g*1000:.3f}",
                f"{rg.mean():.3f}",
                f"{rg.std():.3f}",
            ]
        )
    print(f"  → {out_path}")
    return {
        "theta_c": list(theta_c),
        "theta_g": list(theta_g),
        "rmse_classic_pooled_mV": float(rmse_c * 1000),
        "rmse_general_pooled_mV": float(rmse_g * 1000),
        "rmse_classic_percell": (float(rc.mean()), float(rc.std())),
        "rmse_general_percell": (float(rg.mean()), float(rg.std())),
        "p_star": float(theta_g[5]),
    }


# ══════════════════════════════════════════════════════════════════
#  C.  Practical identifiability via Hessian condition number
# ══════════════════════════════════════════════════════════════════
def hessian_finite_diff(cost_fn, theta, h_rel=1e-3):
    n = len(theta)
    H = np.zeros((n, n))
    f0 = cost_fn(theta)
    h = np.maximum(np.abs(theta) * h_rel, 1e-6)
    for i in range(n):
        for j in range(i, n):
            tpp = np.array(theta, dtype=float)
            tpp[i] += h[i]
            tpp[j] += h[j]
            tpm = np.array(theta, dtype=float)
            tpm[i] += h[i]
            tpm[j] -= h[j]
            tmp = np.array(theta, dtype=float)
            tmp[i] -= h[i]
            tmp[j] += h[j]
            tmm = np.array(theta, dtype=float)
            tmm[i] -= h[i]
            tmm[j] -= h[j]
            H[i, j] = (cost_fn(tpp) - cost_fn(tpm) - cost_fn(tmp) + cost_fn(tmm)) / (
                4 * h[i] * h[j]
            )
            H[j, i] = H[i, j]
    return H, f0


def part_C_identifiability(cells, theta_classic):
    """Compute Hessian / Fisher info around the classic Shepherd optimum.

    The Hessian of the SSE cost is twice the Gauss-Newton approximation of
    the Fisher information matrix; its condition number diagnoses whether
    parameter directions are jointly identifiable.
    """
    print("\n=== C. Practical identifiability — Hessian condition number ===")
    if cells is None or theta_classic is None:
        return None

    def sse_cost(theta_local):
        s = 0.0
        for c in cells:
            V_pred = shepherd_classic(c["soc"], c["I"], theta_local, c["Q"])
            d = V_pred - c["V"]
            s += float(np.sum(d * d))
        return s / sum(len(c["V"]) for c in cells)

    # Scale parameters so the condition number is dimensionally meaningful
    theta = np.array(theta_classic, dtype=float)
    scale = np.maximum(np.abs(theta), 1e-3)

    def scaled_cost(theta_scaled):
        return sse_cost(theta_scaled * scale)

    H, f0 = hessian_finite_diff(scaled_cost, theta / scale, h_rel=2e-3)
    # Symmetrise
    H = 0.5 * (H + H.T)
    eig = np.linalg.eigvalsh(H)
    eig_pos = eig[eig > 0]
    cond = float(eig_pos.max() / eig_pos.min()) if len(eig_pos) >= 2 else float("nan")
    print(f"  Hessian eigenvalues (sorted): {[f'{e:.3e}' for e in eig]}")
    print(f"  Condition number κ(H) = {cond:.3e}")

    # Per-parameter standard errors via Cramer-Rao bound
    # Variance(theta_i) ≈ σ² * (H^{-1})_{ii} ; σ² ≈ MSE
    try:
        Hinv = np.linalg.pinv(H)
        sigma2 = f0  # cost is mean of squared residuals
        # Re-scale variance back to original parameter scale
        var_scaled = np.diag(Hinv) * sigma2
        std_orig = np.sqrt(np.maximum(var_scaled, 0.0)) * scale
        rel_std = std_orig / np.maximum(np.abs(theta), 1e-6) * 100
    except np.linalg.LinAlgError:
        std_orig = np.full_like(theta, np.nan)
        rel_std = std_orig.copy()

    names = ["E0", "K", "A", "B", "R0"]
    out_path = RESULTS_DIR / "jps_identifiability.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["param", "value", "std_err", "rel_std_pct"])
        for n, v, s, r in zip(names, theta, std_orig, rel_std):
            w.writerow([n, f"{v:.6f}", f"{s:.4e}", f"{r:.3f}"])
        w.writerow([])
        w.writerow(["hessian_condition_number", f"{cond:.4e}"])
        w.writerow(["mse_at_optimum_V2", f"{f0:.4e}"])
    print(f"  → {out_path}")
    return {"cond": cond, "rel_std_pct": dict(zip(names, rel_std.tolist()))}


# ══════════════════════════════════════════════════════════════════
#  Main driver
# ══════════════════════════════════════════════════════════════════
def main():
    summary_A = part_A_nasa_ecm()
    res_B = part_B_generalised_shepherd()

    cells = []
    data_dir = BASE / "data" / "XJTU battery dataset" / "Batch-1"
    for i in range(1, 9):
        p = data_dir / f"2C_battery-{i}.mat"
        if p.exists():
            d = load_xjtu_cycle(p, 0)
            if d is not None:
                cells.append(d)
    theta_c = res_B["theta_c"] if res_B else None
    res_C = part_C_identifiability(cells, theta_c)

    bundle = {
        "A_nasa_thev1_summary": summary_A,
        "B_generalised_shepherd": res_B,
        "C_identifiability": res_C,
    }
    json_path = RESULTS_DIR / "jps_analysis_bundle.json"
    with open(json_path, "w") as f:
        json.dump(bundle, f, indent=2, default=str)
    print(f"\nAll outputs collated → {json_path}")


if __name__ == "__main__":
    main()
