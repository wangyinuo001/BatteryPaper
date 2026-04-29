"""
NASA Aging Validation — Q_0(N) on reference discharges
=======================================================
Independently validates the cycle-aging law

    Q_0(N) = Q_0(0) * (1 - alpha * N^beta)

(originally fitted on XJTU Batch-1 with alpha=2.56e-4, beta=1.085)
on the four NASA random-walk cells (RW3, RW4, RW5, RW6).

For each cell:
  1. Extract every "reference discharge" segment (CC @ ~2 A to 3.2 V).
  2. Compute the effective capacity Q_eff = sum(I*dt)/3600.
  3. Use the index of the reference discharge as a coarse cycle counter
     (each "reference" follows ~50 RW cycles of usage; we report N as
     the number of RW cycles elapsed since the cell's first reference).
  4. Fit (alpha, beta) by L-BFGS-B with bounds; report RMSE in Ah.
"""

import sys, os, pathlib, json
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from nasa_dynamic_validation import (
    load_battery, extract_steps,
)

BASE = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CELLS = ["RW3", "RW4", "RW5", "RW6"]


def count_rw_cycles_before(steps, ref_global_index):
    """Count 'discharge (random walk)' steps with global index < ref_global_index.
    Used as a proxy cycle counter."""
    count = 0
    for i in range(ref_global_index):
        s = steps[0, i]
        c = str(s["comment"][0]) if s["comment"].size else ""
        if c == "discharge (random walk)":
            count += 1
    return count


def collect_capacities(name):
    """Return arrays (N_rw, Q_eff) over the cell's lifetime."""
    steps = load_battery(name)
    Ns, Qs = [], []
    for i in range(steps.shape[1]):
        s = steps[0, i]
        c = str(s["comment"][0]) if s["comment"].size else ""
        if c != "reference discharge":
            continue
        I = s["current"][0].flatten()
        t = s["relativeTime"][0].flatten()
        if len(t) < 5:
            continue
        dt = np.diff(t, prepend=t[0])
        Q = float(np.sum(np.abs(I) * dt) / 3600.0)
        if Q < 0.1 or Q > 5.0:
            continue  # skip aborted / outlier reference segments
        Ns.append(count_rw_cycles_before(steps, i))
        Qs.append(Q)
    return np.array(Ns), np.array(Qs)


def fit_aging(N, Q):
    """Q(N) = Q0 * (1 - alpha * N^beta)."""
    Q0_init = Q[N == N.min()].mean() if (N == N.min()).any() else Q[0]
    # Filter: positive N only for fitting (N=0 anchors Q0)
    mask = N > 0
    Nf, Qf = N[mask], Q[mask]

    def loss(theta):
        alpha, beta = theta
        if alpha <= 0 or beta <= 0:
            return 1e6
        Qhat = Q0_init * (1.0 - alpha * Nf**beta)
        Qhat = np.maximum(Qhat, 1e-3)
        return float(np.mean((Qhat - Qf) ** 2))

    res = minimize(
        loss, x0=[1e-4, 1.0], method="L-BFGS-B",
        bounds=[(1e-7, 1e-1), (0.3, 2.5)],
    )
    alpha, beta = res.x
    Qhat_all = Q0_init * (1.0 - alpha * np.maximum(N, 0) ** beta)
    rmse = float(np.sqrt(np.mean((Qhat_all - Q) ** 2)))
    mae = float(np.mean(np.abs(Qhat_all - Q)))
    rel = float(np.mean(np.abs(Qhat_all - Q) / Q) * 100.0)
    return dict(Q0=float(Q0_init), alpha=float(alpha), beta=float(beta),
                rmse_Ah=rmse, mae_Ah=mae, rel_err_pct=rel,
                n_points=int(len(Q)),
                N_max=int(N.max()))


def evaluate_xjtu_law(N, Q, Q0):
    """Apply XJTU-fitted law (alpha=2.56e-4, beta=1.085) without re-fitting."""
    alpha_x, beta_x = 2.56e-4, 1.085
    Qhat = Q0 * (1.0 - alpha_x * np.maximum(N, 0) ** beta_x)
    rmse = float(np.sqrt(np.mean((Qhat - Q) ** 2)))
    rel = float(np.mean(np.abs(Qhat - Q) / Q) * 100.0)
    return dict(rmse_Ah=rmse, rel_err_pct=rel)


def main():
    rows = []
    summary = []
    for cell in CELLS:
        print(f"[{cell}]")
        N, Q = collect_capacities(cell)
        if len(Q) < 4:
            print(f"  insufficient reference discharges ({len(Q)}), skipping")
            continue
        fit = fit_aging(N, Q)
        xjtu = evaluate_xjtu_law(N, Q, fit["Q0"])
        print(f"  fit  : alpha={fit['alpha']:.3e}  beta={fit['beta']:.3f}"
              f"  Q0={fit['Q0']:.3f} Ah  RMSE={fit['rmse_Ah']*1000:.1f} mAh"
              f"  ({fit['rel_err_pct']:.2f}%)")
        print(f"  XJTU : RMSE={xjtu['rmse_Ah']*1000:.1f} mAh"
              f"  ({xjtu['rel_err_pct']:.2f}%)")
        for n_i, q_i in zip(N, Q):
            rows.append(dict(cell=cell, N=int(n_i), Q_Ah=float(q_i)))
        summary.append(dict(cell=cell, **fit,
                            xjtu_rmse_Ah=xjtu["rmse_Ah"],
                            xjtu_rel_err_pct=xjtu["rel_err_pct"]))

    df_raw = pd.DataFrame(rows)
    df_sum = pd.DataFrame(summary)
    df_raw.to_csv(RESULTS_DIR / "nasa_aging_validation_raw.csv", index=False)
    df_sum.to_csv(RESULTS_DIR / "nasa_aging_validation_summary.csv", index=False)

    # Aggregate row
    print("\nAggregate")
    agg = dict(
        alpha_mean=float(df_sum["alpha"].mean()),
        alpha_std=float(df_sum["alpha"].std()),
        beta_mean=float(df_sum["beta"].mean()),
        beta_std=float(df_sum["beta"].std()),
        rmse_mean_mAh=float(df_sum["rmse_Ah"].mean() * 1000.0),
        xjtu_rmse_mean_mAh=float(df_sum["xjtu_rmse_Ah"].mean() * 1000.0),
        xjtu_rel_err_mean_pct=float(df_sum["xjtu_rel_err_pct"].mean()),
    )
    print(json.dumps(agg, indent=2))

    # LaTeX table
    tex = []
    tex.append("\\begin{table}[ht]\n\\centering\n\\small")
    tex.append("\\caption{Independent validation of the cycle-aging law $Q_0(N)=Q_0(0)(1-\\alpha N^{\\beta})$ on NASA random-walk cells. Per-cell parameters are obtained by NASA-only L-BFGS-B fits; the rightmost columns evaluate the XJTU-fitted parameters $(\\alpha=2.56\\times 10^{-4},\\beta=1.085)$ on the same NASA data without retraining.}\n\\label{tab:nasa_aging}")
    tex.append("\\begin{tabular}{lccccccc}\n\\hline")
    tex.append("Cell & $N_{\\max}$ & $Q_0(0)$ (Ah) & $\\alpha$ ($\\times 10^{-4}$) & $\\beta$ & RMSE (mAh) & XJTU RMSE (mAh) & XJTU rel.\\,err.\\,(\\%) \\\\")
    tex.append("\\hline")
    for r in summary:
        tex.append(
            f"{r['cell']} & {r['N_max']} & {r['Q0']:.3f} & "
            f"{r['alpha']*1e4:.2f} & {r['beta']:.2f} & "
            f"{r['rmse_Ah']*1000:.1f} & "
            f"{r['xjtu_rmse_Ah']*1000:.1f} & "
            f"{r['xjtu_rel_err_pct']:.2f} \\\\")
    tex.append("\\hline")
    tex.append(
        f"Mean & -- & -- & {agg['alpha_mean']*1e4:.2f}$\\pm${agg['alpha_std']*1e4:.2f} & "
        f"{agg['beta_mean']:.2f}$\\pm${agg['beta_std']:.2f} & "
        f"{agg['rmse_mean_mAh']:.1f} & "
        f"{agg['xjtu_rmse_mean_mAh']:.1f} & "
        f"{agg['xjtu_rel_err_mean_pct']:.2f} \\\\")
    tex.append("\\hline\n\\end{tabular}}\n\\end{table}")
    (RESULTS_DIR / "nasa_aging_validation_table.tex").write_text(
        "\n".join(tex), encoding="utf-8")
    print("\nWrote results/nasa_aging_validation_*.csv and .tex")


if __name__ == "__main__":
    main()
