"""
Data-driven baseline comparison on NASA Randomised Battery Usage.

Two black-box surrogates are compared against the proposed Shepherd model
on identical held-out random-walk (RW) cycles.

  (i)  Polynomial-Ridge regression  V = f_poly(SOC, I)        — 10 params
  (ii) Single-hidden-layer MLP       V = f_mlp(SOC, I; W,b)    — 21 params
                                     (5 tanh units, L-BFGS train)

Training protocol (fair to the data-driven baselines)
------------------------------------------------------
* Shepherd is fitted on the 5 reference CC discharges only (5 parameters).
* The data-driven baselines additionally see the first 10 RW cycles
  ("calibration") so that their training distribution covers the full
  current range encountered at test time. Evaluation is on RW cycles
  11..50, which neither model has seen.

This setup grants the black-box models the maximum legitimate advantage
they can claim, isolating the value of physical structure rather than
training-set coverage.

Output
------
results/data_driven_baseline_nasa.csv
results/data_driven_baseline_summary.csv
results/data_driven_baseline_nasa_table.tex
"""

import sys, os, pathlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from nasa_dynamic_validation import (
    load_battery,
    extract_steps,
    build_rw_cycles,
    fit_shepherd_on_ref,
    simulate_discharge,
)

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════
# Build (SOC, I) -> V training pairs from reference cycles
# ══════════════════════════════════════════════════════════════════
def build_training_pairs(steps, Q_cap, max_refs=5, calib_cycles=None):
    """Reference CC pairs, optionally augmented with calibration RW cycles."""
    X_list, y_list = [], []
    for k, (V_m, I_m, T_m, t_m) in enumerate(
        extract_steps(steps, "reference discharge")
    ):
        if k >= max_refs:
            break
        # Build SOC trajectory by Coulomb counting
        dt_m = np.diff(t_m, prepend=t_m[0])
        soc_m = np.zeros_like(t_m, dtype=float)
        soc_m[0] = 1.0
        for i in range(1, len(t_m)):
            soc_m[i] = soc_m[i - 1] - I_m[i - 1] * dt_m[i] / (Q_cap * 3600.0)
        soc_m = np.clip(soc_m, 0.001, 1.0)
        X_list.append(np.column_stack([soc_m, I_m]))
        y_list.append(V_m)

    if calib_cycles:
        for cyc in calib_cycles:
            t = cyc["t"]
            I = cyc["I"]
            V = cyc["V"]
            dt = np.diff(t, prepend=t[0])
            soc = np.zeros_like(t, dtype=float)
            soc[0] = 1.0
            for i in range(1, len(t)):
                soc[i] = soc[i - 1] - I[i - 1] * dt[i] / (Q_cap * 3600.0)
            soc = np.clip(soc, 0.001, 1.0)
            X_list.append(np.column_stack([soc, I]))
            y_list.append(V)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    if len(X) > 12000:
        idx = np.linspace(0, len(X) - 1, 12000, dtype=int)
        X = X[idx]
        y = y[idx]
    return X, y


# ══════════════════════════════════════════════════════════════════
# (i) Polynomial Ridge regression  (degree-3, ridge alpha=1e-4)
# ══════════════════════════════════════════════════════════════════
def poly_features(X, degree=3):
    s, i = X[:, 0], X[:, 1]
    feats = [np.ones_like(s)]
    for d in range(1, degree + 1):
        for k in range(d + 1):
            feats.append((s ** (d - k)) * (i**k))
    return np.column_stack(feats)


def fit_poly_ridge(X, y, degree=3, alpha=1e-4):
    Phi = poly_features(X, degree=degree)
    A = Phi.T @ Phi + alpha * np.eye(Phi.shape[1])
    w = np.linalg.solve(A, Phi.T @ y)
    return w, degree


def predict_poly(w, degree, X):
    return poly_features(X, degree=degree) @ w


# ══════════════════════════════════════════════════════════════════
# (ii) Tiny MLP: 2 -> 5 (tanh) -> 1, L-BFGS-trained
# ══════════════════════════════════════════════════════════════════
N_HIDDEN = 5


def _unpack(p):
    n = N_HIDDEN
    W1 = p[: 2 * n].reshape(2, n)
    b1 = p[2 * n : 2 * n + n]
    W2 = p[2 * n + n : 2 * n + n + n]
    b2 = p[-1]
    return W1, b1, W2, b2


def _forward(p, X):
    W1, b1, W2, b2 = _unpack(p)
    h = np.tanh(X @ W1 + b1)
    return h @ W2 + b2


def fit_mlp(X, y, n_restart=2, seed=0):
    # standardise inputs and target for stable training
    mu_x, sd_x = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu_x) / sd_x
    mu_y, sd_y = y.mean(), y.std() + 1e-9
    ys = (y - mu_y) / sd_y
    N = len(ys)

    n = N_HIDDEN
    n_param = 2 * n + n + n + 1
    rng = np.random.default_rng(seed)

    def loss_and_grad(p):
        W1, b1, W2, b2 = _unpack(p)
        z = Xs @ W1 + b1  # (N, n)
        h = np.tanh(z)  # (N, n)
        yhat = h @ W2 + b2  # (N,)
        e = yhat - ys  # residual
        L = float(np.mean(e**2))
        # gradients
        dyhat = (2.0 / N) * e  # (N,)
        gW2 = h.T @ dyhat  # (n,)
        gb2 = float(dyhat.sum())
        dh = np.outer(dyhat, W2)  # (N, n)
        dz = dh * (1 - h**2)  # tanh'
        gW1 = Xs.T @ dz  # (2, n)
        gb1 = dz.sum(axis=0)  # (n,)
        g = np.concatenate([gW1.ravel(), gb1, gW2, [gb2]])
        return L, g

    best_p, best_l = None, np.inf
    for r in range(n_restart):
        p0 = rng.normal(scale=0.3, size=n_param)
        res = minimize(
            loss_and_grad,
            p0,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 400, "ftol": 1e-9, "gtol": 1e-7},
        )
        if res.fun < best_l:
            best_l = res.fun
            best_p = res.x
    return {"p": best_p, "mu_x": mu_x, "sd_x": sd_x, "mu_y": mu_y, "sd_y": sd_y}


def predict_mlp(model, X):
    Xs = (X - model["mu_x"]) / model["sd_x"]
    ys = _forward(model["p"], Xs)
    return ys * model["sd_y"] + model["mu_y"]


# ══════════════════════════════════════════════════════════════════
# Evaluate baselines on the same RW cycles
# ══════════════════════════════════════════════════════════════════
def evaluate_baseline_on_cycles(cycles, predict_fn, Q_cap):
    results = []
    for idx, cyc in enumerate(cycles):
        I = cyc["I"]
        V_meas = cyc["V"]
        t = cyc["t"]
        # SOC by Coulomb counting (identical to Shepherd path)
        dt = np.diff(t, prepend=t[0])
        soc = np.zeros_like(t, dtype=float)
        soc[0] = 1.0
        for k in range(1, len(t)):
            soc[k] = soc[k - 1] - I[k - 1] * dt[k] / (Q_cap * 3600.0)
        soc = np.clip(soc, 0.001, 1.0)

        X = np.column_stack([soc, I])
        V_pred = predict_fn(X)
        err = V_pred - V_meas
        rmse = float(np.sqrt(np.mean(err**2)) * 1000)
        mae = float(np.mean(np.abs(err)) * 1000)
        results.append({"cycle": idx, "rmse_mV": rmse, "mae_mV": mae})
    return results


def shepherd_predict_factory(p):
    def pred(X):
        soc, I = X[:, 0], X[:, 1]
        soc = np.clip(soc, 0.01, 1.0)
        return (
            p["E0"]
            - p["R0"] * I
            - p["K"] * (I / soc)
            + p["A"] * np.exp(-p["B"] * (1 - soc))
        )

    return pred


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("Data-driven baseline comparison on NASA RW cycles")
    print("=" * 65)

    batteries = ["RW3", "RW4", "RW5", "RW6"]
    rows = []
    summary = {"Shepherd": [], "Poly-3 Ridge": [], "MLP (2-5-1)": []}
    n_params = {"Shepherd": 5, "Poly-3 Ridge": 10, "MLP (2-5-1)": 21}

    for bname in batteries:
        print(f"\n--- {bname} ---")
        steps = load_battery(bname)
        params = fit_shepherd_on_ref(steps)
        Q_cap = params["Q_cap"]

        cycles_all = build_rw_cycles(steps, max_cycles=50)
        n_calib = 10
        calib_cycles = cycles_all[:n_calib]
        test_cycles = cycles_all[n_calib:]
        print(
            f"  Calibration RW cycles: {len(calib_cycles)}, test RW cycles: {len(test_cycles)}"
        )

        X_tr, y_tr = build_training_pairs(steps, Q_cap, calib_cycles=calib_cycles)
        print(f"  Training pairs: {len(X_tr)}")

        w_poly, deg = fit_poly_ridge(X_tr, y_tr, degree=3)
        mlp = fit_mlp(X_tr, y_tr)

        res_shep = evaluate_baseline_on_cycles(
            test_cycles, shepherd_predict_factory(params), Q_cap
        )
        res_poly = evaluate_baseline_on_cycles(
            test_cycles, lambda X: predict_poly(w_poly, deg, X), Q_cap
        )
        res_mlp = evaluate_baseline_on_cycles(
            test_cycles, lambda X: predict_mlp(mlp, X), Q_cap
        )

        for s, p_, m in zip(res_shep, res_poly, res_mlp):
            rows.append(
                {
                    "Battery": bname,
                    "Cycle": n_calib + s["cycle"],
                    "RMSE_Shepherd_mV": s["rmse_mV"],
                    "RMSE_Poly_mV": p_["rmse_mV"],
                    "RMSE_MLP_mV": m["rmse_mV"],
                    "MAE_Shepherd_mV": s["mae_mV"],
                    "MAE_Poly_mV": p_["mae_mV"],
                    "MAE_MLP_mV": m["mae_mV"],
                }
            )
        for tag, res in [
            ("Shepherd", res_shep),
            ("Poly-3 Ridge", res_poly),
            ("MLP (2-5-1)", res_mlp),
        ]:
            r = np.array([x["rmse_mV"] for x in res])
            summary[tag].extend(r.tolist())
            print(f"  {tag:<14}  RMSE = {r.mean():6.2f} +/- {r.std():5.2f} mV")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "data_driven_baseline_nasa.csv", index=False)
    print(f"\nSaved -> data_driven_baseline_nasa.csv ({len(df)} rows)")

    # ── summary ──
    print("\n" + "=" * 65)
    print("Aggregate (across all batteries / all cycles)")
    print("=" * 65)
    sum_rows = []
    for tag, vals in summary.items():
        v = np.array(vals)
        print(
            f"  {tag:<14}  RMSE = {v.mean():6.2f} +/- {v.std():5.2f} mV   (n={len(v)}, n_params={n_params[tag]})"
        )
        sum_rows.append(
            {
                "Model": tag,
                "n_params": n_params[tag],
                "RMSE_mean_mV": float(v.mean()),
                "RMSE_std_mV": float(v.std()),
                "n_cycles": int(len(v)),
            }
        )
    pd.DataFrame(sum_rows).to_csv(
        RESULTS_DIR / "data_driven_baseline_summary.csv", index=False
    )

    # ── LaTeX table ──
    tex_path = RESULTS_DIR / "data_driven_baseline_nasa_table.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by data_driven_baseline_nasa.py\n")
        f.write("\\begin{tabular}{@{}lrrr@{}}\n")
        f.write("\\toprule\n")
        f.write("Model & \\#~params & RMSE (mV) & Relative \\\\\n")
        f.write("\\midrule\n")
        ref = sum_rows[0]["RMSE_mean_mV"]
        for r in sum_rows:
            rel = r["RMSE_mean_mV"] / ref
            f.write(
                f"{r['Model']} & {r['n_params']} & ${r['RMSE_mean_mV']:.2f} \\pm {r['RMSE_std_mV']:.2f}$ & ${rel:.2f}\\times$ \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    print(f"LaTeX table -> {tex_path}")


if __name__ == "__main__":
    main()
