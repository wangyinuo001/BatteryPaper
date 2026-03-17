"""
NEW EXPERIMENT: Residual analysis and SOC-segmented error.
  (a) Residual histogram + KDE
  (b) Residual vs SOC scatter
  (c) ACF of residual sequence
  (d) Q-Q plot
Also outputs an SOC-segmented error table.
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pub_style import apply_style, save_fig, COLORS, DOUBLE_COL_TALL, label_panel
from run_all_baselines import load_discharge_data, shepherd_voltage

apply_style()


def acf(x, nlags=50):
    """Compute autocorrelation function."""
    x = x - np.mean(x)
    n = len(x)
    result = np.correlate(x, x, mode="full")
    result = result[n - 1 :]
    result /= result[0]
    return result[: nlags + 1]


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "..", "results")
    data_dir = os.path.join(script_dir, "..", "data", "XJTU battery dataset")

    # Aggregate residuals over all 8 cells
    all_residuals = []
    all_soc = []

    for i in range(1, 9):
        mat_path = os.path.join(data_dir, "Batch-1", f"2C_battery-{i}.mat")
        data = load_discharge_data(mat_path, cycle_idx=0)
        if data is None:
            continue

        v_pred = shepherd_voltage(data["soc"], data["current"], data["Q_total"])
        residual = (v_pred - data["voltage"]) * 1000  # mV
        all_residuals.append(residual)
        all_soc.append(data["soc"])

    res = np.concatenate(all_residuals)
    soc = np.concatenate(all_soc)

    print(f"Total residual points: {len(res)}")
    print(f"Mean: {np.mean(res):.2f} mV, Std: {np.std(res):.2f} mV")
    print(f"Skewness: {stats.skew(res):.3f}, Kurtosis: {stats.kurtosis(res):.3f}")

    # ── Figure ──
    fig, axes = plt.subplots(2, 2, figsize=DOUBLE_COL_TALL)

    # (a) Histogram + KDE
    ax = axes[0, 0]
    ax.hist(
        res,
        bins=80,
        density=True,
        alpha=0.5,
        color=COLORS["shepherd"],
        edgecolor="white",
        linewidth=0.3,
    )
    x_kde = np.linspace(res.min(), res.max(), 300)
    kde = stats.gaussian_kde(res)
    ax.plot(x_kde, kde(x_kde), "-", color=COLORS["shepherd"], lw=1.2, label="KDE")
    # Normal fit overlay
    mu, sigma = np.mean(res), np.std(res)
    ax.plot(
        x_kde,
        stats.norm.pdf(x_kde, mu, sigma),
        "--",
        color=COLORS["accent"],
        lw=1.0,
        label=f"$\\mathcal{{N}}$({mu:.1f}, {sigma:.1f}²)",
    )
    ax.set_xlabel("Residual (mV)")
    ax.set_ylabel("Density")
    label_panel(ax, "a")
    ax.legend(fontsize=6.5)

    # (b) Residual vs SOC
    ax = axes[0, 1]
    # Downsample for plotting
    idx = np.random.RandomState(42).choice(
        len(res), size=min(3000, len(res)), replace=False
    )
    ax.scatter(
        soc[idx], res[idx], s=1.5, alpha=0.3, color=COLORS["shepherd"], rasterized=True
    )
    ax.axhline(0, color="k", lw=0.5)

    # SOC-binned mean and std
    soc_bins = np.arange(0.05, 1.01, 0.05)
    bin_means = []
    bin_stds = []
    bin_centers = []
    for j in range(len(soc_bins) - 1):
        mask = (soc >= soc_bins[j]) & (soc < soc_bins[j + 1])
        if mask.sum() > 5:
            bin_centers.append((soc_bins[j] + soc_bins[j + 1]) / 2)
            bin_means.append(np.mean(res[mask]))
            bin_stds.append(np.std(res[mask]))

    ax.errorbar(
        bin_centers,
        bin_means,
        yerr=bin_stds,
        fmt="o-",
        color=COLORS["accent"],
        ms=3,
        lw=0.8,
        capsize=2,
        label="Bin mean ± σ",
        zorder=5,
    )
    ax.set_xlabel("SOC")
    ax.set_ylabel("Residual (mV)")
    label_panel(ax, "b")
    ax.legend(fontsize=6.5)

    # (c) ACF
    ax = axes[1, 0]
    # Use first cell's residuals for temporal ACF
    acf_vals = acf(all_residuals[0], nlags=80)
    ax.bar(range(len(acf_vals)), acf_vals, width=0.7, color=COLORS["accent"], alpha=0.7)
    n_pts = len(all_residuals[0])
    ci = 1.96 / np.sqrt(n_pts)
    ax.axhline(ci, ls="--", color=COLORS["gray"], lw=0.6, label="95% CI")
    ax.axhline(-ci, ls="--", color=COLORS["gray"], lw=0.6)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    label_panel(ax, "c")
    ax.legend(fontsize=6.5, loc="upper right")
    ax.set_xlim(-1, 82)

    # (d) Q-Q plot
    ax = axes[1, 1]
    (theoretical, sample), (slope, intercept, r) = stats.probplot(res, dist="norm")
    ax.scatter(
        theoretical, sample, s=1, color=COLORS["shepherd"], alpha=0.5, rasterized=True
    )
    x_line = np.linspace(theoretical.min(), theoretical.max(), 100)
    ax.plot(
        x_line,
        slope * x_line + intercept,
        "-",
        color=COLORS["accent"],
        lw=1.0,
        label=f"$R^2$ = {r**2:.4f}",
    )
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles (mV)")
    label_panel(ax, "d")
    ax.legend(fontsize=6.5)

    save_fig(fig, "fig_residual_analysis", results_dir)

    # ── SOC-segmented error table ──
    segments = [(0.05, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.0)]
    import csv

    csv_path = os.path.join(results_dir, "soc_segmented_error.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "SOC_range",
                "N_points",
                "RMSE_mV",
                "MAE_mV",
                "MaxErr_mV",
                "Mean_bias_mV",
                "Std_mV",
            ]
        )
        for lo, hi in segments:
            mask = (soc >= lo) & (soc < hi)
            r_seg = res[mask]
            if len(r_seg) < 5:
                continue
            w.writerow(
                [
                    f"[{lo:.2f}, {hi:.2f})",
                    len(r_seg),
                    f"{np.sqrt(np.mean(r_seg**2)):.2f}",
                    f"{np.mean(np.abs(r_seg)):.2f}",
                    f"{np.max(np.abs(r_seg)):.2f}",
                    f"{np.mean(r_seg):.2f}",
                    f"{np.std(r_seg):.2f}",
                ]
            )
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
