"""
MPC-based Adaptive Energy Management
======================================
Implements a receding-horizon model-predictive control (MPC) strategy
that uses the Shepherd voltage model as the internal prediction engine.

Key idea: the user specifies a **target runtime** T_des (e.g., 6 h).
The MPC budgets energy by tracking a reference SOC trajectory and
gradually adjusting power to ensure the battery reaches T_des.

Three strategies compared:
  A) Baseline:   constant gaming power until depletion
  B) Threshold:  two-stage rule-based throttling (SOC ≤ 20%, SOC ≤ 10%)
  C) MPC:        rolling-horizon optimisation with target runtime

Output:
  - fig_energy_management.pdf  (3-panel: SOC, power, QoS comparison)
  - energy_management_summary.csv
"""

import sys, os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from pub_style import apply_style, save_fig, COLORS, DOUBLE_COL_TALL, label_panel

from main_model import MainBatteryModel

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "ecm_paper", "figures")

# ═══════════════════════════════════════════════════════════════
# Power levels (from paper Table 2)
# ═══════════════════════════════════════════════════════════════
P_GAMING = 4.303  # W — full gaming
P_THROTTLE = 1.727  # W — reading-level throttle
P_LOWPOWER = 0.741  # W — standby-level low-power
P_MIN = 0.741  # W — minimum feasible power
P_MAX = 4.303  # W — maximum power

# Threshold strategy parameters
SOC_THRESH_1 = 0.20
SOC_THRESH_2 = 0.10

# MPC parameters
T_TARGET = 6.0  # desired runtime (hours)
MPC_HORIZON = 1200  # prediction horizon (seconds)
MPC_DT_CTRL = 120  # control interval (seconds)
MPC_N_STEPS = 10  # steps in horizon = 1200/120
SOC_CUTOFF = 0.05  # terminal SOC cutoff
ALPHA_QOS = 0.3  # weight: prefer high power (QoS)
BETA_TRACK = 500.0  # weight: track reference SOC trajectory
GAMMA_SMOOTH = 0.2  # weight: smooth power transitions


# ═══════════════════════════════════════════════════════════════
# Strategy A: Baseline (constant gaming)
# ═══════════════════════════════════════════════════════════════
def simulate_baseline(model, temp_k=298.15, dt=1.0):
    Q_eff = model.get_capacity_at_temp(temp_k)
    soc, t = 1.0, 0.0
    t_max = 40 * 3600
    I_avg = P_GAMING / 3.7

    time_pts, soc_pts, power_pts = [], [], []

    while soc > 0.05 and t < t_max:
        V_term = model.terminal_voltage(soc, I_avg, temp_k)
        if V_term < model.V_cutoff:
            break
        I_new = P_GAMING / V_term if V_term > 0 else I_avg
        I_avg = 0.9 * I_avg + 0.1 * I_new
        dsoc = -I_avg * dt / (Q_eff * 3600)
        soc += dsoc
        t += dt
        if int(t) % 30 == 0:
            time_pts.append(t / 3600)
            soc_pts.append(soc)
            power_pts.append(P_GAMING)

    return {
        "time": np.array(time_pts),
        "soc": np.array(soc_pts),
        "power": np.array(power_pts),
        "tte": time_pts[-1] if time_pts else 0,
    }


# ═══════════════════════════════════════════════════════════════
# Strategy B: Two-stage threshold
# ═══════════════════════════════════════════════════════════════
def simulate_threshold(model, temp_k=298.15, dt=1.0):
    Q_eff = model.get_capacity_at_temp(temp_k)
    soc, t = 1.0, 0.0
    t_max = 40 * 3600
    I_avg = P_GAMING / 3.7

    time_pts, soc_pts, power_pts, phase_pts = [], [], [], []

    while soc > 0.05 and t < t_max:
        if soc > SOC_THRESH_1:
            P_load = P_GAMING
            phase = 0
        elif soc > SOC_THRESH_2:
            P_load = P_THROTTLE
            phase = 1
        else:
            P_load = P_LOWPOWER
            phase = 2

        V_term = model.terminal_voltage(soc, I_avg, temp_k)
        if V_term < model.V_cutoff:
            break
        I_new = P_load / V_term if V_term > 0 else I_avg
        I_avg = 0.9 * I_avg + 0.1 * I_new
        dsoc = -I_avg * dt / (Q_eff * 3600)
        soc += dsoc
        t += dt
        if int(t) % 30 == 0:
            time_pts.append(t / 3600)
            soc_pts.append(soc)
            power_pts.append(P_load)
            phase_pts.append(phase)

    return {
        "time": np.array(time_pts),
        "soc": np.array(soc_pts),
        "power": np.array(power_pts),
        "phase": np.array(phase_pts),
        "tte": time_pts[-1] if time_pts else 0,
    }


# ═══════════════════════════════════════════════════════════════
# Strategy C: Receding-horizon MPC with target runtime
# ═══════════════════════════════════════════════════════════════
def soc_reference(t_now, soc_now, t_target, soc_target=SOC_CUTOFF):
    """
    Linear reference SOC trajectory from (t_now, soc_now) to (t_target, soc_target).
    Returns a function that gives the reference SOC at any future time.
    """
    t_remain = max(t_target - t_now, 1.0)
    slope = (soc_target - soc_now) / t_remain

    def ref(t):
        return soc_now + slope * (t - t_now)

    return ref


def mpc_cost(
    u_vec, model, soc_now, I_avg_now, temp_k, t_now, t_target, dt_ctrl, P_prev
):
    """
    MPC cost function with target-runtime tracking.

    J = Σ_k [ α·(1 - P_k/P_max)²                    (QoS loss)
            + β·(SOC_k - SOC_ref(t_k))²              (trajectory tracking)
            + γ·((P_k - P_{k-1})/P_max)²  ]          (smoothness)
    """
    N = len(u_vec)
    Q_eff = model.get_capacity_at_temp(temp_k)
    soc_ref = soc_reference(t_now, soc_now, t_target)

    soc = soc_now
    I_avg = I_avg_now
    cost = 0.0
    P_prev_step = P_prev
    t_k = t_now

    for k in range(N):
        P_k = u_vec[k]

        # QoS loss: want power as high as possible
        cost += ALPHA_QOS * ((P_MAX - P_k) / P_MAX) ** 2

        # Smoothness penalty
        cost += GAMMA_SMOOTH * ((P_k - P_prev_step) / P_MAX) ** 2

        # Simulate one control interval (use 10s sub-steps for speed)
        sub_dt = 10.0
        n_sub = int(dt_ctrl / sub_dt)
        for _ in range(n_sub):
            V_term = model.terminal_voltage(soc, I_avg, temp_k)
            if V_term < model.V_cutoff or soc <= 0.01:
                cost += BETA_TRACK * 100.0 * (N - k)
                return cost
            I_new = P_k / V_term if V_term > 0 else I_avg
            I_avg = 0.9 * I_avg + 0.1 * I_new
            dsoc = -I_avg * sub_dt / (Q_eff * 3600)
            soc = max(soc + dsoc, 0.0)

        t_k += dt_ctrl

        # Tracking penalty: penalise deviation from reference SOC
        soc_ref_k = soc_ref(t_k)
        cost += BETA_TRACK * (soc - soc_ref_k) ** 2

        P_prev_step = P_k

    return cost


def mpc_solve_step(model, soc_now, I_avg_now, temp_k, t_now, t_target, P_prev):
    """Solve one MPC step and return the optimal first-step power."""
    N = MPC_N_STEPS
    dt_ctrl = MPC_DT_CTRL

    # Initial guess: scale power by ratio of remaining time to target time
    t_remain = max(t_target - t_now, 1.0)
    # Average power to reach target: P_avg ≈ Q_eff * V_avg * (SOC_now - SOC_cutoff) / t_remain
    Q_eff = model.get_capacity_at_temp(temp_k)
    P_est = Q_eff * 3.7 * (soc_now - SOC_CUTOFF) / (t_remain / 3600)
    P_est = np.clip(P_est, P_MIN, P_MAX)
    u0 = np.full(N, P_est)

    bounds = [(P_MIN, P_MAX)] * N

    result = minimize(
        mpc_cost,
        u0,
        args=(model, soc_now, I_avg_now, temp_k, t_now, t_target, dt_ctrl, P_prev),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 60, "ftol": 1e-7},
    )

    return result.x[0]


def simulate_mpc(model, t_target_h=T_TARGET, temp_k=298.15, dt=1.0):
    """Run full MPC simulation with target runtime."""
    Q_eff = model.get_capacity_at_temp(temp_k)
    t_target = t_target_h * 3600  # convert to seconds
    soc, t = 1.0, 0.0
    t_max = (t_target_h + 5) * 3600  # allow some overshoot
    I_avg = P_GAMING / 3.7
    P_current = P_GAMING

    time_pts, soc_pts, power_pts = [], [], []
    ctrl_counter = 0

    print(f"    MPC solving (T_target={t_target_h:.0f} h)", end="", flush=True)

    while soc > SOC_CUTOFF and t < t_max:
        # Re-optimise every MPC_DT_CTRL seconds
        if ctrl_counter % MPC_DT_CTRL == 0:
            P_current = mpc_solve_step(
                model, soc, I_avg, temp_k, t, t_target, P_current
            )
            if ctrl_counter % (MPC_DT_CTRL * 30) == 0:
                print(".", end="", flush=True)

        V_term = model.terminal_voltage(soc, I_avg, temp_k)
        if V_term < model.V_cutoff:
            break
        I_new = P_current / V_term if V_term > 0 else I_avg
        I_avg = 0.9 * I_avg + 0.1 * I_new
        dsoc = -I_avg * dt / (Q_eff * 3600)
        soc += dsoc
        t += dt
        ctrl_counter += 1

        if int(t) % 30 == 0:
            time_pts.append(t / 3600)
            soc_pts.append(soc)
            power_pts.append(P_current)

    print(" done")

    return {
        "time": np.array(time_pts),
        "soc": np.array(soc_pts),
        "power": np.array(power_pts),
        "tte": time_pts[-1] if time_pts else 0,
    }


# ═══════════════════════════════════════════════════════════════
# Figure generation
# ═══════════════════════════════════════════════════════════════
def generate():
    apply_style()
    model = MainBatteryModel(Q0=5.0)

    print("  Simulating baseline (constant gaming)...")
    baseline = simulate_baseline(model)
    print(f"    Baseline TTE: {baseline['tte']:.2f} h")

    print("  Simulating threshold strategy...")
    threshold = simulate_threshold(model)
    print(f"    Threshold TTE: {threshold['tte']:.2f} h")

    print("  Simulating MPC strategy...")
    mpc = simulate_mpc(model, t_target_h=T_TARGET)
    print(f"    MPC TTE: {mpc['tte']:.2f} h")

    ext_thresh = (threshold["tte"] - baseline["tte"]) / baseline["tte"] * 100
    ext_mpc = (mpc["tte"] - baseline["tte"]) / baseline["tte"] * 100
    print(f"    Threshold extension: +{ext_thresh:.1f}%")
    print(f"    MPC extension: +{ext_mpc:.1f}%")

    # ── 3-panel figure ──
    fig, axes = plt.subplots(
        3, 1, figsize=(7.48, 6.5), height_ratios=[1.2, 0.8, 0.8], sharex=True
    )
    ax_soc, ax_power, ax_qos = axes

    # ── Panel (a): SOC trajectories ──
    # Reference SOC line for MPC target
    t_ref = np.linspace(0, T_TARGET, 200)
    soc_ref_line = 1.0 - (1.0 - SOC_CUTOFF) * t_ref / T_TARGET
    ax_soc.plot(
        t_ref,
        soc_ref_line * 100,
        color="gray",
        lw=0.8,
        ls=":",
        label=f"SOC budget ($T_{{des}}$ = {T_TARGET:.0f} h)",
        zorder=1,
    )

    ax_soc.plot(
        baseline["time"],
        baseline["soc"] * 100,
        color=COLORS["gray"],
        lw=1.2,
        ls="--",
        label=f'No management (TTE = {baseline["tte"]:.1f} h)',
    )

    # Threshold: color by phase
    for phase_id, (clr, lbl) in enumerate(
        [
            (COLORS["accent"], "Threshold: full gaming"),
            (COLORS["accent"], "Threshold: throttled"),
            (COLORS["accent"], "Threshold: low-power"),
        ]
    ):
        mask = threshold["phase"] == phase_id
        if mask.any():
            t_seg = threshold["time"][mask]
            s_seg = threshold["soc"][mask] * 100
            if phase_id == 0:
                ax_soc.plot(
                    t_seg,
                    s_seg,
                    color=COLORS["accent"],
                    lw=1.2,
                    ls="-.",
                    label=f'Threshold (TTE = {threshold["tte"]:.1f} h)',
                )
            else:
                ax_soc.plot(t_seg, s_seg, color=COLORS["accent"], lw=1.2, ls="-.")

    ax_soc.plot(
        mpc["time"],
        mpc["soc"] * 100,
        color=COLORS["shepherd"],
        lw=1.5,
        label=f'MPC (TTE = {mpc["tte"]:.1f} h)',
    )

    # Threshold lines
    ax_soc.axhline(SOC_THRESH_1 * 100, color="gray", ls=":", lw=0.4, alpha=0.4)
    ax_soc.axhline(SOC_THRESH_2 * 100, color="gray", ls=":", lw=0.4, alpha=0.4)

    ax_soc.set_ylabel("SOC (%)")
    ax_soc.set_ylim(0, 105)
    ax_soc.legend(loc="upper right", fontsize=6, framealpha=0.9)
    label_panel(ax_soc, "a")

    # ── Panel (b): Power profiles ──
    ax_power.plot(
        baseline["time"],
        baseline["power"],
        color=COLORS["gray"],
        lw=1.0,
        ls="--",
        label="No management",
    )
    ax_power.plot(
        threshold["time"],
        threshold["power"],
        color=COLORS["accent"],
        lw=1.0,
        ls="-.",
        label="Threshold",
    )
    ax_power.plot(
        mpc["time"],
        mpc["power"],
        color=COLORS["shepherd"],
        lw=1.3,
        label="MPC",
    )
    ax_power.fill_between(
        mpc["time"], mpc["power"], alpha=0.10, color=COLORS["shepherd"]
    )

    ax_power.set_ylabel("Power (W)")
    ax_power.set_ylim(0, 5.5)
    ax_power.legend(loc="upper right", fontsize=6.5)
    label_panel(ax_power, "b")

    # ── Panel (c): Cumulative energy delivered ──
    def cum_energy(time_h, power):
        dt_h = np.diff(time_h, prepend=0)
        return np.cumsum(power * dt_h)

    E_base = cum_energy(baseline["time"], baseline["power"])
    E_thresh = cum_energy(threshold["time"], threshold["power"])
    E_mpc = cum_energy(mpc["time"], mpc["power"])

    ax_qos.plot(
        baseline["time"],
        E_base,
        color=COLORS["gray"],
        lw=1.0,
        ls="--",
        label="No management",
    )
    ax_qos.plot(
        threshold["time"],
        E_thresh,
        color=COLORS["accent"],
        lw=1.0,
        ls="-.",
        label="Threshold",
    )
    ax_qos.plot(mpc["time"], E_mpc, color=COLORS["shepherd"], lw=1.3, label="MPC")

    ax_qos.set_xlabel("Time (h)")
    ax_qos.set_ylabel("Cumulative energy\ndelivered (Wh)")
    ax_qos.legend(loc="lower right", fontsize=6.5)
    label_panel(ax_qos, "c")

    fig.align_ylabels(axes)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    save_fig(fig, "fig_energy_management", RESULTS_DIR)

    # Also save directly to ecm_paper/figures
    import shutil

    for fmt in ("pdf", "png"):
        src = os.path.join(RESULTS_DIR, f"fig_energy_management.{fmt}")
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(FIGURES_DIR, f"fig_energy_management.{fmt}"))

    # ── Summary CSV ──
    import csv

    csv_path = os.path.join(RESULTS_DIR, "energy_management_summary.csv")

    E_total_base = E_base[-1] if len(E_base) > 0 else 0
    E_total_thresh = E_thresh[-1] if len(E_thresh) > 0 else 0
    E_total_mpc = E_mpc[-1] if len(E_mpc) > 0 else 0

    eff_base = baseline["tte"] / E_total_base if E_total_base > 0 else 0
    eff_thresh = threshold["tte"] / E_total_thresh if E_total_thresh > 0 else 0
    eff_mpc = mpc["tte"] / E_total_mpc if E_total_mpc > 0 else 0

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Strategy",
                "TTE_h",
                "Energy_Wh",
                "Extension_%",
                "Efficiency_h/Wh",
                "Eff_improvement_%",
            ]
        )
        w.writerow(
            [
                "No management",
                f"{baseline['tte']:.2f}",
                f"{E_total_base:.2f}",
                "0.0",
                f"{eff_base:.4f}",
                "0.0",
            ]
        )
        w.writerow(
            [
                "Threshold",
                f"{threshold['tte']:.2f}",
                f"{E_total_thresh:.2f}",
                f"{ext_thresh:.1f}",
                f"{eff_thresh:.4f}",
                f"{(eff_thresh/eff_base - 1)*100:.1f}",
            ]
        )
        w.writerow(
            [
                "MPC",
                f"{mpc['tte']:.2f}",
                f"{E_total_mpc:.2f}",
                f"{ext_mpc:.1f}",
                f"{eff_mpc:.4f}",
                f"{(eff_mpc/eff_base - 1)*100:.1f}",
            ]
        )

    print(f"  ✓ Summary CSV saved to {csv_path}")
    print(f"\n  === Results Summary ===")
    print(f"  Baseline:  TTE = {baseline['tte']:.2f} h, E = {E_total_base:.1f} Wh")
    print(
        f"  Threshold: TTE = {threshold['tte']:.2f} h, E = {E_total_thresh:.1f} Wh, +{ext_thresh:.1f}%"
    )
    print(
        f"  MPC:       TTE = {mpc['tte']:.2f} h, E = {E_total_mpc:.1f} Wh, +{ext_mpc:.1f}%"
    )
    print(
        f"  Eff improvement: Threshold {(eff_thresh/eff_base-1)*100:.1f}%, MPC {(eff_mpc/eff_base-1)*100:.1f}%"
    )

    return fig


if __name__ == "__main__":
    generate()
