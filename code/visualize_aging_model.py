"""
老化模型可视化
生成4个子图展示老化规律:
1. 内阻vs Cycle (幂律关系)
2. 容量vs Calendar Time (Arrhenius+SOC)
3. 容量vs内阻 (线性关系)
4. E0 vs Cycle (验证几乎不变)
"""

import sys

sys.path.append(".")

from fit_aging_model import (
    load_all_batteries,
    fit_aging_models,
    calendar_acceleration_factor,
    EA_ACTIVATION,
    RG_GAS_CONSTANT,
    SOC_ACCELERATION,
    ASSUMED_STORAGE_SOC,
)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# 设置中文字体和负号显示
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10


def create_aging_visualization():
    """创建老化模型的4面板可视化"""

    # 加载数据
    base_dir = r"d:\WYN_COLLEGE\s21\!MCM_ICM\batteries_new\data\XJTU battery dataset"
    batch_name = "Batch-1"

    print("Loading battery data...")
    batteries = load_all_batteries(base_dir, batch_name)

    # 选择一个代表性电池进行详细展示
    representative_idx = 0  # 使用第一个电池
    battery = batteries[representative_idx]

    print(f"Fitting aging model for {battery['file']}...")
    results = fit_aging_models(battery)

    if results is None:
        print("Failed to fit aging model!")
        return

    # 提取拟合结果
    data = results["data"]
    cycle = data["cycle"]
    capacity = data["capacity"]
    resistance = data["resistance"]
    calendar = data["calendar"]
    temperature = data["temperature"]
    voltage = data["voltage"]

    # 模型参数
    R0_nom = results["R0_nom"]
    beta_R = results["beta_R"]
    gamma = results["gamma"]
    Q0_nom = results["Q0_nom"]
    kQ = results["kQ"]
    alpha_Q = results["alpha_Q"]
    kappa = results["kappa"]
    E0_mean = results["E0_mean"]
    E0_std = results["E0_std"]

    # 创建图形
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # === Panel A: 内阻 vs Cycle (幂律) ===
    ax1 = fig.add_subplot(gs[0, 0])

    # 实验数据
    ax1.scatter(
        cycle, resistance, s=20, alpha=0.5, color="blue", label="Experimental Data"
    )

    # 拟合曲线
    cycle_fit = np.linspace(cycle.min(), cycle.max(), 200)
    R_fit = R0_nom * (1.0 + beta_R * np.power(cycle_fit, gamma))
    ax1.plot(
        cycle_fit,
        R_fit,
        "r-",
        linewidth=2,
        label=f"Power Law Fit: $R = {R0_nom:.4f}(1+{beta_R:.4f}N^{{{gamma:.3f}}})$",
    )

    ax1.set_xlabel("Cycle Number $N$", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Internal Resistance $R$ (Ω)", fontsize=11, fontweight="bold")
    ax1.set_title("(A) Resistance Growth with Cycling", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.text(
        0.95,
        0.05,
        f'$R^2$ = {results["r2_resistance_cycle"]:.4f}',
        transform=ax1.transAxes,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # === Panel B: 容量 vs Calendar Time (Arrhenius+SOC) ===
    ax2 = fig.add_subplot(gs[0, 1])

    # 实验数据
    ax2.scatter(
        calendar, capacity, s=20, alpha=0.5, color="green", label="Experimental Data"
    )

    # 拟合曲线 (calendar aging, alpha_Q * N^gamma + Arrhenius + SOC)
    cal_fit = np.linspace(calendar.min(), calendar.max(), 200)
    cal_days = np.maximum(cal_fit / 24.0, 1e-6)
    temp_fit = np.interp(cal_fit, calendar, temperature)
    accel_fit = calendar_acceleration_factor(temp_fit)
    cycle_fit = np.interp(cal_fit, calendar, cycle)
    Q_fit = Q0_nom * (
        1.0 - alpha_Q * np.power(cycle_fit, gamma) - kappa * np.sqrt(cal_days) * accel_fit
    )
    ax2.plot(
        cal_fit,
        Q_fit,
        "r-",
        linewidth=2,
        label=(
            f"Full: $Q = {Q0_nom:.3f}(1-\\alpha_Q N^\\gamma-{kappa:.5f}\\sqrt{{t}}\\,e^{{-E_a/(R_gT)}}\\,e^{{\\lambda\\,SOC}})$"
        ),
    )

    ax2.set_xlabel("Calendar Time $t$ (hours)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Discharge Capacity $Q$ (Ah)", fontsize=11, fontweight="bold")
    ax2.set_title("(B) Capacity Fade with Usage Time", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.text(
        0.95,
        0.05,
        f'$R^2$ = {results["r2_capacity_calendar"]:.4f}',
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # === Panel C: 容量 vs 内阻 (线性) ===
    ax3 = fig.add_subplot(gs[1, 0])

    # 实验数据
    ax3.scatter(
        resistance, capacity, s=20, alpha=0.5, color="purple", label="Experimental Data"
    )

    # 拟合直线
    R_fit_range = np.linspace(resistance.min(), resistance.max(), 200)
    Q_fit_linear = Q0_nom - kQ * (R_fit_range - R0_nom)
    ax3.plot(
        R_fit_range,
        Q_fit_linear,
        "r-",
        linewidth=2,
        label=f"Linear Fit: $Q = {Q0_nom:.3f} - {kQ:.3f}(R-{R0_nom:.3f})$",
    )

    ax3.set_xlabel("Internal Resistance $R$ (Ω)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Discharge Capacity $Q$ (Ah)", fontsize=11, fontweight="bold")
    ax3.set_title("(C) Capacity-Resistance Correlation", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper right", framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    ax3.text(
        0.95,
        0.05,
        f'$R^2$ = {results["r2_capacity_resistance"]:.4f}',
        transform=ax3.transAxes,
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # === Panel D: E0 vs Cycle (验证恒定) ===
    ax4 = fig.add_subplot(gs[1, 1])

    # 实验数据
    ax4.scatter(
        cycle, voltage, s=20, alpha=0.5, color="orange", label="Mean Discharge Voltage"
    )

    # 平均值线
    ax4.axhline(
        E0_mean,
        color="red",
        linewidth=2,
        linestyle="--",
        label=f"Mean $E_0$ = {E0_mean:.4f} V",
    )

    # 标准差区间
    ax4.fill_between(
        [cycle.min(), cycle.max()],
        E0_mean - E0_std,
        E0_mean + E0_std,
        color="red",
        alpha=0.2,
        label=f"±1σ = {E0_std:.4f} V",
    )

    ax4.set_xlabel("Cycle Number $N$", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Mean Discharge Voltage $E_0$ (V)", fontsize=11, fontweight="bold")
    ax4.set_title("(D) Standard Voltage Stability", fontsize=12, fontweight="bold")
    ax4.legend(loc="lower left", framealpha=0.9, fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.text(
        0.95,
        0.95,
        f'CV = {results["E0_cv"]:.2%}\\n(Nearly Constant)',
        transform=ax4.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
    )

    # 整体标题
    fig.suptitle(
        f'Battery Aging Model Validation - {battery["file"]}',
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    # 保存图形
    output_dir = "../results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "fig_aging_model_validation.png")

    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nFigure saved to: {output_path}")

    # 输出关键发现
    print("\n" + "=" * 60)
    print("KEY AGING MODEL FINDINGS")
    print("=" * 60)
    print(f"\n1. Resistance Growth (Power Law with Offset)")
    print(f"   R(N) = {R0_nom:.4f} × (1 + {beta_R:.6f} × N^{gamma:.4f})")
    print(f"   - At cycle 100: R = {R0_nom * (1 + beta_R * 100**gamma):.4f} Ω")
    print(f"   - At cycle 400: R = {R0_nom * (1 + beta_R * 400**gamma):.4f} Ω")

    print(f"\n2. Capacity Fade (Full Model)")
    print(
        f"   Q(t) = {Q0_nom:.3f} × (1 - {alpha_Q:.6f} × N^{gamma:.4f} - {kappa:.6f} × sqrt(t_days) × exp(-Ea/(Rg*T)) × exp(lambda·SOC))"
    )
    print(f"   Ea = {EA_ACTIVATION:.0f} J/mol, Rg = {RG_GAS_CONSTANT:.3f}")
    print(f"   lambda = {SOC_ACCELERATION:.2f}, SOC = {ASSUMED_STORAGE_SOC:.2f}")
    accel_1000h = calendar_acceleration_factor(np.array([np.mean(temperature)]))[0]
    accel_5000h = accel_1000h
    cycle_1000h = np.interp(1000.0, calendar, cycle)
    cycle_5000h = np.interp(5000.0, calendar, cycle)
    print(
        f"   - At 1000h: Q = {Q0_nom * (1 - alpha_Q * cycle_1000h**gamma - kappa * np.sqrt(1000/24) * accel_1000h):.3f} Ah"
    )
    print(
        f"   - At 5000h: Q = {Q0_nom * (1 - alpha_Q * cycle_5000h**gamma - kappa * np.sqrt(5000/24) * accel_5000h):.3f} Ah"
    )

    print(f"\n3. Capacity-Resistance Linear Correlation")
    print(f"   Q = {Q0_nom:.3f} - {kQ:.3f} × (R - {R0_nom:.3f})")
    print(f"   - Correlation coefficient: R² = {results['r2_capacity_resistance']:.4f}")
    print(
        f"   - Interpretation: Every 0.01Ω increase → {abs(kQ)*0.01:.3f} Ah capacity loss"
    )

    print(f"\n4. Capacity-Cycle Consistency")
    print(f"   Q0(N) = Q0_nom × (1 - alpha_Q × N^{gamma:.4f})")
    print(f"   alpha_Q = {alpha_Q:.6f}")

    print(f"\n4. Standard Voltage Stability")
    print(f"   E0 = {E0_mean:.4f} ± {E0_std:.4f} V")
    print(f"   - Coefficient of Variation: {results['E0_cv']:.2%}")
    print(f"   - Conclusion: E0 remains nearly constant over battery lifetime ✓")

    print("\n" + "=" * 60)

    plt.show()


if __name__ == "__main__":
    create_aging_visualization()
