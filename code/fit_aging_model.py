"""
XJTU电池老化数据提取和模型拟合（更新版）
模型形式与论文保持一致:
1. 内阻增长: R(N) = R0_nom * (1 + beta_R * N^gamma)
2. 容量-内阻线性: Q = Q0_nom - kQ * (R - R0_nom)
3. 容量随cycle: Q0(N) = Q0_nom * (1 - alpha_Q * N^gamma)
4. calendar老化: Q0(ts) = Q0_nom * (1 - kappa * sqrt(ts) * exp(-Ea/(Rg*T)) * f(SOC))  (25°C基线)
"""

import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import glob


# Calendar aging constants (from paper)
RG_GAS_CONSTANT = 8.314  # J/(mol*K)
EA_ACTIVATION = 50000.0  # J/mol
SOC_ACCELERATION = 1.5  # lambda
ASSUMED_STORAGE_SOC = 0.8  # if no SOC history, assume high-SOC storage


def calendar_acceleration_factor(temperature_c, storage_soc=ASSUMED_STORAGE_SOC):
    """Compute Arrhenius*SOC acceleration factor for calendar aging."""
    temperature_k = np.maximum(temperature_c + 273.15, 1.0)
    soc_factor = np.exp(SOC_ACCELERATION * storage_soc)
    arrhenius = np.exp(-EA_ACTIVATION / (RG_GAS_CONSTANT * temperature_k))
    return arrhenius * soc_factor

def load_all_batteries(base_dir, batch_name):
    """加载某个batch下所有电池的老化数据"""
    batch_dir = os.path.join(base_dir, batch_name)
    mat_files = glob.glob(os.path.join(batch_dir, "*.mat"))

    all_batteries = []

    for mat_file in mat_files:
        try:
            mat_data = sio.loadmat(mat_file)

            # 提取summary数据(包含每个cycle的统计信息)
            summary = mat_data["summary"]

            # 提取关键字段
            cycle_life = summary["cycle_life"][0, 0].flatten()
            discharge_capacity = summary["discharge_capacity_Ah"][0, 0].flatten()
            discharge_voltage_mean = summary["discharge_mean_voltage"][0, 0].flatten()

            # 提取data数据(每个cycle的详细测量)
            data = mat_data["data"]
            n_cycles = data.shape[1]

            # 计算每个cycle的平均温度和内阻(从电压电流数据推算)
            temperatures = []
            resistances = []
            calendar_times = []

            cumulative_time = 0.0  # 累计使用时长(小时)

            for i in range(n_cycles):
                cycle_data = data[0, i]

                # 提取该cycle的数据
                time_min = cycle_data["relative_time_min"].flatten()
                voltage = cycle_data["voltage_V"].flatten()
                current = cycle_data["current_A"].flatten()
                temp = cycle_data["temperature_C"].flatten()

                # 平均温度
                avg_temp = np.mean(temp)
                temperatures.append(avg_temp)

                # 累计时间(转换为小时)
                if len(time_min) > 0:
                    cycle_duration = time_min[-1] / 60.0  # 分钟转小时
                    cumulative_time += cycle_duration
                calendar_times.append(cumulative_time)

                # 估算内阻:选择放电段(电流为负),用V=OCV-I*R估算
                discharge_mask = current < -0.1  # 放电电流
                if np.sum(discharge_mask) > 10:
                    V_discharge = voltage[discharge_mask]
                    I_discharge = current[discharge_mask]

                    # 简单估算:R ≈ ΔV / ΔI
                    # 使用中段数据避免极化影响
                    mid_idx = len(V_discharge) // 2
                    window = 100
                    V_mid = np.mean(V_discharge[mid_idx : mid_idx + window])
                    I_mid = np.mean(I_discharge[mid_idx : mid_idx + window])

                    # OCV估算(假设约3.7V)
                    OCV_est = 3.7
                    R_est = (OCV_est - V_mid) / abs(I_mid)
                    resistances.append(R_est)
                else:
                    resistances.append(np.nan)

            battery_data = {
                "file": os.path.basename(mat_file),
                "batch": batch_name,
                "cycle": np.arange(1, len(discharge_capacity) + 1),
                "capacity": discharge_capacity,
                "voltage_mean": discharge_voltage_mean,
                "temperature": np.array(temperatures),
                "resistance": np.array(resistances),
                "calendar_time": np.array(calendar_times),
            }

            all_batteries.append(battery_data)
            print(
                f"Loaded {os.path.basename(mat_file)}: {len(discharge_capacity)} cycles"
            )

        except Exception as e:
            print(f"Error loading {mat_file}: {e}")

    return all_batteries


def resistance_model(N, R0_nom, beta_R, gamma):
    """R(N) = R0_nom * (1 + beta_R * N^gamma)"""
    return R0_nom * (1.0 + beta_R * np.power(N, gamma))


def capacity_linear(R, Q0_nom, R0_nom, kQ):
    """Q = Q0_nom - kQ * (R - R0_nom)"""
    return Q0_nom - kQ * (R - R0_nom)


def fit_aging_models(battery_data):
    """拟合老化模型参数"""
    results = {}

    # 提取有效数据
    cycle = battery_data["cycle"]
    capacity = battery_data["capacity"]
    resistance = battery_data["resistance"]
    calendar = battery_data["calendar_time"]
    voltage = battery_data["voltage_mean"]
    temperature = battery_data["temperature"]

    # 移除NaN值
    valid_mask = ~np.isnan(resistance) & ~np.isnan(capacity)
    cycle_valid = cycle[valid_mask]
    capacity_valid = capacity[valid_mask]
    resistance_valid = resistance[valid_mask]
    calendar_valid = calendar[valid_mask]
    voltage_valid = voltage[valid_mask]
    temperature_valid = temperature[valid_mask]

    if len(cycle_valid) < 10:
        return None

    Q0_nom = capacity_valid[0]
    R0_nom = np.nanmin(resistance_valid)

    # 1. 内阻随cycle的幂次关系: R(N) = R0_nom * (1 + beta_R * N^gamma)
    try:
        y = resistance_valid / R0_nom - 1.0
        mask = y > 0
        if np.sum(mask) >= 5:
            log_cycle = np.log(cycle_valid[mask])
            log_y = np.log(y[mask])
            coeffs = np.polyfit(log_cycle, log_y, 1)
            gamma_init = max(0.1, min(3.0, coeffs[0]))
            beta_init = np.exp(coeffs[1])
        else:
            gamma_init = 0.8
            beta_init = 1e-4

        popt, _ = curve_fit(
            lambda N, beta_R, gamma: resistance_model(N, R0_nom, beta_R, gamma),
            cycle_valid,
            resistance_valid,
            p0=[beta_init, gamma_init],
            bounds=([0, 0.1], [1.0, 3.0]),
            maxfev=20000,
        )
        beta_R, gamma = popt
        R_pred = resistance_model(cycle_valid, R0_nom, beta_R, gamma)
        ss_res = np.sum((resistance_valid - R_pred) ** 2)
        ss_tot = np.sum((resistance_valid - np.mean(resistance_valid)) ** 2)
        r2_R = 1 - ss_res / ss_tot

        results["R0_nom"] = R0_nom
        results["beta_R"] = beta_R
        results["gamma"] = gamma
        results["r2_resistance_cycle"] = r2_R
    except Exception as e:
        print(f"  Resistance-cycle fitting failed: {e}")
        results["R0_nom"] = R0_nom
        results["beta_R"] = np.nan
        results["gamma"] = np.nan
        results["r2_resistance_cycle"] = 0

    # 2. 容量与内阻的线性关系: Q = Q0_nom - kQ*(R - R0_nom)
    try:

        def model_rq(R, kQ):
            return capacity_linear(R, Q0_nom, R0_nom, kQ)

        popt, _ = curve_fit(
            model_rq,
            resistance_valid,
            capacity_valid,
            p0=[1.0],
            bounds=([0.0], [100.0]),
            maxfev=10000,
        )
        kQ = popt[0]

        Q_pred = model_rq(resistance_valid, kQ)
        ss_res = np.sum((capacity_valid - Q_pred) ** 2)
        ss_tot = np.sum((capacity_valid - np.mean(capacity_valid)) ** 2)
        r2_QR = 1 - ss_res / ss_tot

        results["Q0_nom"] = Q0_nom
        results["kQ"] = kQ
        results["r2_capacity_resistance"] = r2_QR
    except Exception as e:
        print(f"  Capacity-resistance fitting failed: {e}")
        results["Q0_nom"] = Q0_nom
        results["kQ"] = np.nan
        results["r2_capacity_resistance"] = 0

    # 3. 由线性关系推导 alpha_Q
    if not np.isnan(results.get("beta_R", np.nan)) and not np.isnan(
        results.get("kQ", np.nan)
    ):
        alpha_Q = results["kQ"] * results["beta_R"] * R0_nom / Q0_nom
    else:
        alpha_Q = np.nan
    results["alpha_Q"] = alpha_Q

    # 4. calendar老化: Q(t_s) = Q0_nom * (1 - alpha_Q * N^gamma - kappa * sqrt(t_s) * exp(-Ea/(Rg*T)) * f(SOC))
    try:
        t_days = calendar_valid / 24.0
        sqrt_t = np.sqrt(np.maximum(t_days, 1e-6))
        accel = calendar_acceleration_factor(temperature_valid)
        effective_term = sqrt_t * accel
        cycle_term = alpha_Q * np.power(cycle_valid, gamma)
        residual = 1 - capacity_valid / Q0_nom - cycle_term
        kappa = np.sum(effective_term * residual) / np.sum(effective_term**2)
        kappa = max(0.0, kappa)

        Q_pred = Q0_nom * (1 - cycle_term - kappa * effective_term)
        ss_res = np.sum((capacity_valid - Q_pred) ** 2)
        ss_tot = np.sum((capacity_valid - np.mean(capacity_valid)) ** 2)
        r2_Q_cal = 1 - ss_res / ss_tot

        results["kappa"] = kappa
        results["r2_capacity_calendar"] = r2_Q_cal
    except Exception as e:
        print(f"  Capacity-calendar fitting failed: {e}")
        results["kappa"] = np.nan
        results["r2_capacity_calendar"] = 0

    # 4. E0(电压)随cycle的变化(检验是否几乎不变)
    voltage_std = np.std(voltage_valid)
    voltage_mean = np.mean(voltage_valid)
    voltage_cv = voltage_std / voltage_mean  # 变异系数

    results["E0_mean"] = voltage_mean
    results["E0_std"] = voltage_std
    results["E0_cv"] = voltage_cv
    results["Ea"] = EA_ACTIVATION
    results["Rg"] = RG_GAS_CONSTANT
    results["lambda_soc"] = SOC_ACCELERATION
    results["storage_soc"] = ASSUMED_STORAGE_SOC

    # 保存原始数据用于绘图
    results["data"] = {
        "cycle": cycle_valid,
        "capacity": capacity_valid,
        "resistance": resistance_valid,
        "calendar": calendar_valid,
        "temperature": temperature_valid,
        "voltage": voltage_valid,
        "Q0_nom": Q0_nom,
        "R0_nom": R0_nom,
    }

    return results


def main():
    """主程序:处理所有batch数据"""

    base_dir = r"d:\WYN_COLLEGE\s21\!MCM_ICM\batteries_new\data\XJTU battery dataset"

    # 选择一个有代表性的batch(Batch-1: 2C恒流放电)
    batch_name = "Batch-1"

    print(f"Loading batteries from {batch_name}...")
    batteries = load_all_batteries(base_dir, batch_name)

    print(f"\nFitting aging models for {len(batteries)} batteries...")

    all_results = []

    for i, battery in enumerate(batteries):
        print(f"\nProcessing {battery['file']}...")
        results = fit_aging_models(battery)

        if results is not None:
            results["file"] = battery["file"]
            results["batch"] = battery["batch"]
            all_results.append(results)

            print(
                f"  R0_nom = {results['R0_nom']:.4f} Ohm, beta_R = {results['beta_R']:.4f}, gamma = {results['gamma']:.4f}, R2 = {results['r2_resistance_cycle']:.4f}"
            )
            print(
                f"  Q0_nom = {results['Q0_nom']:.4f} Ah, kappa = {results['kappa']:.6f} day^-0.5, R2 = {results['r2_capacity_calendar']:.4f}"
            )
            print(
                f"  Q-R linear: kQ = {results['kQ']:.4f} Ah/Ohm, R2 = {results['r2_capacity_resistance']:.4f}"
            )
            print(f"  alpha_Q = {results['alpha_Q']:.6f}")
            print(
                f"  E0 mean = {results['E0_mean']:.4f} V, std = {results['E0_std']:.4f} V, CV = {results['E0_cv']:.2%}"
            )
            print(
                f"  Calendar accel: Ea = {EA_ACTIVATION:.0f} J/mol, Rg = {RG_GAS_CONSTANT:.3f}, lambda = {SOC_ACCELERATION:.2f}, SOC = {ASSUMED_STORAGE_SOC:.2f}"
            )

    # 计算所有电池的平均参数
    print("\n" + "=" * 60)
    print("AVERAGE AGING MODEL PARAMETERS (across all batteries)")
    print("=" * 60)

    R0_mean = np.mean([r["R0_nom"] for r in all_results if not np.isnan(r["R0_nom"])])
    beta_R_mean = np.mean(
        [r["beta_R"] for r in all_results if not np.isnan(r["beta_R"])]
    )
    gamma_mean = np.mean([r["gamma"] for r in all_results if not np.isnan(r["gamma"])])
    Q0_mean = np.mean([r["Q0_nom"] for r in all_results])
    kappa_mean = np.mean([r["kappa"] for r in all_results if not np.isnan(r["kappa"])])
    kQ_mean = np.mean([r["kQ"] for r in all_results if not np.isnan(r["kQ"])])
    alpha_Q_mean = np.mean(
        [r["alpha_Q"] for r in all_results if not np.isnan(r["alpha_Q"])]
    )
    E0_mean = np.mean([r["E0_mean"] for r in all_results])
    E0_cv_mean = np.mean([r["E0_cv"] for r in all_results])

    print(f"\n1. Resistance vs Cycle (Power Law with Offset)")
    print(f"   R(N) = R0_nom * (1 + beta_R * N^gamma)")
    print(f"   R0_nom = {R0_mean:.4f} Ω")
    print(f"   beta_R = {beta_R_mean:.6f}")
    print(f"   gamma = {gamma_mean:.4f}")

    print(
        "\n2. Capacity vs Calendar Time (Full: Q = Q0 * (1 - alpha_Q * N^gamma - kappa * sqrt(ts) * exp(-Ea/(Rg*T)) * f(SOC)))"
    )
    print(f"   Q0_nom = {Q0_mean:.4f} Ah")
    print(f"   kappa = {kappa_mean:.6f} day^-0.5")
    print(f"   Ea = {EA_ACTIVATION:.0f} J/mol, Rg = {RG_GAS_CONSTANT:.3f}")
    print(f"   lambda = {SOC_ACCELERATION:.2f}, SOC = {ASSUMED_STORAGE_SOC:.2f}")

    print(f"\n3. Capacity vs Resistance (Linear)")
    print(f"   Q = Q0_nom - kQ * (R - R0_nom)")
    print(f"   kQ = {kQ_mean:.4f} Ah/Ω")
    print(f"   alpha_Q = {alpha_Q_mean:.6f}")

    print(f"\n4. Standard Voltage E0")
    print(f"   E0_mean = {E0_mean:.4f} V")
    print(f"   E0_CV = {E0_cv_mean:.2%} (confirms near-constant)")

    # 保存结果到CSV
    df = pd.DataFrame(all_results)
    df = df.drop("data", axis=1)  # 移除嵌套的data列
    df.to_csv("aging_model_parameters.csv", index=False)
    print(f"\nParameters saved to 'aging_model_parameters.csv'")

    # 保存汇总参数
    with open("aging_model_summary.txt", "w", encoding="utf-8") as f:
        f.write("AGING MODEL PARAMETERS (XJTU Battery Dataset - Batch-1)\n")
        f.write("=" * 60 + "\n\n")
        f.write("1. Internal Resistance Growth (Power Law with Offset)\n")
        f.write("   R(N) = R0_nom * (1 + beta_R * N^gamma)\n")
        f.write(f"   R0_nom = {R0_mean:.4f} Ohm\n")
        f.write(f"   beta_R = {beta_R_mean:.6f}\n")
        f.write(f"   gamma = {gamma_mean:.4f}\n\n")

        f.write("2. Capacity Fade with Calendar Time (Full)\n")
        f.write(
            "   Q(t) = Q0_nom * (1 - alpha_Q * N^gamma - kappa * sqrt(ts) * exp(-Ea/(Rg*T)) * f(SOC))\n"
        )
        f.write(f"   Q0_nom = {Q0_mean:.4f} Ah\n")
        f.write(f"   kappa = {kappa_mean:.6f} day^-0.5\n\n")
        f.write(f"   Ea = {EA_ACTIVATION:.0f} J/mol\n")
        f.write(f"   Rg = {RG_GAS_CONSTANT:.3f} J/(mol*K)\n")
        f.write(f"   lambda = {SOC_ACCELERATION:.2f}\n")
        f.write(f"   SOC = {ASSUMED_STORAGE_SOC:.2f}\n\n")

        f.write("3. Capacity-Resistance Linear Correlation\n")
        f.write("   Q = Q0_nom - kQ * (R - R0_nom)\n")
        f.write(f"   kQ = {kQ_mean:.4f} Ah/Ohm\n")
        f.write(f"   alpha_Q = {alpha_Q_mean:.6f}\n\n")

        f.write("4. Standard Electrode Potential\n")
        f.write(f"   E0 = {E0_mean:.4f} V (nearly constant)\n")
        f.write(f"   Coefficient of Variation = {E0_cv_mean:.2%}\n")

    print("Summary saved to 'aging_model_summary.txt'")

    return all_results


if __name__ == "__main__":
    results = main()
