"""
生成各场景功耗验证表格
输出为CSV、LaTeX和Markdown格式
"""

import os
import pandas as pd

# 场景定义 (来自表格)
scenarios = {
    "Standby": {"screen": (0, 0), "soc": 0, "radio": 10, "gps": 10, "baseline": 100},
    "Navigation": {
        "screen": (70, 1.0),
        "soc": 20,
        "radio": 30,
        "gps": 100,
        "baseline": 100,
    },
    "Gaming": {"screen": (90, 1.3), "soc": 95, "radio": 95, "gps": 10, "baseline": 100},
    "Video": {"screen": (80, 1.0), "soc": 20, "radio": 90, "gps": 10, "baseline": 100},
    "Reading": {
        "screen": (70, 1.0),
        "soc": 20,
        "radio": 20,
        "gps": 10,
        "baseline": 100,
    },
}

# 功率公式参数
P_SOC_MAX = 2040.0  # mW (修改为2.04W)
P_RADIO_MAX = 1173.0  # mW
P_GPS = 300.0  # mW
P_BASELINE = 120.0  # mW
GAMMA_LS_A = 0.4991
LAMBDA_S_A = 0.1113

# 计算各场景功耗
data = []
for name, params in scenarios.items():
    # Screen
    B_pct, eta = params["screen"]
    B = B_pct / 100.0
    P_screen_W = eta * (GAMMA_LS_A * B + LAMBDA_S_A)
    P_screen = P_screen_W * 1000

    # SoC (包含10%背景负载)
    soc_pct = params["soc"]
    soc_load = (soc_pct / 100.0) + 0.1
    soc_load = min(soc_load, 1.0)
    P_soc = soc_load * P_SOC_MAX

    # Radio
    radio_pct = params["radio"]
    radio_activity = radio_pct / 100.0
    P_radio = radio_activity * P_RADIO_MAX

    # GPS
    gps_pct = params["gps"]
    gps_factor = gps_pct / 100.0
    P_gps = gps_factor * P_GPS

    # Baseline (always 100%)
    baseline_pct = params["baseline"]
    P_baseline_calc = (baseline_pct / 100.0) * P_BASELINE

    # Total
    total = P_screen + P_soc + P_radio + P_gps + P_baseline_calc

    data.append(
        {
            "场景": name,
            "Screen参数": f"{B_pct}%, η={eta}",
            "P_screen_mW": f"{P_screen:.1f}",
            "SoC参数": f"{soc_pct}%+10%bg",
            "P_SoC_mW": f"{P_soc:.1f}",
            "Radio参数": f"{radio_pct}%",
            "P_radio_mW": f"{P_radio:.1f}",
            "GPS参数": f"{gps_pct}%",
            "P_GPS_mW": f"{P_gps:.1f}",
            "Baseline参数": f"{baseline_pct}%",
            "P_baseline_mW": f"{P_baseline_calc:.1f}",
            "总功耗_mW": f"{total:.1f}",
        }
    )

df = pd.DataFrame(data)

# 保存路径
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(results_dir, exist_ok=True)

# ========== 保存CSV ==========
csv_path = os.path.join(results_dir, "scenario_power_verification.csv")
df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"✅ Saved CSV: {csv_path}")

# ========== 生成简化版CSV (用于论文) ==========
data_simple = []
for name, params in scenarios.items():
    B_pct, eta = params["screen"]
    B = B_pct / 100.0
    P_screen_W = eta * (GAMMA_LS_A * B + LAMBDA_S_A)
    P_screen = P_screen_W * 1000

    soc_pct = params["soc"]
    soc_load = (soc_pct / 100.0) + 0.1
    soc_load = min(soc_load, 1.0)
    P_soc = soc_load * P_SOC_MAX

    radio_pct = params["radio"]
    P_radio = (radio_pct / 100.0) * P_RADIO_MAX

    gps_pct = params["gps"]
    P_gps = (gps_pct / 100.0) * P_GPS

    P_baseline_calc = P_BASELINE
    total = P_screen + P_soc + P_radio + P_gps + P_baseline_calc

    data_simple.append(
        {
            "Scenario": name,
            "Screen": f"{B_pct}%, η={eta}→{P_screen:.1f}mW",
            "SoC": f"{soc_pct}%+10%bg→{P_soc:.0f}mW",
            "Radio": f"{radio_pct}%→{P_radio:.1f}mW",
            "GPS": f"{gps_pct}%→{P_gps:.0f}mW",
            "Baseline": f"100%→{P_baseline_calc:.0f}mW",
            "Total": f"{total:.1f}mW",
        }
    )

df_simple = pd.DataFrame(data_simple)
csv_simple_path = os.path.join(results_dir, "scenario_power_simple.csv")
df_simple.to_csv(csv_simple_path, index=False)
print(f"✅ Saved simplified CSV: {csv_simple_path}")

# ========== 生成LaTeX表格 ==========
tex_path = os.path.join(results_dir, "scenario_power_verification.tex")
with open(tex_path, "w", encoding="utf-8") as f:
    f.write("\\begin{table}[htbp]\n")
    f.write("\\centering\n")
    f.write("\\caption{Component Power Consumption Verification by Scenario}\n")
    f.write("\\label{tab:power_verification}\n")
    f.write("\\resizebox{\\textwidth}{!}{\n")
    f.write("\\begin{tabular}{lcccccccccccc}\n")
    f.write("\\hline\n")
    f.write(
        "Scenario & \\multicolumn{2}{c}{Screen} & \\multicolumn{2}{c}{SoC} & \\multicolumn{2}{c}{Radio} & \\multicolumn{2}{c}{GPS} & \\multicolumn{2}{c}{Baseline} & Total \\\\\n"
    )
    f.write(
        "         & Param & Power & Param & Power & Param & Power & Param & Power & Param & Power & Power \\\\\n"
    )
    f.write(
        "         &       & (mW)  &       & (mW)  &       & (mW)  &       & (mW)  &       & (mW)  & (mW) \\\\\n"
    )
    f.write("\\hline\n")

    for name, params in scenarios.items():
        B_pct, eta = params["screen"]
        B = B_pct / 100.0
        P_screen = eta * (GAMMA_LS_A * B + LAMBDA_S_A) * 1000

        soc_pct = params["soc"]
        soc_load = min((soc_pct / 100.0) + 0.1, 1.0)
        P_soc = soc_load * P_SOC_MAX

        radio_pct = params["radio"]
        P_radio = (radio_pct / 100.0) * P_RADIO_MAX

        gps_pct = params["gps"]
        P_gps = (gps_pct / 100.0) * P_GPS

        P_baseline = P_BASELINE
        total = P_screen + P_soc + P_radio + P_gps + P_baseline

        screen_param = f"{B_pct}\\%, $\\eta$={eta}"
        soc_param = f"{soc_pct}\\%+10\\%bg"

        f.write(
            f"{name} & {screen_param} & {P_screen:.1f} & {soc_param} & {P_soc:.0f} & {radio_pct}\\% & {P_radio:.1f} & {gps_pct}\\% & {P_gps:.0f} & 100\\% & {P_baseline:.0f} & {total:.1f} \\\\\n"
        )

    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("}\n")
    f.write("\\end{table}\n")

print(f"✅ Saved LaTeX table: {tex_path}")

# ========== 生成Markdown表格 ==========
md_path = os.path.join(results_dir, "scenario_power_verification.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write("# 各场景功耗验证结果\n\n")
    f.write("## 功率模型参数\n\n")
    f.write(
        "- **Screen**: P_screen = η(0.4991×B + 0.1113) W, where B∈[0,1], η∈{1.0, 1.3}\n"
    )
    f.write(f"- **SoC**: P_SoC = ξ₁ × {P_SOC_MAX}mW (2.04W max)\n")
    f.write(f"- **Radio**: P_radio = ξ₂ × {P_RADIO_MAX}mW (1.173W max)\n")
    f.write(f"- **GPS**: P_GPS = ξ₃ × {P_GPS}mW\n")
    f.write(f"- **Baseline**: P_baseline = {P_BASELINE}mW (always on)\n\n")

    f.write("## 各场景功耗分解\n\n")
    f.write("| 场景 | Screen | SoC | Radio | GPS | Baseline | 总功耗 |\n")
    f.write("|------|--------|-----|-------|-----|----------|--------|\n")

    for name, params in scenarios.items():
        B_pct, eta = params["screen"]
        B = B_pct / 100.0
        P_screen = eta * (GAMMA_LS_A * B + LAMBDA_S_A) * 1000

        soc_pct = params["soc"]
        soc_load = min((soc_pct / 100.0) + 0.1, 1.0)
        P_soc = soc_load * P_SOC_MAX

        radio_pct = params["radio"]
        P_radio = (radio_pct / 100.0) * P_RADIO_MAX

        gps_pct = params["gps"]
        P_gps = (gps_pct / 100.0) * P_GPS

        P_baseline = P_BASELINE
        total = P_screen + P_soc + P_radio + P_gps + P_baseline

        f.write(
            f"| **{name}** | {B_pct}%, η={eta}<br>→ {P_screen:.1f}mW | {soc_pct}%+10%bg<br>→ {P_soc:.0f}mW | {radio_pct}%<br>→ {P_radio:.1f}mW | {gps_pct}%<br>→ {P_gps:.0f}mW | 100%<br>→ {P_baseline:.0f}mW | **{total:.1f}mW** |\n"
        )

    f.write("\n")

print(f"✅ Saved Markdown table: {md_path}")

# ========== 打印美化表格 ==========
print("\n" + "=" * 140)
print("各场景功耗验证结果")
print("=" * 140)
print(
    f"{'场景':<12} | {'Screen':<25} | {'SoC':<20} | {'Radio':<15} | {'GPS':<15} | {'Baseline':<15} | {'总功耗':<12}"
)
print("-" * 140)

for name, params in scenarios.items():
    B_pct, eta = params["screen"]
    B = B_pct / 100.0
    P_screen = eta * (GAMMA_LS_A * B + LAMBDA_S_A) * 1000

    soc_pct = params["soc"]
    soc_load = min((soc_pct / 100.0) + 0.1, 1.0)
    P_soc = soc_load * P_SOC_MAX

    radio_pct = params["radio"]
    P_radio = (radio_pct / 100.0) * P_RADIO_MAX

    gps_pct = params["gps"]
    P_gps = (gps_pct / 100.0) * P_GPS

    P_baseline = P_BASELINE
    total = P_screen + P_soc + P_radio + P_gps + P_baseline

    screen_str = f"{B_pct}%, η={eta}→{P_screen:.1f}mW"
    soc_str = f"{soc_pct}%+10%bg→{P_soc:.0f}mW"
    radio_str = f"{radio_pct}%→{P_radio:.1f}mW"
    gps_str = f"{gps_pct}%→{P_gps:.0f}mW"
    baseline_str = f"100%→{P_baseline:.0f}mW"
    total_str = f"{total:.1f}mW"

    print(
        f"{name:<12} | {screen_str:<25} | {soc_str:<20} | {radio_str:<15} | {gps_str:<15} | {baseline_str:<15} | {total_str:<12}"
    )

print("=" * 140)
print("\n✅ 所有表格文件已生成!")
print(f"   - CSV (详细): {csv_path}")
print(f"   - CSV (简化): {csv_simple_path}")
print(f"   - LaTeX: {tex_path}")
print(f"   - Markdown: {md_path}")
