"""
验证各场景的功耗计算是否与表格参数一致
"""

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
P_SOC_MAX = 2040.0  # mW
P_RADIO_MAX = 1173.0  # mW
P_GPS = 300.0  # mW
P_BASELINE = 120.0  # mW
GAMMA_LS_A = 0.4991
LAMBDA_S_A = 0.1113

print("=" * 120)
print("场景功耗验证 (单位: mW)")
print("=" * 120)
print(
    f"{'场景':<12} | {'Screen参数':<15} | {'P_screen':<10} | {'SoC%':<8} | {'P_SoC':<10} | {'Radio%':<8} | {'P_radio':<10} | {'GPS%':<8} | {'P_GPS':<10} | {'Baseline':<10} | {'总功耗':<10}"
)
print("-" * 120)

for name, params in scenarios.items():
    # Screen
    B_pct, eta = params["screen"]
    B = B_pct / 100.0  # 转换为 [0,1]
    P_screen_W = eta * (GAMMA_LS_A * B + LAMBDA_S_A)  # W
    P_screen = P_screen_W * 1000  # mW

    # SoC (包含10%背景负载)
    soc_pct = params["soc"]
    soc_load = (soc_pct / 100.0) + 0.1  # 加10%背景
    soc_load = min(soc_load, 1.0)  # 上限1.0
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

    screen_str = f"({B_pct}%, {eta})"
    print(
        f"{name:<12} | {screen_str:<15} | {P_screen:>9.2f} | {soc_pct:>6}% | {P_soc:>9.2f} | {radio_pct:>6}% | {P_radio:>9.2f} | {gps_pct:>6}% | {P_gps:>9.2f} | {P_baseline_calc:>9.2f} | {total:>9.2f}"
    )

print("=" * 120)
print("\n表格中的百分比含义:")
print("- Screen (B, η): B是亮度百分比 (0-100), η是刷新率倍数 (1.0或1.3)")
print("- SoC (ξ₁): CPU负载百分比 (0-95) + 10%背景负载")
print("- Radio (ξ₂): 无线电活动百分比 (10-95)")
print("- GPS (ξ₃): GPS使用百分比 (10-100)")
print("- Baseline (x₄): 基线负载百分比 (固定100%)")
print("\n功率公式:")
print(
    f"- P_screen = η(γ_ls·A_screen·B + λ_s·A_screen) = η({GAMMA_LS_A}B + {LAMBDA_S_A}) W"
)
print(f"- P_SoC = ξ₁ × P_SoC^max = ξ₁ × {P_SOC_MAX}mW (2.04W)")
print(f"- P_radio = ξ₂ × P_radio^max = ξ₂ × {P_RADIO_MAX}mW")
print(f"- P_GPS = ξ₃ × {P_GPS}mW")
print(f"- P_baseline = x₄ × {P_BASELINE}mW")
