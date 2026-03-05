"""
展示Video场景的TTE计算过程(使用正确的数值积分方法)
"""

import sys

sys.path.append(".")
from aging_model import BatteryModel
from main_model import MainBatteryModel

# Video场景参数
scenario = {
    "screen": (80, 1.0),  # 80%亮度, η=1.0
    "soc": 20,  # 20% SoC利用率
    "radio": 90,  # 90% Radio
    "gps": 10,  # 10% GPS
}

# 构建输入
screen_B = scenario["screen"][0] / 100.0  # 0.8
screen_eta = scenario["screen"][1]  # 1.0
cpu_load = min((scenario["soc"] / 100.0) + 0.1, 1.0)  # 0.2 + 0.1 = 0.3
radio_activity = scenario["radio"] / 100.0  # 0.9
gps_on = scenario["gps"] > 0  # True

inputs = {
    "brightness_B": screen_B,
    "brightness_eta": screen_eta,
    "cpu_load": cpu_load,
    "network_mode": "4g",
    "network_activity": radio_activity,
    "gps": gps_on,
    "temperature": 298.15,
}

# 计算功耗
power_model = BatteryModel(cycle_count=0, calendar_time_hours=0)
power_mw = power_model.get_power_consumption(0.0, inputs)
power_w = power_mw / 1000.0

print("=== Video场景TTE计算过程 ===")
print()
print("1. 场景参数:")
print(f'   Screen: {scenario["screen"][0]}% 亮度, η={scenario["screen"][1]}')
print(f'   SoC: {scenario["soc"]}% + 10% 后台 = {int(cpu_load*100)}%')
print(f'   Radio: {scenario["radio"]}%')
print(f'   GPS: {scenario["gps"]}%')
print()
print("2. 功耗分解:")
print(
    f"   P_screen = η(0.4991*B + 0.1113) = {screen_eta}*(0.4991*{screen_B} + 0.1113) = {screen_eta*(0.4991*screen_B + 0.1113):.4f}W = {screen_eta*(0.4991*screen_B + 0.1113)*1000:.1f}mW"
)
print(f"   P_SoC = {cpu_load:.2f} * 2040mW = {cpu_load*2040:.1f}mW")
print(f"   P_Radio = {radio_activity:.2f} * 1173mW = {radio_activity*1173:.1f}mW")
print(f"   P_GPS = {0.3 if gps_on else 0}W = {300 if gps_on else 0}mW")
print(f"   P_baseline = 120mW")
print(f"   P_total = {power_w:.4f}W = {power_mw:.1f}mW")
print()
print("3. 电池参数:")
print(f"   Q_nom = 5.0Ah (Samsung Galaxy S25 Ultra)")
print(f"   V_nom = 3.7V")
print(f"   E_battery = 5.0Ah * 3.7V = 18.5Wh")
print()
print("4. TTE数值积分求解:")
print("   求解耦合ODE系统:")
print("   {")
print("     V_terminal(SOC,I) = E₀ - K/SOC·I - R·I + A·exp(-BQ₀(1-SOC))")
print("     I = -Q₀·dSOC/dt")
print(f"     P_load = V_terminal·I = {power_w:.4f}W (恒定)")
print("     SOC(0) = 1.0 (100%)")
print("   }")
print("   TTE = inf{t | SOC(t) ≤ 0.05 或 V_terminal < 2.5V}")
print()
print("   使用Euler前向积分, dt=1.0s:")
print("   每步: SOC(t+dt) = SOC(t) + dt·(-I/Q₀)")
print("         其中 I = P_load / V_terminal(SOC(t), I)")
print()

# 使用Euler积分求解
main_model = MainBatteryModel(Q0=5.0)
result = main_model.predict_discharge(
    P_load=power_w, temp_k=298.15, soc_initial=1.0, dt=1.0
)

print("5. 数值积分结果:")
print(
    f'   TTE = {result["discharge_time"]:.2f}小时 = {result["discharge_time"]*60:.1f}分钟'
)
print(f'   最终SOC = {result["soc"][-1]*100:.1f}%')
print(f'   最终电压 = {result["voltage"][-1]:.2f}V')
print()
print(
    "⚠️  注意: TTE ≠ 18.5Wh / {:.4f}W = {:.2f}h (这是错误的!)".format(
        power_w, 18.5 / power_w
    )
)
print("      因为V_terminal随SOC非线性变化,简单除法假设电压恒定!")
print("      正确方法必须用数值积分求解耦合ODE系统。")
print()
print(f"对比: 错误的简单除法 = {18.5/power_w:.2f}h")
print(f'      正确的数值积分 = {result["discharge_time"]:.2f}h')
print(
    f'      差异 = {abs(18.5/power_w - result["discharge_time"]):.2f}h = {abs(18.5/power_w - result["discharge_time"])/result["discharge_time"]*100:.1f}%'
)
