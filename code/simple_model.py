import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ==================== 设置中文字体 ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.style.use("default")

# ==================== 电池参数设置 ====================
# 电池基本参数
C_battery = 3.0  # 电池容量 (Ah)，典型18650电池 (3000mAh)
SOC_initial = 1.0  # 初始SOC (100%)

# 定义两种模式的参数
PARAMS = {
    '使用中': {
        'I_load': 0.8,  # 恒流放电电流 (A)
        'R0': 0.05,     # 欧姆内阻 (Ω)
        'R1': 0.015, 'C1': 3000,   # RC支路1
        'R2': 0.025, 'C2': 12000,  # RC支路2
        'R3': 0.035, 'C3': 25000,  # RC支路3
    },
    '待机中': {
        'I_load': 0.04,  # 恒流放电电流 (A) - 40mA待机电流
        'R0': 0.08,      # 欧姆内阻 (Ω)
        'R1': 0.025, 'C1': 5000,   # RC支路1
        'R2': 0.040, 'C2': 18000,  # RC支路2
        'R3': 0.050, 'C3': 35000,  # RC支路3
    }
}

# 截止条件
V_cutoff = 2.8  # 电压截止点 (V)，低于此值视为耗尽
SOC_min = 0.05  # SOC截止点，低于5%视为耗尽

# ==================== OCV-SOC关系函数 ====================
def ocv_from_soc(SOC):
    """
    根据SOC计算开路电压(OCV)
    使用基于电化学原理的Nernst模型 + 指数修正项：
    
    V_oc = E0 + (RT/nF) * ln(SOC/(1-SOC)) + K*exp(-α*SOC)
    
    物理意义：
    - E0: 标准电极电位 (参考电压)
    - Nernst项 (RT/nF)*ln(SOC/(1-SOC)): 电化学热力学平衡，
      描述活性锂离子浓度变化对电压的影响
    - 指数项 K*exp(-α*SOC): 低SOC时的浓差极化和质量传输限制
    
    参数针对典型18650锂离子电池(NMC材料)标定
    """
    SOC = np.clip(SOC, 0.001, 0.999)  # 避免对数奇异点
    
    # 电化学参数
    E0 = 3.6        # 标准电极电位 (V)
    R = 8.314       # 气体常数 (J/(mol·K))
    T = 298.15      # 温度 (K)，约25°C
    n = 1           # 转移电子数
    F = 96485       # 法拉第常数 (C/mol)
    
    # 指数修正项参数
    K = -0.2        # 指数项系数，反映低SOC极化
    alpha = 6.0     # 指数衰减常数
    
    # Nernst项: 热力学平衡电压
    nernst_term = (R * T) / (n * F) * np.log(SOC / (1 - SOC))
    
    # 指数修正项: 描述低SOC时的浓差极化
    exp_term = K * np.exp(-alpha * SOC)
    
    V_oc = E0 + nernst_term + exp_term
    
    return V_oc

# ==================== 系统动力学方程 ====================
def battery_model(t, state, I_load, C_battery, R0, R1, C1, R2, C2, R3, C3):
    """
    电池三RC模型的状态方程
    state: [Vc1, Vc2, Vc3, SOC]
    """
    Vc1, Vc2, Vc3, SOC = state
    
    # 确保SOC在合理范围内
    SOC = np.clip(SOC, 0, 1)
    
    # OCV-SOC关系
    V_oc = ocv_from_soc(SOC)
    
    # 状态导数 - 核心方程
    dVc1_dt = (I_load - Vc1/R1) / C1
    dVc2_dt = (I_load - Vc2/R2) / C2
    dVc3_dt = (I_load - Vc3/R3) / C3
    
    # SOC变化率 (单位: 1/s)
    # C_battery * 3600 将Ah转换为As (库仑)
    dSOC_dt = -I_load / (C_battery * 3600)
    
    return [dVc1_dt, dVc2_dt, dVc3_dt, dSOC_dt]

# ==================== 事件函数：检测电池耗尽 ====================
def depletion_event(t, state, I_load, C_battery, R0, R1, C1, R2, C2, R3, C3):
    """检测电池是否耗尽的事件函数"""
    Vc1, Vc2, Vc3, SOC = state
    
    # 确保SOC在合理范围内
    SOC = np.clip(SOC, 0, 1)
    
    # 计算OCV和端电压
    V_oc = ocv_from_soc(SOC)
    V_terminal = V_oc - I_load * R0 - Vc1 - Vc2 - Vc3
    
    # 当电压低于截止电压或SOC低于最小值时停止
    return min(V_terminal - V_cutoff, SOC - SOC_min)

depletion_event.terminal = True  # 事件发生时停止积分
depletion_event.direction = -1   # 仅当值从正变负时触发

# ==================== 仿真函数 ====================
def run_simulation(mode, params):
    """运行单个模式的仿真"""
    I_load = params['I_load']
    R0, R1, C1, R2, C2, R3, C3 = params['R0'], params['R1'], params['C1'], params['R2'], params['C2'], params['R3'], params['C3']
    
    # 初始条件: 电容电压为0，SOC为1
    initial_state = [0.0, 0.0, 0.0, SOC_initial]
    
    # 最大仿真时间 (秒)
    t_max = C_battery * 3600 / I_load
    t_span = (0, t_max * 1.5)
    
    print(f"\n=== {mode}模式参数 ===")
    print(f"电池容量: {C_battery} Ah")
    print(f"放电电流: {I_load} A")
    print(f"理论最大放电时间: {t_max/3600:.2f} 小时 ({t_max:.0f} 秒)")
    print(f"欧姆内阻 R0: {R0} Ω")
    print(f"RC1: R1={R1} Ω, C1={C1} F, 时间常数 τ1={R1*C1:.0f} s")
    print(f"RC2: R2={R2} Ω, C2={C2} F, 时间常数 τ2={R2*C2:.0f} s")
    print(f"RC3: R3={R3} Ω, C3={C3} F, 时间常数 τ3={R3*C3:.0f} s")
    
    # 求解ODE
    solution = solve_ivp(
        battery_model,
        t_span,
        initial_state,
        args=(I_load, C_battery, R0, R1, C1, R2, C2, R3, C3),
        events=depletion_event,
        max_step=5,
        rtol=1e-8,
        atol=1e-10,
        dense_output=True
    )
    
    # 提取结果
    t = solution.t
    Vc1, Vc2, Vc3, SOC = solution.y
    
    # 计算端电压和OCV
    V_terminal = np.zeros_like(t)
    V_oc_array = np.zeros_like(t)
    
    for i in range(len(t)):
        V_oc = ocv_from_soc(SOC[i])
        V_oc_array[i] = V_oc
        V_terminal[i] = V_oc - I_load * R0 - Vc1[i] - Vc2[i] - Vc3[i]
    
    DoD = 1 - SOC
    
    # 找到耗尽时间点
    if len(solution.t_events[0]) > 0:
        depletion_time = solution.t_events[0][0]
        depletion_idx = np.argmin(np.abs(t - depletion_time))
    else:
        depletion_time = t[-1]
        depletion_idx = -1
    
    return {
        't': t, 'Vc1': Vc1, 'Vc2': Vc2, 'Vc3': Vc3, 'SOC': SOC,
        'V_terminal': V_terminal, 'V_oc_array': V_oc_array, 'DoD': DoD,
        'depletion_time': depletion_time, 'depletion_idx': depletion_idx,
        't_max': t_max, 'solution': solution, 'params': params
    }

def plot_figure1_3d_surface():
    """
    Figure 1: 3D Surface + 2D Slice Side by Side
    Left: Voltage vs SOC vs Current 3D Surface
    Right: Voltage-SOC curves at different currents
    """
    print("Generating Figure 1: 3D Surface + 2D Slice...")
    
    # 准备3D数据网格
    currents = np.linspace(0.04, 1.5, 40)  # 电流范围
    soc_values = np.linspace(1.0, 0.05, 80)  # SOC范围
    SOC_mesh, I_mesh = np.meshgrid(soc_values, currents)
    V_mesh = np.zeros_like(SOC_mesh)
    
    # 计算每个点的端电压（稳态近似）
    rows, cols = SOC_mesh.shape
    for r in range(rows):
        for c in range(cols):
            soc_val = SOC_mesh[r, c]
            i_val = I_mesh[r, c]
            V_oc = ocv_from_soc(soc_val)
            # 使用"使用中"模式的参数计算稳态电压
            R0 = PARAMS['使用中']['R0']
            R1, R2, R3 = PARAMS['使用中']['R1'], PARAMS['使用中']['R2'], PARAMS['使用中']['R3']
            R_total = R0 + R1 + R2 + R3  # 稳态时总内阻
            V_mesh[r, c] = V_oc - i_val * R_total
    
    V_mesh[V_mesh < 2.0] = 2.0  # 限制最低电压
    
    # 2D切片数据
    soc_slice = np.linspace(1.0, 0.05, 200)
    
    # 创建图形
    fig = plt.figure(figsize=(16, 7.2), constrained_layout=False)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.15, 0.85],
                           left=0.05, right=0.985, bottom=0.12, top=0.86, wspace=0.08)
    
    # --- 左图：3D曲面 ---
    ax1 = fig.add_subplot(gs[0], projection='3d')
    ax1.plot_surface(SOC_mesh * 100, I_mesh, V_mesh,
                     cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.9)
    
    ax1.set_xlabel('SOC (%)', fontsize=10, labelpad=6)
    ax1.set_ylabel('Current (A)', fontsize=10, labelpad=6)
    ax1.set_zlabel('Voltage (V)', fontsize=10, labelpad=6)
    ax1.set_title('Global Battery Characteristics\nV = f(SOC, I)', fontsize=13, fontweight='bold', pad=6)
    ax1.set_xlim(100, 0)
    ax1.view_init(elev=30, azim=135)
    ax1.tick_params(labelsize=9)
    
    # --- 右图：不同电流下的放电曲线 ---
    ax2 = fig.add_subplot(gs[1])
    
    # 测试多种电流
    test_currents = [0.04, 0.2, 0.4, 0.8, 1.2]  # A
    colors = cm.plasma(np.linspace(0, 0.9, len(test_currents)))
    
    for idx, i_val in enumerate(test_currents):
        v_curve = []
        R0 = PARAMS['使用中']['R0']
        R1, R2, R3 = PARAMS['使用中']['R1'], PARAMS['使用中']['R2'], PARAMS['使用中']['R3']
        R_total = R0 + R1 + R2 + R3
        for s in soc_slice:
            V_oc = ocv_from_soc(s)
            v_curve.append(V_oc - i_val * R_total)
        ax2.plot(soc_slice * 100, v_curve, linewidth=2.4, color=colors[idx], label=f'{i_val}A')
    
    ax2.axhline(y=V_cutoff, color='r', linestyle='--', alpha=0.5, label=f'Cutoff ({V_cutoff}V)')
    ax2.set_xlabel('SOC (%)', fontsize=10)
    ax2.set_ylabel('Terminal Voltage (V)', fontsize=10)
    ax2.set_title('Discharge Curves at Different Load Currents\n(2D Cross-section of 3D Surface)', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(title='Load Current', fontsize=9, frameon=True, loc='lower left')
    ax2.set_xlim(100, 0)
    ax2.set_ylim(2.5, 3.8)
    ax2.tick_params(labelsize=9)
    ax2.set_box_aspect(1)
    
    # 对齐2D面板与3D面板
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    ax2.set_position([pos2.x0, pos1.y0, pos2.width, pos1.height])
    ax2.set_anchor('W')
    
    fig.suptitle('Figure 1: Global Battery Characteristics (3D) vs Specific Load Cases (2D)',
                 fontsize=14, fontweight='bold', y=0.97)
    
    plt.savefig('battery_figure1_3d.png', dpi=300, bbox_inches='tight')
    print("Saved 'battery_figure1_3d.png'")
    plt.show()

def plot_figure2_time_discharge(results_active, results_standby):
    """
    Figure 2: Time-based Discharge Curves with dual x-axes
    Left: Voltage vs Time (bottom axis: Active, top axis: Standby)
    Right: SOC vs Time (bottom axis: Active, top axis: Standby)
    """
    print("Generating Figure 2: Time-based Discharge Curves...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Data for two modes with plasma colormap
    colors = cm.plasma(np.linspace(0, 0.7, 2))
    
    # Extract data
    t_active = results_active['t']
    t_standby = results_standby['t']
    V_active = results_active['V_terminal']
    V_standby = results_standby['V_terminal']
    SOC_active = results_active['SOC']
    SOC_standby = results_standby['SOC']
    
    idx_active = results_active['depletion_idx']
    idx_standby = results_standby['depletion_idx']
    idx_end_active = idx_active + 1 if idx_active >= 0 else len(t_active)
    idx_end_standby = idx_standby + 1 if idx_standby >= 0 else len(t_standby)
    
    # --- Left: Voltage vs Time with dual x-axes ---
    ax1 = axes[0]
    # Bottom axis: Active mode
    line1, = ax1.plot(t_active[:idx_end_active] / 3600, V_active[:idx_end_active], 
             linewidth=2.4, color=colors[0], label=f'Active ({results_active["params"]["I_load"]}A)')
    ax1.axhline(y=V_cutoff, color='r', linestyle='--', alpha=0.5, label=f'Cutoff ({V_cutoff}V)')
    ax1.set_xlabel(f'Time - Active Mode (hours)', fontsize=11, color=colors[0])
    ax1.set_ylabel('Terminal Voltage (V)', fontsize=11)
    ax1.tick_params(axis='x', labelcolor=colors[0], labelsize=9)
    ax1.tick_params(axis='y', labelsize=9)
    ax1.set_xlim(0, 7)
    ax1.set_ylim(2.5, 3.8)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Top axis: Standby mode
    ax1_top = ax1.twiny()
    line2, = ax1_top.plot(t_standby[:idx_end_standby] / 3600, V_standby[:idx_end_standby], 
             linewidth=2.4, color=colors[1], label=f'Standby ({results_standby["params"]["I_load"]}A)')
    ax1_top.set_xlabel(f'Time - Standby Mode (hours)', fontsize=11, color=colors[1])
    ax1_top.tick_params(axis='x', labelcolor=colors[1], labelsize=9)
    ax1_top.set_xlim(0, t_standby[idx_end_standby-1] / 3600)
    
    ax1.set_title('Discharge Voltage vs Time\n(Dual Time Axes)', fontsize=12, fontweight='bold', pad=35)
    ax1.legend(handles=[line1, line2], fontsize=9, frameon=True, loc='lower left')
    
    # --- Right: SOC vs Time with dual x-axes ---
    ax2 = axes[1]
    # Bottom axis: Active mode
    line3, = ax2.plot(t_active[:idx_end_active] / 3600, SOC_active[:idx_end_active] * 100, 
             linewidth=2.4, color=colors[0], label=f'Active ({results_active["params"]["I_load"]}A)')
    ax2.axhline(y=SOC_min * 100, color='r', linestyle='--', alpha=0.5, label=f'SOC Min ({SOC_min*100:.0f}%)')
    ax2.set_xlabel(f'Time - Active Mode (hours)', fontsize=11, color=colors[0])
    ax2.set_ylabel('SOC (%)', fontsize=11)
    ax2.tick_params(axis='x', labelcolor=colors[0], labelsize=9)
    ax2.tick_params(axis='y', labelsize=9)
    ax2.set_xlim(0, 7)
    ax2.set_ylim(0, 105)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # Top axis: Standby mode
    ax2_top = ax2.twiny()
    line4, = ax2_top.plot(t_standby[:idx_end_standby] / 3600, SOC_standby[:idx_end_standby] * 100, 
             linewidth=2.4, color=colors[1], label=f'Standby ({results_standby["params"]["I_load"]}A)')
    ax2_top.set_xlabel(f'Time - Standby Mode (hours)', fontsize=11, color=colors[1])
    ax2_top.tick_params(axis='x', labelcolor=colors[1], labelsize=9)
    ax2_top.set_xlim(0, t_standby[idx_end_standby-1] / 3600)
    
    ax2.set_title('SOC vs Time\n(Dual Time Axes)', fontsize=12, fontweight='bold', pad=35)
    ax2.legend(handles=[line3, line4], fontsize=9, frameon=True, loc='upper right')
    
    fig.suptitle('Figure 2: Battery Discharge Characteristics Over Time',
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig('battery_figure2_time.png', dpi=300, bbox_inches='tight')
    print("Saved 'battery_figure2_time.png'")
    plt.show()

def print_summary(mode, results):
    """打印仿真结果总结"""
    t = results['t']
    Vc1, Vc2, Vc3, SOC = results['Vc1'], results['Vc2'], results['Vc3'], results['SOC']
    V_terminal, V_oc_array, DoD = results['V_terminal'], results['V_oc_array'], results['DoD']
    depletion_time, depletion_idx = results['depletion_time'], results['depletion_idx']
    t_max = results['t_max']
    params = results['params']
    I_load = params['I_load']
    R0, R1, C1, R2, C2, R3, C3 = params['R0'], params['R1'], params['C1'], params['R2'], params['C2'], params['R3'], params['C3']
    
    print(f"\n{'='*50}")
    print(f"=== {mode}模式 仿真结果总结 ===")
    print(f"{'='*50}")
    
    if depletion_idx >= 0:
        print(f"实际耗尽时间: {depletion_time:.1f} 秒 ({depletion_time/3600:.2f} 小时)")
        print(f"电池效率: {depletion_time/t_max*100:.1f}% (实际/理论放电时间比)")
        
        print(f"\n初始时刻 (t=0):")
        print(f"  SOC: {SOC[0]*100:.1f}%")
        print(f"  OCV: {ocv_from_soc(1.0):.3f} V")
        print(f"  端电压: {V_terminal[0]:.3f} V")
        
        print(f"\n耗尽时刻 (t={depletion_time:.1f}s):")
        print(f"  SOC: {SOC[depletion_idx]*100:.2f}%")
        print(f"  DoD: {DoD[depletion_idx]*100:.2f}%")
        print(f"  OCV: {V_oc_array[depletion_idx]:.3f} V")
        print(f"  端电压: {V_terminal[depletion_idx]:.3f} V")
        
        print(f"\n电压降分析:")
        total_drop = V_terminal[0] - V_terminal[depletion_idx]
        print(f"  总电压降: {total_drop:.3f} V")
        print(f"    - OCV下降: {ocv_from_soc(1.0) - V_oc_array[depletion_idx]:.3f} V")
        print(f"    - 欧姆压降 (恒定): {I_load*R0:.3f} V")
        print(f"    - RC1极化压降: {Vc1[depletion_idx]:.3f} V")
        print(f"    - RC2极化压降: {Vc2[depletion_idx]:.3f} V")
        print(f"    - RC3极化压降: {Vc3[depletion_idx]:.3f} V")
    else:
        print("未达到耗尽条件")
        print(f"仿真结束时: t={t[-1]/3600:.2f}h, SOC={SOC[-1]*100:.1f}%, V={V_terminal[-1]:.3f}V")

# ==================== 主程序：运行两种模式的仿真 ====================
print("=" * 60)
print("电池三RC模型仿真 - 使用中 vs 待机中 对比")
print("=" * 60)

# 运行两种模式的仿真
results_active = run_simulation('使用中', PARAMS['使用中'])
results_standby = run_simulation('待机中', PARAMS['待机中'])

# 绘制 Figure 1: 3D曲面 + 2D切片
plot_figure1_3d_surface()

# 绘制 Figure 2: 时间-放电曲线
plot_figure2_time_discharge(results_active, results_standby)

# 打印结果总结
print_summary('使用中', results_active)
print_summary('待机中', results_standby)

# 对比总结
print(f"\n{'='*60}")
print("=== 两种模式对比 ===")
print(f"{'='*60}")
print(f"使用中模式: 放电电流 {PARAMS['使用中']['I_load']}A, 续航时间 {results_active['depletion_time']/3600:.2f} 小时")
print(f"待机中模式: 放电电流 {PARAMS['待机中']['I_load']}A, 续航时间 {results_standby['depletion_time']/3600:.2f} 小时")
print(f"待机续航倍数: {results_standby['depletion_time']/results_active['depletion_time']:.1f}x")