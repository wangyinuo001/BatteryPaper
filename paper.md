# BatteryPaper 项目总结（ECM 改写版）

## 目标与仓库信息

- 目标：将 MCM/ICM 2026 竞赛论文（Team 2600101, Problem A）改写为 *Energy Conversion and Management (ECM)* 期刊论文。
- 仓库：`wangyinuo001/BatteryPaper`
- 分支状态：当前分支 `master`，默认分支 `main`

## 一、核心模型架构

统一连续时间 ODE 系统，由 4 个子模型耦合：

1. **Shepherd 电压子模型**（`main_model.py`）

	$$
	V = E_0 - \frac{K}{SOC} I - R(T,N) I + A\exp\left[-B Q_0(1-SOC)\right],\quad
	I = -Q_0\frac{dSOC}{dt},\quad P = VI
	$$

2. **组件功耗分解**（`aging_model.py`）

	$$
	P = P_{screen} + P_{SoC} + P_{radio} + P_{GPS} + P_{base}
	$$

3. **温度耦合**（`main_model.py`）

	$$
	Q_0(T)=\frac{Q_0'}{1+\exp\left[\alpha_\theta\left(\frac{1}{T}-\frac{1}{T_m}\right)\right]},\qquad
	R(T)=R_0\exp\left[\frac{E_r}{k_B}\left(\frac{1}{T}-\frac{1}{T_0}\right)\right]
	$$

4. **老化动力学**（`aging_model.py`）

	$$
	R(N)=R_0^{nom}(1+\beta_R N^\gamma),\qquad
	Q(N)=Q_0^{nom}(1-\alpha_Q N^\gamma)
	$$

## 二、关键参数（已与 XJTU 拟合对齐）

### 2.1 Shepherd 电压参数（XJTU Batch-1）

| 参数 | 值 | 含义 |
|---|---:|---|
| $E_0$ | 3.3843 V | 满充开路电压 |
| $K$ | 0.0175 Ω | 极化系数 |
| $A$ | 0.8096 V | 初始压降幅值 |
| $B$ | 1.0062 (Ah)$^{-1}$ | 初始压降衰减常数 |
| $R_0$ | 0.035 Ω | 内阻 |
| $Q_0$ | 5.0 Ah | 智能手机电池标称容量 |

### 2.2 老化参数（XJTU Batch-1，8 电池平均）

| 参数 | 值 | 含义 |
|---|---:|---|
| $\alpha_Q$ | 0.000256 | 容量衰减系数 |
| $\beta_R$ | 0.001229 | 电阻增长系数 |
| $\gamma$ | 1.085 | 幂律指数 |
| $\kappa_{cal}$ | $1\times10^{-5}$ | 日历老化系数 |
| $R_0^{nom}$ | 0.0584 Ω | XJTU 标称内阻 |
| $Q_{0,ref}$ | 2.006 Ah | XJTU 参考容量 |

### 2.3 温度参数

| 参数 | 值 | 含义 |
|---|---:|---|
| $\alpha_\theta$ | 5435.3 K | 容量-温度 logistic 系数 |
| $T_m$ | 255.27 K | 参考温度 |
| $E_r$ | 0.2455 eV | 电阻 Arrhenius 活化能 |

### 2.4 功耗组件参数

| 组件 | 最大功率 | 来源 |
|---|---:|---|
| $P_{SoC}^{max}$ | 2040 mW | Carroll & Heiser (2010) |
| $P_{radio}^{max}$ (4G) | 1173 mW | Lauridsen et al. (2014) |
| $P_{GPS}$ | 300 mW | 测量值 |
| $P_{base}$ | 120 mW | 常数 |
| Screen: $\gamma_{ls}A_s$ | 0.4991 W | Lu et al. (2008) |
| Screen: $\lambda_sA_s$ | 0.1113 W | Lu et al. (2008) |

## 三、核心实验结果

### 3.1 基线对比（`run_all_baselines.py`）

| 模型 | RMSE (mV) | MAE (mV) | MaxErr (mV) | Rel.Err (%) |
|---|---:|---:|---:|---:|
| NBM (Nernst) | 72.70 ± 1.23 | 49.99 ± 0.58 | 719.5 | 1.41 |
| Rint | 39.79 ± 0.66 | 16.88 ± 0.28 | 555.7 | 0.50 |
| Thévenin-1RC | 39.75 ± 0.51 | 17.02 ± 0.24 | 553.7 | 0.51 |
| Thévenin-2RC | 39.81 ± 0.44 | 17.33 ± 0.50 | 551.7 | 0.51 |
| Shepherd（本文） | 17.85 ± 0.44 | 9.75 ± 0.72 | 294.1 | 0.28 |

结论：Shepherd 相比最佳 ECM 基线 RMSE 降低约 55%，相比 NBM 降低约 75.5%。

### 3.2 跨批次泛化（`cross_batch_validation.py`）

| 评估 | N | RMSE (mV) |
|---|---:|---:|
| Same-batch (B1→B1) | 8 | 21.58 ± 1.78 |
| Cross-batch (B1→B2) | 15 | 20.36 ± 3.68 |

结论：2C→3C 跨 C-rate 泛化无退化（gap = −1.22 mV，95% CI 跨零）。

### 3.3 Bootstrap 置信区间（`bootstrap_ci.py`）

| 指标 | 估计值 | 95% CI |
|---|---:|---|
| Shepherd RMSE | 21.58 mV | [20.0, 22.4] |
| TTE Standby | 19.67 h | [18.9, 20.4] |
| TTE Navigation | 9.03 h | [8.7, 9.4] |
| TTE Gaming | 3.79 h | [3.7, 3.9] |
| TTE Video | 6.37 h | [6.1, 6.6] |
| TTE Reading | 9.65 h | [9.3, 10.0] |

### 3.4 场景 TTE（`analyze_all_scenarios_tte_loss.py`）

| 场景 | 总功率 (W) | TTE (h) |
|---|---:|---:|
| Standby | 0.741 | 22.69 |
| Navigation | 1.845 | 9.07 |
| Gaming | 4.303 | 3.84 |
| Video Streaming | 2.598 | 6.42 |
| Reading | 1.727 | 9.69 |

### 3.5 消融实验（`ablation_study.py`）

Reading 场景，$3\times3$ 温度×老化网格：

| 去掉模块 | 0°C, 800cyc | 25°C, 800cyc | 40°C, 800cyc |
|---|---:|---:|---:|
| w/o Temperature | +69.5% | 0.0% | −18.2% |
| w/o Aging | +53.6% | +55.5% | +56.2% |
| w/o Component Decomp. | +31.6% | +46.2% | +46.3% |
| w/o Polarization ($K=0$) | +1.0% | +1.7% | +1.8% |

### 3.6 参数灵敏度（`sensitivity_robustness_analysis.py`）

Video 场景，±5% 扰动（baseline TTE ≈ 6.37h）：

| 参数 | Δ₋₅% | Δ₊₅% |
|---|---:|---:|
| $P_{total}$ | +5.37% | −4.86% |
| $Q_0$ | −4.77% | +4.76% |
| $E_0$ | −4.96% | +4.96% |
| $K, R_0, A, B$ | <0.3% | <0.3% |

## 四、目录与产出状态

### 4.1 根目录

| 文件 | 用途 |
|---|---|
| `Smartphone_Battery_Continuous_Time_Model_MCM2026.pdf/.txt` | 原竞赛论文与提取文本 |
| `ECM_revision_blueprint.md` | ECM 投稿改版蓝图 |
| `.gitignore` | 忽略 `.venv`、`data`、`pycache` |

### 4.2 `code/`（核心脚本）

- 关键模型：`main_model.py`、`aging_model.py`、`thevenin_ecm.py`
- 关键实验：`run_all_baselines.py`、`ablation_study.py`、`cross_batch_validation.py`、`bootstrap_ci.py`
- 场景与灵敏度：`analyze_all_scenarios_tte_loss.py`、`scenario_time_to_empty.py`、`sensitivity_robustness_analysis.py`
- 验证与可视化：`section5_nbm_validation.py`、`voltage_error_analysis.py`、`plot_*.py`

### 4.3 `results/`（已清理）

- 当前共 **50 个文件**（清理前 75 个，已删除 25 个冗余/中间/被替代产物）
- 保留内容覆盖：基线、消融、跨批次、Bootstrap、TTE、参数灵敏度、核心图表

### 4.4 其他目录

- `data/`：NASA + XJTU 原始数据
- `ecm_paper/`：ECM 投稿目录（待放置 `main.tex` 与 `refs.bib`）

## 五、已完成修复

| # | 问题 | 修复 | 文件 |
|---:|---|---|---|
| 1 | 老化参数不匹配 XJTU 拟合 | 改为 power-law（$\alpha_Q=0.000256,\gamma=1.085$） | `aging_model.py` |
| 2 | 灵敏度脚本使用 $Q_0=1.991$（电芯） | 改为 $Q_0=5.0$（手机） | `sensitivity_analysis.py` |
| 3 | Thévenin-2RC 数值溢出（RMSE=134mV） | 增加 `np.clip(v_rc, -2, 2)` + `np.isfinite` | `thevenin_ecm.py` |
| 4 | 温度消融仅测 25°C | 增加 `[0,25,40]°C` 多温度循环 | `ablation_study.py` |
| 5 | 场景定义不统一（缺 `background_extra`） | 统一为 `background_extra=0.1` | `ablation_study.py` |

## 六、ECM 论文状态

- ✅ 摘要、引言、相关工作、模型、实验、结果、结论已完成
- ✅ 参考文献（`refs.bib`，25 条）已完成
- ⬜ 待补：作者/单位/基金信息
- ⬜ 待补：`fig_framework.pdf`
- ⬜ 待做：`cas-dc` 编译验证

## 七、复现实验命令

```powershell
cd d:\WYN_COLLEGE\s20\!MCM_ICM\paper
.venv\Scripts\Activate.ps1
cd code

# 1) 基线对比
python run_all_baselines.py

# 2) 消融实验
python ablation_study.py

# 3) 跨批次验证
python cross_batch_validation.py

# 4) Bootstrap CI
python bootstrap_ci.py

# 5) 场景 TTE
python analyze_all_scenarios_tte_loss.py
python scenario_time_to_empty.py

# 6) 灵敏度
python sensitivity_robustness_analysis.py
```
