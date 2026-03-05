# ECM 投稿大改蓝图（基于 Team 2600101 现稿）

## 1) 当前稿件与 Energy Conversion and Management 的主要差距

### A. 文章定位差距（竞赛报告 vs 期刊论文）
- 现稿偏“任务响应式”叙事（Requirement 1-4），需改为“问题驱动 + 学术贡献驱动”。
- 现稿强调场景预测结果，但期刊更看重：
  1) 方法学新意是否清晰；
  2) 与 SOTA/强基线相比是否显著提升；
  3) 可复现性与统计稳健性是否充分。

### B. 创新表达不足
- 现稿提出 NBM + Shepherd + 温度 + 老化 + 功耗分解，但“创新点边界”尚不够聚焦。
- 建议重写为 2–3 条明确贡献（每条可验证）：
  - 贡献1：构建“电化学可解释 + 组件负载可分解 + 生命周期衰退耦合”的统一连续时间框架。
  - 贡献2：提出参数温度响应与老化耦合校正策略，并给出跨数据集可迁移验证。
  - 贡献3：给出可部署的 TTE 预测与能耗归因流程（含不确定性/敏感性）。

### C. 验证深度不足
- 当前主要是拟合与少量场景比较，需补：
  - 与更强基线模型对比（ECM / Rint / Thevenin / 数据驱动回归基线）。
  - 统计置信区间、显著性检验、跨电池/跨批次泛化能力。
  - 消融实验（去掉温度项/老化项/组件分解项后性能变化）。

### D. 可复现性与学术规范
- 需补全数据切分、训练/验证协议、超参数来源、代码与随机种子设置。
- AI 使用说明应迁移到投稿系统披露，不建议保留为正文大段章节。

---

## 2) 建议的 ECM 论文结构（重构目录）

1. Introduction
2. Related work
3. Problem formulation and preliminaries
4. Proposed unified continuous-time battery model
   - 4.1 Electrochemical voltage submodel
   - 4.2 Component-level power decomposition
   - 4.3 Temperature-response coupling
   - 4.4 Aging dynamics and lifetime coupling
   - 4.5 Identifiability and parameter estimation
5. Experimental setup
   - 5.1 Datasets and preprocessing
   - 5.2 Baselines and evaluation metrics
   - 5.3 Implementation details and reproducibility
6. Results and discussion
   - 6.1 Overall prediction accuracy
   - 6.2 Cross-scenario TTE performance
   - 6.3 Ablation studies
   - 6.4 Sensitivity/uncertainty analysis
   - 6.5 Engineering implications
7. Limitations and future work
8. Conclusions

---

## 3) 必做实验清单（投稿前）

### E1. 强基线对比（必须）
- 基线至少包含：
  - NBM（你们已有）
  - 标准 Shepherd（你们已有）
  - 一阶/二阶 ECM（建议新增）
  - 简单数据驱动基线（如 XGBoost 或 LSTM，小规模即可）
- 指标：RMSE/MAE/MaxE + TTE error + 推理耗时。

### E2. 泛化与鲁棒性（必须）
- Train on XJTU Batch-1/2, test on other batch 或 NASA 子集。
- 报告跨域误差变化，证明非过拟合。

### E3. 消融实验（必须）
- Full model
- w/o temperature
- w/o aging
- w/o component decomposition
- 结果用统一表格 + 关键图。

### E4. 统计可靠性（强烈建议）
- 每组结果给 95% CI（bootstrap）。
- 关键比较给显著性检验（如配对 t-test / Wilcoxon）。

---

## 4) 结合你们现有代码的执行映射

- 温度/敏感性：
  - `code/assumption_sensitivity.py`
  - `code/plot_parameter_sensitivity.py`
  - `code/sensitivity_analysis.py`
- 场景 TTE：
  - `code/scenario_time_to_empty.py`
  - `code/scenario_sensitivity.py`
  - `code/analyze_all_scenarios_tte_loss.py`
- 老化模型：
  - `code/aging_model.py`
  - `code/fit_aging_model.py`
  - `code/visualize_aging_model.py`
- 验证与对比：
  - `code/section5_nbm_validation.py`
  - `code/discharge_time_comparison.py`
  - `code/voltage_error_analysis.py`

建议先统一输出目录与字段命名，再批量生成期刊图表。

---

## 5) 图表与写作重做优先级

### P0（第一周内完成）
1. 重写摘要（结构化：背景-方法-结果-贡献-意义）
2. 重写引言与贡献点（3条可验证贡献）
3. 统一符号与方程编号，删竞赛措辞（Requirement 1/2/3/4）

### P1
4. 新增 Related Work（至少 20–30 篇近5年文献）
5. 补强基线 + 消融 + 跨数据集泛化
6. 结果部分改为“事实 + 解释 + 工程含义”三段式

### P2
7. 语言润色、图表排版、投稿格式化（Elsevier 模板）
8. 附录补充参数表、伪代码、训练配置

---

## 6) 高风险问题（需要先修）

1. 结果可信度风险：目前“75.5% RMSE improvement”等结论需在统一切分协议下重算。
2. 数据泄漏风险：需明确参数拟合区间与测试区间完全分离。
3. 叙事风险：目前更像竞赛答题，不像学术论文。
4. 术语/符号风险：SOC/SoC、E0/OCV 等写法要统一。

---

## 7) 你们可以直接采用的目标标题（备选）

1. A Unified Continuous-Time Electrochemical-Power Coupled Model for Smartphone Battery Time-to-Empty Prediction
2. Physics-Informed Continuous-Time Modeling of Smartphone Battery Dynamics with Temperature and Aging Coupling
3. Interpretable Time-to-Empty Prediction via Coupled Electrochemical and Component-Level Power Modeling

---

## 8) 下一步建议（最小可执行动作）

- 先重写摘要 + 引言 + 贡献点（2天内）。
- 同步跑一次“统一协议”的结果重现实验（3天内）。
- 再进入方法与结果整稿改写。

如果需要，可在下一步直接产出：
1) ECM 风格英文摘要（250词）；
2) 引言前两节可直接替换稿；
3) 实验部分模板（含数据划分和统计检验写法）。
