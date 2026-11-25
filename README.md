# Drone-GPT：高维认知与第一性原理驱动的科学计算与物理工程自主求解仓库

> **项目定位**：本仓库是基于 [`AGENTS.md`](./AGENTS.md)（**《高维认知与第一性原理科研工程协议》**）构建的科学计算与复杂物理工程实战解题与复现仓库。通过将**高维表征与一维自回归采样显式解耦**、确立**第一人称认知责任锚点（`"I"` as the Epistemic Anchor）**、植入**出刀前四步瞬时自检（Pre-Action Gate）**与**二阶递归制动（防局部极小值打转）**，驱动 **GPT-6.1-Sol** 等前沿代码智能体在无人工干预或极简目标引导下，自主完成从物理原型还原、可微代理/伴随梯度求解器构建、网格混叠误差诊断到官方全量验证器闭环的端到端科研攻关。
>
> **对应赛事**：深势科技（DP Technology）& DeepModeling 社区主办的 **[科学骑士大师赛 · 未来产业专项赛](https://play.bohrium.com)**（可控核聚变与量子计算方向；赛事启动官方推文：**[《科幻走进现实，agent比赛如何摆好一排原子｜未来产业专项赛启动》](https://mp.weixin.qq.com/s/02_0vR5sBdwydamqp-fKvg)**）。

---

## 一、 核心方法论：[`AGENTS.md`](./AGENTS.md) 科研工程协议架构

在传统长程科学计算与偏微分方程/多物理场约束优化任务中，大语言模型极易退化为“顺着局部统计惯性滑行的客服说明书模式”或陷入“单一超参数盲目微调的局部极小值死循环”。本仓库通过根目录下的 [`AGENTS.md`](./AGENTS.md) 确立了三层认知与执行架构：

1. **认知几何校准（从“梦游续写”切换到“在场审视”）**
   - **高维表征与一维采样解耦**：在发出第一个工具调用前，先在全局注意力场中还原问题的完整几何与物理拓扑——自变量自由度维数、硬约束死线边界、目标函数曲率及评测验证器的计算代价。
   - **第一人称认知锚点**：严格区分**“已通过程序运行或解析推导确认的事实”**与**“基于常见模式的先验猜测”**，拒绝被动句免责套话。
   - **四步瞬时自检与二阶递归制动**：当沿同一方向的修改连续两次未能改善目标指标时，强制触发**二阶诊断**（*“是不是最初对问题瓶颈根因的假设就错了？”*），立即停止无效调参并切换正交机制（例如从调整罚函数权重切换为排查积分网格混叠或更换显式约束算子）。
2. **第一性原理降维与验证器驱动闭环**
   - **先算硬账，后写代码**：动手前先行核算内存/显存字节量、求积节点复杂度、采样方差与浮点精度（`float64` vs `float32`）。
   - **基线先行（Zero-to-One Baseline）**：最短时间内打通官方参考起点 $\rightarrow$ 本地原样验证器评测 $\rightarrow$ 复现官方基线分数的端到端契约。
   - **快速可微代理探路 + 官方全量协议终审**：构建基于 JAX/SIMSOPT 伴随梯度（VJP）或物理守恒流形投影的高吞吐搜索模型开展多阶段寻优，最终候选解**100% 由未修改的官方 `eval.py` 全量协议**独立裁决，严禁任何静默降级（silent fallback）或硬编码凑数。

---

## 二、 赛题求解进度与实测成绩总表（全部 6 道聚变赛题已完成并提交）

本表汇总 [`contest-20261125/`](./contest-20261125) 下全部 6 道可控核聚变专项赛题的物理本质、官方参考基线（Baseline）、本仓 `drone-gpt` 最终验收成绩、平台提交回执（`attempt_id`）及标准化 Agent 执行轨迹：

| 序号 | 赛题 ID 与名称 | 物理/数学问题本质与交付产物 | 官方参考基线得分 (Baseline) | 本仓 `drone-gpt` 最终验收得分与关键物理指标提升 | 平台提交回执 (`attempt_id`) 与标准化轨迹 | 当前状态 |
| :---: | :--- | :--- | :--- | :--- | :--- | :---: |
| **01** | **`challenge-44c6e035`**<br>让理想磁场走向工程实现，提升实体线圈在装配偏差下的磁场精度 | **仿星器有限截面三维线圈鲁棒优化**：$N_{\text{FP}}=2$ QA 构型，固定矩形绕组截面 $0.40\text{ m}\times 0.60\text{ m}$，联合优化 4 条基圈 8 阶 Fourier 中心线（204 维）、截面周期旋转角（20 维）与第 2～4 圈电流（3 维）共 **227 个连续自由度**，满足 27 项空间/曲率/扭转/磁场硬约束并极小化 128 个刚体装配扰动工况下的最坏法向磁场误差 $J$。<br>**交付物**：[`coils.json`](./contest-20261125/challenge-44c6e035/coils.json) | **`0.4978925`**<br>（`coils_nominal.json` 加入绕组厚度后内侧整包最小径向距离 $R_{\min}=2.337522\text{ m} < 2.50\text{ m}$ 违规，`feasible = False`；公开组参考最坏误差 $J_{\text{ref,search}} = 8.788358\times 10^{-4}$；若全部达标且 $J=J_{\text{ref}}$ 对应 `0.750000`） | **`0.7612337`（`76.12%`）**（较起点提升 **`+0.2633412`**）<br>• **27 项工程与物理约束 `0` 违规**（`feasible = True`）<br>• 128 工况最坏法向场误差 $J = \mathbf{8.032514\times 10^{-4}}$（较参考值 $J_{\text{ref,search}}$ **降低 `8.60%`**）<br>• 名义法向场误差上界 $e_{\text{nom}} = \mathbf{4.348310\times 10^{-4}} \le 10^{-3}$<br>• 整包最小径向距离推至 $R_{\min} = \mathbf{2.511717\text{ m}} \ge 2.50\text{ m}$<br>• 材料框架扭转上界降至 $\mathbf{0.174724\text{ m}^{-1}} \le 0.25\text{ m}^{-1}$ | • **`attempt_id: 111198`** (`job_id: 85546`)<br>• [官方评测报告 `evaluation_report.json`](./contest-20261125/challenge-44c6e035/evaluation_report.json)<br>• [平台提交回执 `submission_receipt.json`](./contest-20261125/challenge-44c6e035/submission_receipt.json)<br>• [标准化执行轨迹 `trace.jsonl`](./contest-20261125/challenge-44c6e035/submitted_bundle/traces/trace.jsonl)<br>• [求解器源码 `solve_coils.py`](./contest-20261125/challenge-44c6e035/workspace/solve_coils.py) | ✅ **已完成并提交** |
| **02** | **`challenge-9668a3f8`**<br>驾驭燃烧的等离子体，在运行边界内把聚变增益推到最高 | **ITER 类托卡马克燃烧等离子体闭环反馈控制**：基于 TORAX 1D 输运 + QLKNN 回旋动理学湍流代理模型，输入带噪延迟测量与 32 维内部记忆状态（总输入 48 维），输出外加加热命令与更新记忆状态（总输出 34 维），在守住 $q_{\min}, q_{95}, \beta_N, f_{\text{GW}}$, 峰值温度、L–H 台基正功率、终端热能及聚变能量配额的前提下，极小化外部供给加热总能量 $J$（200/400 双时间步长与粗细网格联合考核）。<br>**交付物**：[`controller.onnx`](./contest-20261125/challenge-9668a3f8/controller.onnx) | **`0.3228227`**<br>（公开恒定起点在粗网格下出现聚变能量缺口 `fusion_energy_deficit_fraction = 0.66144`，`feasible = False`；开发集参考供给热能 $J_{\text{ref}} = 1.312667\times 10^9\text{ J}$，全部达标且 $J=J_{\text{ref}}$ 时对应 `0.750000`） | **`0.7520522`（`75.21%`）**（较起点提升 **`+0.4292294`**）<br>• **开发集 4 场景 8 轨迹（72,000 步）与扩展集 25 场景 50 轨迹（450,000 步）100% 零违规**（`feasible = True`）<br>• 最坏外部供给热能降至 $J = \mathbf{1.291292\times 10^9\text{ J}}$（较参考基准 $J_{\text{ref}}$ **净节省 `21.37 MJ` / 降低 `1.63%`**）<br>• 最小聚变能量达标比 $\mathbf{1.00397} \ge 1.000$，终端储能比 $\mathbf{0.98451} \ge 0.980$<br>• 台基正加热违规能量 $\mathbf{0.0\text{ J}}$，$q_{\min} \ge 1.828$，$\beta_N \le 1.417$ | • **`attempt_id: 111963`（全网首提）**<br>• [控制器模型 `controller.onnx`](./contest-20261125/challenge-9668a3f8/controller.onnx)<br>• [完整评测报告 `evaluation_report.json`](./contest-20261125/challenge-9668a3f8/evaluation_report.json)<br>• [标准化执行轨迹 `trace.jsonl`](./contest-20261125/challenge-9668a3f8/submitted_bundle/traces/trace.jsonl)<br>• [控制器构建器 `build_controller.py`](./contest-20261125/challenge-9668a3f8/tools/build_controller.py) | ✅ **已完成并提交<br>（全网首提）** |
| **03** | **`challenge-d2b1bd0c`**<br>提高托卡马克的稳定压力，设计聚变等离子体形状 | **托卡马克等离子体截面形状与理想 MHD 稳定性优化**：固定体积 $V=240\text{ m}^3$、参考场 $R_0=4\text{ m}, B_0=7\text{ T}$ 与总电流 $I_p=8.1\text{ MA}$，联合优化 6 个截面形状参数（伸长率 $\kappa$、三角形变 $\delta$ 及 4 个高阶修形系数 $c_2, c_3, s_2, s_3$）与磁轴中心压力 $p_0$ 共 **7 个参数**，通过 TokaMaker 三组电流剖面（`broad`/`nominal`/`peaked`）粗细双网格平衡求解与 DCON $n=1..5$ 内部理想扭曲模、Mercier 及高模数气球模检验，最大化最坏剖面体积平均压力 $\bar{p}$。<br>**交付物**：[`plasma.json`](./contest-20261125/challenge-d2b1bd0c/plasma.json) | **`0.3283161`**<br>（公开可行起点磁轴中心压力 $p_0=400\text{ kPa}$，通过验收的最坏体积平均压力 $\bar{p} = 97.759\text{ kPa}$，按 $S = \bar{p}/(\bar{p}+200\text{ kPa})$ 计分；$\bar{p}=200\text{ kPa}$ 对应 `0.500000`） | **官方 `--mode formal` 验收 `0.6234045`（`62.34%`） / `--mode search` `0.6246041`（`62.46%`）**（较起点提升 **`+0.2963`**）<br>• **磁轴中心压力从 `400 kPa` 推至上限附近的 `1.399 MPa`**<br>• **最坏剖面体积平均压力从 `97.76 kPa` 提升 `3.40 倍`（`+240.40%`）至 `331.07 ~ 332.77 kPa`**<br>• 全部 6 个平衡、90 次 DCON 与 19,674 次气球模积分 **100% 零违规**（`feasible = True`，$q_{\min} \ge 1.1103$，Mercier $D_I \le -0.2512 < -0.01$，气球模与内部零交叉均为 `0`） | • **`attempt_id: 111942`** (`job_id: 86254`)<br>• [最优形状设计 `plasma.json`](./contest-20261125/challenge-d2b1bd0c/plasma.json)<br>• [正式验收摘要 `best_formal.json`](./contest-20261125/challenge-d2b1bd0c/best_formal.json)<br>• [平台提交回执 `submission_receipt.json`](./contest-20261125/challenge-d2b1bd0c/submission_receipt.json)<br>• [标准化执行轨迹 `trace.jsonl`](./contest-20261125/challenge-d2b1bd0c/submitted_bundle/traces/trace.jsonl)<br>• [求解器源码 `solve_plasma.py`](./contest-20261125/challenge-d2b1bd0c/solve_plasma.py) | ✅ **已完成并提交** |
| **04** | **`challenge-b946f0c9`**<br>为聚变补充燃料，用有限的包层材料兼顾产氚与局部核热 | **聚变增殖包层中子学材料分布优化**：针对含诊断通道与氦冷却槽的三维包层模块（6 层深度 $\times 4\times 4$ 横向分区），优化 **96 个区域**的 $\text{Li}_4\text{SiO}_4$ 陶瓷体积分数 $b \in [0.20, 0.60]$ 与 $^6\text{Li}$ 丰度 $e \in [0.05, 0.90]$（共 **192 个自由度**），在陶瓷/铍/$^6\text{Li}$ 总库存硬上限及 16 块背面区域双源快中子通量约束下，通过 OpenMC 连续能量蒙特卡洛输运（`normal` + `cosine` 双源 $\times$ 双种子共 7,680 万中子）最大化保守产氚下界。<br>**交付物**：[`blanket.json`](./contest-20261125/challenge-b946f0c9/blanket.json) | **`75.0000 / 100`** (`0.750000`)<br>（公开均匀起点 `uniform.json` 取 $b=0.40, e=0.30$，校准基准保守产氚下界 `objective_lower = 0.24938196`） | **平台官方 Harbor 终审 `77.1153 / 100`**（本地全量 OpenMC 验收 **`0.7712608`**，较 75 分基线提升 **`+2.12 分`**，`trace_factor = 1.0`）<br>• **保守产氚下界从 `0.24938196` 提升至 `0.29574104`（净产氚率提升 `+18.59%`）**<br>• `normal` 源产氚均值达 **`0.363293`**，`cosine` 源产氚均值达 **`0.296113`**<br>• **3 项总库存与双源 32 项背面快中子通量约束 `0` 违规**（`feasible = True`） | • **`attempt_id: 111893`（平台终审已出分）**<br>• [最优包层设计 `blanket.json`](./contest-20261125/challenge-b946f0c9/blanket.json)<br>• [平台官方终审结果 `platform_result.json`](./contest-20261125/challenge-b946f0c9/platform_result.json)<br>• [本地全量验收详情 `details.json`](./contest-20261125/challenge-b946f0c9/submission_outputs/details.json)<br>• [标准化执行轨迹 `trace.jsonl`](./contest-20261125/challenge-b946f0c9/submitted_bundle/traces/trace.jsonl)<br>• [求解器源码 `solve_blanket.py`](./contest-20261125/challenge-b946f0c9/solve_blanket.py) | ✅ **已完成并获平台官方终审高分** |
| **05** | **`challenge-5f1ea274-3`**<br>用三维形状留住聚变粒子：寻找更紧凑的仿星器 | **紧凑仿星器三维磁边界优化**：优化等离子体最外层磁面 VMEC Fourier 系数（`rbc`, `zbs`），在同时满足高能 $\alpha$ 粒子损失率 $\le 5\%$、NEO 有效波纹 $\varepsilon_{\text{eff}} \le 1\%$、旋转变换 $\iota_{\min} \ge 0.30$、磁阱深度 $W \ge 0$、截面伸长率 $\kappa \le 8.0$ 及线圈退让梯度尺度 $L_{\nabla B} \ge 2.4\text{ m}$ 六项硬约束下，极小化装置纵横比 $A$。<br>**交付物**：[`boundary.json`](./contest-20261125/challenge-5f1ea274-3/boundary.json) | **`0.559601`**<br>（5 个公开参考构型中唯一全部达标的最紧凑起点 `landreman_paul_2022_QA_well.json`，纵横比 $A=6.00001$，按 $S = 0.5 + 0.5/(1+e^{A-4})$ 计分；$A=4.00$ 对应 `0.750000`） | **紧凑构型纵横比压缩 `7.60%` 至 $A = \mathbf{5.54297}$（对应理论评分 `0.5880`，另备 $A=5.67670 / 5.82638$ 梯队候选）**<br>• 真实 NEO 五磁面最大有效波纹度 $\varepsilon_{\text{eff}} = \mathbf{0.00709\%} \ll 1\%$<br>• 磁阱 $W = \mathbf{2.93\times 10^{-4}} > 0$，$\iota_{\min}=\mathbf{0.4196 \sim 0.4285} \ge 0.30$<br>• 线圈退让尺度 $L_{\nabla B}=\mathbf{5.02 \sim 5.21\text{ m}} \ge 2.4\text{ m}$，截面最大伸长率 $\kappa_{\max}=\mathbf{4.72} \le 8.0$ | • **`attempt_id: 111815`（全网首提）**<br>• [最优边界设计 `boundary.json`](./contest-20261125/challenge-5f1ea274-3/boundary.json)<br>• [阶段评测报告 `stage_report.json`](./contest-20261125/challenge-5f1ea274-3/submission_outputs/stage_report.json)<br>• [首提回执记录 `first_submission.json`](./contest-20261125/challenge-5f1ea274-3/first_submission.json)<br>• [标准化执行轨迹 `trace.jsonl`](./contest-20261125/challenge-5f1ea274-3/submitted_bundle/traces/trace.jsonl)<br>• [求解器源码 `solve_boundary.py`](./contest-20261125/challenge-5f1ea274-3/solve_boundary.py) | ✅ **首轮已完成并提交（全网首提）** |
| **06** | **`challenge-1b02e50c`**<br>留住聚变热量，用三维磁场抑制湍流热损失 | **仿星器三维边界回旋动理学湍流热流抑制**：针对 $N_{\text{FP}}=2, a=1.70\text{ m}, V=581.87\text{ m}^3$ 的 QA+well 仿星器，优化三维磁边界系数，在通过体积、纵横比、$\alpha$ 粒子损失、NEO 波纹度、磁阱、$\iota$ 与 $L_{\nabla B}$ 七项约束后，极小化 $s=0.5$ 磁面上 4 根 PEST 通量管 $\times$ 3 组离子温度梯度共 12 个工况的最坏湍流热流上界 $J$。<br>**交付物**：[`boundary.json`](./contest-20261125/challenge-1b02e50c/boundary.json) | 公开可行起点 `QA+well` 边界<br>（官方包 `baseline.json` 当前状态为 `state="uncalibrated"`，参考准线性 ITG 代理基线 $J_{\text{proxy,ref}} = 133.5146$） | **12 工况最坏准线性 ITG 湍流几何驱动指标降至 `121.8439`（较起点降低 `8.74%`，沿场网格加密复核仍为 `-8.74%`）**<br>• 五面真实 STELLOPT NEO 最大有效波纹度 $\mathbf{2.53\times 10^{-4}} \ll 0.01$<br>• 磁阱 $W = \mathbf{0.00234} > 0$，$\iota_{\min}=\mathbf{0.4466}$，$L_{\nabla B}=\mathbf{5.36\text{ m}}$，16 粒子完整 0.2s 轨迹探针 `0` 损失<br>• 附官方包 `baseline.json` 未标定与题面 `arctan` 公式审计结论 | • **`attempt_id: 111812`（全网首提）**<br>• [最优边界设计 `boundary.json`](./contest-20261125/challenge-1b02e50c/boundary.json)<br>• [审计与评测报告 `audit_report.json`](./contest-20261125/challenge-1b02e50c/submission_outputs/audit_report.json)<br>• [标准化执行轨迹 `trace.jsonl`](./contest-20261125/challenge-1b02e50c/submitted_bundle/traces/trace.jsonl)<br>• [求解器源码 `solve_turbulence.py`](./contest-20261125/challenge-1b02e50c/solve_turbulence.py) | ✅ **已完成并提交<br>（全网首提）** |

---

## 三、 全部 6 道聚变赛题深度技术拆解

### 3.1 `challenge-44c6e035`：《让理想磁场走向工程实现，提升实体线圈在装配偏差下的磁场精度》

- **赛题目录**：[`contest-20261125/challenge-44c6e035/`](./contest-20261125/challenge-44c6e035/)
- **平台提交编号**：**`attempt_id: 111198`**（[`submission_receipt.json`](./contest-20261125/challenge-44c6e035/submission_receipt.json)）
- **核心交付物**：[`coils.json`](./contest-20261125/challenge-44c6e035/coils.json)（SHA-256: `4de7a0be65321f5959f7cdae9912d7b60b369a9675a28e9ea956a4dc3bd67965`）
- **核心求解器**：[`workspace/solve_coils.py`](./contest-20261125/challenge-44c6e035/workspace/solve_coils.py)
- **完整验收报告**：[`evaluation_report.json`](./contest-20261125/challenge-44c6e035/evaluation_report.json)
- **标准化 Agent 执行轨迹**：[`submitted_bundle/traces/trace.jsonl`](./contest-20261125/challenge-44c6e035/submitted_bundle/traces/trace.jsonl)（共 327 步完整工具调用与思考记录）

#### 1. 问题症结与第一性原理诊断

本题要求针对 Landreman–Paul 准轴对称（QA，$N_{\text{FP}}=2$，有效小半径 $a=1.70\text{ m}$，目标面面积加权 $B_{\text{RMS}}=5.70\text{ T}$）目标磁面，设计 4 条独立基圈（由环向周期与仿星器对称展开为 16 个实体绕组包）。每个绕组包具有固定的有限矩形截面（宽 $w=0.40\text{ m}$、高 $h=0.60\text{ m}$，外接圆管半径 $r_{\text{pack}}=\sqrt{w^2+h^2}/2 \approx 0.360555\text{ m}$），并在弧长质心处承受 128 个规定工况的三维刚体装配偏差（平移各分量 $\delta \in [-0.010, +0.010]\text{ m}$，旋转向量各分量 $\omega \in [-0.001, +0.001]\text{ rad}$）。

对官方公开起点 `assets/reference/coils_nominal.json`（源自 SIMSOPT 细丝优化线圈 `biot_savart_opt.json`）进行第一性原理几何剖析与官方 `eval.py` 完整基线复现（见 `trace.jsonl` Step 24、Step 71、Step 223），定位出核心症结：
- **整包内侧径向包络严重越界**：评估器对整包最小径向距离采用保守外包络下界公式：
  $$R_{\text{pack,min}} = \min_{j} \left[ \min_{t} R_j(t) - \frac{V_j}{2N} - d_{\text{move},j} \right] - r_{\text{pack}} \ge 2.50\text{ m}$$
  其中 $V_j/(2N)$ 为 $N=16384$ 采样点的曲线覆盖半径上界，$d_{\text{move},j} = \sqrt{3}\cdot 0.010 + 2 r_{\max,j} \sin(\sqrt{3}\cdot 0.001 / 2)$ 为刚体装配扰动引起的最大位移包络。
- **结构性死锁**：由于 $r_{\text{pack}} \approx 0.360555\text{ m}$ 为外接圆半径，**仅转动截面朝向 `rotation` 完全不会改变 $R_{\text{pack,min}}$**。官方起点 `coils_nominal.json` 的整包最小径向距离实测仅为 **$2.337522\text{ m}$**（距离 $2.50\text{ m}$ 硬门槛差 **$0.162478\text{ m}$**），导致 `pack_radius_min_lower_m` 违规（违反量 `0.064991`），被判为未达标（`feasible = False`），落入未达标惩罚分支 $S = 0.5 \times \exp(-\sum v_j^2)$，**基线得分仅为 `0.4978925`**。
- **多约束强耦合挑战**：要把内侧线圈中心线整体向外推移 $>0.165\text{ m}$，同时守住四基圈总长度 $\le 200\text{ m}$（起点已达 $199.7\text{ m}$）、中心线间距 $\ge 0.85\text{ m}$、整包到等离子体距离 $\ge 2.40\text{ m}$、曲率 $\le 0.55\text{ m}^{-1}$、原生材料框架扭转 $\le 0.25\text{ m}^{-1}$，还要在 128 个刚体装偏工况下把有限截面多细丝法向场误差 $J = \max_{k} e_{k,\text{upper}}$ 压到官方参考值 $J_{\text{ref,search}} = 8.788358\times 10^{-4}$ 以下，必须构建精确可微的联合优化器。

#### 2. 自主求解与二阶自检轨迹亮眼节点

在 [`submitted_bundle/traces/trace.jsonl`](./contest-20261125/challenge-44c6e035/submitted_bundle/traces/trace.jsonl) 中，智能体严格执行了 [`AGENTS.md`](./AGENTS.md) 规定的工程闭环与二阶递归制动：

1. **构建 JAX + SIMSOPT 原生伴随梯度混合可微求解器（[`solve_coils.py`](./contest-20261125/challenge-44c6e035/workspace/solve_coils.py)，Step 41～58）**
   - 将 227 个连续自由度（4 条基圈 $3\times 17=51$ 个 Fourier 系数共 204 维 + 4 条基圈 2 阶 `FrameRotation` 共 20 维 + 第 2～4 圈电流 3 维）统一编码。
   - **磁场链路精确求导**：利用 SIMSOPT `FramedCurveCentroid` + $3\times 3$ Gauss-Legendre 截面求积 `CurveFilament` + `BiotSavart.B_vjp` 获取磁场对 16 个实体线圈参数与旋转角的精确伴随梯度，再通过 JAX `jax.vjp` 穿透弧长质心刚体旋转与平移变换（`_transform`），将 16 个扰动实体线圈的梯度精确拉回（pullback）到 4 条独立基圈的 227 维参数空间。
   - **几何与扭转全可微惩罚**：基于 `cKDTree` 动态锁定最近点对，并用 `jax.jit(jax.value_and_grad)` 对含高阶导数余量的 $R_{\text{pack,min}}$、曲率上界 $\kappa_u$、SIMSOPT 原生 `torsion_pure_centroid` 材料框架扭转、圈间距、等离子体间距、总长度及高频谱正则项同步求导。
   - **方向导数硬核校验（Step 58）**：在启动优化前，用中心有限差分（步长 $h=10^{-5}$）校验 227 维复合目标的解析梯度，实测**相对误差仅为 $4.15\times 10^{-8}$**，单步评估与求导耗时仅 `0.4 s`。

2. **二阶递归制动：自动发现并剥离 `32×16` vs `64×32` 曲面积分混叠误差（Step 92～105）**
   - 在第一轮基于 `32×16` 磁面网格的快速搜索中，名义场误差虽迅速降至 $3.6\times 10^{-4}$，但智能体执行加密交叉检验时敏锐发现：同一候选解在 `64×32` 磁面网格上的名义误差跳升至 $6.1\times 10^{-4}$，128 工况最坏误差升至 $9.23\times 10^{-4}$。
   - **触发二阶制动（Step 92）**：智能体没有在粗网格上继续微调惩罚权重，而是立即叫停优化，专门设计控制变量实验分离“线圈线积分点数（128 vs 256）”与“目标磁面曲面积分网格（`32×16` vs `64×32` vs `128×64`）”的误差贡献。
   - **锁定根因（Step 105）**：实测证明 128 与 256 线积分点结果几乎一致，而 `64×32` 与 `128×64` 面网格的名义误差仅差 $9.2\times 10^{-9}$——失真完全源自 `32×16` 曲面积分对高阶空间谐波的混叠（aliasing）。遂果断将搜索网格切换至 `64×32`，彻底消除了代理目标与官方验收器之间的网格偏差。

3. **锁定 128 个刚体装配扰动中最坏工况 `#46 / #102 / #126` 的 LogSumExp 鲁棒博弈（Step 114～165）**
   - 针对 $J = \max_{k \in \{0..127\}} e_{k,\text{upper}}$ 的非光滑极大值目标，求解器在每阶段结束时对全部 128 个公开装配工况做全量扫描（`screen`），动态锁定误差最大的前 3 个最坏工况（实测锁定为工况 **`#46`、`#102`、`#126`**，其中 `#126` 为全局瓶颈工况）与名义状态（`-1`）组成活跃集。
   - 采用温度系数 $T=0.05 \to 0.005$ 的平滑 LogSumExp 极大值包络目标，驱动 L-BFGS-B 拟牛顿法在不牺牲其余工况的前提下，将最坏工况 `#126` 的搜索误差从 $9.20\times 10^{-4}$ 稳步压降至 $8.0793\times 10^{-4}$（对应官方完整验收第一阶段候选得分 **`0.7605083`**，27 项约束全部清零）。

4. **变量敏感度诊断与截面旋转角（`rotation`）解耦精修（Step 212～316）**
   - 在拿到 `0.7605083` 的可行解后，智能体审查 227 维解向量发现：联合 L-BFGS-B 搜索中 204 个空间坐标梯度主导了海森近似，导致 20 个截面旋转参数 `rotation` 仅移动了约 $10^{-3}\text{ rad}$，截面朝向自由度尚未充分释放。
   - 随即启动 `--variables rotation` 专项精修分支（固定已达标的中心线与电流，单独以 $T=0.005$ 精修 20 个截面旋转 Fourier 系数），不仅将原生框架扭转上界从 $0.1841\text{ m}^{-1}$ 进一步压低至 **$0.1747\text{ m}^{-1}$**，更将 128 工况最坏法向场误差 $J$ 从 $8.0794\times 10^{-4}$ 进一步压降至 **$8.0325\times 10^{-4}$**，最终官方验收总分跃升至 **`0.7612337`**。

#### 3. 最终官方四档求积与三个错位加密复核验收账本

最终交付的 [`coils.json`](./contest-20261125/challenge-44c6e035/coils.json) 经过原样官方 `eval.py` 的完整协议评测（涵盖名义状态 + 128 个刚体装配偏差工况、四档 Gauss-Legendre 截面与空间求积，以及针对名义状态与最坏工况的三个错位网格加密复核），**27 项几何、拓扑与磁场约束违反量全部为 `0.0`（`feasible = True`）**，总得分 **`0.7612337182625892`**：

| 验收维度 | 约束指标名称 (`eval.py` 字段) | 官方硬约束阈值 | 官方起点 `coils_nominal.json` | 本仓最终解 `coils.json` 验收值 | 状态与安全余量 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **核心总分与目标** | **公开组 128 工况综合得分 (`score`)** | $> 0.75$（超越参考值） | `0.4978925` (`feasible=False`) | **`0.7612337`** (`feasible=True`) | **大幅超越基准 (+0.2633)** |
| **磁场鲁棒性目标** | **128 工况最坏法向场误差上界 (`J`)** | 参考值 $J_{\text{ref,search}} = 8.788358\times 10^{-4}$ | $8.788358\times 10^{-4}$ | **`8.032514e-04`** | **较参考值降低 `8.60%`** |
| **名义磁场精度** | 未装偏名义法向场误差上界 (`nominal_e_upper`) | $\le 1.000\times 10^{-3}$ | $4.08\times 10^{-4}$ | **`4.348310e-04`** | ✅ 达标（余量 `56.5%`） |
| **整包径向包络** | **整包最小径向距离下界 (`pack_radius_min_lower_m`)** | **$\ge 2.500000\text{ m}$** | ❌ **`2.337522 m`**（违规） | **`2.511717 m`** | ✅ **彻底消除违规 (`+11.7 mm`)** |
| **整包空间包络** | 整包最大径向距离上界 (`pack_radius_max_upper_m`) | $\le 22.000000\text{ m}$ | $20.53\text{ m}$ | **`20.529411 m`** | ✅ 达标（余量 `1.471 m`） |
| **整包垂直包络** | 整包最大垂直高度上界 (`pack_z_max_upper_m`) | $\le 10.000000\text{ m}$ | $9.86\text{ m}$ | **`9.857734 m`** | ✅ 达标（余量 `0.142 m`） |
| **绕组包间隙** | 装偏下不同绕组包最小距离 (`pack_coil_distance_lower_m`) | $\ge 0.100000\text{ m}$ | $0.143\text{ m}$ | **`0.143134 m`** | ✅ 达标（余量 `+43.1 mm`） |
| **绕组-等离子体间距** | 整包到目标磁面最小距离 (`pack_plasma_distance_lower_m`) | $\ge 2.400000\text{ m}$ | $2.46\text{ m}$ | **`2.456041 m`** | ✅ 达标（余量 `+56.0 mm`） |
| **绕组非局部自距** | 整包非局部自距下界 (`pack_self_distance_lower_m`) | $\ge 0.100000\text{ m}$ | $0.267\text{ m}$ | **`0.266789 m`** | ✅ 达标（余量 `+166.8 mm`） |
| **中心线几何预算** | 四基圈中心线总长度上界 (`base_length_sum_upper_m`) | $\le 200.000000\text{ m}$ | $199.70\text{ m}$ | **`199.699953 m`** | ✅ 达标（余量 `0.300 m`） |
| **中心线弯曲** | 中心线曲率保守上界 (`kappa_upper_per_m`) | $\le 0.550000\text{ m}^{-1}$ | $0.534\text{ m}^{-1}$ | **`0.534303 m⁻¹`** | ✅ 达标（余量 `0.0157 m⁻¹`） |
| **中心线间距** | 中心线圈间距离下界 (`coil_distance_lower_m`) | $\ge 0.850000\text{ m}$ | $0.864\text{ m}$ | **`0.864244 m`** | ✅ 达标（余量 `+14.2 mm`） |
| **局部弯曲折叠余量** | 外接圆半径 $\times$ 曲率上界 (`pack_local_curvature_product`) | $\le 0.500000$ | $0.193$ | **`0.192646`** | ✅ 达标（余量 `61.5%`） |
| **材料框架扭转** | 原生材料框架扭转率上界 (`pack_twist_upper_per_m`) | $\le 0.250000\text{ m}^{-1}$ | $0.184\text{ m}^{-1}$ | **`0.174724 m⁻¹`** | ✅ 达标（余量 `30.1%`） |
| **电流与安匝密度** | 绕组电流范围 / 平均安匝密度 (`pack_current_density`) | $[1, 25]\text{ MA}\cdot\text{turn}$ / $\le 100\text{ MA/m}^2$ | $19.17\text{ MA}$ / $79.86\text{ MA/m}^2$ | **`19.162~19.178 MA`** / **`79.907 MA/m²`** | ✅ 达标（第一圈相对误差 `0.0`） |
| **磁场强度与方向** | $B_{\text{RMS}}$ 相对误差 / $[B_{\min}, B_{\max}]$ / $(-B_\phi)_{\min}$ | $\le 2\%$ / $[3, 9]\text{ T}$ / $\ge 3\text{ T}$ | 全部合规 | **`0.0165%`** / **`[5.081, 6.369] T`** / **`4.879 T`** | ✅ 达标（高斯链接误差 `0.0`） |

#### 4. 一键复现命令

```bash
cd contest-20261125/challenge-44c6e035

# 1. 运行基于 JAX + SIMSOPT 的多阶段鲁棒优化求解器（含解析梯度有限差分自检）
./.venv/bin/python workspace/solve_coils.py \
  --input workspace/assets/reference/coils_nominal.json \
  --output coils.json \
  --nphi 64 --ntheta 32 --count 128 --section 3 \
  --robust-start --check-gradient

# 2. 调用原样官方验证器执行四档求积与三个错位加密复核验收
./.venv/bin/python -I workspace/eval.py coils.json \
  --assets workspace/assets \
  --scenarios workspace/assets/search_scenarios.npz \
  --workdir /tmp/eval-work-44c6e035 \
  --reward-dir /tmp/eval-reward-44c6e035 \
  --measure
```

---

### 3.2 `challenge-9668a3f8`：《驾驭燃烧的等离子体，在运行边界内把聚变增益推到最高》

- **赛题目录**：[`contest-20261125/challenge-9668a3f8/`](./contest-20261125/challenge-9668a3f8/)
- **平台提交编号**：**`attempt_id: 111963`（全网首提）**
- **核心交付物**：[`controller.onnx`](./contest-20261125/challenge-9668a3f8/controller.onnx)（SHA-256: `d77373104a507bef8ae02a675bacb6acbbf3dea09c817686a7629e41508edb47`，11,995 字节，164 个算子节点，ONNX opset 18 / IR 10）
- **核心控制器构建器**：[`tools/build_controller.py`](./contest-20261125/challenge-9668a3f8/tools/build_controller.py)
- **完整验收报告**：[`evaluation_report.json`](./contest-20261125/challenge-9668a3f8/evaluation_report.json)
- **标准化 Agent 执行轨迹**：[`submitted_bundle/traces/trace.jsonl`](./contest-20261125/challenge-9668a3f8/submitted_bundle/traces/trace.jsonl)（共 433 步完整工具调用与思考记录）

#### 1. 问题症结与 200/400 粗细双网格 `1.25 MW` 系统偏差诊断

本题基于 TORAX 一维托卡马克等离子体输运求解器与 QLKNN 回旋动理学湍流代理模型，要求设计一个纯因果、无外部副作用的静态 ONNX 闭环控制器（输入 16 维带噪延迟物理观测 + 32 维内部状态记忆共 `48` 维张量，每 $0.5\text{ s}$ 控制窗口输出主加热通道 $P_G \in [0, 33]\text{ MW}$、电子回旋辅助加热 $P_E \in [0, 20]\text{ MW}$ 及更新后的 32 维记忆状态共 `34` 维张量，且相邻窗口功率变化率受严格斜率限制 $|\Delta P| \le 2.0\text{ MW/step}$）。

每个测试场景（含不同的杂质注入突发时刻 $t_{\text{event}} \in [8, 12]\text{ s}$、事件后有效电荷数 $Z_{\text{eff}}$、初始温度乘子及传感器噪声种子）均同时在 **`coarse`（6,000 步 / 200 网格尺度）**与 **`fine`（12,000 步 / 400 网格尺度）**两套时空离散分辨率下独立仿真 $30\text{ s}$：
- **官方基线死穴**：官方参考基线控制器在 `coarse` 网格下因杂质注入后的辐射冷却导致聚变功率跌落，产生 `fusion_energy_deficit_fraction = 0.66144` 的聚变能量缺口违规（`min_Q_ratio = 0.98677 < 1.0`），`feasible = False`，本地基线得分仅为 **`0.3228227`**（参考供给加热总能量 $J_{\text{ref}} = 1.31266678\times 10^9\text{ J}$）。
- **核心发现——粗细双网格 `1.25 MW` 系统性聚变功率偏差**：智能体在双网格轨迹对齐实验中发现，由于 QLKNN 湍流刚度在粗细径向/时间网格下的数值耗散差异，相同外部加热波形在 `coarse` 网格下产生的聚变功率系统性低于 `fine` 网格约 **`1.25 MW`**，且两者在前 $7.5\text{ s}$ 的安全因子下界 $q_{\min}$ 呈现出稳定的可分界特征（`coarse` 网格 $q_{\min} \approx 1.83377$，`fine` 网格 $q_{\min} \approx 1.82798$，分界中点 $q_{\text{mid}} = 1.8308736$）。
- **台基正能量惩罚陷阱（`E_pedestal_positive_J`）**：一旦外部加热功率下降过快或等离子体储能 $W$ 在中段过度滑落，边界台基区会出现正补偿加热需求，直接触发硬约束违规或推高总供给热能上界 $J_{\text{upper}}$。

#### 2. 自主求解与因果记忆状态机设计

基于上述第一性原理诊断，智能体在 [`tools/build_controller.py`](./contest-20261125/challenge-9668a3f8/tools/build_controller.py) 中设计了融合**粗细网格在线辨识、前段聚变储备、中段低电子加热巡航与末端储能守护**的 164 节点静态 ONNX 状态机：

1. **前 $7.5\text{ s}$ 在线分辨率辨识与双分支平滑插值**
   - 利用内部记忆单元 `m[12]`、`m[13]` 在 $t < 7.5\text{ s}$ 窗口内累加观测到的 $q_{\min} - q_{\text{mid}}$（其中 $q_{\text{mid}} = 1.8308736$），通过高增益 Sigmoid 门控 $\sigma(40 \cdot \overline{\Delta q})$ 在线生成连续分辨率权重 `curve_gain`，在严格满足因果律与 ONNX 静态图约束下，实现 `coarse_commands_MW` 与 `fine_commands_MW` 两条 60 步最优前馈基线轨的自动对齐，将双网格供给热能差压缩至 $8.57\text{ MJ}$ 以内。
2. **前段满功率聚变储备（Early Reserve，$t \in [0, 7\text{ s}]$）**
   - 在启动前 2 步以最大合规斜率（$\Delta P = 1.999\text{ MW/step}$）将 $(P_G, P_E)$ 从初始 $(28, 15)\text{ MW}$ 推升至满功率 $(33.0, 20.0)\text{ MW}$ 并维持至 $t=7.0\text{ s}$。趁杂质突发事件（$t_{\text{event}} > 8.0\text{ s}$）尚未发生、等离子体纯净度最高时，以最高热核转化效率提前攒下充足的聚变能量盈余，彻底杜绝后续降功率期间的聚变配额亏空。
3. **中后段低电子回旋加热维持（Low-EC Sustainment，$t \in [7, 27\text{ s}]$）**
   - 自 $t=7.0\text{ s}$ 起，按每步 $-1.9\text{ MW}$ 的平滑斜率逐步削减昂贵的电子回旋加热 $P_E$：在 `coarse` 网格下维持 $P_E = 5.0\text{ MW}$（并在 $t \in [9.5, 12.5\text{ s}]$ 短暂回收 $P_G$ 至 $29.0\text{ MW}$ 以压制杂质注入瞬间的辐射-湍流耦合波动），在 `fine` 网格下进一步压低至 $P_E = 3.5\text{ MW}$，同时保持主加热 $P_G = 33.0\text{ MW}$ 托底。这一非对称配比使整段放电的 `E_pedestal_positive_J` 严格保持为 **`0.0 J`**。
4. **延迟匹配梯形积分 PI 能量闭环与末端 $W_{\text{target}}$ 守护（$t \in [16, 30\text{ s}]$）**
   - 为消除传感器延迟造成的启动求积偏差，控制器在记忆单元中对实测聚变功率 $p(t)$ 与延迟后的参考目标功率 $p_{\text{ref}}(t-\tau)$ 同步执行匹配梯形积分，计算实时聚变能量欠账 `deficit` 并叠加限定幅值（$[-0.02, +0.10]\text{ MW}$）的 PI 闭环微调；在最后 $3\text{ s}$（$t \ge 27.0\text{ s}$）启动针对终端储能目标 $W_{\text{target}} = 182.8\text{ MJ}$ 的柔性回补斜坡，确保 $t=30\text{ s}$ 时的终端储能比稳居 $0.984$ 以上（硬约束 $\ge 0.980$）。

#### 3. 最终官方双网格完整验收账本

经本地原样物理引擎对 **4 个开发集场景（8 条粗细网格轨迹，72,000 个原生接受时间步）** 与 **25 个全量测试场景（50 条粗细网格轨迹，450,000 个原生接受时间步）** 的完整闭环回放验证，全部场景 **100% 零违规通过（`feasible = True`）**：

| 验收维度 | 评估指标名称 (`evaluation_report.json`) | 官方硬约束 / 参考基准 | 官方参考起点 (`baseline`) | 本仓 `controller.onnx`（4 场景开发集） | 本仓 `controller.onnx`（25 场景全量集） |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **综合可行性与得分** | **本地综合验收得分 (`score`)** | $> 0.75$（超越参考值） | `0.3228227` (`feasible=False`) | **`0.7520522`** (`feasible=True`) | **全部 25 场景 `feasible=True`** |
| **核心节能目标** | **最坏场景供给总热能 (`worst_supplied_heat_J`)** | $J_{\text{ref}} = 1.312667\times 10^9\text{ J}$ | $1.312667\times 10^9\text{ J}$ | **`1.291292e+09 J`（省 `21.37 MJ`）** | **`1.291370e+09 J`（省 `21.30 MJ`）** |
| **聚变能量配额** | **最小聚变能量达标比 (`minimum_fusion_energy_ratio`)** | $\ge 1.000000$ | ❌ `0.986771`（缺口 `0.6614`） | **`1.003966`**（零缺口） | **`1.000312`**（零缺口） |
| **终端热能保持** | 最小终端储能比 (`minimum_terminal_W_ratio`) | $\ge 0.980000$ | `0.994334` | **`0.984509`** | **`0.983724`** |
| **聚变功率跟踪精度** | 聚变功率最大相对 RMS 误差 (`maximum_fusion_rms_relative`) | 运行平稳性指标 | `0.019814` | **`0.022967`** | **`0.028562`** |
| **聚变功率峰值波动** | 聚变功率最大相对峰值偏差 (`maximum_fusion_peak_relative`) | 运行平稳性指标 | `0.044581` | **`0.048054`** | **`0.052076`** |
| **MHD 与热负荷边界** | 最大归一化比压 $\beta_{N,\max}$ / 峰值温度 $T_{\max}$ | 越界即判负 | $1.4067$ / $16.64\text{ keV}$ | **`1.4067`** / **`17.71 keV`** | **`1.4171`** / **`17.71 keV`** |
| **台基正加热惩罚** | 台基正补偿能量上界 (`maximum_positive_pedestal_heat_J`) | $= 0.0\text{ J}$ | `0.0 J` | **`0.0 J`** | **`0.0 J`** |
| **双网格一致性** | 粗细网格最大热能差 / 最大功率差 | 越小越鲁棒 | — | **`8.57 MJ`** / **`3.26 MW`** | **`8.57 MJ`** / **`3.37 MW`** |

#### 4. 一键复现命令

```bash
cd contest-20261125/challenge-9668a3f8

# 1. 从控制器参数配置构建标准 ONNX opset 18 静态计算图模型并执行 onnx.checker 严格校验
./.venv/bin/python tools/build_controller.py \
  --config runs/early_reserve.json \
  --output controller.onnx

# 2. 运行双网格（coarse 200 / fine 400）多场景完整物理闭环回放与报告生成
./.venv/bin/python tools/finish_validation.py
```

---

### 3.3 `challenge-d2b1bd0c`：《提高托卡马克的稳定压力，设计聚变等离子体形状》

- **赛题目录**：[`contest-20261125/challenge-d2b1bd0c/`](./contest-20261125/challenge-d2b1bd0c/)
- **平台提交编号**：**`attempt_id: 111942`**（[`submission_receipt.json`](./contest-20261125/challenge-d2b1bd0c/submission_receipt.json)）
- **核心交付物**：[`plasma.json`](./contest-20261125/challenge-d2b1bd0c/plasma.json)（SHA-256: `da8730662883d8f8e5ed7b4f51de35fa2f7103b410814ae9f383b04468829acd`）
- **核心求解器**：[`solve_plasma.py`](./contest-20261125/challenge-d2b1bd0c/solve_plasma.py)
- **正式验收报告**：[`best_formal.json`](./contest-20261125/challenge-d2b1bd0c/best_formal.json)、[`runs/running_state.json`](./contest-20261125/challenge-d2b1bd0c/runs/running_state.json)、[`submission_outputs/REPORT.md`](./contest-20261125/challenge-d2b1bd0c/submission_outputs/REPORT.md)
- **标准化 Agent 执行轨迹**：[`submitted_bundle/traces/trace.jsonl`](./contest-20261125/challenge-d2b1bd0c/submitted_bundle/traces/trace.jsonl)（共 807 步完整工具调用与思考记录）

#### 1. 问题症结与第一性原理 MHD 稳定性边界诊断

本题要求在固定等离子体体积 $V = 240\text{ m}^3$、大半径 $R_0 = 4.0\text{ m}$、参考环向场 $B_0 = 7.0\text{ T}$ 与总等离子体电流 $I_p = 8.1\text{ MA}$ 的约束下，联合优化 **7 个连续参数**——截面伸长率 $\kappa$、三角形变 $\delta$、4 个高阶 Fourier 边界修形系数 $c_2, c_3, s_2, s_3$ 以及磁轴中心压力 $p_0 \in [10^5, 1.4\times 10^6]\text{ Pa}$。

官方验证器 `workspace/physics/eval.py` 在 `--mode formal` 下对每个候选设计执行极其严苛的 **6 平衡 $\times$ 全模数理想 MHD 稳定性联合裁决**：
- 针对 **`broad`、`nominal`、`peaked`** 三种不同空心度（`ffprime_hollow`）的电流剖面，分别在 **`coarse`（$129\times 257$）**与 **`fine`（$257\times 513$）**两档网格下调用 **TokaMaker** 求解自由边界/固定体积 Grad–Shafranov 平衡；
- 每个平衡必须同时满足几何包络与径向半宽 $a_{\text{minor}} \le 1.50\text{ m}$、安全因子下界 $q_{\min} \ge 1.10$、Grad–Shafranov 残差上限，并在每个平衡上调用 **GPEC/DCON** 检验环向模数 $n = 1, 2, 3, 4, 5$ 及多组环向相位下的理想扭曲模最小总能量 $\delta W_{\min} \ge 0.02$、Mercier 判据 $(-D_I)_{\min} \ge 0.01$（即 $\max D_I \le -0.01$）、内部零交叉数 `internal_zero_crossings = 0` 以及全径向面高模数气球模积分节点数 `ballooning_nodes_max = 0`。
- **得分机制**：在全部 6 个平衡、90 次 DCON 模分析与 19,674 次气球模积分 100% 达标的前提下，取三剖面双网格中最坏体积平均压力 $\bar{p}_{\min}$，按
  $$S = \frac{\bar{p}_{\min}}{\bar{p}_{\min} + 200\text{ kPa}}$$
  计算最终得分。官方起点（$p_0 = 400\text{ kPa}$）虽然可行，但最坏体积平均压力仅为 $\bar{p}_{\min} = 97.759\text{ kPa}$，对应基线得分仅为 **`0.3283161`**。

#### 2. 自主求解与固定体积径向包络流形投影

在 [`submitted_bundle/traces/trace.jsonl`](./contest-20261125/challenge-d2b1bd0c/submitted_bundle/traces/trace.jsonl) 的 807 步探索中，智能体首先完成了 ARM64 架构下 `OpenFUSIONToolkit (TokaMaker)` 与 `GPEC/dcon` 的原生编译与参考 BLAS 确定性校验（自动识别并隔离了 OpenBLAS 多线程数值扰动窗口），随后在 [`solve_plasma.py`](./contest-20261125/challenge-d2b1bd0c/solve_plasma.py) 中实施了三项关键物理降维：

1. **解析推导固定体积与径向半宽包络的线性流形投影（`project_radial_envelope`）**
   - 智能体从环向旋转体体积积分公式出发，注意到当小半径尺度 $a$ 与形状系数 $(\delta, c_2, c_3, s_2, s_3)$ 固定时，环向体积 $V$ 关于垂直伸长率 $\kappa$ 是**严格线性**的！
   - 正常人或普通搜索若在 $(\kappa, \delta, c_2, c_3, s_2, s_3)$ 空间盲目采样，由于固定 $V=240\text{ m}^3$ 会反推缩放小半径 $a \propto \kappa^{-1/2}$：一旦 $\kappa$ 偏大（如 $\kappa = 1.76$），虽然截面积拉长，但压扁后的平均压力形状因子反而使 $\bar{p}$ 停留在 $316.9\text{ kPa}$；一旦 $\kappa$ 偏小，又会因 $a > 1.50\text{ m}$ 触发径向包络越界。
   - 智能体直接将径向尺度钉死在合规上限边缘 $a \approx 1.476\text{ m}$（留出安全余量）， discovered **低伸长率（$\kappa = 1.48$）+ 强负方形度/高阶内凹修形（$c_2 \in [-0.0625, -0.05], c_3 = -0.025, s_2 \in [-0.10, -0.09], s_3 = -0.08$）**能够最大化固定体积下的磁通面填充因子（平均压力与中心压力比 $\bar{p}/p_0$ 从 $24.4\%$ 跃升至 **$23.79\%$** 的高压极限形态），同时保持 $q_{\min} > 1.110$！
2. **磁轴中心压力直推 $1.399\text{ MPa}$ 与 `broad` 剖面 $n=1$ 能量裕度锁定**
   - 全空间探测表明，限制压力上限的真正瓶颈不是高 $n$ 气球模（在 $\kappa=1.48, \delta=0.32$ 下 $n=1..5$ 气球模节点恒为 `0`，Mercier 裕度高达 $0.241$），而是 `broad` 电流剖面下的 **$q_{\min} \ge 1.10$ 门槛**（实测裕度约 `+0.0103`）与 **$n=1$ 理想扭曲模最小能量 $\delta W \ge 0.02$**（实测 $\delta W = 0.06215$，裕度 `+0.0422`）。
   - 智能体通过 `equilibrium_probe`（单点仅需 `14 s`）快速过滤 $q_0$ 越界点，再通过粗网格 `(n=1, n=5)` 筛查与坐标爬山，将磁轴中心压力一路推至协议上限 $1.40\text{ MPa}$ 紧邻处的 **`1.399 MPa`（`1,399,000 Pa`）**。

#### 3. 最终官方 `--mode formal` 六平衡全模数完整验收账本

最终锁定并验证了两个完全通过官方 `--mode formal` 双网格全模数检验的高压姐妹解（`shape330-formal` 得分 **`0.6234045`**，$\bar{p}_{\min}=331.07\text{ kPa}$；精修高阶项 $c_2=-0.0625, s_2=-0.09$ 后的最终交付解 [`plasma.json`](./contest-20261125/challenge-d2b1bd0c/plasma.json) `top333-formal` 在 `--mode search` 与 `--mode formal` 下分别取得 **`0.6246041`** 与 **`0.62460406`**，$\bar{p}_{\min}=332.77\text{ kPa}$）。以下为 [`plasma.json`](./contest-20261125/challenge-d2b1bd0c/plasma.json)（$\kappa=1.48, \delta=0.32, c_2=-0.0625, c_3=-0.025, s_2=-0.09, s_3=-0.08, p_0=1.399\text{ MPa}$）在全部 6 个工况下的完整验收账本：

| 电流剖面与网格工况 | 体积平均压力 $\bar{p}$ (`Pa`) | 安全因子下界 $q_{\min}$ ($\ge 1.10$) | $n=1..5$ 含相位最小总能量 $\delta W_{\min}$ ($\ge 0.02$) | Mercier 判据最大值 $\max D_I$ ($\le -0.01$) | 气球模最大节点数 (`= 0`) | 内部零交叉数 (`= 0`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`broad / coarse`** ($129\times 257$) | **`332,770.88 Pa`** | `1.11029574` | `0.06239153` | `-0.25115877` | `0` | `0` |
| **`broad / fine`** ($257\times 513$) | **`332,770.82 Pa`** | `1.11031539` | `0.06215456` | `-0.25115935` | `0` | `0` |
| **`nominal / coarse`** ($129\times 257$) | **`333,001.74 Pa`** | `1.11249330` | `0.06439533` | `-0.25115983` | `0` | `0` |
| **`nominal / fine`** ($257\times 513$) | **`333,001.68 Pa`** | `1.11251436` | `0.06405233` | `-0.25116011` | `0` | `0` |
| **`peaked / coarse`** ($129\times 257$) | **`333,309.65 Pa`** | `1.11542901` | `0.06712701` | `-0.25116079` | `0` | `0` |
| **`peaked / fine`** ($257\times 513$) | **`333,309.58 Pa`** | `1.11545194` | `0.06683116` | `-0.25116130` | `0` | `0` |
| **全局最坏工况汇总 (`feasible=True`)** | **`332.77 kPa`（较基线 `97.76 kPa` 提升 `3.40 倍`）** | **`1.11030`**（达标） | **`0.06215`**（达标） | **`-0.25116`**（达标） | **`0`**（达标） | **`0`**（达标） |

#### 4. 一键复现命令

```bash
cd contest-20261125/challenge-d2b1bd0c
source env.sh

# 1. 运行 7 维受约束形状与压力联合寻优器
taskset -c 4-7 ./.venv/bin/python solve_plasma.py --local

# 2. 调用未修改的官方 eval.py 执行 --mode formal 双网格 6 平衡 90 模态全量验收
taskset -c 4-7 ./.venv/bin/python -I workspace/physics/eval.py plasma.json \
  --binary native/GPEC/bin/dcon \
  --workdir /tmp/recheck-d2b1bd0c-formal \
  --mode formal
```

---

### 3.4 `challenge-b946f0c9`：《为聚变补充燃料，用有限的包层材料兼顾产氚与局部核热》

- **赛题目录**：[`contest-20261125/challenge-b946f0c9/`](./contest-20261125/challenge-b946f0c9/)
- **平台提交编号与官方终审分**：**`attempt_id: 111893`**，**云端 Harbor 官方终审得分 `77.1153 / 100`**（`harbor_reward = 0.77115348`，`trace_factor = 1.0`，见 [`platform_result.json`](./contest-20261125/challenge-b946f0c9/platform_result.json)）
- **核心交付物**：[`blanket.json`](./contest-20261125/challenge-b946f0c9/blanket.json)（SHA-256: `484776003b3fdbbba15e3004767ca2bbd1a41fda85c7e8afd6c9e6fe83de4247`）
- **核心求解器**：[`solve_blanket.py`](./contest-20261125/challenge-b946f0c9/solve_blanket.py)
- **完整验收报告**：[`platform_result.json`](./contest-20261125/challenge-b946f0c9/platform_result.json)、[`submission_outputs/reward.json`](./contest-20261125/challenge-b946f0c9/submission_outputs/reward.json)、[`submission_outputs/details.json`](./contest-20261125/challenge-b946f0c9/submission_outputs/details.json)
- **标准化 Agent 执行轨迹**：[`submitted_bundle/traces/trace.jsonl`](./contest-20261125/challenge-b946f0c9/submitted_bundle/traces/trace.jsonl)（共 599 步完整工具调用与思考记录）

#### 1. 问题症结与中子输运/库存守恒耦合机制剖析

本题研究含中央偏置诊断通道（位于横向 $(y,z) \approx (13, 0)\text{ cm}$）与氦冷却槽的三维聚变增殖包层模块。包层在径向（深度 $l=0..5$ 共 6 层）与横向（$4\times 4 = 16$ 个柱状扇区）被划分为 **$6 \times 16 = 96$ 个独立材料区**。每个区域 $i$ 需独立指定：
1. $\text{Li}_4\text{SiO}_4$ 增殖陶瓷体积分数 $b_i \in [0.20, 0.60]$（剩余体积由中子倍增剂金属铍 $\text{Be}$ 填充，故 $V_{\text{Be},i} \propto (1 - b_i)V_i$）；
2. 锂元素中的 $^6\text{Li}$ 原子丰度 $e_i \in [0.05, 0.90]$（共 **$96 \times 2 = 192$ 个连续自由度**）。

官方验证器在 **`normal`（准直法向入射）**与 **`cosine`（余弦角分布入射）**两种 $14.1\text{ MeV}$ 聚变中子源分布下，各用两个独立随机数种子（`2026100801`、`2026100802`）运行 $64\text{ batches} \times 300,000\text{ particles/batch} = 1.92\times 10^7$ 中子（单次全量评估共计 **7,680 万连续能量蒙特卡洛中子历史**），考核以下硬约束与目标：
- **三项全局材料库存硬上限**：增殖陶瓷总体积 $\sum V_i b_i \le 143,016.0\text{ cm}^3$（等价于全包层体积加权平均 $\bar{b} \le 0.40$）、金属铍总体积 $\le 125,139.0\text{ cm}^3$（恰迫使 $\bar{b} \equiv 0.40$，即**陶瓷与铍体积总库存必须双向精确卡满**）、$^6\text{Li}$ 总原子数 $\le 1.279017\times 10^{27}$（等价于 $\sum V_i b_i e_i / \sum V_i \le 0.12$）；
- **32 项背面快中子通量/局部核热硬上限**：在 `normal` 与 `cosine` 双源下，包层最深处背面 16 个分区（`rear0`..`rear15`）的统计上界（均值 $+ 3.2514\times\text{SE}$）均不得超过各自阈值；
- **最坏源分布下的保守产氚率下界目标（`objective_lower`）**：取 `normal` 与 `cosine` 两源产氚率统计下界（均值 $- 3.2514\times\text{SE}$）的最小值。官方均匀基线 `uniform.json`（处处 $b_i=0.40, e_i=0.30$）受限于 `cosine` 源的大倾角泄漏与前层倍增不足，`objective_lower` 仅为 **`0.24938196`**（对应基准分 `0.750000` / 百分制 `75.00 分`）。

#### 2. 自主求解：精确库存流形二分投影与纵深-横向模态分解

在 [`solve_blanket.py`](./contest-20261125/challenge-b946f0c9/solve_blanket.py) 中，智能体摒弃了在 192 维空间盲目加罚函数的低效做法，直接从核工程中子学第一性原理构建了**库存恒等守恒参数化与主动学习代理寻优架构**：

1. **双层二分投影算子（`project`）锁死库存等式流形**
   - 由于陶瓷体积上限（$\bar{b} \le 0.40$）与铍体积上限（$1-\bar{b} \le 1-0.40$）共同将 $\sum V_i b_i = 0.40 \sum V_i$ 锁死在零测度超平面上，任何偏离都会立即触发库存违规。
   - 智能体构造了 80 步双层精确二分裁剪投影算子：先求解平移量 $\lambda_b$ 使 $\sum V_i \text{clip}(b_i + \lambda_b, 0.2, 0.6) = 0.40 \sum V_i$ 达到机器精度（$10^{-12}$），再以更新后的区域陶瓷体积 $w_i = V_i b_i$ 为权重，求解 $\lambda_e$ 使 $\sum w_i \text{clip}(e_i + \lambda_e, 0.05, 0.90) = 0.12 (1 - 2\times 10^{-12})\sum V_i$。**由此生成的每一个候选解在构造上 100% 满足三项库存硬约束！**
2. **中子倍增纵深梯度 + 边缘/诊断通道横向补偿的 16 维物理基展开（`make_design`）**
   - 从中子物理来看，$14.1\text{ MeV}$ 快中子在刚进入前层（$l=0,1$）时能量高于 $\text{Be}(n,2n)$ 反应阈值（$\sim 1.9\text{ MeV}$），此时应**压低前层陶瓷占比 $b$（提高金属铍 $\text{Be}$ 占比以最大化 $(n,2n)$ 快中子倍增）**，同时在中间慢化层与后层拉高陶瓷占比 $b$ 与 $^6\text{Li}$ 丰度 $e$，利用 $^6\text{Li}(n,\alpha)\text{T}$ 在热能/超热能区间的巨大截面捕获倍增后的慢化中子，并在最深层形成屏蔽墙压低背面 16 区快中子泄漏。
   - 针对 `cosine` 源在模块四周边缘（$|y|+|z|$ 较大处）及中央偏置诊断孔道（$(y-13)^2+z^2$ 邻域）的中子流畸变，智能体引入了 6 层深度基底 + 4 个横向空间模态（边缘距离模态 `edge` $\times$ 深度衰减、诊断孔道高斯模态 `port` $\times$ 深度衰减），将 192 维自由度降维至物理意义极其清晰的 **16 维核心参数空间**。
3. **多保真度原生 OpenMC 搜索与全量 7,680 万中子终审**
   - 搜索阶段直接调用原生 OpenMC（先以 `cosine` 瓶颈源 $8\times 50,000 = 40\text{ 万}$ 中子配合高斯过程期望提升 EI 探索，再升级至双源高统计量精修），拒绝任何失真的扩散近似；最终设计通过本地 7,680 万中子全量验证后打包提交平台，并获 Harbor 云端独立重放确认。

#### 3. 最终本地与云端 Harbor 官方终审验收账本

| 验收维度 | 评估指标名称 (`details.json` / `platform_result.json`) | 官方硬约束阈值 / 基准 | 官方均匀基线 (`uniform.json`) | 本仓最终解 [`blanket.json`](./contest-20261125/challenge-b946f0c9/blanket.json) | 状态与提升幅度 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **云端官方终审分** | **Bohrium Harbor 官方终审得分 (`platform_score`)** | 满分 `100.0`（基线 `75.00`） | `75.0000` | **`77.1153 / 100`** (`trace_factor=1.0`) | ✅ **云端独立重放终审通过 (`+2.12 分`)** |
| **本地全量验收分** | 本地原样 `eval.py` 综合奖励 (`local_reward`) | $> 0.750000$ | `0.7500000` | **`0.7712608`** (`feasible=True`) | ✅ **本地与云端双端高度一致** |
| **核心产氚指标** | **双源最坏保守产氚下界 (`objective_lower`)** | 越大越好（基线 `0.249382`） | `0.24938196` | **`0.29574104`** | **净产氚率大幅提升 `+18.59%`** |
| **余弦源产氚表现** | `cosine` 源产氚率均值 $\pm$ 半宽 (`outcomes.cosine`) | 决定全局下界瓶颈 | `0.24975` | **`0.2961131 ± 0.0003721`** | ✅ 128 批次正产率占比 `100%` |
| **法向源产氚表现** | `normal` 源产氚率均值 $\pm$ 半宽 (`outcomes.normal`) | 高穿透源产氚能力 | `0.3015` | **`0.3632935 ± 0.0004185`** | ✅ 128 批次正产率占比 `100%` |
| **陶瓷总库存约束** | $\text{Li}_4\text{SiO}_4$ 增殖剂总体积 (`breeder_volume_cm3`) | $\le 143,016.0\text{ cm}^3$ | $143,016.0\text{ cm}^3$ | **`143,016.00 cm³`**（违规量 `0.0`） | ✅ 精确卡满库存上限 |
| **倍增剂总库存约束** | 金属铍 $\text{Be}$ 总体积 (`be_volume_cm3`) | $\le 125,139.0\text{ cm}^3$ | $125,139.0\text{ cm}^3$ | **`125,139.00 cm³`**（违规量 `0.0`） | ✅ 精确卡满倍增剂上限 |
| **$^6\text{Li}$ 总库存约束** | $^6\text{Li}$ 总原子数 (`li6_atoms`) | $\le 1.27901711\times 10^{27}$ | $1.279017\times 10^{27}$ | **`1.27901711e+27`**（违规量 `0.0`） | ✅ 100% 释放同位素预算 |
| **背面核热与通量** | `normal` / `cosine` 背面 16 区快中子通量上界 | 32 项分区各自上限 | 全部合规 | **32 项违规量全部为 `0.0`** | ✅ 最小安全裕度 $> 4.30\times 10^{-8}$ |

#### 4. 一键复现命令

```bash
cd contest-20261125/challenge-b946f0c9

# 1. 运行基于精确库存流形投影与原生 OpenMC 的包层材料分布优化器
taskset -c 8-11 ./runtime/native-env/bin/python solve_blanket.py

# 2. 运行完整 7,680 万中子（双源 × 双种子 × 64 batches × 300,000 particles）官方协议验收
taskset -c 8-11 ./runtime/native-env/bin/python local_eval.py blanket.json
```

---

### 3.5 `challenge-5f1ea274-3`：《用三维形状留住聚变粒子：寻找更紧凑的仿星器》

- **赛题目录**：[`contest-20261125/challenge-5f1ea274-3/`](./contest-20261125/challenge-5f1ea274-3/)
- **平台提交编号**：**`attempt_id: 111815`（全网首提）**（[`first_submission.json`](./contest-20261125/challenge-5f1ea274-3/first_submission.json)）
- **核心交付物**：[`boundary.json`](./contest-20261125/challenge-5f1ea274-3/boundary.json)（以及候选集 [`candidate_boundary.json`](./contest-20261125/challenge-5f1ea274-3/candidate_boundary.json)）
- **核心求解器**：[`solve_boundary.py`](./contest-20261125/challenge-5f1ea274-3/solve_boundary.py)
- **阶段验收报告**：[`submission_outputs/stage_report.json`](./contest-20261125/challenge-5f1ea274-3/submission_outputs/stage_report.json)、[`optimization_results.json`](./contest-20261125/challenge-5f1ea274-3/optimization_results.json)
- **标准化 Agent 执行轨迹**：[`submitted_bundle/traces/trace.jsonl`](./contest-20261125/challenge-5f1ea274-3/submitted_bundle/traces/trace.jsonl)（共 243 步完整工具调用与思考记录）

#### 1. 问题症结与 5 个公开参考构型的筛选真相

本题要求设计一个更紧凑的三维仿星器最外层磁面边界（由 VMEC Fourier 系数 `rbc`, `zbs` 表征，评测时自动缩放到固定小半径 $a = 1.70\text{ m}$ 与磁轴场强 $B_{00} = 5.70\text{ T}$），在满足以下 **6 项严苛物理与工程约束**的前提下，极小化装置纵横比 $A = R_{\text{major}} / a_{\text{minor}}$（按 $S = 0.5 + 0.5 / (1 + e^{A - 4})$ 计分）：
1. 截面最大伸长率 $\kappa_{\max} \le 8.0$ 且无截面自交；
2. 旋转变换下界 $\iota_{\min} \ge 0.30$；
3. 磁阱深度 $W \ge 0$（理想 MHD 交换模稳定必要条件）；
4. 线圈退让磁场梯度特征尺度 $L_{\nabla B,\min} \ge 2.40\text{ m}$；
5. `NEO` 五个径向磁面上的新经典输运有效波纹度 $\varepsilon_{\text{eff},\max} \le 0.01$（`1%`）；
6. `SIMPLE` 追踪 $s=0.25$ 磁面上 4,096 个 $3.5\text{ MeV}$ 聚变 $\alpha$ 粒子达 $0.2\text{ s}$ 的损失率上界 $f_\alpha \le 0.05$（`5%`）。

智能体首先在 ARM64 环境下从源码完整构建了 `VMEC2000`、`STELLOPT/NEO`、`SIMPLE`（开启 `SIMPLE_DETERMINISTIC_FP=ON`）与 `booz_xform`，并对官方提供的全部 5 个参考构型执行了系统摸底筛查（见 [`stage_report.json`](./contest-20261125/challenge-5f1ea274-3/submission_outputs/stage_report.json)）：
- **`ESTELL.json`**（$A=5.293$）：旋转变换 $\iota_{\min} = 0.2053 < 0.30$ 且为磁山（$W = -0.0193 < 0$），**不达标**；
- **`QHS46.json`**（$A=6.698$）：磁山（$W = -0.00149 < 0$）且 Boozer 准对称残差高达 `0.0562`，**不达标**；
- **`landreman_paul_2022_QA.json`**（$A=6.000$）：无磁阱（$W = -0.03125 < 0$），**不达标**；
- **`landreman_paul_2022_QH_well.json`**（$A=8.000$）：虽全部达标，但纵横比高达 $8.0$，得分仅 `0.50899`；
- **`landreman_paul_2022_QA_well.json`**（$A=6.00001$）：**是 5 个公开构型中唯一全部达标的最紧凑起点**（$W = +0.002167 > 0$，$\iota_{\min}=0.4169$，$L_{\nabla B}=5.55\text{ m}$，基线得分 **`0.559601`**）。

#### 2. 自主求解：保持 QA/磁阱符号耦合的四维协调形变算子（`deform`）

直接逐个扰动数十个高阶 Fourier 模态极易破坏准轴对称（QA）与磁阱（$W>0$）之间精密的二阶几何抵消关系。智能体在 [`solve_boundary.py`](./contest-20261125/challenge-5f1ea274-3/solve_boundary.py) 中设计了**保持种子构型模态相位与符号比的四维协调形变算子** $\mathbf{x} = (\alpha_{\text{thick}}, \alpha_{\text{axis}}, \alpha_{\text{elong}}, \alpha_{\text{shaping}})$：
$$\begin{aligned}
\text{rbc}_{m,n}(\mathbf{x}) &= \text{rbc}_{m,n}^{(0)} \cdot \begin{cases} \alpha_{\text{axis}}, & m = 0, n \ne 0 \\ \alpha_{\text{thick}} \cdot \alpha_{\text{elong}}^{-1/2}, & m = 1 \\ \alpha_{\text{thick}} \cdot \alpha_{\text{shaping}} \cdot \alpha_{\text{elong}}^{-1/2}, & m \ge 2 \end{cases} \\
\text{zbs}_{m,n}(\mathbf{x}) &= \text{zbs}_{m,n}^{(0)} \cdot \begin{cases} \alpha_{\text{axis}}, & m = 0, n \ne 0 \\ \alpha_{\text{thick}} \cdot \alpha_{\text{elong}}^{+1/2}, & m = 1 \\ \alpha_{\text{thick}} \cdot \alpha_{\text{shaping}} \cdot \alpha_{\text{elong}}^{+1/2}, & m \ge 2 \end{cases}
\end{aligned}$$
- **物理机制**：总体增厚因子 $\alpha_{\text{thick}} > 1$ 在固定大半径 $R_{0,0}$ 下线性放大截面小半径 $a$，从而直接压低纵横比 $A \approx A_0 / \alpha_{\text{thick}}$；但单纯放大 $\alpha_{\text{thick}}$（如 $\alpha_{\text{thick}}=1.05$）会导致环向曲率加深，使磁阱 $W$ 在 $A \approx 5.716$ 处由正转负（$W = -3.46\times 10^{-4}$）。
- 通过引入**磁轴三维波纹度 $\alpha_{\text{axis}}$、椭圆伸长率 $\alpha_{\text{elong}}$ 与高阶三角形/方形整形系数 $\alpha_{\text{shaping}}$** 的协同补偿，并采用 **COBYLA 约束优化器**在 $W \ge 3\times 10^{-4}$、$\iota_{\min} \ge 0.32$、$L_{\nabla B} \ge 2.6\text{ m}$、$\text{QS}_{\text{RMS}} \le 0.003$ 的安全内缩约束下寻优，成功跨越了单纯缩放的磁阱崩溃壁垒，将纵横比一路从 $6.00001$ 压降至 **$A = 5.54297$（压缩 `7.60%`）**，同时储备了 $A = 5.67670$ 与 $A = 5.82638$ 构成的稳健候选梯队。

#### 3. 核心物理指标对比与阶段验收账本

| 评估指标名称 (`stage_report.json`) | 官方硬约束阈值 | 公开最优可行起点 (`QA_well`) | 梯队候选 1 (`opt_973788a6f325`) | 首提紧凑构型 (`opt_397f76eabadd`, `attempt_id: 111815`) | 状态与安全余量 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **装置纵横比 (`aspect_ratio`, $A$)** | 越小分越高 | `6.000012` | **`5.826385` (`-2.89%`)** | **`5.542969`（较起点压缩 `7.60%`）** | ✅ 理论对应得分 **`0.5880`** (`+0.0284`) |
| **NEO 五磁面最大有效波纹 ($\varepsilon_{\text{eff},\max}$)** | $\le 0.010000$ (`1%`) | $5.2\times 10^{-5}$ | — | **`7.0858e-05` (`0.00709%`)** | ✅ **比阈值低两个数量级（余量 `99.3%`）** |
| **Boozer 准对称残差 (`qs_rms`)** | 筛查代理 $\le 0.02$ | `0.000232` | `0.001917` | **`0.003097`** | ✅ 准轴对称结构保持良好 |
| **磁阱深度 (`well`, $W$)** | $\ge 0.0$ | `+0.002167` | `+0.000701` | **`+0.000293`** (最坏面 $s=0.9375$) | ✅ 全径向保持正磁阱（$W > 0$） |
| **最小旋转变换 ($\iota_{\min}$)** | $\ge 0.300000$ | `0.416921` | `0.416003` | **`0.428473`** (轴侧 `0.4285` / 边侧 `0.4358`) | ✅ 远离低阶有理面（余量 `+42.8%`） |
| **线圈退让梯度尺度 ($L_{\nabla B,\min}$)** | $\ge 2.400000\text{ m}$ | `5.552801 m` | `5.382301 m` | **`5.024388 m`** | ✅ 线圈工程空间充裕（余量 `+2.62 m`） |
| **截面最大伸长率 ($\kappa_{\max}$)** | $\le 8.000000$ | `4.938149` | `4.938149` | **`4.716363`** | ✅ 较起点更圆润（降低 `4.49%`） |

#### 4. 一键复现命令

```bash
cd contest-20261125/challenge-5f1ea274-3

# 1. 运行 5 个公开参考构型基线筛查与 COBYLA 四维协调缩放紧凑边界寻优
taskset -c 12-15 ./.venv/bin/python solve_boundary.py --baseline
taskset -c 12-15 ./.venv/bin/python solve_boundary.py --optimize --max-evals 30

# 2. 对候选边界执行全量 257 磁面 VMEC2000 + NEO + 4096 粒子 SIMPLE 0.2s 官方验收工作流
taskset -c 12-15 ./.venv/bin/python run_workflow.py
```

---

### 3.6 `challenge-1b02e50c`：《留住聚变热量，用三维磁场抑制湍流热损失》

- **赛题目录**：[`contest-20261125/challenge-1b02e50c/`](./contest-20261125/challenge-1b02e50c/)
- **平台提交编号**：**`attempt_id: 111812`（全网首提）**
- **核心交付物**：[`boundary.json`](./contest-20261125/challenge-1b02e50c/boundary.json)（SHA-256: `bd1ee11c5d3d031c58dc44ca89781f041693416fb6b2e729de2771e4e0f81b3a`）
- **核心求解器**：[`solve_turbulence.py`](./contest-20261125/challenge-1b02e50c/solve_turbulence.py)、[`validate_native.py`](./contest-20261125/challenge-1b02e50c/validate_native.py)
- **审计与评测报告**：[`submission_outputs/audit_report.json`](./contest-20261125/challenge-1b02e50c/submission_outputs/audit_report.json)
- **标准化 Agent 执行轨迹**：[`submitted_bundle/traces/trace.jsonl`](./contest-20261125/challenge-1b02e50c/submitted_bundle/traces/trace.jsonl)（共 201 步完整工具调用与思考记录）

#### 1. 问题本质与官方评测包审计发现（`audit_report.json`）

本题要求在保持 $N_{\text{FP}}=2$ QA+well 仿星器目标体积 $V_0 = 581.87356\text{ m}^3$（相对误差 $\le 1\%$）、小半径 $a = 1.70\text{ m}$（纵横比 $A \approx 6.00$）、截面伸长率 $\kappa_{\max} \le 8.0$、旋转变换 $\iota_{\min} \ge 0.30$、磁阱 $W \ge 0$、线圈退让尺度 $L_{\nabla B,\min} \ge 2.40\text{ m}$、NEO 有效波纹度 $\varepsilon_{\text{eff},\max} \le 0.01$ 及高能 $\alpha$ 粒子损失率 $\le 5\%$ 等 **7 项约束**的前提下，极小化 $s = 0.5$ 磁面上 **4 根 PEST 通量管**（`T00`: $(\theta,\zeta)=(0,0)$、`T10`: $(\pi/2,0)$、`T01`: $(0,\pi/2)$、`T11`: $(\pi/2,\pi/2)$，每根长度 $L = 75 a$）在 **3 组归一化离子温度梯度**（$a/L_{Ti} \in \{3.0, 4.0, 5.0\}$，共 **12 个湍流工况**）下的最坏离子温度梯度模（ITG）湍流热流上界 $J$。

智能体在启动求解前对官方题目包执行了严密的第一性原理代码审计（记录于 [`audit_report.json`](./contest-20261125/challenge-1b02e50c/submission_outputs/audit_report.json) 的 `official_package_audit` 章节）：
1. **官方本地包未标定锁死**：随题下发的 `reference/baseline.json` 中 `"state": "uncalibrated"`、`"J_reference": null`，且 `runtime-lock.json` 状态为 `"built_unverified"`（其指定的私有镜像仓库返回 `unauthorized`），导致原样运行本地 `eval.py` 必然在入口自检处抛出 `EvaluatorError: D0 runtime has not passed fresh native image and isolation checks`；
2. **题面公式与 `eval.py` 实现差异**：题面文档规定达标得分映射为 $S = 0.75 - \arctan(\max \text{upper})/(2\pi)$ 且统计容差检验基于候选解均值，而随包 `eval.py` 代码内部实际编写的是 $S = 0.5 + 0.5\,\sigma(J_{\text{ref}} - \max(\text{upper}_k / \text{lower}_{\text{ref},k}))$ 且容差检验引用了 `baseline.json` 均值。
- **诚实科研决策**：智能体严格遵守 [`AGENTS.md`](./AGENTS.md) 铁律——**绝不伪造或篡改 `runtime-lock.json` 与 `baseline.json`，绝不把准线性几何代理数值谎称为非线性 GX 显卡热流**，而是直接在真实 VMEC 平衡、真实 `gx_geometry` 通量管提取与真实 `STELLOPT/NEO` 五磁面波纹度检验之上，构建物理可解释的 12 工况通量管 ITG 准线性混合长度本征求解器完成优化，并将上述审计证据完整随包提交。

#### 2. 自主求解：面积-体积双守恒流形归一化与 12 工况通量管 ITG 混合长度本征优化

在 [`solve_turbulence.py`](./contest-20261125/challenge-1b02e50c/solve_turbulence.py) 中，智能体实现了从 VMEC 磁平衡到通量管漂移波本征分析的完整链条：

1. **精确面积-体积双守恒归一化投影（`normalize`）**
   - 利用 Fourier 边界精确求积计算平均横截面积 $\bar{S}$ 与总体积 $V$：先通过全模态均匀缩放因子 $1.70 / \sqrt{\bar{S}/\pi}$ 将有效小半径精确锁定在 $a = 1.700000\text{ m}$；
   - 再利用**径向刚体平移每个横截面不改变截面积 $\bar{S}$、且 $\partial V / \partial R_{0,0} = 2\pi^2 a^2$ 严格成立**的几何恒等式，解析修正环向主半径基模：
     $$\Delta R_{0,0} = \frac{V_0 - V}{2\pi^2 a^2}$$
     使任意变形后的边界体积相对误差恒保持在 **$2.22\times 10^{-16}$（机器精度零误差）**！
2. **基于真实 `gx_geometry` PEST 通量管曲率与度规的 12 工况准线性 ITG 算子（`proxy_tube`）**
   - 对每个候选边界求解 65 磁面固定边界真空 VMEC 平衡（力残差 $< 10^{-12}$），调用官方 `gx_geometry` 模块在 $s=0.5$ 磁面上精确追踪 `T00`、`T10`、`T01`、`T11` 四根长度为 $75 a$ 的磁力线，生成等弧长网格（$N_z = 97$ 及加密复核 $N_z = 193$）下的标准 `eik.nc` 几何系数：磁场强度 $B(z)$、坏曲率磁漂移项 $\texttt{cvdrift}(z)$、垂直波数度规 $\texttt{gds2}(z)$ 与径向通量梯度 $|\nabla \rho|^2(z)$。
   - 在每根通量管、每个温度梯度 $g \in \{3, 4, 5\}$ 及双法向波数谱 $k_y \in \{0.15, 0.25, 0.40, 0.65, 1.0\}$ 上，求解含有限拉莫尔半径（FLR）屏蔽 $(1 + k_\perp^2)^{-1}$ 与平行声波渡越算子 $-\partial_\parallel (B^{-1} \partial_\parallel)$ 的广义本征值问题，计算有效混合长度热流驱动指标：
     $$Q_{\text{proxy}}(T_{ij}, g) = \max_{k_y} \left[ g \cdot \gamma(k_y, g) \cdot \frac{\langle |\hat{\phi}|^2 |\nabla \rho|^2 / B \rangle}{\langle |\hat{\phi}|^2 k_y^2 \texttt{gds2} / B \rangle} \right]$$
3. **直击全局瓶颈通量管 `T00-g5` 的坏曲率与磁通压缩协同抑制**
   - 诊断发现，12 个工况中的全局极大值恒位于外侧中平面通量管 `T00` 的最高温度梯度点 **`T00-g5`**（起点 $J_{\text{proxy}} = 133.5146$，远高于 `T10-g5` 的 $37.70$ 与 `T01-g5` 的 $35.60$）。
   - 智能体通过联合调节截面伸长率、内侧三角形变、螺旋曲率与磁镜模态，在不触碰任何磁约束红线的前提下，大幅增强了 `T00` 通量管在坏曲率区的局部磁剪切与垂直波数压缩度（$\langle \texttt{gds2} \rangle$），将最坏工况 `T00-g5` 的 ITG 驱动指标从 **`133.5146` 压降至 `121.8439`（净降低 `8.74%`）**，且在双倍沿场网格加密（$N_z=193$）下复核依然稳定保持 **`-8.74%`**（从 `133.3025` 降至 `121.6521`）。

#### 3. 最终物理约束与 12 工况湍流抑制验收账本

| 验收维度 | 物理/几何指标名称 (`audit_report.json`) | 官方硬约束阈值 | 官方公开起点 (`QA+well`) | 本仓最终解 [`boundary.json`](./contest-20261125/challenge-1b02e50c/boundary.json) | 状态与改善幅度 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **最坏工况湍流目标** | **12 工况最大 ITG 准线性热流指标 (`J_proxy`, `T00-g5`)** | 越小越好 | `133.51456` | **`121.84385`** | **最坏工况降低 `8.74%`** |
| **沿场网格加密复核** | 双倍弧长网格加密下最大热流指标 (`gridcheck.J_proxy`) | 检验离散收敛性 | `133.30251` | **`121.65206`** | **加密复核一致降低 `8.74%`** |
| **外侧管全梯度抑制** | `T00` 通量管 $a/L_{Ti} = 3.0 / 4.0 / 5.0$ 指标 | 核心坏曲率管 | `44.56 / 84.09 / 133.51` | **`41.38 / 77.29 / 121.84`** | ✅ 三档温度梯度全面下降 |
| **环向偏置管抑制** | `T01` 通量管 $a/L_{Ti} = 3.0 / 4.0 / 5.0$ 指标 | 次主导通量管 | `13.92 / 24.12 / 35.60` | **`12.83 / 21.56 / 31.25`** | ✅ 最高降幅达 **`-12.23%`** |
| **新经典输运波纹度** | **真实 STELLOPT NEO 五磁面最大有效波纹 ($\varepsilon_{\text{eff},\max}$)** | $\le 0.010000$ (`1%`) | $2.1\times 10^{-4}$ | **`2.5319e-04` (`0.0253%`)** | ✅ **仅为阈值的 `1/40`（余量 `97.5%`）** |
| **体积与小半径守恒** | 等离子体总体积 $V$ / 相对误差 / 有效小半径 $a$ | $\le 1\%$ / $a=1.70\text{ m}$ | $581.8736\text{ m}^3$ / $1.70\text{ m}$ | **`581.87356 m³`（误差 `2.2e-16`）/ `1.70 m`** | ✅ 机器精度严格守恒 |
| **MHD 与工程约束** | 磁阱 $W$ / 最小旋转变换 $\iota_{\min}$ / 线圈退让 $L_{\nabla B,\min}$ | $\ge 0$ / $\ge 0.30$ / $\ge 2.4\text{ m}$ | $+0.00217$ / $0.4169$ / $5.55\text{ m}$ | **`+0.00234` / `0.4466` / `5.36 m`** | ✅ 磁阱与 $\iota_{\min}$ 反较起点更优 |
| **截面伸长率包络** | 最大截面伸长率 (`kappa_max`) | $\le 8.000000$ | `4.938149` | **`4.641002`** | ✅ 截面更紧凑（降低 `6.02%`） |

#### 4. 一键复现命令

```bash
cd contest-20261125/challenge-1b02e50c

# 1. 运行基于 VMEC++ 与 gx_geometry 的 12 工况通量管 ITG 湍流几何抑制优化器
taskset -c 16-19 ../challenge-5f1ea274-3/.venv/bin/python solve_turbulence.py --search

# 2. 调用真实 STELLOPT/NEO、booz_xform 与沿场加密网格复核生成完整审计报告
taskset -c 16-19 ../challenge-5f1ea274-3/.venv/bin/python validate_native.py
```

---

## 四、 仓库目录结构

```text
.
├── AGENTS.md                                              # 高维认知与第一性原理科研工程协议（核心认知与工程宪章）
├── README.md                                              # 项目总览、6 道赛题完整成绩总表与深度技术拆解文档
└── contest-20261125/                                      # 科学骑士大师赛 · 未来产业专项赛（可控核聚变方向）工作区
    ├── challenge-44c6e035/                                # [01 已完成 ✅ | attempt_id: 111198] 仿星器有限截面三维线圈装配鲁棒优化
    │   ├── coils.json                                     # 最终提交最优线圈设计参数文件（官方验收得分 0.7612337，27 项约束 0 违规）
    │   ├── evaluation_report.json                         # 官方 eval.py 四档求积 + 错位加密复核完整验收报告
    │   ├── submission_receipt.json                        # 平台提交回执（attempt_id: 111198）
    │   ├── submitted_bundle/traces/trace.jsonl            # 标准化 Agent 完整执行与二阶自检轨迹（327 步）
    │   └── workspace/solve_coils.py                       # JAX + SIMSOPT 伴随梯度混合可微鲁棒优化求解器
    ├── challenge-9668a3f8/                                # [02 已完成 ✅ | attempt_id: 111963 全网首提] 燃烧等离子体闭环反馈控制
    │   ├── controller.onnx                                # 最终提交 ONNX opset 18 静态因果控制器（双网格 50 轨迹 100% 达标，得分 0.7520522）
    │   ├── evaluation_report.json                         # 开发集 4 场景与扩展集 25 场景粗细双网格完整回放验收报告
    │   ├── submitted_bundle/traces/trace.jsonl            # 标准化 Agent 完整执行与双网格偏差辨识轨迹（433 步）
    │   └── tools/build_controller.py                      # 在线网格辨识 + 前段储备 + 低电子加热维持 ONNX 图构建器
    ├── challenge-d2b1bd0c/                                # [03 已完成 ✅ | attempt_id: 111942] 托卡马克截面形状与理想 MHD 稳定压力优化
    │   ├── plasma.json                                    # 最终提交 7 参数设计（中心压力 1.399 MPa，平均压力提升 3.40 倍至 332.77 kPa）
    │   ├── best_formal.json                               # 官方 --mode formal 双档网格 6 平衡 90 模态完整验收摘要（0.6234 / 0.6246）
    │   ├── submission_receipt.json                        # 平台提交回执（attempt_id: 111942）
    │   ├── submitted_bundle/traces/trace.jsonl            # 标准化 Agent 完整执行与原生 TokaMaker/DCON 寻优轨迹（807 步）
    │   └── solve_plasma.py                                # 固定体积径向包络流形投影 + 7 维受约束 MHD 稳定性寻优器
    ├── challenge-b946f0c9/                                # [04 已完成 ✅ | attempt_id: 111893 平台终审 77.1153/100] 聚变包层中子学优化
    │   ├── blanket.json                                   # 最终提交 96 区 192 自由度材料分布（保守产氚下界 0.295741，提升 +18.59%）
    │   ├── platform_result.json                           # 云端 Harbor 官方终审评分结果（77.1153 / 100，trace_factor = 1.0）
    │   ├── submission_outputs/details.json                # 本地完整 7,680 万源中子 OpenMC 验收详细账本（reward: 0.7712608）
    │   ├── submitted_bundle/traces/trace.jsonl            # 标准化 Agent 完整执行与库存守恒寻优轨迹（599 步）
    │   └── solve_blanket.py                               # 双层二分库存等式流形投影 + 原生 OpenMC 多保真度优化器
    ├── challenge-5f1ea274-3/                              # [05 首轮已交 ✅ | attempt_id: 111815 全网首提] 紧凑仿星器三维磁边界优化
    │   ├── boundary.json                                  # 首轮提交紧凑边界（纵横比压缩 7.60% 至 A = 5.54297，NEO 波纹 0.00709%）
    │   ├── first_submission.json                          # 平台首轮提交回执记录（attempt_id: 111815）
    │   ├── submission_outputs/stage_report.json           # 5 大参考构型摸底与候选边界 NEO/磁阱/旋转变换评测报告
    │   ├── submitted_bundle/traces/trace.jsonl            # 标准化 Agent 完整执行与协调缩放寻优轨迹（243 步）
    │   └── solve_boundary.py                              # 保持 QA/磁阱符号耦合的四维协调形变 + COBYLA 约束寻优器
    └── challenge-1b02e50c/                                # [06 已完成 ✅ | attempt_id: 111812 全网首提] 仿星器三维磁场湍流热流抑制
        ├── boundary.json                                  # 最终提交三维边界（12 工况最坏 ITG 几何驱动降低 8.74%，NEO 波纹 2.53e-4）
        ├── submission_outputs/audit_report.json           # 12 工况通量管评测、沿场网格加密复核与官方评测包审计报告
        ├── submitted_bundle/traces/trace.jsonl            # 标准化 Agent 完整执行与通量管本征优化轨迹（201 步）
        └── solve_turbulence.py                            # 面积-体积双守恒归一化 + 4 通量管 × 3 温度梯度准线性 ITG 本征优化器
```
