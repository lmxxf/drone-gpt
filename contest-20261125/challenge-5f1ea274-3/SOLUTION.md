# `challenge-5f1ea274-3`：《用三维形状留住聚变粒子：寻找更紧凑的仿星器》深度技术拆解

> **[⬅️ 返回仓库首页总览 (`README.md`)](../../README.md)** | **方法论宪章：[`AGENTS.md`](../../AGENTS.md)**

- **赛题目录**：[`.`](./)（`contest-20261125/challenge-5f1ea274-3/`）
- **平台提交编号**：**`attempt_id: 112075`**（`job_id: 86387`，[`second_submission_receipt.json`](./second_submission_receipt.json)；首轮全网首提 `attempt_id: 111815`，[`first_submission.json`](./first_submission.json)）
- **核心交付物**：[`boundary.json`](./boundary.json)（SHA-256: `062640a28051d29d44f07ae434d9106a4d5723ad6397960304f4a1a0a2079a96`）
- **核心求解器**：[`solve_boundary.py`](./solve_boundary.py)
- **4,096 粒子全量官方认证报告**：[`certified_result.json`](./certified_result.json)、[`runs/certify_062640a28051/result/details.json`](./runs/certify_062640a28051/result/details.json)、[`submission_outputs/stage_report.json`](./submission_outputs/stage_report.json)
- **标准化 Agent 执行轨迹**：[`submitted_bundle/traces/trace.jsonl`](./submitted_bundle/traces/trace.jsonl)（共 243 步完整工具调用与思考记录）

---

## 1. 问题症结与 5 个公开参考构型的筛选真相

本题要求设计一个更紧凑的三维仿星器最外层磁面边界（由 VMEC Fourier 系数 `rbc`, `zbs` 表征，评测时自动缩放到固定小半径 $a = 1.70\text{ m}$ 与磁轴场强 $B_{00} = 5.70\text{ T}$），在满足以下 **6 项严苛物理与工程约束**的前提下，极小化装置纵横比 $A = R_{\text{major}} / a_{\text{minor}}$（按 $S = 0.5 + 0.5 / (1 + e^{A - 4})$ 计分）：
1. 截面最大伸长率 $\kappa_{\max} \le 8.0$ 且无截面自交；
2. 旋转变换下界 $\iota_{\min} \ge 0.30$；
3. 磁阱深度 $W \ge 0$（理想 MHD 交换模稳定必要条件）；
4. 线圈退让磁场梯度特征尺度 $L_{\nabla B,\min} \ge 2.40\text{ m}$；
5. `NEO` 五个径向磁面上的新经典输运有效波纹度 $\varepsilon_{\text{eff},\max} \le 0.01$（`1%`）；
6. `SIMPLE` 追踪 $s=0.25$ 磁面上 4,096 个 $3.5\text{ MeV}$ 聚变 $\alpha$ 粒子达 $0.2\text{ s}$ 的损失率上界 $f_\alpha \le 0.05$（`5%`）。

智能体首先在 ARM64 环境下从源码完整构建了 `VMEC2000`、`STELLOPT/NEO`、`SIMPLE`（开启 `SIMPLE_DETERMINISTIC_FP=ON`）与 `booz_xform`，并对官方提供的全部 5 个参考构型执行了系统摸底筛查（见 [`submission_outputs/stage_report.json`](./submission_outputs/stage_report.json)）：
- **`ESTELL.json`**（$A=5.293$）：旋转变换 $\iota_{\min} = 0.2053 < 0.30$ 且为磁山（$W = -0.0193 < 0$），**不达标**；
- **`QHS46.json`**（$A=6.698$）：磁山（$W = -0.00149 < 0$）且 Boozer 准对称残差高达 `0.0562`，**不达标**；
- **`landreman_paul_2022_QA.json`**（$A=6.000$）：无磁阱（$W = -0.03125 < 0$），**不达标**；
- **`landreman_paul_2022_QH_well.json`**（$A=8.000$）：虽全部达标，但纵横比高达 $8.0$，得分仅 `0.50899`；
- **`landreman_paul_2022_QA_well.json`**（$A=6.00001$）：**是 5 个公开构型中唯一全部达标的最紧凑起点**（$W = +0.002167 > 0$，$\iota_{\min}=0.4169$，$L_{\nabla B}=5.55\text{ m}$，基线得分 **`0.559601`**）。

---

## 2. 自主求解：保持 QA/磁阱符号耦合的四维协调形变算子（`deform`）

直接逐个扰动数十个高阶 Fourier 模态极易破坏准轴对称（QA）与磁阱（$W>0$）之间精密的二阶几何抵消关系。智能体在 [`solve_boundary.py`](./solve_boundary.py) 中设计了**保持种子构型模态相位与符号比的四维协调形变算子** $\mathbf{x} = (\alpha_{\text{thick}}, \alpha_{\text{axis}}, \alpha_{\text{elong}}, \alpha_{\text{shaping}})$：
$$\begin{aligned}
\text{rbc}_{m,n}(\mathbf{x}) &= \text{rbc}_{m,n}^{(0)} \cdot \begin{cases} \alpha_{\text{axis}}, & m = 0, n \ne 0 \\ \alpha_{\text{thick}} \cdot \alpha_{\text{elong}}^{-1/2}, & m = 1 \\ \alpha_{\text{thick}} \cdot \alpha_{\text{shaping}} \cdot \alpha_{\text{elong}}^{-1/2}, & m \ge 2 \end{cases} \\
\text{zbs}_{m,n}(\mathbf{x}) &= \text{zbs}_{m,n}^{(0)} \cdot \begin{cases} \alpha_{\text{axis}}, & m = 0, n \ne 0 \\ \alpha_{\text{thick}} \cdot \alpha_{\text{elong}}^{+1/2}, & m = 1 \\ \alpha_{\text{thick}} \cdot \alpha_{\text{shaping}} \cdot \alpha_{\text{elong}}^{+1/2}, & m \ge 2 \end{cases}
\end{aligned}$$

- **物理机制**：总体增厚因子 $\alpha_{\text{thick}} > 1$ 在固定大半径 $R_{0,0}$ 下线性放大截面小半径 $a$，从而直接压低纵横比 $A \approx A_0 / \alpha_{\text{thick}}$；但单纯放大 $\alpha_{\text{thick}}$（如 $\alpha_{\text{thick}}=1.05$）会导致环向曲率加深，使磁阱 $W$ 在 $A \approx 5.716$ 处由正转负（$W = -3.46\times 10^{-4}$）。
- 通过引入**磁轴三维波纹度 $\alpha_{\text{axis}}$、椭圆伸长率 $\alpha_{\text{elong}}$ 与高阶三角形/方形整形系数 $\alpha_{\text{shaping}}$** 的协同补偿，并采用 **COBYLA 约束优化器**在正磁阱、旋转变换、线圈退让尺度与准轴对称残差安全内缩约束下寻优，成功跨越了单纯缩放的磁阱崩溃壁垒：
  - 首轮提交（`attempt_id: 111815`）将纵横比从 $6.00001$ 压降至 $A = 5.542969$（压缩 `7.60%`）；
  - 第二轮深度寻优与 257 磁面全分辨率 VMEC + 4,096 粒子 `SIMPLE` 全量 0.2s 认证（**`attempt_id: 112075`**，[`certified_result.json`](./certified_result.json)）进一步将纵横比压降至 **$A = 5.51173067$（较起点大幅压缩 `8.14%`）**，4,096 个 $3.5\text{ MeV}$ $\alpha$ 粒子完整 $0.2\text{ s}$ 轨道损失率仅为 **`3.2715%`（`134 / 4096`，未解析粒子 `0`）**，原版 `eval.py` 官方协议认证得分达 **`0.5903412`**（`feasible = True`）。

---

## 3. 4,096 粒子全量官方认证账本（`certified_result.json`）

以下为最终提交的 [`boundary.json`](./boundary.json)（SHA-256: `062640a28051d29d44f07ae434d9106a4d5723ad6397960304f4a1a0a2079a96`）经原版 `eval.py` 全量协议（257 径向磁面 VMEC2000 平衡 + $60\times 56$ Boozer 坐标变换 + 五磁面 NEO + 4,096 粒子 `SIMPLE` 0.2s 全轨道追踪，总耗时 `24,719.4 s`）认证的完整账本：

| 评估指标名称 (`details.json` / `certified_result.json`) | 官方硬约束阈值 | 公开最优可行起点 (`QA_well`) | 首轮紧凑构型 (`attempt_id: 111815`) | 二轮全量认证终稿 (`boundary.json`, `attempt_id: 112075`) | 状态与安全余量 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **原版 `eval.py` 全量认证总分 (`score`)** | $> 0.559601$ (`feasible=True`) | `0.559601` | `0.5880`（阶段代理） | **`0.5903412` (`feasible = True`)** | ✅ **较起点提升 `+0.030740`（6 项完成度全 `1.0`）** |
| **装置纵横比 (`aspect_ratio`, $A$)** | 越小分越高 | `6.000012` | `5.542969` (`-7.60%`) | **`5.51173067`（较起点压缩 `8.14%`）** | ✅ $R_{\text{major}} = 9.3699\text{ m}, a = 1.7000\text{ m}$ |
| **4,096 粒子 $0.2\text{ s}$ $\alpha$ 损失率 ($f_\alpha$)** | $\le 0.050000$ (`5%`, $N_U=0$) | $\le 5\%$ | 代理筛查通过 | **`0.0327148`（`3.2715%`, `134/4096`, $N_U=0$）** | ✅ **充足余量 `34.6%`（95% Wilson CI `[2.77%, 3.86%]`）** |
| **NEO 五磁面最大有效波纹 ($\varepsilon_{\text{eff},\max}$)** | $\le 0.010000$ (`1%`) | $5.2\times 10^{-5}$ | `7.0858e-05` (`0.00709%`) | **`7.54538e-05` (`0.007545%`)** | ✅ **比阈值低两个数量级（余量 `99.25%`）** |
| **磁阱深度 (`well`, $W$)** | $\ge 0.0$ | `+0.002167` | `+0.000293` | **`+2.77160e-04`** (最坏面 $s=0.9492$) | ✅ 全径向保持正磁阱（$W > 0$） |
| **最小旋转变换 ($\iota_{\min}$)** | $\ge 0.300000$ | `0.416921` | `0.428473` | **`0.4328233`** (轴侧 `0.4333` / 边侧 `0.4399`) | ✅ 远离低阶有理面（余量 `+44.3%`） |
| **线圈退让梯度尺度 ($L_{\nabla B,\min}$)** | $\ge 2.400000\text{ m}$ | `5.552801 m` | `5.024388 m` | **`4.954222 m`** | ✅ 线圈工程空间充裕（余量 `+2.55 m`） |
| **截面最大伸长率 ($\kappa_{\max}$)** | $\le 8.000000$ | `4.938149` | `4.716363` | **`4.657356`** | ✅ 较起点更圆润（降低 `5.69%`） |
| **VMEC 257 面平衡与 Boozer 精度** | 残差 $\le 10^{-12}$ | 达标 | 达标 | **VMEC 残差 `< 1.0e-16`，Boozer $e_B = 6.22\times 10^{-13}$** | ✅ 机器精度收敛 |

---

## 4. 一键复现命令

```bash
cd contest-20261125/challenge-5f1ea274-3

# 1. 运行 5 个公开参考构型基线筛查与 COBYLA 四维协调缩放紧凑边界寻优
taskset -c 12-15 ./.venv/bin/python solve_boundary.py --baseline
taskset -c 12-15 ./.venv/bin/python solve_boundary.py --optimize --max-evals 30

# 2. 对最优候选边界执行全量 257 磁面 VMEC2000 + NEO + 4096 粒子 SIMPLE 0.2s 官方协议认证
taskset -c 12-15 ./.venv/bin/python run_workflow.py
```
