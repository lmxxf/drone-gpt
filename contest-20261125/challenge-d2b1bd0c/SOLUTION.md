# `challenge-d2b1bd0c`：《提高托卡马克的稳定压力，设计聚变等离子体形状》深度技术拆解

> **[⬅️ 返回仓库首页总览 (`README.md`)](../../README.md)** | **方法论宪章：[`AGENTS.md`](../../AGENTS.md)**

- **赛题目录**：[`.`](./)（`contest-20261125/challenge-d2b1bd0c/`）
- **平台提交编号**：**`attempt_id: 111942`** (`job_id: 86254`，[`submission_receipt.json`](./submission_receipt.json))
- **核心交付物**：[`plasma.json`](./plasma.json)（SHA-256: `da8730662883d8f8e5ed7b4f51de35fa2f7103b410814ae9f383b04468829acd`）
- **核心求解器**：[`solve_plasma.py`](./solve_plasma.py)
- **正式验收报告**：[`best_formal.json`](./best_formal.json)、[`runs/running_state.json`](./runs/running_state.json)、[`submission_outputs/REPORT.md`](./submission_outputs/REPORT.md)
- **标准化 Agent 执行轨迹**：[`submitted_bundle/traces/trace.jsonl`](./submitted_bundle/traces/trace.jsonl)（共 807 步完整工具调用与思考记录）

---

## 1. 问题症结与第一性原理 MHD 稳定性边界诊断

本题要求在固定等离子体体积 $V = 240\text{ m}^3$、大半径 $R_0 = 4.0\text{ m}$、参考环向场 $B_0 = 7.0\text{ T}$ 与总等离子体电流 $I_p = 8.1\text{ MA}$ 的约束下，联合优化 **7 个连续参数**——截面伸长率 $\kappa$、三角形变 $\delta$、4 个高阶 Fourier 边界修形系数 $c_2, c_3, s_2, s_3$ 以及磁轴中心压力 $p_0 \in [10^5, 1.4\times 10^6]\text{ Pa}$。

官方验证器 `workspace/physics/eval.py` 在 `--mode formal` 下对每个候选设计执行极其严苛的 **6 平衡 $\times$ 全模数理想 MHD 稳定性联合裁决**：

- 针对 **`broad`、`nominal`、`peaked`** 三种不同空心度（`ffprime_hollow`）的电流剖面，分别在 **`coarse`（$129\times 257$）**与 **`fine`（$257\times 513$）**两档网格下调用 **TokaMaker** 求解自由边界/固定体积 Grad–Shafranov 平衡；
- 每个平衡必须同时满足几何包络与径向半宽 $a_{\text{minor}} \le 1.50\text{ m}$、安全因子下界 $q_{\min} \ge 1.10$、Grad–Shafranov 残差上限，并在每个平衡上调用 **GPEC/DCON** 检验环向模数 $n = 1, 2, 3, 4, 5$ 及多组环向相位下的理想扭曲模最小总能量 $\delta W_{\min} \ge 0.02$、Mercier 判据 $(-D_I)_{\min} \ge 0.01$（即 $\max D_I \le -0.01$）、内部零交叉数 `internal_zero_crossings = 0` 以及全径向面高模数气球模积分节点数 `ballooning_nodes_max = 0`。
- **得分机制**：在全部 6 个平衡、90 次 DCON 模分析与 19,674 次气球模积分 100% 达标的前提下，取三剖面双网格中最坏体积平均压力 $\bar{p}_{\min}$，按
  $$S = \frac{\bar{p}_{\min}}{\bar{p}_{\min} + 200\text{ kPa}}$$
  计算最终得分。官方起点（$p_0 = 400\text{ kPa}$）虽然可行，但最坏体积平均压力仅为 $\bar{p}_{\min} = 97.759\text{ kPa}$，对应基线得分仅为 **`0.3283161`**。

---

## 2. 自主求解与固定体积径向包络流形投影

在 [`submitted_bundle/traces/trace.jsonl`](./submitted_bundle/traces/trace.jsonl) 的 807 步探索中，智能体首先完成了 ARM64 架构下 `OpenFUSIONToolkit (TokaMaker)` 与 `GPEC/dcon` 的原生编译与参考 BLAS 确定性校验（自动识别并隔离了 OpenBLAS 多线程数值扰动窗口），随后在 [`solve_plasma.py`](./solve_plasma.py) 中实施了三项关键物理降维：

1. **解析推导固定体积与径向半宽包络的线性流形投影（`project_radial_envelope`）**
   - 智能体从环向旋转体体积积分公式出发，注意到当小半径尺度 $a$ 与形状系数 $(\delta, c_2, c_3, s_2, s_3)$ 固定时，环向体积 $V$ 关于垂直伸长率 $\kappa$ 是**严格线性**的！
   - 若在 $(\kappa, \delta, c_2, c_3, s_2, s_3)$ 空间盲目采样，由于固定 $V=240\text{ m}^3$ 会反推缩放小半径 $a \propto \kappa^{-1/2}$：一旦 $\kappa$ 偏大（如 $\kappa = 1.76$），虽然截面积拉长，但压扁后的平均压力形状因子反而使 $\bar{p}$ 停留在 $316.9\text{ kPa}$；一旦 $\kappa$ 偏小，又会因 $a > 1.50\text{ m}$ 触发径向包络越界。
   - 智能体直接将径向尺度钉死在合规上限边缘 $a \approx 1.476\text{ m}$（留出安全余量），发现**低伸长率（$\kappa = 1.48$）+ 强负方形度/高阶内凹修形（$c_2 \in [-0.0625, -0.05], c_3 = -0.025, s_2 \in [-0.10, -0.09], s_3 = -0.08$）**能够最大化固定体积下的磁通面填充因子（平均压力与中心压力比 $\bar{p}/p_0$ 达 **$23.79\%$** 的高压极限形态），同时保持 $q_{\min} > 1.110$！
2. **磁轴中心压力直推 $1.399\text{ MPa}$ 与 `broad` 剖面 $n=1$ 能量裕度锁定**
   - 全空间探测表明，限制压力上限的真正瓶颈不是高 $n$ 气球模（在 $\kappa=1.48, \delta=0.32$ 下 $n=1..5$ 气球模节点恒为 `0`，Mercier 裕度高达 $0.241$），而是 `broad` 电流剖面下的 **$q_{\min} \ge 1.10$ 门槛**（实测裕度约 `+0.0103`）与 **$n=1$ 理想扭曲模最小能量 $\delta W \ge 0.02$**（实测 $\delta W = 0.06215$，裕度 `+0.0422`）。
   - 智能体通过 `equilibrium_probe`（单点仅需 `14 s`）快速过滤 $q_0$ 越界点，再通过粗网格 `(n=1, n=5)` 筛查与坐标爬山，将磁轴中心压力一路推至协议上限 $1.40\text{ MPa}$ 紧邻处的 **`1.399 MPa`（`1,399,000 Pa`）**。

---

## 3. 最终官方 `--mode formal` 六平衡全模数完整验收账本

最终锁定并验证了两个完全通过官方 `--mode formal` 双网格全模数检验的高压姐妹解（`shape330-formal` 得分 **`0.6234045`**，$\bar{p}_{\min}=331.07\text{ kPa}$；精修高阶项 $c_2=-0.0625, s_2=-0.09$ 后的最终交付解 [`plasma.json`](./plasma.json) `top333-formal` 在 `--mode search` 与 `--mode formal` 下分别取得 **`0.6246041`** 与 **`0.62460406`**，$\bar{p}_{\min}=332.77\text{ kPa}$）。以下为 [`plasma.json`](./plasma.json)（$\kappa=1.48, \delta=0.32, c_2=-0.0625, c_3=-0.025, s_2=-0.09, s_3=-0.08, p_0=1.399\text{ MPa}$）在全部 6 个工况下的完整验收账本：

| 电流剖面与网格工况 | 体积平均压力 $\bar{p}$ (`Pa`) | 安全因子下界 $q_{\min}$ ($\ge 1.10$) | $n=1..5$ 含相位最小总能量 $\delta W_{\min}$ ($\ge 0.02$) | Mercier 判据最大值 $\max D_I$ ($\le -0.01$) | 气球模最大节点数 (`= 0`) | 内部零交叉数 (`= 0`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`broad / coarse`** ($129\times 257$) | **`332,770.88 Pa`** | `1.11029574` | `0.06239153` | `-0.25115877` | `0` | `0` |
| **`broad / fine`** ($257\times 513$) | **`332,770.82 Pa`** | `1.11031539` | `0.06215456` | `-0.25115935` | `0` | `0` |
| **`nominal / coarse`** ($129\times 257$) | **`333,001.74 Pa`** | `1.11249330` | `0.06439533` | `-0.25115983` | `0` | `0` |
| **`nominal / fine`** ($257\times 513$) | **`333,001.68 Pa`** | `1.11251436` | `0.06405233` | `-0.25116011` | `0` | `0` |
| **`peaked / coarse`** ($129\times 257$) | **`333,309.65 Pa`** | `1.11542901` | `0.06712701` | `-0.25116079` | `0` | `0` |
| **`peaked / fine`** ($257\times 513$) | **`333,309.58 Pa`** | `1.11545194` | `0.06683116` | `-0.25116130` | `0` | `0` |
| **全局最坏工况汇总 (`feasible=True`)** | **`332.77 kPa`（较基线 `97.76 kPa` 提升 `3.40 倍`）** | **`1.11030`**（达标） | **`0.06215`**（达标） | **`-0.25116`**（达标） | **`0`**（达标） | **`0`**（达标） |

---

## 4. 一键复现命令

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
