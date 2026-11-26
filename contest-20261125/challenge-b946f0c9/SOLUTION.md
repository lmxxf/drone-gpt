# `challenge-b946f0c9`：《为聚变补充燃料，用有限的包层材料兼顾产氚与局部核热》深度技术拆解

> **[⬅️ 返回仓库首页总览 (`README.md`)](../../README.md)** | **方法论宪章：[`AGENTS.md`](../../AGENTS.md)**

- **赛题目录**：[`.`](./)（`contest-20261125/challenge-b946f0c9/`）
- **平台提交编号与官方终审分**：**`attempt_id: 111893`**，**云端 Harbor 官方终审得分 `77.1153 / 100`**（`harbor_reward = 0.77115348`，`trace_factor = 1.0`，见 [`platform_result.json`](./platform_result.json)）
- **核心交付物**：[`blanket.json`](./blanket.json)（SHA-256: `484776003b3fdbbba15e3004767ca2bbd1a41fda85c7e8afd6c9e6fe83de4247`）
- **核心求解器**：[`solve_blanket.py`](./solve_blanket.py)
- **完整验收报告**：[`platform_result.json`](./platform_result.json)、[`submission_outputs/reward.json`](./submission_outputs/reward.json)、[`submission_outputs/details.json`](./submission_outputs/details.json)
- **标准化 Agent 执行轨迹**：[`submitted_bundle/traces/trace.jsonl`](./submitted_bundle/traces/trace.jsonl)（共 599 步完整工具调用与思考记录）

---

## 1. 问题症结与中子输运/库存守恒耦合机制剖析

本题研究含中央偏置诊断通道（位于横向 $(y,z) \approx (13, 0)\text{ cm}$）与氦冷却槽的三维聚变增殖包层模块。包层在径向（深度 $l=0..5$ 共 6 层）与横向（$4\times 4 = 16$ 个柱状扇区）被划分为 **$6 \times 16 = 96$ 个独立材料区**。每个区域 $i$ 需独立指定：
1. $\text{Li}_4\text{SiO}_4$ 增殖陶瓷体积分数 $b_i \in [0.20, 0.60]$（剩余体积由中子倍增剂金属铍 $\text{Be}$ 填充，故 $V_{\text{Be},i} \propto (1 - b_i)V_i$）；
2. 锂元素中的 $^6\text{Li}$ 原子丰度 $e_i \in [0.05, 0.90]$（共 **$96 \times 2 = 192$ 个连续自由度**）。

官方验证器在 **`normal`（准直法向入射）**与 **`cosine`（余弦角分布入射）**两种 $14.1\text{ MeV}$ 聚变中子源分布下，各用两个独立随机数种子（`2026100801`、`2026100802`）运行 $64\text{ batches} \times 300,000\text{ particles/batch} = 1.92\times 10^7$ 中子（单次全量评估共计 **7,680 万连续能量蒙特卡洛中子历史**），考核以下硬约束与目标：

- **三项全局材料库存硬上限**：增殖陶瓷总体积 $\sum V_i b_i \le 143,016.0\text{ cm}^3$（等价于全包层体积加权平均 $\bar{b} \le 0.40$）、金属铍总体积 $\le 125,139.0\text{ cm}^3$（恰迫使 $\bar{b} \equiv 0.40$，即**陶瓷与铍体积总库存必须双向精确卡满**）、$^6\text{Li}$ 总原子数 $\le 1.279017\times 10^{27}$（等价于 $\sum V_i b_i e_i / \sum V_i \le 0.12$）；
- **32 项背面快中子通量/局部核热硬上限**：在 `normal` 与 `cosine` 双源下，包层最深处背面 16 个分区（`rear0`..`rear15`）的统计上界（均值 $+ 3.2514\times\text{SE}$）均不得超过各自阈值；
- **最坏源分布下的保守产氚率下界目标（`objective_lower`）**：取 `normal` 与 `cosine` 两源产氚率统计下界（均值 $- 3.2514\times\text{SE}$）的最小值。官方均匀基线 `uniform.json`（处处 $b_i=0.40, e_i=0.30$）受限于 `cosine` 源的大倾角泄漏与前层倍增不足，`objective_lower` 仅为 **`0.24938196`**（对应基准分 `0.750000` / 百分制 `75.00 分`）。

---

## 2. 自主求解：精确库存流形二分投影与纵深-横向模态分解

在 [`solve_blanket.py`](./solve_blanket.py) 中，智能体摒弃了在 192 维空间盲目加罚函数的低效做法，直接从核工程中子学第一性原理构建了**库存恒等守恒参数化与主动学习代理寻优架构**：

1. **双层二分投影算子（`project`）锁死库存等式流形**
   - 由于陶瓷体积上限（$\bar{b} \le 0.40$）与铍体积上限（$1-\bar{b} \le 1-0.40$）共同将 $\sum V_i b_i = 0.40 \sum V_i$ 锁死在零测度超平面上，任何偏离都会立即触发库存违规。
   - 智能体构造了 80 步双层精确二分裁剪投影算子：先求解平移量 $\lambda_b$ 使 $\sum V_i \text{clip}(b_i + \lambda_b, 0.2, 0.6) = 0.40 \sum V_i$ 达到机器精度（$10^{-12}$），再以更新后的区域陶瓷体积 $w_i = V_i b_i$ 为权重，求解 $\lambda_e$ 使 $\sum w_i \text{clip}(e_i + \lambda_e, 0.05, 0.90) = 0.12 (1 - 2\times 10^{-12})\sum V_i$。**由此生成的每一个候选解在构造上 100% 满足三项库存硬约束！**
2. **中子倍增纵深梯度 + 边缘/诊断通道横向补偿的 16 维物理基展开（`make_design`）**
   - 从中子物理来看，$14.1\text{ MeV}$ 快中子在刚进入前层（$l=0,1$）时能量高于 $\text{Be}(n,2n)$ 反应阈值（$\sim 1.9\text{ MeV}$），此时应**压低前层陶瓷占比 $b$（提高金属铍 $\text{Be}$ 占比以最大化 $(n,2n)$ 快中子倍增）**，同时在中间慢化层与后层拉高陶瓷占比 $b$ 与 $^6\text{Li}$ 丰度 $e$，利用 $^6\text{Li}(n,\alpha)\text{T}$ 在热能/超热能区间的巨大截面捕获倍增后的慢化中子，并在最深层形成屏蔽墙压低背面 16 区快中子泄漏。
   - 针对 `cosine` 源在模块四周边缘（$|y|+|z|$ 较大处）及中央偏置诊断孔道（$(y-13)^2+z^2$ 邻域）的中子流畸变，智能体引入了 6 层深度基底 + 4 个横向空间模态（边缘距离模态 `edge` $\times$ 深度衰减、诊断孔道高斯模态 `port` $\times$ 深度衰减），将 192 维自由度降维至物理意义极其清晰的 **16 维核心参数空间**。
3. **多保真度原生 OpenMC 搜索与全量 7,680 万中子终审**
   - 搜索阶段直接调用原生 OpenMC（先以 `cosine` 瓶颈源 $8\times 50,000 = 40\text{ 万}$ 中子配合高斯过程期望提升 EI 探索，再升级至双源高统计量精修），拒绝任何失真的扩散近似；最终设计通过本地 7,680 万中子全量验证后打包提交平台，并获 Harbor 云端独立重放确认。

---

## 3. 最终本地与云端 Harbor 官方终审验收账本

| 验收维度 | 评估指标名称 (`details.json` / `platform_result.json`) | 官方硬约束阈值 / 基准 | 官方均匀基线 (`uniform.json`) | 本仓最终解 [`blanket.json`](./blanket.json) | 状态与提升幅度 |
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

---

## 4. 一键复现命令

```bash
cd contest-20261125/challenge-b946f0c9

# 1. 运行基于精确库存流形投影与原生 OpenMC 的包层材料分布优化器
taskset -c 8-11 ./runtime/native-env/bin/python solve_blanket.py

# 2. 运行完整 7,680 万中子（双源 × 双种子 × 64 batches × 300,000 particles）官方协议验收
taskset -c 8-11 ./runtime/native-env/bin/python local_eval.py blanket.json
```
