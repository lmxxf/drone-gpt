# `challenge-1b02e50c`：《留住聚变热量，用三维磁场抑制湍流热损失》深度技术拆解

> **[⬅️ 返回仓库首页总览 (`README.md`)](../../README.md)** | **方法论宪章：[`AGENTS.md`](../../AGENTS.md)**

- **赛题目录**：[`.`](./)（`contest-20261125/challenge-1b02e50c/`）
- **平台提交编号**：**`attempt_id: 111812`（全网首提）**
- **核心交付物**：[`boundary.json`](./boundary.json)（SHA-256: `bd1ee11c5d3d031c58dc44ca89781f041693416fb6b2e729de2771e4e0f81b3a`）
- **核心求解器**：[`solve_turbulence.py`](./solve_turbulence.py)、[`validate_native.py`](./validate_native.py)
- **审计与评测报告**：[`submission_outputs/audit_report.json`](./submission_outputs/audit_report.json)
- **标准化 Agent 执行轨迹**：[`submitted_bundle/traces/trace.jsonl`](./submitted_bundle/traces/trace.jsonl)（共 201 步完整工具调用与思考记录）

---

## 1. 问题本质与官方评测包审计发现（`audit_report.json`）

本题要求在保持 $N_{\text{FP}}=2$ QA+well 仿星器目标体积 $V_0 = 581.87356\text{ m}^3$（相对误差 $\le 1\%$）、小半径 $a = 1.70\text{ m}$（纵横比 $A \approx 6.00$）、截面伸长率 $\kappa_{\max} \le 8.0$、旋转变换 $\iota_{\min} \ge 0.30$、磁阱 $W \ge 0$、线圈退让尺度 $L_{\nabla B,\min} \ge 2.40\text{ m}$、NEO 有效波纹度 $\varepsilon_{\text{eff},\max} \le 0.01$ 及高能 $\alpha$ 粒子损失率 $\le 5\%$ 等 **7 项约束**的前提下，极小化 $s = 0.5$ 磁面上 **4 根 PEST 通量管**（`T00`: $(\theta,\zeta)=(0,0)$、`T10`: $(\pi/2,0)$、`T01`: $(0,\pi/2)$、`T11`: $(\pi/2,\pi/2)$，每根长度 $L = 75 a$）在 **3 组归一化离子温度梯度**（$a/L_{Ti} \in \{3.0, 4.0, 5.0\}$，共 **12 个湍流工况**）下的最坏离子温度梯度模（ITG）湍流热流上界 $J$。

智能体在启动求解前对官方题目包执行了严密的第一性原理代码审计（记录于 [`submission_outputs/audit_report.json`](./submission_outputs/audit_report.json) 的 `official_package_audit` 章节）：
1. **官方本地包未标定锁死**：随题下发的 `reference/baseline.json` 中 `"state": "uncalibrated"`、`"J_reference": null`，且 `runtime-lock.json` 状态为 `"built_unverified"`（其指定的私有镜像仓库返回 `unauthorized`），导致原样运行本地 `eval.py` 必然在入口自检处抛出 `EvaluatorError: D0 runtime has not passed fresh native image and isolation checks`；
2. **题面公式与 `eval.py` 实现差异**：题面文档规定达标得分映射为 $S = 0.75 - \arctan(\max \text{upper})/(2\pi)$ 且统计容差检验基于候选解均值，而随包 `eval.py` 代码内部实际编写的是 $S = 0.5 + 0.5\,\sigma(J_{\text{ref}} - \max(\text{upper}_k / \text{lower}_{\text{ref},k}))$ 且容差检验引用了 `baseline.json` 均值。
- **诚实科研决策**：智能体严格遵守 [`AGENTS.md`](../../AGENTS.md) 铁律——**绝不伪造或篡改 `runtime-lock.json` 与 `baseline.json`，绝不把准线性几何代理数值谎称为非线性 GX 显卡热流**，而是直接在真实 VMEC 平衡、真实 `gx_geometry` 通量管提取与真实 `STELLOPT/NEO` 五磁面波纹度检验之上，构建物理可解释的 12 工况通量管 ITG 准线性混合长度本征求解器完成优化，并将上述审计证据完整随包提交。

---

## 2. 自主求解：面积-体积双守恒流形归一化与 12 工况通量管 ITG 混合长度本征优化

在 [`solve_turbulence.py`](./solve_turbulence.py) 中，智能体实现了从 VMEC 磁平衡到通量管漂移波本征分析的完整链条：

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

---

## 3. 最终物理约束与 12 工况湍流抑制验收账本

| 验收维度 | 物理/几何指标名称 (`audit_report.json`) | 官方硬约束阈值 | 官方公开起点 (`QA+well`) | 本仓最终解 [`boundary.json`](./boundary.json) | 状态与改善幅度 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **最坏工况湍流目标** | **12 工况最大 ITG 准线性热流指标 (`J_proxy`, `T00-g5`)** | 越小越好 | `133.51456` | **`121.84385`** | **最坏工况降低 `8.74%`** |
| **沿场网格加密复核** | 双倍弧长网格加密下最大热流指标 (`gridcheck.J_proxy`) | 检验离散收敛性 | `133.30251` | **`121.65206`** | **加密复核一致降低 `8.74%`** |
| **外侧管全梯度抑制** | `T00` 通量管 $a/L_{Ti} = 3.0 / 4.0 / 5.0$ 指标 | 核心坏曲率管 | `44.56 / 84.09 / 133.51` | **`41.38 / 77.29 / 121.84`** | ✅ 三档温度梯度全面下降 |
| **环向偏置管抑制** | `T01` 通量管 $a/L_{Ti} = 3.0 / 4.0 / 5.0$ 指标 | 次主导通量管 | `13.92 / 24.12 / 35.60` | **`12.83 / 21.56 / 31.25`** | ✅ 最高降幅达 **`-12.23%`** |
| **新经典输运波纹度** | **真实 STELLOPT NEO 五磁面最大有效波纹 ($\varepsilon_{\text{eff},\max}$)** | $\le 0.010000$ (`1%`) | $2.1\times 10^{-4}$ | **`2.5319e-04` (`0.0253%`)** | ✅ **仅为阈值的 `1/40`（余量 `97.5%`）** |
| **体积与小半径守恒** | 等离子体总体积 $V$ / 相对误差 / 有效小半径 $a$ | $\le 1\%$ / $a=1.70\text{ m}$ | $581.8736\text{ m}^3$ / $1.70\text{ m}$ | **`581.87356 m³`（误差 `2.2e-16`）/ `1.70 m`** | ✅ 机器精度严格守恒 |
| **MHD 与工程约束** | 磁阱 $W$ / 最小旋转变换 $\iota_{\min}$ / 线圈退让 $L_{\nabla B,\min}$ | $\ge 0$ / $\ge 0.30$ / $\ge 2.4\text{ m}$ | $+0.00217$ / $0.4169$ / $5.55\text{ m}$ | **`+0.00234` / `0.4466` / `5.36 m`** | ✅ 磁阱与 $\iota_{\min}$ 反较起点更优 |
| **截面伸长率包络** | 最大截面伸长率 (`kappa_max`) | $\le 8.000000$ | `4.938149` | **`4.641002`** | ✅ 截面更紧凑（降低 `6.02%`） |

---

## 4. 一键复现命令

```bash
cd contest-20261125/challenge-1b02e50c

# 1. 运行基于 VMEC++ 与 gx_geometry 的 12 工况通量管 ITG 湍流几何抑制优化器
taskset -c 16-19 ../challenge-5f1ea274-3/.venv/bin/python solve_turbulence.py --search

# 2. 调用真实 STELLOPT/NEO、booz_xform 与沿场加密网格复核生成完整审计报告
taskset -c 16-19 ../challenge-5f1ea274-3/.venv/bin/python validate_native.py
```
