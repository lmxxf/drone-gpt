# `challenge-44c6e035`：《让理想磁场走向工程实现，提升实体线圈在装配偏差下的磁场精度》深度技术拆解

> **[⬅️ 返回仓库首页总览 (`README.md`)](../../README.md)** | **方法论宪章：[`AGENTS.md`](../../AGENTS.md)**

- **赛题目录**：[`.`](./)（`contest-20261125/challenge-44c6e035/`）
- **平台提交编号**：**`attempt_id: 111198`** (`job_id: 85546`，[`submission_receipt.json`](./submission_receipt.json))
- **核心交付物**：[`coils.json`](./coils.json)（SHA-256: `4de7a0be65321f5959f7cdae9912d7b60b369a9675a28e9ea956a4dc3bd67965`）
- **核心求解器**：[`workspace/solve_coils.py`](./workspace/solve_coils.py)
- **完整验收报告**：[`evaluation_report.json`](./evaluation_report.json)
- **标准化 Agent 执行轨迹**：[`submitted_bundle/traces/trace.jsonl`](./submitted_bundle/traces/trace.jsonl)（共 327 步完整工具调用与思考记录）

---

## 1. 问题症结与第一性原理诊断

本题要求针对 Landreman–Paul 准轴对称（QA，$N_{\text{FP}}=2$，有效小半径 $a=1.70\text{ m}$，目标面面积加权 $B_{\text{RMS}}=5.70\text{ T}$）目标磁面，设计 4 条独立基圈（由环向周期与仿星器对称展开为 16 个实体绕组包）。每个绕组包具有固定的有限矩形截面（宽 $w=0.40\text{ m}$、高 $h=0.60\text{ m}$，外接圆管半径 $r_{\text{pack}}=\sqrt{w^2+h^2}/2 \approx 0.360555\text{ m}$），并在弧长质心处承受 128 个规定工况的三维刚体装配偏差（平移各分量 $\delta \in [-0.010, +0.010]\text{ m}$，旋转向量各分量 $\omega \in [-0.001, +0.001]\text{ rad}$）。

对官方公开起点 [`workspace/assets/reference/coils_nominal.json`](./workspace/assets/reference/coils_nominal.json)（源自 SIMSOPT 细丝优化线圈 `biot_savart_opt.json`）进行第一性原理几何剖析与官方 [`workspace/eval.py`](./workspace/eval.py) 完整基线复现（见 [`trace.jsonl`](./submitted_bundle/traces/trace.jsonl) Step 24、Step 71、Step 223），定位出核心症结：

- **整包内侧径向包络严重越界**：评估器对整包最小径向距离采用保守外包络下界公式：
  $$R_{\text{pack,min}} = \min_{j} \left[ \min_{t} R_j(t) - \frac{V_j}{2N} - d_{\text{move},j} \right] - r_{\text{pack}} \ge 2.50\text{ m}$$
  其中 $V_j/(2N)$ 为 $N=16384$ 采样点的曲线覆盖半径上界，$d_{\text{move},j} = \sqrt{3}\cdot 0.010 + 2 r_{\max,j} \sin(\sqrt{3}\cdot 0.001 / 2)$ 为刚体装配扰动引起的最大位移包络。
- **结构性死锁**：由于 $r_{\text{pack}} \approx 0.360555\text{ m}$ 为外接圆半径，**仅转动截面朝向 `rotation` 完全不会改变 $R_{\text{pack,min}}$**。官方起点 `coils_nominal.json` 的整包最小径向距离实测仅为 **$2.337522\text{ m}$**（距离 $2.50\text{ m}$ 硬门槛差 **$0.162478\text{ m}$**），导致 `pack_radius_min_lower_m` 违规（违反量 `0.064991`），被判为未达标（`feasible = False`），落入未达标惩罚分支 $S = 0.5 \times \exp(-\sum v_j^2)$，**基线得分仅为 `0.4978925`**。
- **多约束强耦合挑战**：要把内侧线圈中心线整体向外推移 $>0.165\text{ m}$，同时守住四基圈总长度 $\le 200\text{ m}$（起点已达 $199.7\text{ m}$）、中心线间距 $\ge 0.85\text{ m}$、整包到等离子体距离 $\ge 2.40\text{ m}$、曲率 $\le 0.55\text{ m}^{-1}$、原生材料框架扭转 $\le 0.25\text{ m}^{-1}$，还要在 128 个刚体装偏工况下把有限截面多细丝法向场误差 $J = \max_{k} e_{k,\text{upper}}$ 压到官方参考值 $J_{\text{ref,search}} = 8.788358\times 10^{-4}$ 以下，必须构建精确可微的联合优化器。

---

## 2. 自主求解与二阶自检轨迹亮眼节点

在 [`submitted_bundle/traces/trace.jsonl`](./submitted_bundle/traces/trace.jsonl) 中，智能体严格执行了 [`AGENTS.md`](../../AGENTS.md) 规定的工程闭环与二阶递归制动：

1. **构建 JAX + SIMSOPT 原生伴随梯度混合可微求解器（[`workspace/solve_coils.py`](./workspace/solve_coils.py)，Step 41～58）**
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

---

## 3. 最终官方四档求积与三个错位加密复核验收账本

最终交付的 [`coils.json`](./coils.json) 经过原样官方 [`workspace/eval.py`](./workspace/eval.py) 的完整协议评测（涵盖名义状态 + 128 个刚体装配偏差工况、四档 Gauss-Legendre 截面与空间求积，以及针对名义状态与最坏工况的三个错位网格加密复核），**27 项几何、拓扑与磁场约束违反量全部为 `0.0`（`feasible = True`）**，总得分 **`0.7612337182625892`**：

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

---

## 4. 一键复现命令

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
