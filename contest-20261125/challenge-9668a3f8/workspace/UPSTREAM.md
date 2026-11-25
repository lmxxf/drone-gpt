# T0 的维护来源与适配

## 物理与环境来源

| 包 / 数据 | 固定版本与来源 | 许可证 |
|---|---|---|
| TORAX | [官方仓库](https://github.com/google-deepmind/torax)，发布包 1.4.3 | Apache-2.0 |
| Gym-TORAX | [官方仓库](https://github.com/antoine-mouchamps/gymtorax)，发布包 1.1.1 | MIT |
| QLKNN_7_11 | TORAX 原生 `qlknn_7_11_v1` 入口与 `fusion_surrogates==0.4.6` 发布模型 | 已原样保留 Fusion Surrogates 随包 LICENSE；附加模型声明仍须核查 |
| CHEASE 几何 | TORAX 1.4.3 `torax/data/third_party/geo/iterhybrid.mat2cols` | 保留上游数据来源与 Apache-2.0 包许可证；仍须核查第三方数据附加声明 |
| 参考配置 | TORAX 1.4.3 `torax/examples/iterhybrid_predictor_corrector.py` | Apache-2.0 |

包许可证与原生 QLKNN 模型说明原样放在 `environment/reference/`。发布包来源 SHA256：

```text
torax-1.4.3.tar.gz
b5d72d6a871615db800190447c8a5e050617540ac8ea1205b02f83d9d70a8470
gymtorax-1.1.1.tar.gz
8adf07ac4140fba81e350a057831ad897b889efa5ffcdd4df3d5e93854d00977
iterhybrid.mat2cols
11d43613bf3b508b1b3d41c63ddefe642d725bf9de8dc0da230ffd3223400e08
```

版本来自 PyPI 官方元数据。当前完整40条原生轨迹、全程时间加密及独立 Bohrium 双网格重放已完成；维护代码、模型和执行库731项指纹在独立环境核对一致。GitHub 当前 main 的提交不是这里的发布包版本。

## 对公开配置的明确改变

当前定义为 `thermal-feedback-supplied-heat-20261008-v4`。保留 CHEASE 几何、D/T、QLKNN、α 自热、能量交换、辐射和欧姆热，原生只演化电子/离子热方程。密度固定为 ρ=0/0.9/1 上的 8/6.2/2.5×10¹⁹ m⁻³；移除不再生效的粒子源。初始温度为 15/4.5/0.2 keV，不冒称稳态或实验剖面。

电流用原生 `initial_psi_mode=j`、`initial_j_is_total_current=true`、`current_profile_nu=0.5` 初始化，`hires_factor=16`，然后固定。bootstrap/ECCD 原生诊断不代表本题求解自洽电流扩散；维持背景的外部伺服成本不计入。电流与密度阈值是背景准入，非候选约束。

通用加热初始 28 MW、中心/宽度 0.25/0.25，电子比例 0.66；ECRH 初始 15 MW、中心/宽度 0.15/0.15，原生 current_drive_efficiency=0.25。执行器上限仍 33/20 MW。扰动为原生 Z_eff 插值及温度初态乘数；无粒子补给控制。原生线性 predictor-corrector 四次校正、固定步长，200/400 格和 0.005/0.0025 s 是当前固定配置，参考全场景与400格时间加密已通过原2%/.02门槛。原生 adaptive_T_source_prefactor=2e13，用于减少固定pedestal惩罚边界的网格偏差；原2e10的失败结果仍保留。旧全耦合及更粗网格失败保留于证据目录，不再作为当前配置。

总产出新增原生 P_fusion 接受步梯形积分，所有场景的共同最低交付量等于已测唯一名义400格参考积分；终端储能独立约束。该统计不是重写物理方程。

修改记录在 `environment/torax-config.json` 与 `case.json`。没有修改维护包的求解器、输运通量、聚变反应率、α 沉积或电流驱动物理。

## 适配器调用位置

`tests/plant.py` 的基类是 `gymtorax.envs.base_env.BaseEnv`；每个转移调用上游 `BaseEnv.step` → `ToraxApp.run` → TORAX 固定时间步入口。配置和执行器空间是上游抽象接口要求的实现，不是另写一套模拟器或优化 driver。

调用 `GenericHeatAction` 与 `EcrhAction`，只开放功率，中心与宽度用相同的上下界固定。Gym-TORAX 的 `ramp_rate` 表示每次调用的最大差，本题按 `4 MW/s × 原生固定步长` 转换；0.5 s 策略周期的总变化上限为每路 2 MW，单窗口线性斜坡在原生边界处分割。上游命令会裁剪；本题先检查越界，并把任何意外裁剪当成错误，避免裁剪动作获益。

本题不采用 Gym-TORAX 1.1.1 的 `NbiAction`。源码中默认换算 `nbi_w_to_ma=1/16e6` 直接赋给 TORAX `generic_current.I_generic`；前者名字及默认数值采用 MA 口径，后者原生配置单位为 A，需要单独确认单位契约。避开该入口，也不把 Gaussian 热源写成真实中性束模型。

上游在失败时可能保留前一模拟状态。本题额外检查原生 `SimError`、非线性求解器状态和实际推进时间；有限的旧诊断不等于成功转移。上游 restart 时间处理还需要单独复核，所以当前不利用它伪造稳态初始轨迹。

QLKNN 范围检查调用原生 `_prepare_qualikiz_inputs` 与 `get_model_inputs_from_qualikiz_inputs`；唯一额外坐标变换原样遵循 TORAX 1.4.3 `QLKNNTransportModel._combined` 的 ε 归一化。合并模型的 `inputs_and_ranges` 把边缘 7D 与核心 11D 数据范围合并，不能把边缘的大梯度上限用于核心。本题在实际激活核心面使用发布模型 README 的 **11D core** 范围，原样说明与源文件 hash 在 `qlknn-core-domain.json`。在训练矩形内也不构成误差认证。

激活区依据上游内外补丁严格边界确定；原生平滑矩阵排除补丁区贡献，补丁边界面不算 QLKNN 贡献。固定时间步下初始态及每个接受态都检查，Newton 内部试探态与接受态之间没有连续有效性声明。细网格曾在 0.17 s 检出负 Ate，实际拒绝了该候选；不会通过裁剪、把外推当成功或只检查 0.5 s 末态掩盖它。检查只读模型输入，没有重新计算通量。私有原生入口受固定版本与源文件哈希约束；版本升级须重新复核，不能静默兼容。

`E_external_total`、`P_fusion`、`W_thermal_total`、`q_min`、`q95`、`beta_N`、`n_e_line_avg` 和 `fgw_n_e_line_avg` 来自原生后处理。密度是 m⁻³，温度是 keV，功率是 W，累计能量是 J。所谓 axis 输入实际取最内有限网格，题面明确这一点。

原生 `P_external_total` 实际统计吸收前的辅助功率与欧姆功率，`E_external_total` 对它做梯形累计。本题固定两路吸收系数为 1，逐接受步验证 `P_external_total=P_aux_total+P_ohmic_e`，并独立核对原生累计能量，因此这里才可称外部吸收热能。不能把这个口径推广到部分吸收情形。原生后处理定义 `P_fusion=5×P_alpha_total`，带有该维护包的功率换算近似；不另行修改它生成更好看的 Q。

Gym-TORAX 导出的 NetCDF 没有这些变量的单位 attrs。单位依据固定 TORAX 源码和已知初始剖面交叉检查，不伪造 NC 标签。4开发与16正式场景、每场景两档30 s轨迹，以及D=1/T=0时全部101接受态聚变/alpha源精确为零的退化检查均已有真实凭据。`P_SOL_total` 使用平滑后的 dW/dt；不能用它冒充独立边界热通量或严格能量闭合。规定基座通过原生 adaptive source 与 cell_integration 独立审计：40条参考接受步右端点正向净供热均为0；初态梯形正项另列，不能冒称连续时间从未正向供热。v4对任意候选把接受步的正向净pedestal热量计入主目标，负向热汇不抵扣。原生辅助/欧姆梯形累计与右端点积分的最大参考差约214 J；FVM原生项闭合和储能差分残差单独报告，不冒称完整严格能量守恒。

## FusionX 编写的小工具

`tests/eval.py`、`tests/controller.py` 与 `batch/` 没有上游来源；新增的native_budget_audit、extract_reference、summarize_ensemble和runtime_fingerprint分别审计原生离散项、提取实测目标、复用正式评分聚合及记录实际源/二进制指纹；它们只验证候选格式、配置调用、因果时序、误差、物理诊断统计与公开评分，不实现领域物理。标准参考图是常功率图，既不是领域优化 driver，也不是未经验证的专家基线。原生失败、缺数据、模型越界和发布证据缺失会明确报错。

本地复核使用独立 conda。即使使用QLKNN，TORAX1.4.3的eager注册也要求TGLFNN、TensorStore和IMAS数据字典。Bohrium科学镜像用85份官方SHA256核验制品完成安装、pip check与CPU原生导入；本地与远端Python3.12.14二进制因环境构建/前缀不同而分别记录散列，不冒称二进制相同。独立作业20862432完成合法ONNX的两条30 s轨迹，物理量跨CPU仅有约10^-12量级相对浮点差异；完整正式入口重放另列收据。镜像数据层只添加公开题目资源，不安装依赖、不包含正式种子；干净grader另行注入可信tests并只接受controller.onnx。

## 论文与定义依据

- [TORAX 原始论文](https://arxiv.org/abs/2406.06718)：维护中的核心输运求解和 RAPTOR 比较，不是本题所有场景的独立验证。
- [Gym-TORAX 论文](https://arxiv.org/abs/2510.11283)、[2026 Software Impacts 发表版本](https://doi.org/10.1016/j.simpa.2026.100829)：原生闭环 RL 接口与公开示例，不是本题已经优化成功的证据。
- [TORAX 官方物理说明](https://torax.readthedocs.io/en/stable/physics_models.html)：源项、输运与模型范围。在线 stable 可能更新，复算以固定发布源码为准。
- 飞书领域图用于专家兴趣映射；不作为数值门槛、训练模型有效域或实际控制数据的来源。

正式入口实测：Bohrium作业20862717在c32_m128_cpu以16个独立场景进程完成全部32条30 s轨迹；实际test.sh给出valid1、feasible0、evaluator_error0，进展分0.27055144846898393，墙钟3467.58694s。下载后再次检查6001/12001接受态及全部模型域。此成绩不代表可行或专家最优；与作者重算差7.56e-13。正式r1因baseline导出字段缺失而失败的收据保留。最终公开镜像只含开发资源，正式种子由干净grader注入；完整收据与源身份在作者evidence/native-v4-20261008中。
