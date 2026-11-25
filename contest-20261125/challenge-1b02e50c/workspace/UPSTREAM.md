# 程序与数据来源

本题不实现平衡、导心轨道或回旋动理学方程。新增代码只准备输入、调用维护中的程序、解析原生结果、计算统计估计与公开分数。

| 来源 | 固定版本或维护位置 | 用途 |
|---|---|---|
| A 磁评估 | ../task_A/tests/eval.py；SHA-256 f8af3835fd88130409385891a695f21c774d0ce3d0c990fdf5ca5af7b6bcd83a | 同条件几何、VMEC、booz_xform、NEO、SIMPLE、SIMSOPT 接口；打包生成副本 |
| STELLOPT | 780f5db557eacc7ca1160f4ca5a10a80922e4f26 | VMEC2000、NEO；沿用 A 中已记录的 NEO 手性修正 |
| SIMPLE | 9269e8614e581ede8768e0bccb003b52db87e3be | 4096 条 α 导心轨道 |
| libneo | f1bef44e80c4508f04594d4d2ca120eaf17564f8 | SIMPLE 依赖 |
| booz_xform | 21aff118735de4255641aa1b64fc7ab35e8c9767 | Boozer 坐标变换 |
| SIMSOPT | 1.10.6 | 磁场梯度尺度等原生诊断 |
| GX | b88d763a7e622b6597531bfabc8696161a791979 | 原生非线性静电 ITG；与 2025 年数据研究采用的提交一致 |
| gx_geometry | d080a16deaca328aa970f4f0036e61786ac4d67e | 维护方的 VMEC 通量管、沿场长度与 GX 几何导出 |
| QA+well 边界 | doi:10.5281/zenodo.5645413，A 的公开 JSON | 真实几何输入；复制时核对 SHA-256 |

GX 的官方仓库为 [Bitbucket](https://bitbucket.org/gyrokinetics/gx)，源码核对另用论文作者 Jong Youl Choi 的 [GitHub 副本](https://github.com/jychoi-hpc/gx/tree/b88d763a7e622b6597531bfabc8696161a791979)。固定提交的原生非线性 W7-X 输入在 `benchmarks/nonlinear/w7x/w7x_adiabatic_electrons.in`；本题采用其中的电子响应、初始化、耗散与诊断接口，改变题面明确规定的几何、梯度、网格和统计时间。

`eval.py`、`batch/build_package.py`、`batch/native_probe.py`、`image/build-gx.sh`、`image/render_dockerfile.py`和回归测试没有上游来源。它们是FusionX的输入、统计、评分、原生测量、平台构建与打包薄封装，不包含新的物理求解器。统计采用块均值、Student t分布与公开的联合比较规则。公式回归只验证接口和数学关系，不宣称验证了GX物理。

GX平台构建仅调整原生Makefile的目标级头文件搜索顺序，避免conda编译器头文件与GX、VMEC几何模块同名；CUDA/C++物理源码未修改。CUDA12.4依赖实际提供`libnvToolsExt.so.1`，据此显式链接，不自行替代库。GPU主机驱动不打入镜像，构建阶段只允许这一已知外部驱动未解析，其他科学依赖缺失立即失败；新GPU沙箱必须实测驱动。

gx_geometry固定提交的`write_eik_netcdf`正常路径没有关闭Dataset，实际复现中最终绘图占位变量Zplot在GC时才刷新，文件SHA因此改变。原生完整源文件见[image/eik-close/eik_files.original.py](image/eik-close/eik_files.original.py)，修正文件见[eik_files.py](image/eik-close/eik_files.py)，唯一源码改动为最终赋值后的`file.close()`。上游/修正SHA为`14e837643e823fc9aa0574bc863f63e0990d61fa5dd9970ce731977766f5b099` / `bfe0158a5f2f1afa33d5ceb1478ffa6b8acbdc4f464b1f73c1071d1b19d4a6e6`。GX原生reader不读Zplot；全部物理数组逐项相同。镜像172940 FROM原172756 digest，不重编译GX或重装依赖，GX实际SHA仍为`0f14d22252f39f1d3d79af81a180871689396a1c6aab58c41ec91e5557670ca0`。真实新镜像零解已取回并独立读出104个精确零样本。该关闭修正属于维护源码的I/O生命周期修复，不实现新的物理；[构建和原生检查记录](image/eik-close/build-and-native-zero.json)不宣称完整参考校准通过。

固定GX与gx_geometry组合已在Bohrium完成原生零热流与一个非零工况，独立D0镜像172756已构建并完成新沙箱的平衡与零解复算；完整热流与数值适用性尚待标定。原生字段、公共单位和版本组合依证据验证；失败明确报错，不自动改投另一个物理程序或归一化定义。

v0.2的并行入口没有上游来源，是对独立原生GX进程的执行封装，不是新的物理算法或优化driver。每个输入有独立工作目录，GPU设备明确绑定；几何生成、NetCDF读取与统计汇总保持单线程。出现运行失败时释放本次子进程，不改物理模型或把失败补成热流。公开测量入口与可信评分调用相同的输入生成、几何和热流读取函数，测量不生成正式分数。

原生输出的gzip归档没有上游来源，仅是FusionX的无损字节存储封装。每个完成文件在压缩前、解压校验时记录并比较原始SHA-256，确认相等后才释放未压缩副本；测量时在主线程恢复相同原生字节再读NetCDF。关闭的是大体积场快照，热流与自由能诊断、原生采样节奏、时间窗和物理设置均保留。已用已知正负热流的原生字段布局验证归档前后测量完全相同，并验证错误SHA明确报错。

`batch/measure_reference.py` 为 FusionX 原创作者测量入口，无上游同名脚本。它调用现有评估器的 geometry_stage/gx_input/run_gx_jobs/read_flux/block_means/level_statistics；同一几何只调用维护接口生成一次，各原生任务独立目录，NetCDF只在主线程读取。准备与收集产物保留case/源码/平衡/原生二进制身份，失败或未完成只记measured/incomplete_not_ranked，不写评分、calibrated基线或发布通过状态。脚本自身不实现回旋动理学或磁平衡方程。

`native_probe.py magnetic`及`batch/run_native_magnetic_job.sh`没有上游来源，直接调用同一评估器的magnetic_stage完成全部磁约束测量，保留原生指标、违反量和运行事实，不生成比赛分数。此入口使作者能在Bohrium CPU作业中与GX独立并行验证；不重复实现轨道或磁性能计算。

`batch/run_reference_job.sh` 是上述作者测量入口的 Bohrium 作业包装，无上游来源；显式绑定实际申请并核实的GPU，按同case重新求解VMEC并通过原生零驱动自检，执行现有测量程序并归档真实输出和退出码。它不改物理输入，不声明校准通过。

参考证据契约为 `evidence_schema: fusionx-d0-evidence/v2`，是FusionX的发布验收语义，没有上游来源。`load_configuration`按已知九项分别核对assessment；各record必须有非空scope、finding和artifacts，文件继续逐一验证path/SHA。zero_drive要求expected_limit_recovered；parallel_resolution、velocity_resolution、box_size要求adequate_for_declared_scope，needs_refinement或unresolved不能放行；tube_length、parallel_boundary接受definition_verified或sensitivity_measured，finding须如实说明模型敏感性，不要求改变模型后热流相等；reference_replay、runtime_isolation分别要求replay_verified、isolation_verified。

native_solver_benchmark的comparison_met要求upstream_result=passed。documented_historical_discrepancy则要求upstream_result=failed且scope严格为fixed_native_discrete_protocol，并以artifact的role明确引用original_thresholds、measured_difference、discrepancy_analysis三类真实资料；同一真实文件可承担多个角色，但每项引用均核SHA。此分支保留原上游失败，不改阈值，不等同于benchmark通过或独立代码一致。schema只核验记录契约和文件完整性；资料的科学解释仍需审核，不能以哈希或状态词替代数值充分性。所有其他参考身份、十二点正下界、统计及评分要求保留。当前baseline仍uncalibrated，points、J_reference和evidence均为空。

本次只改变canonical评估器的load_configuration证据验收；在途计算使用的17e8677f…快照、case483及所有物理/统计/评分函数未改变。新增回归使用明确标注的临时契约fixture验证数值未解析拒绝、模型敏感性可记录、历史FAIL不可冒充comparison_met与证据SHA完整性，不运行替代物理、不生成已标定参考。
