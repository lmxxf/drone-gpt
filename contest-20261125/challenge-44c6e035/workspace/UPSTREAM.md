# C0 第三方来源与本地转换

物理程序使用维护中的 SIMSOPT 1.10.6、NumPy、SciPy。C0 不实现 Biot–Savart、轨道求解、平衡求解或另一个优化 driver。

固定输入来源为[SIMSOPT](https://github.com/hiddenSymmetries/simsopt)，开发时参考checkout`9e027eac38028d57aa23777be52a781aa860e347`，使用MIT许可并随资源提供全文。优化示例只给出该固定源版本的位置，不在题包复制driver。

| 来源文件 | 本地材料 | 改动 |
|---|---|---|
| `tests/test_files/input.LandremanPaul2021_QA` | `environment/reference/target.vmec` | 原始输入逐字复制；评估时仅按固定倍率缩放表面 Fourier 系数 |
| `examples/1_Simple/inputs/biot_savart_opt.json` | `environment/reference/coils_nominal.json` | 原生加载前四个独立圈，长度/电流统一缩放，阶数 5 补零至 8，转换为严格最小 JSON |
| `examples/2_Intermediate/stage_two_optimization.py` | `environment/upstream/README.md` | 仅固定链接及使用事实，不复制优化流程 |
| `examples/2_Intermediate/stage_two_optimization_stochastic.py` | `environment/upstream/README.md` | 仅固定链接；说明其形状误差与本题刚体装配模型不同 |

文件 SHA256、转换倍率和原生来源指标见 `environment/reference/source.json`。`batch/prepare_inputs.py` 是作者一次性转换工具，运行时题包不依赖 `refs/`。

`tests/eval.py`、`tests/check_native.py` 与打包工具没有上游同名来源，由 FusionX 编写。它们只负责输入校验、原生对象调用、精确刚体坐标变换、几何采样误差界、指标解析和连续分数。全部磁场由原生 SIMSOPT 重算。解析电流圆环、零电流、电流线性缩放、旋转协变和对称展开重建是专门的接口检查。

`batch/run_native_verification.sh`没有上游同名来源，仅调用上述维护入口并记录Python标准库time/resource成本、原生库与输入身份；不含优化方法或物理实现。独立Bohrium构建记录保留于`calibration/evidence/image-*-build-log.json`，各次失败与成功不覆盖。

`batch/check_field_evidence.py`没有上游同名来源，仅从已有原生数组独立核对分量、面积权重和归一化解析，作为作者审计，不进入物理求解或正式评分。它支持已有本地原生目录与真实Bohrium导出归档两个实际证据来源，缺项或不一致直接失败。

制造扫描和安装误差文献提供真实研究动机；本题的独立有界均匀刚体模型是显式任务选择，不声称拟合了某个装置的制造数据库。给定误差组的最大法向场误差不自动认证磁面或粒子性能。

`batch/check_geometry_review.py` 为 FusionX 原创验收小工具，无第三方脚本上游；仅调用 SIMSOPT 和本题评估器，在双倍曲线/曲面/链接网格上重评公开参考几何，不实现磁场或物理求解器。

`batch/check_isolated_oracle.py` 为 FusionX 原创验收脚本，无第三方同名上游。仅在独立干净的 Bohr 评分实例中调用原样 `tests/test.sh` 和原生已知答案检查，记录输入/程序哈希、低权限不可写检查、参考与无效/违规候选分类；不实现物理、不替代线上平台访问验收。求解实例只运行公开参考解，其 JSON 提交经独立传输进入评分实例。

v0.2 的实体排序只使用 SIMSOPT 原生弧长质心、已展开 Fourier 几何和电流建立确定性标签，不实现磁物理。`batch/check_permutation_equivalence.py` 是无上游同名来源的表示不变性验收：通过原生展开及 BiotSavart/SquaredFlux，对第二至第四条基曲线的全部六种置换验证同一128工况下每项原生场指标在原生浮点舍入精度内相同；不把粗档身份检查当积分收敛认证。

`batch/check_canonical_release.py` 是v0.2作者验收脚本，无第三方同名上游；只调用原样test.sh，比较原序/换序的完整原生指标和得分，并核查可信输入前后身份。不实现磁场或优化器。

## v0.3：有限截面与材料框架（2026-10-09）

调用 SIMSOPT 1.10.6 原生 `create_multifilament_grid`、`CurveFilament`、`FramedCurveCentroid`、`FrameRotation` 及其 `frame_torsion()`，通过已有 BiotSavart/SquaredFlux 求场与目标。原生来源为同版本 `src/simsopt/geo/finitebuild.py`、`framedcurve.py`；没有复制或改写物理内核、优化流程。

FusionX 的无上游小工具职责：严格 JSON 接口、固定截面 Gauss–Legendre 求积参数、整包刚体位移、规范实体排序、截面外接圆管几何界、独立空间/截面细化诊断、分数及证据归档。原生圆环轴上场的解析求和、零厚度极限、圆环正弦材料扭转、非零旋转下的对称展开和数组换序均由 `tests/check_native.py` 检查。缺数据、奇异框架或精度未解析明确失败，绝不补零。

v0.2 的细丝校准仍保存在历史证据中，不支持 v0.3 有限截面成绩。当前参考的 `rotation=[0,0,0,0,0]` 是新增设计起点，不称为专业优化过的截面朝向。

本版最终截面数值协议采用 NumPy `numpy.polynomial.legendre.leggauss`，通过原生 FrameRotation / FramedCurveCentroid / CurveFilament 组成绕组，不改写磁场物理。Gauss 节点与归一电流权重由维护库生成；等间隔多细丝探查仅是作者的收敛对照，不作为本版最终评分协议。
