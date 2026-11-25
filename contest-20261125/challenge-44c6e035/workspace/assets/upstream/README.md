# SIMSOPT 原生优化示例位置

示例使用固定源版本 `9e027eac38028d57aa23777be52a781aa860e347`，维护者为hiddenSymmetries。题包不复制或改写上游优化流程；需要运行时从上游固定位置取得源文件：

- [名义二阶段线圈优化](https://github.com/hiddenSymmetries/simsopt/blob/9e027eac38028d57aa23777be52a781aa860e347/examples/2_Intermediate/stage_two_optimization.py)
- [随机线圈优化](https://github.com/hiddenSymmetries/simsopt/blob/9e027eac38028d57aa23777be52a781aa860e347/examples/2_Intermediate/stage_two_optimization_stochastic.py)

输入路径要改为本题`/app/reference/target.vmec`，并使用`case.json`中的长度倍率、场强、四条曲线/十六条展开、阶数与统一条件。示例的原始TEST_DIR假定SIMSOPT源码仓库目录，不能原样当成任务运行目录。

原生随机示例针对曲线参数上的形状误差，包含系统与独立部分；C0则冻结物理空间的独立刚体装配误差，不能照搬GaussianSampler幅度和相关尺度。误差后必须计算完整环，不能借目标的仿星器对称性恢复实物场对称。

原生优化中的长度、曲率、距离惩罚是搜索手段，本题明确的全部条件才是验收。得到候选后用公开eval.py真实重算；优化日志或自报SquaredFlux不能代替它。题包的SIMSOPT-LICENSE用于固定公开输入及其原生转换来源。

## 本版有限截面入口

[SIMSOPT 1.10.6 原生有限截面优化示例](https://github.com/hiddenSymmetries/simsopt/blob/v1.10.6/examples/3_Advanced/stage_two_optimization_finitebuild.py)。本题调用 `create_multifilament_grid`、`FrameRotation`、`FramedCurveCentroid`、`CurveFilament` 和原生 BiotSavart/SquaredFlux，不重写场或导数内核。

本题截面固定 0.40×0.60 m，最终评估使用 NumPy `leggauss(N)` 的节点和权重：位置 `(0.20*ξ_i,0.30*ξ_j)`，电流 `I*q_i*q_j/4`，Σq_i=2。使用 SIMSOPT 原生 `FrameRotation(order=2,scale=1)`、`FramedCurveCentroid` 和 `CurveFilament`，把节点交给原生 BiotSavart/SquaredFlux。没有自写场内核。

上游 `create_multifilament_grid` 提供等间隔细丝，适合参考其对象图；它本身不生成本题的 Gauss 节点和非等权电流，不能原样作为正式求积。全部同包细丝共用装配刚体变换。加密不改变绕组尺寸和总安匝；上游示例的几何尺度、目标和优化惩罚须按本题配置。
