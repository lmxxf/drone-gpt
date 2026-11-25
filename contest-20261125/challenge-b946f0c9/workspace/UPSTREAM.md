# 来源与自有代码边界

- OpenMC 原生 C++ 输运与 Python API：MIT 许可，v0.16.0，提交 `617d35a5063c57796b43428bc401e627d2011046`，维护方 [openmc-dev/openmc](https://github.com/openmc-dev/openmc/tree/617d35a5063c57796b43428bc401e627d2011046)。没有改写输运、碰撞、材料混合、反应截面或产氚算法。
- 核数据库必须由维护方公开 ENDF/B-VIII.0 处理库或经审计的对应处理流程提供，记录处理来源、温度、文件 SHA-256。本版已按 `environment/provenance/` 与 `environment/runtime-lock.json` 冻结实际来源和文件散列，并在独立固定镜像中逐项核验；不得用库名替代实际文件锁。
- [neutronics-workshop](https://github.com/fusion-energy/neutronics-workshop) 提供维护中的固定源、材料及包层示例。本题使用相同原生 API，不复制上游优化流程。研究资料缓存不充当运行依赖。
- `tests/eval.py` 没有上游来源：仅验证输入，构建本题固定 CSG 几何及材料，调用 OpenMC，读取真实 statepoint，按批汇总与计算公开评分。CSG 面和材料均调用维护中的 OpenMC 对象；不实现新的中子求解器。
- `batch/freeze_reference.py` 和 `batch/audit_runtime.py` 没有上游来源，只检查实际证据与运行时哈希、写入标定清单；不生成物理数据。
- `batch/calibrate.py` 没有上游来源，只依次调用已维护的原生检查和同一评估入口，产生参考、独立种子和粒子数加倍证据；不实现物理或优化算法。
- `batch/build_package.py` 只生成公开副本和标准题包；`batch/measure.py` 只调用同一个评估入口的非排名测量模式，没有第二份物理流程。
- 题目几何是 FusionX 的公开组件测试定义，与任何现成完整反应堆 CAD 不同。固定源的入射归一只能建立组件产氚率。

- 2026-10-08：300k双源参考、独立复验、200k非排名对照和正式隐藏种子重放均使用同一原生实现。正式重放真实得分0.7499724633473455；256个statepoint散列与全部可信评估文件已核对。历史未校准状态及100k/200k精度失败记录保存在 `environment/reference/calibration/history/` 和测量文件中。校准与包装工具不识别参考JSON给予预设分。
