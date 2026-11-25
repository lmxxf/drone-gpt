# C0 公开评测复现

本包对应当前 `instruction.md` 与 `case.json`。`background.zip` 从唯一维护源码生成，包含 `eval.py`、原生已知答案检查、公开输入和文件清单。正式保留误差向量、Oracle 和可信入口不在公开包里。

```bash
cd /app
python -m zipfile -e background.zip background
/opt/stellarator/env/bin/python -I background/eval.py reference/coils_nominal.json \
  --assets background/assets --scenarios search_scenarios.npz \
  --workdir runs/reference-measurement --reward-dir results/reference-measurement --measure
```

评分源码不从镜像中的旧路径加载：上面的入口始终指向当次题包解压所得的 `background/eval.py`，其同目录文件由包内manifest记录身份；每个版本使用全新的解压与运行目录。固定镜像只提供原生物理依赖，不内置题目的eval.py或background.zip。

解释器属于 C0 的固定镜像；当前 `release-lock.json` 记录实际镜像状态。未核验镜像前，在已安装 SIMSOPT 1.10.6、NumPy 和 SciPy 的 conda 环境中可调用相同入口取得实际物理测量，不能宣称固定镜像已经可公开复现。

`--measure` 保存真实场及物理量，不产生竞赛成绩，reward 保持 0 且状态为 `measured_unfrozen`。正式模式要求对应误差组的基线、固定镜像以及干净可信环境全部通过；出题或基础设施缺项报 `evaluator_error=1`，不算选手物理失败。

`details.json` 保存约束、两组最坏误差、各工况的四档积分值和误差。工作目录保存当前提交、工况快照和实际线圈原生场 `field_*.npz`；数组包含位置、磁场、单位法向和面积权重。新评测使用空目录，禁止复用其他候选的原生输出。

正式保留组与公开搜索组具有相同分布。有限截面版必须使用本版 `search-baseline.json` 的完整公开组校准，不能沿用细丝版数值。参考中心线加上厚度后并不自动满足整包约束。四档场积分分别检验空间与截面求积，扭转使用两档原生框架计算。运行时间与当前校准状态见参考记录，不沿用旧版分钟数。

最终截面协议为固定矩形上的 Gauss–Legendre 求积，电流按积分权重分配。四档中最后一档使用5×5截面；名义和各组最坏工况另用7×7截面、加密及半格错位目标面复核。中点求积探查仅用于作者数值对照，不提供正式参考分数。
