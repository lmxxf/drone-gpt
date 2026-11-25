# 公开复算入口

从 `/app` 解压 `background.zip`，公开 `eval.py` 与可信入口由同一源码构建。

```bash
python -m zipfile -e background.zip background
/opt/neutronics/env/bin/python -I background/eval.py reference_configurations/uniform.json \
  --workdir runs/uniform-001 --reward-dir results/uniform-001 --threads 8
```

工作目录必须为空，结果写入指定目录的 `reward.json`、`details.json`。物理数据缺失、锁不匹配或基线未校准为 `evaluator_error`，不是有效零成绩。原始 statepoint 保留在工作目录，不能把旧文件作为新计算输出。

固定镜像提供 OpenMC 0.16.0 及核数据；是否已经完成物理标定，以 `runtime-lock.json` 和 `reference/baseline.json` 为准。作者模式 `--physics-only` 可使用实际核数据执行输运，保留原生结果并标记 `not_ranked`；它不会通过正式冻结检查或输出可排名成绩。

正式评分在干净且只读的可信运行环境重新生成输入，公开脚本供搜索和复现使用。核数据与程序不放入 background.zip，随固定镜像单独提供。外部镜像的匿名拉取及权限仍需发布前验证。

当前冻结状态：300k双源参考与独立复验通过；正式原生评分重放feasible，reward=0.7499724633，耗时982.6秒。见 `reference/calibration/official-replay-summary.json`；历史低统计失败仍保留。本次验证范围为固定镜像中的作者控制可信评估沙箱，不代表未来平台隔离认证或在线发布。
