# 从一个边界，复现一次完整评测

`background.zip` 提供本题的公开评测背景包。拿到一个边界 JSON，就可以用与题目相同的物理程序和设置检查它。先复算公开构型，再比较自己的候选，可以看清每次形状改变带来的后果。

包内文件如下：

| 文件 | 用途 |
|---|---|
| `instruction.md` | 设计问题、物理背景、提交格式、六项门槛和评分公式 |
| `eval.py` | 复现用评估入口，调用 VMEC2000、booz_xform、NEO、SIMPLE 和 SIMSOPT |
| `image-reference.json` | 题目使用的固定镜像引用、算力配置和预装软件版本 |
| `reference_configurations/` | 五个可直接读取的公开边界，格式与提交相同 |
| `resources.yaml` | 物理程序、数据和文献的版本与位置清单 |
| `manifest.json` | 包内文件的 SHA-256，以及各文件的来源路径 |
| `README.md` | 本指南 |

镜像提供物理程序，`eval.py` 串起各阶段的计算。镜像通过 `image-reference.json` 中的引用单独拉取；ZIP 中不包含镜像层。

## 在本题运行环境中

物理程序已经安装。从 `/app` 执行：

```bash
python -m zipfile -e background.zip background
/opt/stellarator/env/bin/python -I background/eval.py \
  reference_configurations/landreman_paul_2022_QA_well.json \
  --workdir runs/qa-well-001 \
  --reward-dir results/qa-well-001 \
  --threads 96
```

`--workdir` 必须不存在或为空。每次复算换一个运行目录；结果保存在 `--reward-dir` 指定的位置。评估自己的设计时，把参考 JSON 路径换成 `boundary.json`。

## 在自己的机器上

解压 `background.zip`，进入包含 `eval.py` 的目录。下面使用题目当前引用的镜像，在 Linux x86_64 环境中运行。主机需要 Docker，以及该镜像的拉取授权；没有镜像权限时，可以使用平台提供的任务环境，按上一节的命令复算。

```bash
docker pull registry.dp.tech/dptech/dp/native/prod-21128/21221/fusionx-compact-vacuum-stellarator:4
docker run --rm --cpus 96 --memory 128g \
  --mount type=bind,src="$PWD",dst=/app \
  --workdir /app \
  registry.dp.tech/dptech/dp/native/prod-21128/21221/fusionx-compact-vacuum-stellarator:4 \
  /opt/stellarator/env/bin/python -I eval.py \
  reference_configurations/landreman_paul_2022_QA_well.json \
  --workdir runs/qa-well-001 \
  --reward-dir results/qa-well-001 \
  --threads 96
```

程序输出通过挂载保存在主机的 `runs/` 和 `results/` 下。不同算力会影响耗时与超时结果；若要与题目的算力条件一致，使用 96 核和 128 GiB 内存。`--threads` 控制 SIMPLE 的 OpenMP 线程数，不改变物理门槛。

## 看结果，再改变形状

先打开 `reward.json`：`valid=1` 表示得到有效评测，`feasible=1` 表示满足全部六项要求，`reward` 是分数。`evaluator_error=1` 表示评估器本身发生错误，具体原因见 `details.json`。

再看 `details.json` 的 `metrics` 和 `completion`。例如 `f_alpha=0.10` 表示测试 α 粒子损失 10%，超过 5% 门槛；它对应的完成度为 0.5。`eps_eff_max` 同样使用比例，`lgradb_min` 用米。`stages` 保存平衡、Boozer 变换、NEO 和粒子轨道的诊断，便于区分某项物理指标未过与计算没有完成。

Landreman–Paul QA+well 的一次完整评测约需 3 小时。可以先用计算量较小的物理目标筛选边界，再对有希望的候选做完整复算。最终需在 `/app/boundary.json` 留下准备提交的边界。
