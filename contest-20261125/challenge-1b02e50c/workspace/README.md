# 复算多工况仿星器热流

本题使用固定版本 VMEC2000、booz_xform、NEO、SIMPLE、SIMSOPT、GX 与 gx_geometry。公共评估器与正式入口从同一份维护源码生成。

从任务工作目录执行：

```bash
python -m zipfile -e background.zip background
/opt/stellarator/env/bin/python -I background/eval.py \
  reference_configurations/landreman_paul_2022_QA_well.json \
  --workdir runs/qa-well --reward-dir results/qa-well --threads 64
```

`--workdir` 必须为空。评测重新读取边界并创建原生平衡、通量管与 GX 输入，输出不读取提交者自报热流。资源位置见 resources.yaml。

当前runtime-lock.json为built_unverified，reference/baseline.json为uncalibrated，入口会返回evaluator_error=1和原因。它不会把缺失参考数值变成有效零热流。task.toml引用Bohrium镜像172940的固定digest：它在原生镜像172756上只修正gx_geometry写出NetCDF后缺少关闭的问题，GX字节和物理数组未变。修正镜像的真实计算作业已取回104个精确零热流样本；十二点完整参考和数值适用性仍待校准。主参考批次继续保留其实际原镜像与原输入身份，不重标成新镜像计算。

讨论版保留十二点，拟用64/128网格及八GPU独立运行；原32/64实测不通过精度要求。`--gx-gpus 0,1,2,3,4,5,6,7`可明确指定八卡；实测或作者分组也可显式指定实际可用设备，例如`--gx-gpus 0,1,2,3`。物理输入和评分不随并行度改变。原生NetCDF读取与几何生成在主线程执行，避免HDF5线程安全问题。

大体积场快照关闭，热流和自由能诊断开启且采样节奏保留。完成的NetCDF先gzip无损归档、核对还原SHA，再释放未压缩副本；测量在主线程还原完全相同的原始文件。完整存储和并行耗时以原生批次为准，不能仅用单点小文件估计全流程预算。

完整评分尚未标定时，公开`measure_native.py`仍能检查同一原生接口，命令见题面。它从边界求平衡或运行一个指定GX工况，输出原生热流和统计诊断，不产生比赛分数；不是识别参考边界并返回预存数值。

冻结后提供独立 D0 镜像 digest、只读物理程序、安装来源和真实依赖校验值；公开背景包中的 image-reference.json 记录它们。外部复算还需可拉取的镜像和 NVIDIA GPU。未核对拉取权限、完整原生计算与成本前，不宣称可以完整复现。

reward.json 给出平台分数与 valid、feasible、evaluator_error。details.json 给出明确状态、七项要求、每点时间平均热流与区间、J、原生输入输出校验值和资源统计。低分首先按状态解释；同一个缺失数据不会被填成物理损失或零热流。
