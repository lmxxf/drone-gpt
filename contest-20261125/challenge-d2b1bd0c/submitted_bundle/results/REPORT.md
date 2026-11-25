正式本地验收：feasible=True，p_bar=332770.817557358 Pa，S=0.624604063494。
起始设计正式基线：p_bar=97759.126574723 Pa，S=0.328316138280。平均压力提高 240.3987%。
独立打包复核了全部六个平衡、90 次正常终止的 DCON 和 19674 次完整气球模积分。所有 n=1–5、规定相位、Mercier、q 下限、平衡残差和粗细网格一致性均通过。

| 工况 | 平均压力 Pa | q_min | 最小总能量（含相位） | Mercier 最大值 | 气球模节点 |
|---|---:|---:|---:|---:|---:|
| broad/coarse | 332770.883257 | 1.11029574 | 0.06239153 | -0.25115877 | 0 |
| broad/fine | 332770.817557 | 1.11031539 | 0.06215456 | -0.25115935 | 0 |
| nominal/coarse | 333001.744728 | 1.11249330 | 0.06439533 | -0.25115983 | 0 |
| nominal/fine | 333001.678394 | 1.11251436 | 0.06405233 | -0.25116011 | 0 |
| peaked/coarse | 333309.646550 | 1.11542901 | 0.06712701 | -0.25116079 | 0 |
| peaked/fine | 333309.581388 | 1.11545194 | 0.06683116 | -0.25116130 | 0 |

搜索使用原生粗网格筛选、七维空间填充、坐标爬山、形状/压力联合边界探测及高压力区原生平衡审计。最终仅从完整 formal 通过的设计中选择本次搜索得到的最高平均压力解；未证明连续设计空间的全局最优。

公开 evaluator、equilibrium、dcon、geometry 与 protocol 均逐字节匹配 background.zip。原生源码包的 SHA-256 已核对。构建环境及兼容性修改详见 build-notes.txt。OpenBLAS 对照失败的运行已隔离；最终报告使用参考 BLAS 重跑结果。

在原题目录运行 `source env.sh`，再执行 `taskset -c 4-7 .venv/bin/python -I workspace/physics/eval.py submission_outputs/plasma.json --binary native/GPEC/bin/dcon --workdir runs/recheck-final --mode formal`。结果目录必须尚不存在。

重建入口为 build_native.sh（Ubuntu Noble ARM64）；优化入口为 solve_plasma.py，支持 --latin、--audit、--climb、--pressure-boundary、--de、--local 和 --formal。formal 标签与候选绑定，筛选记录不能替代正式验收。
