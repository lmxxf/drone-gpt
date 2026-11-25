# 物理程序来源

磁平衡采用 [Open FUSION Toolkit / TokaMaker](https://github.com/OpenFUSIONToolkit/OpenFUSIONToolkit) `26.9`，上游标签 `v26.9`，提交 `73d8eb09bce0a4f4c56b88f13fcf36f951c53e91`，许可 LGPL-3.0-only。Python 接口来自官方发行版，原生库从随附的对应源码构建，没有改写平衡方程。

`src/physics/grad_shaf_util.F90` 的 `gs_save_eqdsk` 有两项导出修正：移除写文件前的单精度转换；在序列化前将磁通整体平移，使磁轴磁通为零。常数平移不改变磁场、磁通导数或归一化磁通，可以减小磁轴附近有限位数存储造成的相消误差。保留标准 `E16.9` 格式和 COCOS 约定。原生平衡求解与压力、电流剖面不变。

完整对应源码、许可及上游构建入口包含在 `oft-source.tar.gz` 中，SHA-256 为 `ccb49ba66cc2b9b8f06f438e74198ff9fe0a99b02e3d86d1b5dce149d2351d67`。归档保留动态库的完整对应源码、构建文件和上游解析测试，省略展示用笔记本、大型示例网格及文档图片。原生库采用可替换的动态库形式。源码构建使用 CMake Release、GNU 编译器 15.2、HDF5、METIS、SuperLU、ARPACK 和 libxml2；具体依赖由同目录 conda 清单固定。

稳定性采用 [Princeton University / GPEC](https://github.com/PrincetonUniversity/GPEC) 的 DCON，提交 `e3b6ec6986a7cb349d043a1a25eb8ef9cbebbc18`，许可 MIT。随附源码归档包含原生程序及上游许可。

`dcon/bal.f` 和 `dcon/dcon.F` 增加气球相位、积分容差、初值归一化和完整积分诊断。归一化利用线性齐次方程的尺度不变性，防止极小初值落入绝对误差容限；稳定性方程未改变，未启用这些选项时保留原默认行为。归档不包含试验性的 TokaMaker i-file 读取分支，正式接口使用 GEQDSK。

源码归档 `gpec-source.tar.gz` 的 SHA-256：

```
35c08f36361199636b188e4992a082294d1fbc8e661b6ad874841e20009556b3
```

几何参数化、输入验证、调用编排与结果汇总由 FusionX 编写，没有独立上游来源。它们调用上述维护程序及 Shapely、SciPy，不另行实现磁平衡或 MHD 稳定性求解器。

本题研究规定剖面下的目标平衡。其设计方法参考 [ARIES-AT 的平衡与稳定性研究](https://aries.pppl.gov/LIB/REPORT/ARIES-AT/FINAL/aat-3-equilibrium.pdf)，但不复用该研究的成绩或完整工程结论。
