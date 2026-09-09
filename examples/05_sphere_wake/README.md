# 8 m 球体尾流 incflo + Travel-Time Tomography 基准

本目录把长水道球体绕流的 incflo 正向模拟、四侧壁多圈阵列、解析有限频率
Travel-Time 正向模型、局部无散 curl 基反演、阵元数量/圈数/频率消融及关键图片整理为
一个独立可复现子项目。它不会改变仓库原有的方管和 unit-ball benchmark。

## 先读结论与结果状态

- incflo 物理域为 `1.5 x 1.5 x 8.0 m^3`，球直径 `0.30 m`，`Re_D=500`，主流沿
  `+z`，速度 `0.5 m/s`。
- 主阵列只使用四个 `x/y` 侧壁，不使用入口/出口端面。参考布局为 41 圈、每圈 48
  阵元，共 1,968 个物理阵元；每个局部切片只使用最近 5 圈，即 240 阵元和 21,600
  条有向互易跨壁射线。
- 主反演使用 100 kHz、5 点 Fresnel fat-ray、6,075 个局部无散 curl 模态、GCV
  选谱秩/岭参数以及固定种子 `20260807`。
- 已完成的 incflo 多圈 19 截面结果平均联合 L2 为 **26.44%**，平均分量 L2 为
  **28.77% / 28.20% / 27.17%**，平均 correlation 为
  **0.951 / 0.948 / 0.962**。

> **重要状态说明：** incflo 流场图片使用同一网格、同一时刻的 incflo 无球基线；但下方
> 已完成的多圈 TT 数值表仍使用 `provisional_cross_solver` 基线，即 incflo sphere
> snapshot 减 PyTorch/MAC 无球基线。它们适合比较阵元密度、圈数和基函数的相对趋势，
> 不能作为最终 matched-incflo 绝对误差。匹配的 incflo baseline HDF5 已存在并记录在
> `results/provenance/raw_artifacts.json`，但尚未按同一协议重跑整套 TT sweep。

![incflo 匹配基线尾流总览](results/figures/incflo_forward/incflo_matched_wake_overview_t30p969.png)

## incflo 流体模拟设定

### 3.1 物理与边界

| 项目 | 冻结设定 |
|---|---|
| 方程 | 三维、常密度、不可压 Navier-Stokes |
| 域 | `[0,1.5] x [0,1.5] x [0,8.0] m`，`z` 为主流方向 |
| 球体 | 半径 `0.15 m`，球心 `(0.75,0.75,0.75) m`，no-slip EB |
| 密度 | `998 kg/m^3` |
| 动力/运动黏度 | `0.2994 Pa s` / `3e-4 m^2/s` |
| 入口 | `zlo` mass inflow，`(Ux,Uy,Uz)=(0,0,0.5) m/s` |
| 出口 | `zhi` pressure outflow，表压 0 |
| 侧壁 | `xlo/xhi/ylo/yhi` no-slip |
| Reynolds 数 | `Re_D = U D / nu = 500` |
| 球体/无球真值时刻 | `t=30.969 s` |

这是有效黏度低 Reynolds 数基准，不代表真实水黏度下的高 Reynolds 数湍流。

### 3.2 离散和运行

| 项目 | 冻结设定 |
|---|---|
| 精度 | incflo strict double；紧凑 HDF5 速度存为 float32 |
| 网格 | `144 x 144 x 768`，均匀各向同性 `dx=0.0104167 m` |
| 球直径分辨率 | 28.8 cells/D |
| 时间步 | `0.003 s` |
| mature run | 13,320 步，`39.96 s`；每 333 步输出 |
| snapshot | 第 10,323 步，`30.969 s` |
| 离散 | cell-centred velocity、nodal pressure、MOL、显式扩散、EB StateRedist |
| 单卡短跑成本 | 高分辨率约 `1.3775 s/step`，采样峰值显存约 30,402 MiB |

冻结源码见 `../../incflo_sphere_wake_comparison/versions.env`：incflo `46de3367...`、AMReX
`2cf4fbcd...`、AMReX-Hydro `e49df248...`，CUDA 12.8，`sm_120`。

已测的中期 `t=16.983 s` plotfile 为有限值；球体离散体积误差 `0.2349%`，入口/出口
流量代理失配 `6.95e-14`，按最大速度计算的 velocity CFL 为 `0.1799`。incflo 没有在
这些产物中导出原生 nodal/EB divergence，所以原生离散散度门禁仍是 **not measured**；
cell-centred Cartesian divergence proxy 不能替代它。

球体与无球模拟的共同保存时刻对比如下：

![球体与无球 incflo 对比](results/figures/incflo_forward/incflo_sphere_vs_no_sphere_common_times.png)

在 `t=30.969 s`，两者差场三分量全域 RMS 为
`0.02032 / 0.01976 / 0.07456 m/s`。在声学 ROI `z=1.0--7.05 m`，截面平均扰动
RMS 为 `0.00854 / 0.01043 / 0.02589 m/s`。

## Travel-Time 正向模型与反演

扰动真值定义为

```text
U'(x,y,z) = U_sphere(x,y,z) - U_no-sphere(x,y,z)
```

直线小 Mach 走时基线为

```text
Delta T_ab ~= -2/c0^2 * integral_path U'(x) dot t_hat ds,
c0 = 1480 m/s.
```

有限频率版本在每条中心射线周围使用 5 个 Fresnel 横向采样点形成 fat-ray。主实验为
100 kHz，波长 14.8 mm。噪声由 20 kHz 参考 delay sigma
`2.2371317e-9 s` 按 `1/f` 缩放；每档固定 4 个噪声重复，随机种子 `20260807`。
这些是解析 finite-frequency Travel-Time 数据，不是全波声学波形。

反演采用局部各向异性 curl basis：侧壁 envelope 使 `x/y` 四壁速度为零，curl 恒等式
使连续场严格满足 `div U'=0`。参考基为 `13 x 13 x 7 = 3,549` 模态；主密基为
`15 x 15 x 9 = 6,075` 模态。谱秩和 ridge 只由观测数据 GCV 选择，不使用速度真值。

## 传感器布置

![41 圈四侧壁阵列](results/figures/incflo_tt/acquisition.png)

| 参数 | 参考布局 |
|---|---:|
| 使用表面 | `x=0, x=1.5, y=0, y=1.5`；不使用 `z` 端面 |
| 轴向覆盖 | `z=1.05--7.05 m` |
| 全局圈数/圈距 | 41 / `0.15 m` |
| 每圈阵元 | 48，即每面每圈 12 个 |
| 全局物理阵元 | 1,968 |
| 目标切片 | 19 个，`z=1.35--6.75 m`，间隔 `0.30 m` |
| 每目标局部窗口 | 最近 5 圈，物理轴向孔径 `0.60 m` |
| 局部阵元 | 240 |
| 局部有向互易跨壁射线 | 21,600 |
| 端面阵元 | 0 |

完整 1,968 个坐标和 wall/ring 标签在
`results/metrics/incflo_tt/acquisition.json`。

## 主结果：41 圈、48 阵元/圈、100 kHz

19 个切片的 provisional noisy 指标为：

| 指标 | Joint | Ux' | Uy' | Uz' |
|---|---:|---:|---:|---:|
| 平均相对 L2 | **26.44%** | 28.77% | 28.20% | 27.17% |
| 中位相对 L2 | 25.86% | 29.96% | 26.25% | 27.34% |
| 平均 correlation | - | **0.951** | **0.948** | **0.962** |
| 最低 correlation | - | 0.827 | 0.745 | 0.924 |

联合 L2 最好/最差切片分别为 `15.39%`（`z=5.85 m`）和 `37.67%`
（`z=2.25 m`）；12/19 个切片同时满足 joint `<=30%`、逐分量 L2 `<=45%`、逐分量
correlation `>=0.80`。

![19 截面轴向指标](results/figures/incflo_tt/axial_metrics.png)

![Ux 多截面](results/figures/incflo_tt/ux_multiwindow_truth_reconstruction_error.png)

## 每圈传感器数量消融：固定 41 圈

所有档固定 19 个目标、最近 5 圈、100 kHz、6,075 模态与相同噪声；只改变每圈阵元
和由此产生的射线数。以下均为 provisional noisy 平均。

| 每圈阵元 | 全局/局部阵元 | 局部射线 | Joint L2 均值/中位/最差 | Ux/Uy/Uz L2 均值 | Ux/Uy/Uz corr 均值 |
|---:|---:|---:|---:|---:|---:|
| 12 | 492 / 60 | 1,350 | 71.31 / 67.12 / 97.37% | 81.20 / 81.48 / 71.95% | 0.612 / 0.589 / 0.703 |
| 24 | 984 / 120 | 5,400 | 51.44 / 39.20 / 129.90% | 77.23 / 73.28 / 46.56% | 0.762 / 0.779 / 0.893 |
| 36 | 1,476 / 180 | 12,150 | 27.53 / 26.42 / 46.43% | 32.57 / 31.65 / 27.87% | 0.935 / 0.936 / 0.960 |
| 48 | 1,968 / 240 | 21,600 | **26.44 / 25.86 / 37.67%** | **28.77 / 28.20 / 27.17%** | **0.951 / 0.948 / 0.962** |

36/圈是平均质量的明显拐点；48/圈相对 36/圈平均 joint 只改善 1.09 个百分点，但最差
切片改善 8.76 个百分点。因此成本优先可选 36/圈，稳健性优先保留 48/圈。

![阵元密度消融](results/figures/incflo_tt/sensor_density_tradeoff.png)

## 圈数消融：固定 36 阵元/圈

以下比较固定全局 `z=1.05--7.05 m` 覆盖，并在共同的 14 个目标位置
`z=2.1--6.0 m` 评价。每个目标始终取最近 5 圈，所以圈数变化也改变了局部物理孔径。

| 圈数 | 圈距 | 全局阵元 | 5 圈孔径 | Joint L2 均值/中位/最差 | Ux/Uy/Uz L2 均值 | Ux/Uy/Uz corr 均值 |
|---:|---:|---:|---:|---:|---:|---:|
| 15 | 0.4286 m | 540 | 1.714 m | 28.01 / 26.61 / 47.23% | 40.84 / 35.93 / 27.47% | 0.900 / 0.920 / 0.960 |
| 20 | 0.3158 m | 720 | 1.263 m | 24.81 / 23.68 / 36.43% | 31.87 / 29.27 / 24.65% | 0.950 / 0.949 / 0.968 |
| 25 | 0.2500 m | 900 | 1.000 m | **24.47 / 24.03 / 35.10%** | **28.39 / 27.50 / 24.81%** | **0.956 / 0.955 / 0.968** |
| 41 | 0.1500 m | 1,476 | 0.600 m | 28.73 / 27.87 / 41.24% | 35.56 / 34.40 / 28.93% | 0.924 / 0.924 / 0.957 |

20 圈是传感器总数/质量拐点；25 圈略优。41 圈并不证明“圈越多越差”：固定只取 5 圈
会把局部孔径缩短到 0.6 m，混入了轴向角度变化。后续公平圈密度实验应固定约
`1.0--1.3 m` 的局部物理孔径，让密阵列使用更多局部圈。

![圈数消融](results/figures/incflo_tt/ring_count_tradeoff.png)

## 基函数密度

同一 `z=1.65 m`、48/圈、5 圈和 21,600 射线下：

| 基函数 | 模态数 | clean Joint L2 | noisy Joint L2 | noisy Ux/Uy/Uz L2 | noisy corr Ux/Uy/Uz |
|---|---:|---:|---:|---:|---:|
| `13x13x7` reference | 3,549 | 19.93% | 20.12% | 18.98 / 18.08 / 21.24% | 0.982 / 0.983 / 0.978 |
| `15x15x9` dense | 6,075 | **18.40%** | **19.85%** | **17.57 / 16.28** / 21.78% | **0.985 / 0.986** / 0.980 |

加密基对 clean 和横向分量有小幅收益，但 noisy joint 只改善 0.27 个百分点，Uz 略差；
因此继续单纯加密参数网格已经不是主要改进方向。

![基函数对比](results/figures/incflo_tt/basis_comparison_z1p65_provisional.png)

## 综合结论

1. 当前四侧壁条件下，**阵元/射线的方向覆盖比把频率从 100 kHz 提到 300 kHz 更重要**。
2. 局部窗口主实验的成本拐点是约 36 阵元/圈；48/圈主要改善最坏切片。
3. 固定 36/圈时，20--25 圈和约 `1.0--1.3 m` 的局部孔径优于过稀的 15 圈；不能在
   固定 5 圈窗口下把 41 圈结果解读为圈密度的单调规律。
4. 6,075 个无散 curl 模态已足够消除明显块状表达瓶颈；继续增加模式的收益有限。
5. 当前数值表是 analytic finite-frequency TT，并且主 incflo 表仍是跨求解器基线。下一
   个正式动作应是用匹配 incflo baseline 重跑冻结的 20/25 圈与 36/48 阵元方案，然后
   才进入全波声学验证。

## 已知限制

- incflo 原生 nodal/EB divergence 尚未导出，不能用 Cartesian proxy 冒充正式散度门禁。
- 已完成 TT 表的 baseline 是 `provisional_cross_solver`；匹配 incflo 重跑尚缺。
- analytic fat-ray 不含全波多径、壁面反射、有限孔径和换能器响应。
- 当前没有给出 resolution-matrix/PSF 实测 FWHM，不能用波长或基函数间距代替。
- 不同章节明确使用了不同真值、ROI 或 pair budget；只比较各自表内的控制变量趋势。


## 本仓库的复现入口

从仓库根目录执行；Python 模块通过根目录 `pip install -e '.[test]'` 安装。
CFD 冻结输入见 [inputs](../../incflo_sphere_wake_comparison/inputs)，源码版本见
[versions.env](../../incflo_sphere_wake_comparison/versions.env)。已收纳本例所需的 sphere、matched-incflo
和 provisional-PyTorch 三个 HDF5 文件，见 [数据校验清单](../../docs/data_manifest.json)。

```bash
python -m incflo_multiring_tt_study.design_array --output examples/05_sphere_wake/outputs/acquisition
python examples/05_sphere_wake/run.py --mode reference
```

`reference` 使用现有数值表的跨求解器基线，重算 19 个窗口；`--mode matched` 使用
同网格同一时刻 incflo 基线，属于新对照，不能冒用原表误差。计算量远高于阵列绘图。

单窗口命令：

```bash
python -m incflo_multiring_tt_study.run_window \
  --snapshot examples/05_sphere_wake/data/incflo_snapshot_t30p969.h5 \
  --baseline examples/05_sphere_wake/data/provisional_pytorch_baseline.h5 \
  --output examples/05_sphere_wake/outputs/z1p65 \
  --center-z 1.65 --basis dense --sensors-per-ring 48 \
  --baseline-status provisional_cross_solver
```

重建 CFD 的本机命令如下，需要 CUDA 编译器、GNU make、C++ 编译工具和空闲 GPU；
`CUDA_ARCH` 需与实际 GPU 相符，原始参考为 120。构建脚本下载冻结 commit 的上游源码。

```bash
export INCFLO_WORK_ROOT="$HOME/flow_estimation_incflo"
export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_ARCH=120
INCFLO_PRECISION=DOUBLE bash incflo_sphere_wake_comparison/scripts/build_remote.sh
INCFLO_VARIANT=double bash incflo_sphere_wake_comparison/scripts/run_remote_case.sh sphere_mature40_double_strict_high144 0
INCFLO_VARIANT=double bash incflo_sphere_wake_comparison/scripts/run_remote_case.sh baseline_t30p969_double_strict_high144 0
```

脚本名字保留 `remote` 以对应历史实现，但在执行命令的机器上运行，不会自动连接服务器。
plotfile 转换使用 `python -m incflo_multiring_tt_study.export_incflo_snapshot --plotfile PATH --output PATH`，需安装 `.[cfd]`。
原始 AMReX plotfiles 未复制，紧凑 HDF5 已包含用于 TT 的三分量速度。

辅助协议独立见 [例 06：频率消融](../06_frequency_ablation/README.md) 和
[例 07：全 slab 阵元消融](../07_sensor_count_ablation/README.md)。

![Uy 多截面](results/figures/incflo_tt/uy_multiwindow_truth_reconstruction_error.png)

![Uz 多截面](results/figures/incflo_tt/uz_multiwindow_truth_reconstruction_error.png)
