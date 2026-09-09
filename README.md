# Flow Estimation from Ultrasound Travel Times

用超声互易走时差估计三维流速：从方管轴向流、球内解析旋涡到球体绕流尾迹。
仓库包含 Python 求解与反演代码、逐例配置、现有成像结果、数值指标和测试。
主要说明使用中文，代码模块沿用原实验名称，方便将公式对应到实现。

本仓库于 2026-09-10 从 `Water_USCT_3D_benchmark` 整理。
**下表是归档结果，不是本次重新运行完整模拟得到的结果。** 本次执行了迁移后的测试和轻量复现检查。

## 1. 问题与结果导航

| 例子 | 观测与反演 | 具体规模 | 已有成像结果 | 状态 |
|---|---|---|---|---|
| [01 方管轴向流](examples/01_square_duct/README.md) | 线性化 Euler 波形；走时候选与幅值先验选择；开发 FWI | 0.1×0.1×0.2 m；24 阵元；50 kHz | 先验候选剖面 L2 4.814%；开发 FWI 4.473% | 不是一般三分量恢复；最终选择幅值先验 |
| [02 球内旋涡，R=10 cm](examples/02_ball_10cm/README.md) | 全波压力→走时差→TSVD | 96 阵元；100 kHz；364 模态 | clean / 5 ns / 10 ns：2.535 / 8.700 / 14.043% | clean 未达到 2% 门槛 |
| [03 球内旋涡，R=50 cm](examples/03_ball_50cm/README.md) | 同上，声学相似缩放 | 96 阵元；20 kHz；364 模态 | 2.535 / 3.067 / 4.618% | clean 未达到 2% 门槛 |
| [04 相似缩放](examples/04_scale_comparison/README.md) | 比较例 02/03 | 长度和时间×5，频率÷5 | 固定 ns 噪声下大尺度误差较低 | 可直接用已收纳数据重绘 |
| [05 incflo 球体尾流](examples/05_sphere_wake/README.md) | CFD→解析有限频率走时→局部 curl/GCV | 1.5×1.5×8 m；41 圈×48 阵元；19 截面 | 截面平均 joint L2 26.44%；分量相关 0.951/0.948/0.962 | 现有 TT 表为跨求解器基线 |
| [06 频率消融](examples/06_frequency_ablation/README.md) | 4 m Python/MAC 真值；解析有限频率 TT | 240 阵元；100/150/300 kHz | joint L2 48.46/46.11/44.49% | 辅助协议，不与例 05 比绝对误差 |
| [07 阵元数量消融](examples/07_sensor_count_ablation/README.md) | 8 m Python/MAC pilot；全 slab TT | 固定 15 圈；180–720 阵元 | joint L2 398.63→31.71% | 图表完整，原始 pilot 输入尚缺 |

每个例子 README 包含参数表、输入来源、运行命令和图片。不同真值、ROI、噪声重复及基线下的误差不能跨表直接排序。

![球体尾流阵列](examples/05_sphere_wake/results/figures/incflo_tt/acquisition.png)

![球体尾流阵元密度与重建质量](examples/05_sphere_wake/results/figures/incflo_tt/sensor_density_tradeoff.png)

## 2. 安装与最快检查

Python ≥3.10。从本仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python -m water_usct_3d.cli --help
python -m water_usct_3d.sphere.cli --help
python scripts/verify_repository.py
```

CPU 可运行测试、阵列设计、图表复核和线性反演；正式全波生成建议 CUDA。
球内配置明确指定 CUDA，不会静默回退到 CPU。incflo CFD 另需 CUDA 编译器和 C++ 构建工具，
其安装与冻结版本见例 05；plotfile 读取依赖可用 `pip install -e '.[cfd]'` 安装。
本机验证环境另见 [validation.json](docs/validation.json)，依赖范围不是跨平台位级复现保证。

无需重跑声学或 CFD 的演示：

```bash
python -m incflo_multiring_tt_study.design_array --output examples/05_sphere_wake/outputs/acquisition
python -m water_usct_3d.sphere.scale_study
```

已有图表在各例 `results/`，新运行写入 `outputs/`。后者被 Git 忽略，避免覆盖归档。
`config.yaml` 用于新运行；例 01–03 的 `reference_config.yaml` 指向归档，仅供读取/对照。
若将其传给写入型命令，会改写归档，因此复用波形应先按例子说明复制到 `outputs/`。
所有相对路径命令均从仓库根目录执行。

## 3. 数学原理

### 3.1 互易走时如何包含流速

设背景声速为常数 $c_0$，稳态流速为 $\mathbf U(\mathbf x)$，从阵元 $a$ 指向 $b$ 的单位方向为
$\widehat{\mathbf t}_{ab}$。小 Mach 数 $M=\|\mathbf U\|/c_0\ll1$ 下，固定同一条直射线 $\gamma_{ab}$：

$$
T_{a\to b}\approx\int_{\gamma_{ab}}\frac{ds}{c_0+\mathbf U\cdot\widehat{\mathbf t}_{ab}},\qquad
T_{b\to a}\approx\int_{\gamma_{ab}}\frac{ds}{c_0-\mathbf U\cdot\widehat{\mathbf t}_{ab}}.
$$

对分母作小流速展开，取差：

$$
\boxed{\Delta T_{ab}=T_{a\to b}-T_{b\to a}
\approx-\frac{2}{c_0^2}\int_{\gamma_{ab}}\mathbf U\cdot\widehat{\mathbf t}_{ab}\,ds.}
$$

顺流方向到时更早，因此沿 $a\to b$ 的正流速对应负走时差。单位为秒：
$c_0^{-2}$ 的单位是 s²/m²，线积分的单位为 m²/s。
同路径、常声速模型下，互易差消除了零流几何走时 $L/c_0$。
这里采用直线和一阶流速近似，没有包含弯曲声线、全波散射或真实换能器响应。

实际全波实验先测相对静水的移动量 $\delta t_{a\to b}$，再计算
$\Delta T_{ab}=\delta t_{a\to b}-\delta t_{b\to a}$；静水与有流数据使用相同阵列和离散网格。

### 3.2 可辨识性与不可压缩约束

单条射线只测沿射线方向的投影，不能恢复一个点上的完整三分量。
不同方向、跨层射线提供互补信息；同层阵列对轴向变化尤其缺乏约束。
对任意边界为零的标量 $\varphi$，

$$
\int_{\gamma_{ab}}\nabla\varphi\cdot\widehat{\mathbf t}_{ab}\,ds
=\varphi(b)-\varphi(a)=0.
$$

因此一般向量场的纵向射线变换具有势场零空间。本仓库通过物理先验缩小未知空间：

- 方管：$\mathbf U=(0,0,u(x,y))$，侧壁 $u=0$，沿轴向不变。
- 球内旋涡：在无散、球壁速度为零的有限维空间中反演。
- 球体尾流：恢复扰动 $\mathbf U'=\mathbf U_{sphere}-\mathbf U_{baseline}$，使用侧壁零速度的局部无散基。

有限维矩阵满秩只说明选定基函数在当前采集下可辨识，不能推出无限维问题良定或更细尺度可恢复。

### 3.3 无散旋度基与矩阵离散

球内使用无量纲坐标 $\boldsymbol\xi=\mathbf x/R$，定义

$$
\boldsymbol\psi_{jk}(\boldsymbol\xi)=\nabla_{\xi}\times
\left[(1-\|\boldsymbol\xi\|^2)^2
\exp\left(-\frac{\|\boldsymbol\xi-\mathbf c_j\|^2}{2\sigma^2}\right)\mathbf e_k\right].
$$

旋度的散度恒为零；包络及其一阶导数在球面为零，从而速度满足零壁值。
123 个中心×3 个方向得到 369 个原始模式。构造
$G_{ij}=\int_B\boldsymbol\psi_i\cdot\boldsymbol\psi_j\,d\boldsymbol\xi$，
删除相对特征值不超过 $10^{-10}$ 的相关方向并进行 $L^2$ 正交归一化，得到 364 个模式 $\boldsymbol\phi_n$。
$\mathbf U$ 仍以 m/s 表示，积分中的物理长度因子 $R$ 单独保留：

$$
\mathbf U=\sum_n a_n\boldsymbol\phi_n,\qquad
A_{in}=-\frac{2R}{c_0^2}\int_{\mathrm{chord}_i}
\boldsymbol\phi_n\cdot\widehat{\mathbf t}_i\,d\ell,\qquad
\mathbf d=A\mathbf a+\boldsymbol\varepsilon.
$$

96 阵元给出 $\binom{96}{2}=4560$ 行，矩阵为 $4560\times364$；
32 点 Gauss–Legendre 积分用于算子，独立 64 点积分用于规定射线真值。
细节见 [基函数与奇异谱](docs/basis_and_spectrum.md)。

尾流使用物理坐标的各向异性 Gaussian：

$$
f_j=b(x/L)b(y/L)\exp\left[-\frac{(x-c_{jx})^2+(y-c_{jy})^2}{2\sigma_{xy}^2}
-\frac{(z-c_{jz})^2}{2\sigma_z^2}\right],\qquad b(q)=q^2(1-q)^2,
\qquad\boldsymbol\psi_{jk}=\nabla\times(f_j\mathbf e_k).
$$

侧壁包络保证四个 x/y 壁面零速度；z 方向局部窗口不施加入口/出口零值。
主配置 $15\times15\times9$ 个中心、三个方向，共 6075 模态；按速度范数归一化。
连续无散恒等式不等于 CFD 的离散散度已经通过验证。

### 3.4 TSVD、Morozov 与 GCV

球内算子 $A=Q\operatorname{diag}(s_k)V^T$，截断解为

$$
\widehat{\mathbf a}_r=\sum_{k=1}^r\frac{\mathbf q_k^T\mathbf d}{s_k}\mathbf v_k.
$$

clean 保留 $s_k/s_1\ge10^{-8}$ 的模式；有噪声时用已知走时噪声标准差 $\sigma_t$，
选使 $\|A\widehat{\mathbf a}_r-\mathbf d\|_2$ 最接近 $\sqrt{m}\sigma_t$ 的秩（Morozov）。
当前 96 阵元配置条件数约 2.254，说明这 364 维空间内的几何条件较好。

局部尾流使用带谱截断的 ridge：

$$
\widehat{\mathbf a}_{r,\lambda}=\sum_{k=1}^r
\frac{s_k}{s_k^2+\lambda^2}(\mathbf q_k^T\mathbf d)\mathbf v_k,
\qquad
\mathrm{GCV}=\frac{\|A\widehat{\mathbf a}-\mathbf d\|_2^2}
{\left(m-\sum_{k=1}^r s_k^2/(s_k^2+\lambda^2)\right)^2}.
$$

在预设候选谱秩和 ridge 上最小化 GCV，不读取速度真值。代码将走时乘以 $-c_0^2/2$
转换成线积分单位后求解，对应噪声标准差也同步变换。
例 07 的大型全 slab 算子使用观测行留出验证选 ridge，再用共轭梯度求解，区别于局部窗口 GCV。

### 3.5 三种观测生成方式

**规定射线参考：** 直接计算流致延迟，并使用确定性脉冲检验延迟提取。
这是球内 `results/velocity_slices.png` 和顶层 `inversion_metrics.json` 的来源。

**全波波形观测：** 球内在开放水域求解一阶小 Mach 对流声学模型，主体为

$$
\partial_{tt}p+2\mathbf U\cdot\nabla\partial_t p-c_0^2\nabla^2p=s,
$$

外围使用分裂场 PML；分别计算有流与静水压力波形。在直达波窗口内拟合

$$
p_U(t)\approx\alpha p_0(t)+\beta\partial_t p_0(t)+b_0,\qquad
\delta t=-\beta/\alpha.
$$

向静水和有流压力波形加入独立 Gaussian 噪声，再重新提取走时；只用加噪前后延迟之差
标定 5 ns/10 ns 的全局噪声幅度。正式反演仍使用射线矩阵，因此属于**全波观测的走时层析**。
相关结果在 `results/fullwave/`。

方管的波形求解器采用含背景剪切项的四状态线性化 Euler 系统，八阶空间差分与 RK4，
x/y 刚性奇偶延拓，z 端 pressure-only sponge。方管剖面候选的选择按波形失配进行；
额外保留的开发 FWI 使用完整波形目标，它与走时反演是不同阶段。

**解析有限频率走时：** 例 05–07 在射线周围采用 5 点 Fresnel 横向采样形成 fat-ray，
近似有限频率空间平均。参考噪声 $\sigma_t(20\,\mathrm{kHz})=2.2371317$ ns，按 $1/f$ 缩放。
它没有传播完整声场，Fresnel 宽度是比较代理，不是实测 PSF 分辨率。

### 3.6 误差与统计口径

对指定评价网格或球内体积积分权重，定义

$$
E_{joint}=\frac{\|\widehat{\mathbf U}-\mathbf U\|_2}{\|\mathbf U\|_2},\qquad
E_k=\frac{\|\widehat U_k-U_k\|_2}{\|U_k\|_2}.
$$

球内相关系数为带体积权重的归一化内积；方管使用归一化内积；尾流逐分量使用 Pearson 相关系数。
接近零的真值分量会使相对误差不稳定，因此需同时看绝对幅值和图片。
尾流对扰动评分，不能用 0.5 m/s 主流主导的总轴向速度掩盖扰动误差。
例 05 的 26.44% 是 19 个截面指标的算术平均，不能当作整个 8 m 三维体积的 L2。

## 4. 成像结果解读

**球内旋涡：** 全波细网格为 97³，粗网格为 73³。10 cm 配置提取到时相对射线参考 RMSE 为 0.777 ns，
粗细网格差 RMS 为 0.423 ns。clean L2 2.535% 超过冻结的 2% 门槛，
不能因为波形提取通过就把整个反演 benchmark 标为通过。

![10 cm 全波反演误差](examples/02_ball_10cm/results/fullwave/noise_inversion_summary.png)

**相似缩放：** R 从 0.1 m 增至 0.5 m，同时频率由 100 kHz 降至 20 kHz。
保持声学相似性后，走时信号约放大五倍，固定绝对 ns 噪声下反演误差减小，clean 模型失配基本不变。

![球内尺度比较](examples/04_scale_comparison/results/scale_comparison.png)

**球体尾流：** incflo 真值使用直径 0.30 m 球、入口 0.5 m/s、有效运动黏度 $3\times10^{-4}$ m²/s、
$Re_D=500$，144×144×768 网格。下图 CFD 差场使用同网格同一时刻 incflo 无球基线。
但现有 TT 数值表使用 incflo 球体场减 PyTorch/MAC 无球基线，状态为 `provisional_cross_solver`。
匹配 incflo 基线已收纳，完整 TT sweep 尚未按该基线重跑。

![匹配基线 CFD 尾流](examples/05_sphere_wake/results/figures/incflo_forward/incflo_matched_wake_overview_t30p969.png)

![跨求解器基线 TT 的 Ux 扰动重建](examples/05_sphere_wake/results/figures/incflo_tt/ux_multiwindow_truth_reconstruction_error.png)

例 05 详细列出了三分量成像、19 截面误差、12/24/36/48 阵元每圈的消融，
以及 15/20/25/41 圈的比较。圈数实验始终只取最近 5 圈，圈距变化同时改变物理孔径，
所以不能解释为“圈越多越差”。incflo 原生 nodal/EB 离散散度在已有产物中未测量。

## 5. 仓库结构与代码对应

```text
Flow_estimation_official/
├── README.md
├── pyproject.toml
├── water_usct_3d/                 # 方管声学/流体/反演，以及 sphere 球内基准
├── incflo_sphere_wake_comparison/  # 冻结 CFD inputs、构建及后处理
├── incflo_multiring_tt_study/      # 多圈阵列、局部窗口与切片汇总
├── multislice_xyz_tt_study/        # fat-ray、curl basis、GCV
├── midplane_xy_tt_study/           # 四壁阵列与共用几何工具
├── long_domain_multislice_tt_study/# 全 slab 大矩阵辅助实验
├── examples/                      # 01–07：每例 README、配置/入口、results、可用 data
├── docs/                          # 数学补充、数据来源与校验记录
├── scripts/                       # 仓库完整性检查、从归档数据重绘
└── tests/                         # 58 项现有几何/物理/反演测试
```

| 数学或计算环节 | 实现 |
|---|---|
| 球内真值与无散基 | [flow.py](water_usct_3d/sphere/flow.py)、[basis.py](water_usct_3d/sphere/basis.py) |
| 球内全波、提取和噪声标定 | [fullwave.py](water_usct_3d/sphere/fullwave.py) |
| TSVD/Morozov | [inversion.py](water_usct_3d/sphere/inversion.py) |
| 方管剖面与候选选择 | [profile_inversion.py](water_usct_3d/profile_inversion.py) |
| 尾流有限频率算子、curl/GCV | [multislice_tt.py](multislice_xyz_tt_study/multislice_tt.py) |
| 多圈局部窗口 | [run_multiwindow.py](incflo_multiring_tt_study/run_multiwindow.py) |

数组和单位以各读取函数为准：方管速度为 `(3,nx,ny,nz)`，压力为 `(source,receiver,time)`；
球内点集速度为 `(n_points,3)`；CFD 紧凑 HDF5 的 `velocity` 为 `(3,nx,ny,nz)`。
长度 m、时间 s、速度 m/s；图表若使用 ns 或百分数会显式标注。

## 6. 数据完整性与共享

已复制原目录中的相关 HDF5/NPZ/CSV/JSON/PNG/GIF，以及可定位的尾流输入。
旧文章 PDF、重复 PDF 图、plan/progress、会话记忆、缓存与旧 Git 历史未复制。
CFD 源码由构建脚本获取冻结 commit；不将上游代码伪装为本仓库原创。

[数据清单](docs/data_manifest.json) 记录新增输入的路径、尺寸及 SHA-256。
[归档结果清单](docs/results_manifest.json) 用于验证结果复制前后字节一致。
历史指标里的旧 `artifact`/`source` 路径保留为来源信息，不是本仓库运行时依赖。
例 07 准确的远程 pilot 输入尚缺；其报告明确记录原时刻、形状及路径，未用 smoke 文件替代。

大数组采用 Git LFS：`.gitattributes` 已覆盖 `.h5`、`.npz`、`.pt`。
发布及克隆这些文件的环境需要 Git LFS；本地实际数据完整保留，没有用空占位文件冒充数据。
远程仓库：[yl602019618/Flow_estimation_official](https://github.com/yl602019618/Flow_estimation_official)。
克隆后运行 `git lfs pull` 获取大数组文件。整理来源目录未改动。
