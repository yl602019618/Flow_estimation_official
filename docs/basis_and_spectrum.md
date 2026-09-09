# 单位球走时层析：基函数与纵向射线奇异谱说明

本文说明当前单位球 Water-USCT 基准采用的速度场基函数、基函数的正交化方式，以及
`singular_spectrum.png` 中纵向射线奇异谱的物理与数值含义。这里的结论对应
`examples/02_ball_10cm/config.yaml` 和冻结输出
`examples/02_ball_10cm/results/operator.npz`。

## 1. 我们要表示什么速度场

反演目标是单位球

$$
B=\{\boldsymbol\xi\in\mathbb R^3:\lVert\boldsymbol\xi\rVert<1\}
$$

内的三维速度场 $\mathbf U(\boldsymbol\xi)$。采用的表示必须满足两个硬约束：

$$
\nabla_{\boldsymbol\xi}\!\cdot\mathbf U=0,
\qquad
\mathbf U\big|_{\partial B}=0.
$$

前者表示不可压缩流，后者表示球壁无滑移。当前方法不是先反演三个任意速度分量再附加散度惩罚，
而是直接选择逐个满足这两个约束的基函数，因此任意线性组合仍然严格无散并在球壁为零。

## 2. 原始无散高斯旋度基函数

对每个中心 $\mathbf c_j$ 和笛卡尔方向 $\mathbf e_k$，定义标量包络

$$
f_j(\boldsymbol\xi)
= (1-\lVert\boldsymbol\xi\rVert^2)^2
  \exp\!\left[-\frac{\lVert\boldsymbol\xi-\mathbf c_j\rVert^2}{2\sigma^2}\right],
$$

以及原始向量基函数

$$
\boldsymbol\psi_{jk}(\boldsymbol\xi)
=\nabla_{\boldsymbol\xi}\times\left[f_j(\boldsymbol\xi)\mathbf e_k\right]
=\nabla_{\boldsymbol\xi}f_j(\boldsymbol\xi)\times\mathbf e_k,
\qquad k\in\{1,2,3\}.
$$

它有以下性质：

1. **严格无散**：旋度的散度恒为零，所以
   $\nabla\cdot\boldsymbol\psi_{jk}=0$。
2. **球壁速度为零**：平方边界因子 $(1-\lVert\boldsymbol\xi\rVert^2)^2$
   使 $f_j$ 及其梯度在 $\lVert\boldsymbol\xi\rVert=1$ 上同时为零。
3. **局部性**：高斯因子把每组模式集中在中心 $\mathbf c_j$ 附近；
   $\sigma$ 控制其空间宽度。
4. **三维方向性**：每个中心使用三个旋度轴 $\mathbf e_1,\mathbf e_2,\mathbf e_3$。

当前配置在 $[-0.75,0.75]^3$ 上取 $7^3$ 个候选中心，只保留
$\lVert\mathbf c_j\rVert\leq0.76$ 的中心，共 123 个；$\sigma=0.34$。因此原始模式数为

$$
N_{\rm raw}=123\times3=369.
$$

这里的坐标、中心和 $\sigma$ 都是在单位球坐标 $\boldsymbol\xi$ 下定义的。由
$\mathbf x=R\boldsymbol\xi$，实际物理宽度为 $R\sigma$：10 cm 半径时为 3.4 cm，
50 cm 半径时为 17 cm。

## 3. 为什么从 369 个模式变成 364 个模式

369 个原始模式并不彼此正交，而且存在 gauge/近线性相关方向。程序在单位球体积求积网上构造
速度内积 Gram 矩阵

$$
G_{ab}=\int_B \boldsymbol\psi_a(\boldsymbol\xi)\cdot
                 \boldsymbol\psi_b(\boldsymbol\xi)\,d\boldsymbol\xi.
$$

令 $G=Q\Lambda Q^\mathsf T$。程序删除满足

$$
\lambda_i\leq10^{-10}\lambda_{\max}
$$

的近相关方向，并用

$$
W=Q_{\rm keep}\Lambda_{\rm keep}^{-1/2}
$$

形成新的基函数

$$
\boldsymbol\phi_n=\sum_{a=1}^{369}\boldsymbol\psi_a W_{an}.
$$

这样得到 364 个近似 $L^2(B)$ 正交归一的物理模式，5 个近相关方向被删除。最终速度场表示为

$$
\mathbf U(\boldsymbol\xi)=\sum_{n=1}^{364}a_n\boldsymbol\phi_n(\boldsymbol\xi).
$$

需要注意，$\boldsymbol\phi_n$ 通常是多个不同中心、不同旋度轴的全局线性组合，已经不再是单个
高斯涡旋。`operator.npz` 中的 `transform` 就是大小为 $369\times364$ 的矩阵 $W$。

## 4. 纵向射线矩阵是什么

对第 $i$ 条收发弦，单位切向为 $\mathbf t_i$。一阶 reciprocal travel-time difference 为

$$
\Delta T_i\simeq-\frac{2R}{c_0^2}
\int_{\mathrm{chord}_i}\mathbf U(\boldsymbol\xi)\cdot\mathbf t_i\,d\ell.
$$

代入基函数展开后得到线性系统

$$
\mathbf d=A\mathbf a+\boldsymbol\varepsilon,
\qquad
A_{in}=-\frac{2R}{c_0^2}
\int_{\mathrm{chord}_i}\boldsymbol\phi_n(\boldsymbol\xi)
\cdot\mathbf t_i\,d\ell.
$$

96 个阵元的全部 reciprocal pairs 给出 4560 条独立弦，因此当前矩阵尺寸为

$$
A\in\mathbb R^{4560\times364}.
$$

矩阵积分采用独立的 32 点 Gauss--Legendre 弦求积。全波模块负责产生观测波形和提取到时，
但当前反演算子 $A$ 仍是上述一阶纵向直射线模型；它不是全波 Jacobian。

## 5. `singular_spectrum.png` 表示什么

![纵向射线奇异谱](../examples/02_ball_10cm/results/singular_spectrum.png)

对射线矩阵进行奇异值分解：

$$
A=U\,\mathrm{diag}(s_1,\ldots,s_{364})V^\mathsf T,
\qquad s_1\geq s_2\geq\cdots\geq s_{364}>0.
$$

图的横轴是按奇异值从大到小排列的 **SVD 模态编号**，纵轴是归一化奇异值
$s_k/s_1$。每个右奇异向量（$V$ 的一列）是 364 个正交空间基函数的一个线性组合；
因此图上的“mode”不是单个高斯中心，也不是第 $k$ 个原始基函数。

奇异值衡量一个速度组合在走时数据中的可观测强度：

- $s_k$ 大：该速度组合会产生较强的 reciprocal delay，较容易从数据恢复；
- $s_k$ 很小：该组合对数据影响弱，求逆时会把噪声放大约 $1/s_k$；
- $s_k=0$：该组合位于当前离散观测算子的零空间，无法由这些数据恢复。

当前冻结矩阵的实测值为：

| 数量 | 数值 |
|---|---:|
| 矩阵尺寸 | $4560\times364$ |
| 最大奇异值 $s_1$ | $2.3369648\times10^{-6}$ |
| 最小奇异值 $s_{364}$ | $1.0368063\times10^{-6}$ |
| 最小归一化奇异值 $s_{364}/s_1$ | 0.443655 |
| 条件数 $s_1/s_{364}$ | 2.254003 |
| clean 相对截断线 | $10^{-8}$ |

虚线 $10^{-8}$ 是无噪声 TSVD 使用的相对奇异值截断门限。全部 364 个奇异值都远高于该线，
所以 clean prescribed-ray 反演保留全部 364 个模态。谱线较平、条件数较低，说明在**当前选定的
364 维无散子空间和当前 96 阵元全配对采集下**，不存在明显的离散弱可观测方向。

这个结论有明确边界：它不能证明连续无限维纵向射线逆问题良定，也不能证明比当前高斯基更细的
空间结构能够稳定恢复。平坦谱在一定程度上来自先做了速度 $L^2$ 正交化，同时也反映了 96 阵元
全配对采集对当前有限维空间的覆盖较充分。

## 6. TSVD 如何使用这张谱

无噪声情况下，程序保留所有满足

$$
s_k/s_1\geq10^{-8}
$$

的模态；本例因此保留 364 阶。

有噪声情况下并不固定使用图中的 $10^{-8}$ 线，而是使用 Morozov discrepancy principle：
在各个截断阶数中选择使数据残差最接近 $\sqrt{m}\sigma_{\Delta T}$ 的阶数，其中
$m=4560$。当前 prescribed-ray 冻结结果为：

| 数据 | TSVD 阶数 | 残差 RMS | 相对速度场 $L^2$ 误差 |
|---|---:|---:|---:|
| clean | 364 | 0.0911 ns | 0.468% |
| noise 5 ns | 277 | 5.0001 ns | 8.025% |
| noise 10 ns | 206 | 9.9997 ns | 12.722% |

噪声越大，Morozov 原则保留的阶数越少，以牺牲部分空间自由度换取更小的噪声放大。即使本图中
最小奇异值并不接近零，有限噪声下截断仍然起到统计正则化作用。

## 7. 一句话区分三个“模态”层次

可以把整个过程记为：

$$
\boxed{
369\ \text{个局部原始旋度模式}
\xrightarrow{\text{Gram 去相关及 }L^2\text{ 正交化}}
364\ \text{个速度空间基函数}
\xrightarrow{\text{射线矩阵 SVD}}
364\ \text{个按可观测性排序的奇异模态}
}
$$

图中画的是最后一层，即观测算子的奇异模态强度，而不是第一层或第二层基函数的空间形状。

## 8. 对应实现与可复核文件

- 原始旋度模式：`water_usct_3d/sphere/flow.py`
- Gram 正交化和射线矩阵：`water_usct_3d/sphere/basis.py`
- TSVD 与 Morozov 截断：`water_usct_3d/sphere/inversion.py`
- 绘图函数：`water_usct_3d/sphere/visualization.py`
- 基函数配置：`examples/02_ball_10cm/config.yaml`
- 冻结矩阵及谱：`examples/02_ball_10cm/results/operator.npz`
- 冻结反演指标：`examples/02_ball_10cm/results/inversion_metrics.json`

