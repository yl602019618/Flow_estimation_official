# 例 03：球内三维旋涡，半径 0.5 m / 20 kHz

[共用数学推导、数据链路与采集说明](../../docs/unit_ball.md) · [配置](config.yaml)

这是全波生成压力观测、提取互易走时差、用直射线 TSVD 重建的合成问题。
水声速 1480 m/s，密度 998 kg/m³，峰值流速 1 m/s，96 个 Fibonacci 球面阵元，
4560 个互易对，364 个正交无散速度模态。倾斜旋涡轴为 (1,2,3)/√14。

| 全波观测 | 实现走时噪声 / ns | TSVD 秩 | 相对 L2 | 场相关系数 |
|---|---:|---:|---:|---:|
| clean | 0.000 | 364 | 2.535% | 0.999709 |
| noise_5ns | 5.001 | 358 | 3.067% | 0.999559 |
| noise_10ns | 9.994 | 352 | 4.618% | 0.998962 |

数值来自 [fullwave/metrics.json](results/fullwave/metrics.json)。clean 超过 2% 门槛；两个噪声条件通过原阈值。
全波生产网格为 73³/97³，时间步 2.0 µs，记录时长 900 µs。

![球面阵列](results/sensors.png)

![全波提取走时诊断](results/fullwave/diagnostics.png)

![全波反演误差汇总](results/fullwave/noise_inversion_summary.png)

下图为**规定射线参考**的真值与重建切片；其观测模型与上表不同。

![规定射线参考切片](results/velocity_slices.png)

## 复现

从仓库根目录运行：

```bash
python -m water_usct_3d.sphere.cli benchmark --config examples/03_ball_50cm/config.yaml
```

默认请求 CUDA，完整 benchmark 在 clean 门槛不通过时返回 2。查看产物中的具体门禁，不能仅检查采集阶段的 `status`。
只跑规定射线参考可用 `geometry`、`generate`、`invert`、`evaluate` 子命令；只跑全波阶段用 `fullwave`。
新结果进入本例 `outputs/`。参考结果保留在 `results/`，包含波形 HDF5、算子 NPZ 和指标。
如需复用归档波形，先 `cp -a examples/03_ball_50cm/results examples/03_ball_50cm/outputs`（目标尚不存在时），
再运行上述命令并添加 `--reuse-waveforms`。
