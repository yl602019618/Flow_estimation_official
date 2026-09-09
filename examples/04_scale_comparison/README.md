# 例 04：球内相似缩放对照

比较半径 0.10 m / 100 kHz 与半径 0.50 m / 20 kHz。空间、时间放大 5 倍，频率缩小 5 倍；
保持 D/λ=13.51、网格数、每波长网格点数与 CFL 相同。流速不变，互易走时信号约增大 5 倍。

| 配置 | clean L2 | 5 ns L2 | 10 ns L2 |
|---|---:|---:|---:|
| 0.10 m / 100 kHz | 2.535% | 8.700% | 14.043% |
| 0.50 m / 20 kHz | 2.535% | 3.067% | 4.618% |

![尺度、走时和误差对照](results/scale_comparison.png)

数据见 [metrics.json](results/metrics.json)。固定绝对计时误差时，大尺度信噪比更高；clean 的模型失配并未被缩放消除。
50 cm / 100 kHz 的 481³ 网格和约 358 GiB 状态存储是估算，未运行该配置。

从仓库根目录运行下列命令即可用已收纳数据重绘，无须重新传播：

```bash
python -m water_usct_3d.sphere.scale_study
```

默认读取例 02/03 的 `reference_config.yaml`，输出到本例 `outputs/`。
比较新实验时用 `--small-config examples/02_ball_10cm/config.yaml --large-config examples/03_ball_50cm/config.yaml`。
