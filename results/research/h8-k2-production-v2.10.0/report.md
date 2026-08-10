# K2 相对生产 K1 的正式晋级实验

- 配对服务样本：30 组；每个 arm 均进入真实 Paged 图且零 fallback。
- 客户端配对中位差（K2-K1）：0.150 ms；bootstrap 95% 区间 [-0.109, 0.308] ms。
- 客户端 P95：K1 30.102 ms，K2 30.559 ms，变化 1.52%。
- 服务内部 prompt 配对中位差：0.044 ms。
- Bootstrap 95% upper regression bounds (median/P95): 2.86% / 4.78%; gate metrics: median.
- NSYS：K1 480 次 / 8.174 ms；K2 480 次 / 4.051 ms（仅机制证据）。
- 生产晋级：通过。门槛为客户端 median/P95 回退均不超过 5.0%，且 kernel 总时长至少降低 20.0%。

该结论只回答 K2 能否替代同一 Paged 路径中的 K1；不把它改写成 Paged 已优于 Direct。
