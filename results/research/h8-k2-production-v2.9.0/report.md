# K2 相对生产 K1 的正式晋级实验

- 配对服务样本：30 组；每个 arm 均进入真实 Paged 图且零 fallback。
- 客户端配对中位差（K2-K1）：-0.041 ms；bootstrap 95% 区间 [-0.180, 0.033] ms。
- 客户端 P95：K1 7.189 ms，K2 9.340 ms，变化 29.92%。
- 服务内部 prompt 配对中位差：-0.009 ms。
- Bootstrap 95% upper regression bounds (median/P95): 8.99% / 59.97%.
- NSYS：K1 480 次 / 8.137 ms；K2 480 次 / 4.057 ms（仅机制证据）。
- 生产晋级：未通过。门槛为客户端 median/P95 回退均不超过 5.0%，且 kernel 总时长至少降低 20.0%。

该结论只回答 K2 能否替代同一 Paged 路径中的 K1；不把它改写成 Paged 已优于 Direct。
