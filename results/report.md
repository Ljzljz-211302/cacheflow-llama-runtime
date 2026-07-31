# CacheFlow Runtime 可复现验收报告

> 该报告由实验脚本从原始 JSON/CSV 自动生成；速度只代表当前机器、固定版本和固定配置。

## 环境

- 平台：`Windows-11-10.0.22631-SP0`
- Python：`3.13.13`
- GPU：`NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB, 561.19`
- llama.cpp：`b9632` / `acd79d603`

## 离线算子基线

| Case | 量化 | 后端 | 测试 | tokens/s | 标准差 | 模型 MiB | Run 峰值 VRAM MiB | Run 增量 VRAM MiB |
|---|---|---|---|---:|---:|---:|---:|---:|
| gpu-quantization | Q4_K_M | CUDA | pp256 | 15419.74 | 2084.82 | 463.0 | 619 | 619 |
| gpu-quantization | Q4_K_M | CUDA | tg64 | 306.19 | 13.01 | 463.0 | 619 | 619 |
| gpu-quantization | Q8_0 | CUDA | pp256 | 17470.93 | 2875.69 | 638.7 | 747 | 747 |
| gpu-quantization | Q8_0 | CUDA | tg64 | 258.85 | 6.79 | 638.7 | 747 | 747 |
| gpu-quantization | F16 | CUDA | pp256 | 12797.55 | 3395.05 | 1202.1 | 1201 | 1201 |
| gpu-quantization | F16 | CUDA | tg64 | 148.20 | 1.09 | 1202.1 | 1201 | 1201 |
| cpu-thread-6 | Q4_K_M | CPU-only | pp256 | 3011.55 | 640.10 | 463.0 | 431 | 431 |
| cpu-thread-6 | Q4_K_M | CPU-only | tg64 | 53.07 | 2.10 | 463.0 | 431 | 431 |
| cpu-thread-12 | Q4_K_M | CPU-only | pp256 | 2754.45 | 405.54 | 463.0 | 431 | 431 |
| cpu-thread-12 | Q4_K_M | CPU-only | tg64 | 45.87 | 4.04 | 463.0 | 431 | 431 |

`pp` 表示 prompt processing，`tg` 表示 token generation。同一 run 同时产生 pp/tg 记录，因此两行共享整次进程的显存峰值，并非阶段级峰值。`llama-bench` 不包含 tokenization 和 sampling 时间，因此在线指标需看下一节。

## 在线流式服务

| 并发 | 请求数 | TTFT p50 ms | TTFT p95 ms | TPOT p95 ms | 总延迟 p95 ms | 单请求平均 TPS | 聚合 TPS | 峰值 VRAM MiB | 服务增量 MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 30 | 15.07 | 36.94 | 3.73 | 207.35 | 277.68 | 254.06 | 553 | 553 |
| 2 | 60 | 27.78 | 44.05 | 5.30 | 281.24 | 206.07 | 370.90 | 553 | 553 |
| 4 | 120 | 36.17 | 50.55 | 8.12 | 424.00 | 133.91 | 490.51 | 553 | 553 |

## C++ KV Cache 调度 A/B

| 淘汰惩罚 | trials | 当前请求 ms | 后续长会话 ms | 序列总延迟 ms | 重复 prefill tokens | 选择淘汰 tokens |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 5 | 31.53 | 1456.70 | 1490.00 | 415 | 405 |
| 0.50 | 5 | 102.79 | 54.85 | 157.03 | 35 | 10 |

惩罚 0 等价于上游最长公共前缀选择；正惩罚使用本项目新增的净收益评分。当前请求可能少复用 token，但能避免破坏更有价值的长会话缓存，因此必须比较请求序列而非单请求。

## 真实 CUDA Tail COW

| 方法 | Median E2E ms | P95 E2E ms | Bytes/op | Launch/op | Extra device bytes |
|---|---:|---:|---:|---:|---:|
| tail_block_cow | 0.0089 | 0.0258 | 196608 | 1 | 196608 |
| whole_sequence_copy | 0.7786 | 0.9980 | 12582912 | 128 | 12582912 |

3 个 fresh process、每个方法 300 samples；Tail COW P95 改善 97.41%。

## Adaptive Prefill

| Backend | Mode | Median wall ms | P95 wall ms | Effective chunk median |
|---|---|---:|---:|---:|
| cpu | greedy | 14141.66 | 14306.91 | 0 |
| cpu | fixed64 | 14563.67 | 14651.96 | 64 |
| cpu | fixed256 | 14140.74 | 14339.43 | 256 |
| cpu | adaptive | 14070.41 | 14148.23 | 0 |
| cuda | greedy | 668.66 | 674.74 | 0 |
| cuda | fixed64 | 751.94 | 758.22 | 64 |
| cuda | fixed256 | 666.13 | 684.03 | 256 |
| cuda | adaptive | 674.16 | 683.61 | 62 |

Adaptive 能避开错误 fixed-64，但当前 CUDA trace 未稳定击败 greedy/fixed-256；该负结果保留。

## Adaptive Speculation

| Backend | Mode | Median wall ms | P95 wall ms | Acceptance |
|---|---|---:|---:|---:|
| cpu | none | 4501.84 | 4521.16 | 0.0% |
| cpu | fixed | 3565.58 | 3675.80 | 85.4% |
| cpu | adaptive | 3672.59 | 3682.78 | 97.2% |
| cuda | none | 421.50 | 424.57 | 0.0% |
| cuda | fixed | 239.58 | 243.68 | 85.4% |
| cuda | adaptive | 209.39 | 213.48 | 97.2% |

## Mixed Prefill / Decode

| Backend | Policy | Trials | TTFT median / P95 ms | TPOT P95 ms | Latency P95 ms | Aggregate output TPS |
|---|---|---:|---:|---:|---:|---:|
| cpu | upstream | 3 | 1603.06 / 7883.35 | 437.20 | 12779.54 | 13.94 |
| cpu | cacheflow | 3 | 1625.82 / 3073.61 | 277.82 | 7043.05 | 17.65 |
| cuda | upstream | 3 | 68.66 / 155.31 | 20.30 | 543.48 | 235.26 |
| cuda | cacheflow | 3 | 74.45 / 93.71 | 12.76 | 618.81 | 264.90 |

CPU 本轮 latency P95 -44.9%、aggregate TPS +26.6%；CUDA 本轮 latency P95 +13.9%、aggregate TPS +12.6%。这是当前轮结果，不外推为稳定收益。

### 跨验收轮重复性

| Run | Backend | Policy | Trials | Latency P95 ms | Aggregate output TPS |
|---|---|---|---:|---:|---:|
| pre_refactor | cpu | upstream | 3 | 11458.46 | 15.07 |
| pre_refactor | cpu | cacheflow | 3 | 6347.13 | 22.22 |
| pre_refactor | cuda | upstream | 3 | 524.16 | 278.85 |
| pre_refactor | cuda | cacheflow | 3 | 610.14 | 249.33 |
| baseline_repeat | cpu | upstream | 3 | 12012.17 | 13.42 |
| baseline_repeat | cpu | cacheflow | 3 | 7644.68 | 22.13 |
| baseline_repeat | cuda | upstream | 3 | 602.19 | 241.95 |
| baseline_repeat | cuda | cacheflow | 3 | 556.07 | 268.48 |

两轮 CUDA latency/throughput 结论反号，因此当前 3-trial laptop GPU 证据不足以声称稳定端到端提升；production 应保留 upstream fallback，并增加 backend/workload-aware gating。

## Production Engine Phase Trace

| Phase | Duration us | Share |
|---|---:|---:|
| execute | 9544509 | 99.9145% |
| commit | 6441 | 0.0674% |
| plan | 1411 | 0.0148% |
| prepare | 315 | 0.0033% |

共 242 个 production Engine spans。该数据来自 `in-process Chrome trace; WPR sampled stacks require elevated SeSystemProfilePrivilege`；它是 phase-duration trace，不是 sampled-stack CPU flame graph。

## 最小质量护栏

| 量化 | 通过题数 | 总题数 | 规则准确率 |
|---|---:|---:|---:|
| Q4_K_M | 2 | 5 | 40% |
| Q8_0 | 1 | 5 | 20% |
| F16 | 1 | 5 | 20% |

## 自动计算观察

- Q4_K_M 权重大小是 F16 的 38.5%，CUDA decode 速度是 F16 的 2.07 倍。
- 同一 Q4_K_M 配置下，CUDA decode 速度是 12 线程 CPU-only 的 6.68 倍。
- 并发从 1 增至 4 时，聚合输出吞吐提高到 1.93 倍；同时应结合 TTFT/TPOT 尾延迟判断交互体验。
- Cache-aware 调度在冲突序列中减少 91.6% 的重复 prefill token、减少 97.5% 的缓存淘汰，中位序列延迟提升 9.49 倍。
- 3 种精度在 5 题 smoke set 上的规则准确率为 20%–40%；样本过少，不能据此比较量化精度。

## 解释边界

- 量化对照使用同一 Qwen2.5-0.5B-Instruct GGUF 仓库和固定 revision。
- 结论不可直接外推到更大模型、其他 GPU 或不同上下文分布。
- 5 道规则题只是 smoke-level 质量护栏，不能替代标准 benchmark 或 perplexity。
- VRAM 为 nvidia-smi 对整块 GPU 的轮询值；增量以 case/服务启动前为基线，可能受其他 GPU 进程及采样间隔影响。
- TTFT 按首个非空 SSE content 事件计时，是首 token 延迟的服务端接口近似；TPOT 使用服务返回的 completion token 数。
- 原始输出位于 `results/raw/`，重新运行会覆盖汇总文件但保留固定配置。
