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
| gpu-quantization | Q4_K_M | CUDA | pp256 | 15613.83 | 2176.40 | 463.0 | 625 | 625 |
| gpu-quantization | Q4_K_M | CUDA | tg64 | 320.25 | 15.83 | 463.0 | 625 | 625 |
| gpu-quantization | Q8_0 | CUDA | pp256 | 18388.85 | 2460.85 | 638.7 | 747 | 747 |
| gpu-quantization | Q8_0 | CUDA | tg64 | 266.42 | 8.19 | 638.7 | 747 | 747 |
| gpu-quantization | F16 | CUDA | pp256 | 14152.55 | 3516.15 | 1202.1 | 1201 | 1201 |
| gpu-quantization | F16 | CUDA | tg64 | 155.16 | 2.49 | 1202.1 | 1201 | 1201 |
| cpu-thread-6 | Q4_K_M | CPU-only | pp256 | 3624.85 | 349.82 | 463.0 | 431 | 431 |
| cpu-thread-6 | Q4_K_M | CPU-only | tg64 | 65.12 | 0.62 | 463.0 | 431 | 431 |
| cpu-thread-12 | Q4_K_M | CPU-only | pp256 | 3415.72 | 298.06 | 463.0 | 431 | 431 |
| cpu-thread-12 | Q4_K_M | CPU-only | tg64 | 68.89 | 1.26 | 463.0 | 431 | 431 |

`pp` 表示 prompt processing，`tg` 表示 token generation。同一 run 同时产生 pp/tg 记录，因此两行共享整次进程的显存峰值，并非阶段级峰值。`llama-bench` 不包含 tokenization 和 sampling 时间，因此在线指标需看下一节。

## 在线流式服务

| 并发 | 请求数 | TTFT p50 ms | TTFT p95 ms | TPOT p95 ms | 总延迟 p95 ms | 单请求平均 TPS | 聚合 TPS | 峰值 VRAM MiB | 服务增量 MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 30 | 21.86 | 35.90 | 3.48 | 195.29 | 299.05 | 266.73 | 553 | 553 |
| 2 | 60 | 27.95 | 39.04 | 4.36 | 239.62 | 238.68 | 425.97 | 553 | 553 |
| 4 | 120 | 31.54 | 51.49 | 6.64 | 348.90 | 168.35 | 610.99 | 553 | 553 |

## C++ KV Cache 调度 A/B

| 淘汰惩罚 | trials | 当前请求 ms | 后续长会话 ms | 序列总延迟 ms | 重复 prefill tokens | 选择淘汰 tokens |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 5 | 27.09 | 1334.84 | 1361.16 | 415 | 405 |
| 0.50 | 5 | 89.40 | 48.46 | 138.76 | 35 | 10 |

惩罚 0 等价于上游最长公共前缀选择；正惩罚使用本项目新增的净收益评分。当前请求可能少复用 token，但能避免破坏更有价值的长会话缓存，因此必须比较请求序列而非单请求。

## 真实 CUDA Tail COW

| 方法 | Median E2E ms | P95 E2E ms | Bytes/op | Launch/op | Extra device bytes |
|---|---:|---:|---:|---:|---:|
| tail_block_cow | 0.0083 | 0.0236 | 196608 | 1 | 196608 |
| whole_sequence_copy | 0.7631 | 0.9990 | 12582912 | 128 | 12582912 |

3 个 fresh process、每个方法 300 samples；Tail COW P95 改善 97.64%。

## Adaptive Prefill

| Backend | Mode | Median wall ms | P95 wall ms | Effective chunk median |
|---|---|---:|---:|---:|
| cpu | greedy | 12922.72 | 13058.38 | 0 |
| cpu | fixed64 | 13740.13 | 13774.64 | 64 |
| cpu | fixed256 | 12662.20 | 12673.42 | 256 |
| cpu | adaptive | 12748.54 | 12857.65 | 31 |
| cuda | greedy | 642.61 | 645.76 | 0 |
| cuda | fixed64 | 746.89 | 751.35 | 64 |
| cuda | fixed256 | 652.70 | 657.32 | 256 |
| cuda | adaptive | 650.00 | 654.95 | 78 |

Adaptive 能避开错误 fixed-64，但当前 CUDA trace 未稳定击败 greedy/fixed-256；该负结果保留。

## Adaptive Speculation

| Backend | Mode | Median wall ms | P95 wall ms | Acceptance |
|---|---|---:|---:|---:|
| cpu | none | 3787.11 | 3858.12 | 0.0% |
| cpu | fixed | 3197.05 | 3235.78 | 85.4% |
| cpu | adaptive | 3175.64 | 3215.33 | 97.2% |
| cuda | none | 397.45 | 410.55 | 0.0% |
| cuda | fixed | 202.58 | 204.15 | 85.4% |
| cuda | adaptive | 196.18 | 200.72 | 97.2% |

## 最小质量护栏

| 量化 | 通过题数 | 总题数 | 规则准确率 |
|---|---:|---:|---:|
| Q4_K_M | 2 | 5 | 40% |
| Q8_0 | 1 | 5 | 20% |
| F16 | 1 | 5 | 20% |

## 自动计算观察

- Q4_K_M 权重大小是 F16 的 38.5%，CUDA decode 速度是 F16 的 2.06 倍。
- 同一 Q4_K_M 配置下，CUDA decode 速度是 12 线程 CPU-only 的 4.65 倍。
- 并发从 1 增至 4 时，聚合输出吞吐提高到 2.29 倍；同时应结合 TTFT/TPOT 尾延迟判断交互体验。
- Cache-aware 调度在冲突序列中减少 91.6% 的重复 prefill token、减少 97.5% 的缓存淘汰，中位序列延迟提升 9.81 倍。
- 3 种精度在 5 题 smoke set 上的规则准确率为 20%–40%；样本过少，不能据此比较量化精度。

## 解释边界

- 量化对照使用同一 Qwen2.5-0.5B-Instruct GGUF 仓库和固定 revision。
- 结论不可直接外推到更大模型、其他 GPU 或不同上下文分布。
- 5 道规则题只是 smoke-level 质量护栏，不能替代标准 benchmark 或 perplexity。
- VRAM 为 nvidia-smi 对整块 GPU 的轮询值；增量以 case/服务启动前为基线，可能受其他 GPU 进程及采样间隔影响。
- TTFT 按首个非空 SSE content 事件计时，是首 token 延迟的服务端接口近似；TPOT 使用服务返回的 completion token 数。
- 原始输出位于 `results/raw/`，重新运行会覆盖汇总文件但保留固定配置。
