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
| gpu-quantization | Q4_K_M | CUDA | pp256 | 15618.28 | 2233.08 | 463.0 | 634 | 625 |
| gpu-quantization | Q4_K_M | CUDA | tg64 | 318.65 | 15.43 | 463.0 | 634 | 625 |
| gpu-quantization | Q8_0 | CUDA | pp256 | 17903.55 | 2612.62 | 638.7 | 756 | 747 |
| gpu-quantization | Q8_0 | CUDA | tg64 | 267.04 | 6.08 | 638.7 | 756 | 747 |
| gpu-quantization | F16 | CUDA | pp256 | 14105.03 | 4036.96 | 1202.1 | 1250 | 1241 |
| gpu-quantization | F16 | CUDA | tg64 | 154.73 | 0.62 | 1202.1 | 1250 | 1241 |
| cpu-thread-6 | Q4_K_M | CPU-only | pp256 | 3549.09 | 666.58 | 463.0 | 440 | 431 |
| cpu-thread-6 | Q4_K_M | CPU-only | tg64 | 68.84 | 1.39 | 463.0 | 440 | 431 |
| cpu-thread-12 | Q4_K_M | CPU-only | pp256 | 2986.16 | 549.17 | 463.0 | 440 | 431 |
| cpu-thread-12 | Q4_K_M | CPU-only | tg64 | 64.27 | 0.46 | 463.0 | 440 | 431 |

`pp` 表示 prompt processing，`tg` 表示 token generation。同一 run 同时产生 pp/tg 记录，因此两行共享整次进程的显存峰值，并非阶段级峰值。`llama-bench` 不包含 tokenization 和 sampling 时间，因此在线指标需看下一节。

## 在线流式服务

| 并发 | 请求数 | TTFT p50 ms | TTFT p95 ms | TPOT p95 ms | 总延迟 p95 ms | 单请求平均 TPS | 聚合 TPS | 峰值 VRAM MiB | 服务增量 MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 30 | 25.53 | 36.04 | 3.73 | 209.40 | 287.47 | 256.56 | 562 | 553 |
| 2 | 60 | 30.12 | 42.66 | 4.54 | 249.16 | 229.40 | 406.26 | 562 | 553 |
| 4 | 120 | 38.38 | 52.33 | 6.50 | 343.87 | 164.86 | 587.60 | 562 | 553 |

## C++ KV Cache 调度 A/B

| 淘汰惩罚 | trials | 当前请求 ms | 后续长会话 ms | 序列总延迟 ms | 重复 prefill tokens | 选择淘汰 tokens |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 5 | 28.37 | 1394.40 | 1422.29 | 415 | 405 |
| 0.50 | 5 | 94.35 | 51.16 | 145.20 | 35 | 10 |

惩罚 0 等价于上游最长公共前缀选择；正惩罚使用本项目新增的净收益评分。当前请求可能少复用 token，但能避免破坏更有价值的长会话缓存，因此必须比较请求序列而非单请求。

## 真实 CUDA Tail COW

| 方法 | Median E2E ms | P95 E2E ms | Bytes/op | Launch/op | Extra device bytes |
|---|---:|---:|---:|---:|---:|
| tail_block_cow | 0.0081 | 0.0270 | 196608 | 1 | 196608 |
| whole_sequence_copy | 0.7580 | 0.8493 | 12582912 | 128 | 12582912 |

3 个 fresh process、每个方法 300 samples；Tail COW P95 改善 96.82%。

## Adaptive Prefill

| Backend | Mode | Median wall ms | P95 wall ms | Effective chunk median |
|---|---|---:|---:|---:|
| cpu | greedy | 12475.67 | 12549.60 | 0 |
| cpu | fixed64 | 13341.32 | 13460.18 | 64 |
| cpu | fixed256 | 13038.55 | 13056.91 | 256 |
| cpu | adaptive | 13126.46 | 13147.95 | 31 |
| cuda | greedy | 636.55 | 648.84 | 0 |
| cuda | fixed64 | 725.20 | 748.32 | 64 |
| cuda | fixed256 | 661.44 | 665.00 | 256 |
| cuda | adaptive | 665.31 | 677.16 | 82 |

Adaptive 能避开错误 fixed-64，但当前 CUDA trace 未稳定击败 greedy/fixed-256；该负结果保留。

## Adaptive Speculation

| Backend | Mode | Median wall ms | P95 wall ms | Acceptance |
|---|---|---:|---:|---:|
| cpu | none | 3964.32 | 4065.35 | 0.0% |
| cpu | fixed | 3344.77 | 3351.09 | 85.4% |
| cpu | adaptive | 3294.07 | 3294.69 | 97.2% |
| cuda | none | 414.17 | 420.02 | 0.0% |
| cuda | fixed | 212.38 | 215.79 | 85.4% |
| cuda | adaptive | 209.83 | 217.97 | 97.2% |

## Mixed Prefill / Decode

| Backend | Policy | Trials | TTFT median / P95 ms | TPOT P95 ms | Latency P95 ms | Aggregate output TPS |
|---|---|---:|---:|---:|---:|---:|
| cpu | upstream | 3 | 1521.25 / 7449.65 | 418.92 | 11395.63 | 14.35 |
| cpu | cacheflow | 3 | 1507.24 / 2877.07 | 151.45 | 7038.50 | 21.87 |
| cuda | upstream | 3 | 63.63 / 155.30 | 15.13 | 524.43 | 254.44 |
| cuda | cacheflow | 3 | 61.40 / 91.64 | 11.93 | 489.79 | 286.57 |

CPU 本轮 latency P95 -38.2%、aggregate TPS +52.4%；CUDA 本轮 latency P95 -6.6%、aggregate TPS +12.6%。这是当前轮结果，不外推为稳定收益。

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

## Conservative Benefit Gating

| Backend | Mode | Trials | Objective median ms | Paired upstream regression | CacheFlow decisions | Exploration | Positive lower bound |
|---|---|---:|---:|---:|---:|---:|---:|
| cpu | upstream | 10 | 4789.32 | — | 0 | 0 | 0 |
| cpu | always | 10 | 3480.59 | — | 196 | 0 | 0 |
| cpu | rule | 10 | 3560.27 | — | 67 | 0 | 0 |
| cpu | learned | 10 | 3997.56 | -12.46% | 19 | 19 | 0 |
| cpu | oracle | 10 | 3428.03 | — | 0 | 0 | 0 |
| cuda | upstream | 10 | 226.47 | — | 0 | 0 | 0 |
| cuda | always | 10 | 233.78 | — | 174 | 0 | 0 |
| cuda | rule | 10 | 204.18 | — | 50 | 0 | 0 |
| cuda | learned | 10 | 209.74 | -6.17% | 0 | 0 | 0 |
| cuda | oracle | 10 | 196.95 | — | 0 | 0 | 0 |

相对 upstream 的 learned objective 变化：CPU -12.46%；CUDA -6.17%。
真实短 trace 的 learned 路径共出现 0 次 positive-lower-bound 决策；该 fresh-process 结果只证明受限探索和 fail-closed，长周期收敛证据见下一节。

### 长驻在线收敛与分布切换

同一 CUDA server PID 连续执行 53 个 wave；Ridge 最小样本数 12，confidence beta=1.00。

| Phase | Upstream | CacheFlow | Exploration | Positive lower bound | Positive waves / max streak | Drift | Safety fallback | TTFT P95 ms | Terminal benefit / uncertainty ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cold_start | 3 | 0 | 0 | 0 | 0 / 0 | 0 | 1 | 110.30 | 0.00 / 0.00 |
| stable_reuse | 33 | 160 | 17 | 143 | 39 / 35 | 0 | 22 | 211.75 | 11.24 / 5.27 |
| distribution_shift | 3 | 0 | 0 | 0 | 0 / 0 | 0 | 3 | 95.32 | 11.24 / 5.27 |

该实验补齐 fresh-process 短 trace 的证据缺口：稳定阶段必须出现非探索的置信下界启用，切换到独立 throughput-only 请求后必须 fail closed。

## CUDA Profiling 因果链

对 upstream/always 进行 paired Latin 干预；中间变量来自调度指标、CUDA Event 和 100 ms GPU 活跃度采样，结果来自 Engine trace 与请求 TTFT。该范围只覆盖本项目 CacheFlow KV kernel，不冒充完整 Nsight Compute kernel census。

| Paired trials | Δ CacheFlow decisions | Δ Prefill chunks | Δ Prefill tokens | Δ KV kernel launches | Δ KV copy bytes | Δ CUDA Event ms | Δ GPU busy | Δ max idle gap ms | Δ Execute us | Δ TTFT P95 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 15 | 23 | -145 | 0 | -2519000 | -0.260 | +1.19% | 0 | 31230 | 44.79 |

本轮强制 CacheFlow 改变了 prefill/CUDA 中间变量并恶化 Engine execute 与 TTFT；这是 learned gate 必须按上下文拒绝有害动作的直接系统证据。

## Production Engine Phase Trace

| Phase | Duration us | Share |
|---|---:|---:|
| execute | 9113907 | 99.9136% |
| commit | 6527 | 0.0716% |
| plan | 1127 | 0.0124% |
| prepare | 231 | 0.0025% |

共 218 个 production Engine spans。该数据来自 `in-process Chrome trace; WPR sampled stacks require elevated SeSystemProfilePrivilege`；它是 phase-duration trace，不是 sampled-stack CPU flame graph。

## 最小质量护栏

| 量化 | 通过题数 | 总题数 | 规则准确率 |
|---|---:|---:|---:|
| Q4_K_M | 2 | 5 | 40% |
| Q8_0 | 1 | 5 | 20% |
| F16 | 1 | 5 | 20% |

## 自动计算观察

- Q4_K_M 权重大小是 F16 的 38.5%，CUDA decode 速度是 F16 的 2.06 倍。
- 同一 Q4_K_M 配置下，CUDA decode 速度是 12 线程 CPU-only 的 4.96 倍。
- 并发从 1 增至 4 时，聚合输出吞吐提高到 2.29 倍；同时应结合 TTFT/TPOT 尾延迟判断交互体验。
- Cache-aware 调度在冲突序列中减少 91.6% 的重复 prefill token、减少 97.5% 的缓存淘汰，中位序列延迟提升 9.80 倍。
- 3 种精度在 5 题 smoke set 上的规则准确率为 20%–40%；样本过少，不能据此比较量化精度。

## 解释边界

- 量化对照使用同一 Qwen2.5-0.5B-Instruct GGUF 仓库和固定 revision。
- 结论不可直接外推到更大模型、其他 GPU 或不同上下文分布。
- 5 道规则题只是 smoke-level 质量护栏，不能替代标准 benchmark 或 perplexity。
- VRAM 为 nvidia-smi 对整块 GPU 的轮询值；增量以 case/服务启动前为基线，可能受其他 GPU 进程及采样间隔影响。
- TTFT 按首个非空 SSE content 事件计时，是首 token 延迟的服务端接口近似；TPOT 使用服务返回的 completion token 数。
- 原始输出位于 `results/raw/`，重新运行会覆盖汇总文件但保留固定配置。
