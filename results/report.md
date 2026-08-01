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
| gpu-quantization | Q4_K_M | CUDA | pp256 | 15777.59 | 2061.28 | 463.0 | 634 | 625 |
| gpu-quantization | Q4_K_M | CUDA | tg64 | 319.44 | 15.32 | 463.0 | 634 | 625 |
| gpu-quantization | Q8_0 | CUDA | pp256 | 18076.00 | 2556.22 | 638.7 | 762 | 753 |
| gpu-quantization | Q8_0 | CUDA | tg64 | 266.77 | 5.24 | 638.7 | 762 | 753 |
| gpu-quantization | F16 | CUDA | pp256 | 13732.37 | 4220.33 | 1202.1 | 1210 | 1201 |
| gpu-quantization | F16 | CUDA | tg64 | 153.84 | 1.21 | 1202.1 | 1210 | 1201 |
| cpu-thread-6 | Q4_K_M | CPU-only | pp256 | 3536.82 | 550.61 | 463.0 | 440 | 431 |
| cpu-thread-6 | Q4_K_M | CPU-only | tg64 | 59.24 | 2.16 | 463.0 | 440 | 431 |
| cpu-thread-12 | Q4_K_M | CPU-only | pp256 | 3351.62 | 580.82 | 463.0 | 440 | 431 |
| cpu-thread-12 | Q4_K_M | CPU-only | tg64 | 56.26 | 7.68 | 463.0 | 440 | 431 |

`pp` 表示 prompt processing，`tg` 表示 token generation。同一 run 同时产生 pp/tg 记录，因此两行共享整次进程的显存峰值，并非阶段级峰值。`llama-bench` 不包含 tokenization 和 sampling 时间，因此在线指标需看下一节。

## 在线流式服务

| 并发 | 请求数 | TTFT p50 ms | TTFT p95 ms | TPOT p95 ms | 总延迟 p95 ms | 单请求平均 TPS | 聚合 TPS | 峰值 VRAM MiB | 服务增量 MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 30 | 24.30 | 38.93 | 3.83 | 210.27 | 276.64 | 246.27 | 562 | 553 |
| 2 | 60 | 24.24 | 37.59 | 5.11 | 267.83 | 210.60 | 385.07 | 562 | 553 |
| 4 | 120 | 32.86 | 48.85 | 7.65 | 399.83 | 143.01 | 525.63 | 562 | 553 |

## C++ KV Cache 调度 A/B

| 淘汰惩罚 | trials | 当前请求 ms | 后续长会话 ms | 序列总延迟 ms | 重复 prefill tokens | 选择淘汰 tokens |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 5 | 30.00 | 1499.24 | 1528.96 | 415 | 405 |
| 0.50 | 5 | 97.13 | 52.21 | 148.73 | 35 | 10 |

惩罚 0 等价于上游最长公共前缀选择；正惩罚使用本项目新增的净收益评分。当前请求可能少复用 token，但能避免破坏更有价值的长会话缓存，因此必须比较请求序列而非单请求。

## 真实 CUDA Tail COW

| 方法 | Median E2E ms | P95 E2E ms | Bytes/op | Launch/op | Extra device bytes |
|---|---:|---:|---:|---:|---:|
| tail_block_cow | 0.0091 | 0.0282 | 196608 | 1 | 196608 |
| whole_sequence_copy | 0.7885 | 1.0236 | 12582912 | 128 | 12582912 |

3 个 fresh process、每个方法 300 samples；Tail COW P95 改善 97.25%。

## Adaptive Prefill

| Backend | Mode | Median wall ms | P95 wall ms | Effective chunk median |
|---|---|---:|---:|---:|
| cpu | greedy | 13491.19 | 13614.93 | 0 |
| cpu | fixed64 | 14459.42 | 14540.78 | 64 |
| cpu | fixed256 | 13925.76 | 14077.14 | 256 |
| cpu | adaptive | 14026.05 | 14592.57 | 31 |
| cuda | greedy | 655.34 | 658.98 | 0 |
| cuda | fixed64 | 748.96 | 758.04 | 64 |
| cuda | fixed256 | 674.20 | 686.30 | 256 |
| cuda | adaptive | 667.70 | 669.78 | 77 |

Adaptive 能避开错误 fixed-64，但当前 CUDA trace 未稳定击败 greedy/fixed-256；该负结果保留。

## Adaptive Speculation

| Backend | Mode | Median wall ms | P95 wall ms | Acceptance |
|---|---|---:|---:|---:|
| cpu | none | 4414.43 | 4559.77 | 0.0% |
| cpu | fixed | 3699.81 | 3727.43 | 85.4% |
| cpu | adaptive | 3576.24 | 3598.39 | 97.2% |
| cuda | none | 428.23 | 432.53 | 0.0% |
| cuda | fixed | 225.23 | 238.85 | 85.4% |
| cuda | adaptive | 206.61 | 207.51 | 97.2% |

## Mixed Prefill / Decode

| Backend | Policy | Trials | TTFT median / P95 ms | TPOT P95 ms | Latency P95 ms | Aggregate output TPS |
|---|---|---:|---:|---:|---:|---:|
| cpu | upstream | 3 | 2008.40 / 9333.77 | 383.81 | 14314.42 | 13.30 |
| cpu | cacheflow | 3 | 1810.05 / 3385.73 | 218.59 | 7995.47 | 18.15 |
| cuda | upstream | 3 | 57.95 / 163.42 | 21.22 | 525.99 | 284.68 |
| cuda | cacheflow | 3 | 54.95 / 83.66 | 12.32 | 458.01 | 296.53 |

CPU 本轮 latency P95 -44.1%、aggregate TPS +36.5%；CUDA 本轮 latency P95 -12.9%、aggregate TPS +4.2%。这是当前轮结果，不外推为稳定收益。

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
| cpu | upstream | 10 | 5273.82 | — | 0 | 0 | 0 |
| cpu | always | 10 | 3729.06 | — | 189 | 0 | 0 |
| cpu | rule | 10 | 3619.65 | — | 82 | 0 | 0 |
| cpu | learned | 10 | 4177.39 | -18.11% | 15 | 15 | 0 |
| cpu | oracle | 10 | 3619.65 | — | 0 | 0 | 0 |
| cuda | upstream | 10 | 213.85 | — | 0 | 0 | 0 |
| cuda | always | 10 | 221.76 | — | 173 | 0 | 0 |
| cuda | rule | 10 | 202.39 | — | 60 | 0 | 0 |
| cuda | learned | 10 | 190.70 | -2.82% | 0 | 0 | 0 |
| cuda | oracle | 10 | 193.70 | — | 0 | 0 | 0 |

相对 upstream 的 learned objective 变化：CPU -18.11%；CUDA -2.82%。
真实短 trace 的 learned 路径共出现 0 次 positive-lower-bound 决策；该 fresh-process 结果只证明受限探索和 fail-closed，长周期收敛证据见下一节。

### 长驻在线收敛与分布切换

同一 CUDA server PID 连续执行 53 个 wave；Ridge 最小样本数 12，confidence beta=1.00。

| Phase | Upstream | CacheFlow | Exploration | Positive lower bound | Positive waves / max streak | Drift | Safety fallback | TTFT P95 ms | Terminal benefit / uncertainty ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cold_start | 3 | 0 | 0 | 0 | 0 / 0 | 0 | 1 | 108.24 | 0.00 / 0.00 |
| stable_reuse | 49 | 160 | 18 | 142 | 33 / 13 | 0 | 21 | 323.91 | 21.29 / 8.82 |
| distribution_shift | 3 | 0 | 0 | 0 | 0 / 0 | 0 | 3 | 113.84 | 21.29 / 8.82 |

该实验补齐 fresh-process 短 trace 的证据缺口：稳定阶段必须出现非探索的置信下界启用，切换到独立 throughput-only 请求后必须 fail closed。

## CUDA Profiling 因果链

对 upstream/always 进行 paired Latin 干预；中间变量来自调度指标、CUDA Event 和 100 ms GPU 活跃度采样，结果来自 Engine trace 与请求 TTFT。该范围只覆盖本项目 CacheFlow KV kernel，不冒充完整 Nsight Compute kernel census。

| Paired trials | Δ CacheFlow decisions | Δ Prefill chunks | Δ Prefill tokens | Δ KV kernel launches | Δ KV copy bytes | Δ CUDA Event ms | Δ GPU busy | Δ max idle gap ms | Δ Execute us | Δ TTFT P95 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 13 | 23 | -354 | 2 | 20066300 | 0.808 | +0.00% | 0 | -11446 | 85.61 |

本轮强制 CacheFlow 改变了 prefill/CUDA 中间变量并恶化 Engine execute 与 TTFT；这是 learned gate 必须按上下文拒绝有害动作的直接系统证据。

## Production Engine Phase Trace

| Phase | Duration us | Share |
|---|---:|---:|
| execute | 9512266 | 99.8984% |
| commit | 7997 | 0.0840% |
| plan | 1397 | 0.0147% |
| prepare | 280 | 0.0029% |

共 250 个 production Engine spans。该数据来自 `in-process Chrome trace; WPR sampled stacks require elevated SeSystemProfilePrivilege`；它是 phase-duration trace，不是 sampled-stack CPU flame graph。

## 最小质量护栏

| 量化 | 通过题数 | 总题数 | 规则准确率 |
|---|---:|---:|---:|
| Q4_K_M | 2 | 5 | 40% |
| Q8_0 | 1 | 5 | 20% |
| F16 | 1 | 5 | 20% |

## 自动计算观察

- Q4_K_M 权重大小是 F16 的 38.5%，CUDA decode 速度是 F16 的 2.08 倍。
- 同一 Q4_K_M 配置下，CUDA decode 速度是 12 线程 CPU-only 的 5.68 倍。
- 并发从 1 增至 4 时，聚合输出吞吐提高到 2.13 倍；同时应结合 TTFT/TPOT 尾延迟判断交互体验。
- Cache-aware 调度在冲突序列中减少 91.6% 的重复 prefill token、减少 97.5% 的缓存淘汰，中位序列延迟提升 10.28 倍。
- 3 种精度在 5 题 smoke set 上的规则准确率为 20%–40%；样本过少，不能据此比较量化精度。

## 解释边界

- 量化对照使用同一 Qwen2.5-0.5B-Instruct GGUF 仓库和固定 revision。
- 结论不可直接外推到更大模型、其他 GPU 或不同上下文分布。
- 5 道规则题只是 smoke-level 质量护栏，不能替代标准 benchmark 或 perplexity。
- VRAM 为 nvidia-smi 对整块 GPU 的轮询值；增量以 case/服务启动前为基线，可能受其他 GPU 进程及采样间隔影响。
- TTFT 按首个非空 SSE content 事件计时，是首 token 延迟的服务端接口近似；TPOT 使用服务返回的 completion token 数。
- 原始输出位于 `results/raw/`，重新运行会覆盖汇总文件但保留固定配置。
