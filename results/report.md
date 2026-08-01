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
| gpu-quantization | Q4_K_M | CUDA | pp256 | 14178.02 | 4151.11 | 463.0 | 634 | 625 |
| gpu-quantization | Q4_K_M | CUDA | tg64 | 317.60 | 10.78 | 463.0 | 634 | 625 |
| gpu-quantization | Q8_0 | CUDA | pp256 | 17802.69 | 2966.44 | 638.7 | 762 | 753 |
| gpu-quantization | Q8_0 | CUDA | tg64 | 264.82 | 1.85 | 638.7 | 762 | 753 |
| gpu-quantization | F16 | CUDA | pp256 | 14168.44 | 3625.26 | 1202.1 | 1210 | 1201 |
| gpu-quantization | F16 | CUDA | tg64 | 153.07 | 1.21 | 1202.1 | 1210 | 1201 |
| cpu-thread-6 | Q4_K_M | CPU-only | pp256 | 3496.20 | 839.50 | 463.0 | 440 | 431 |
| cpu-thread-6 | Q4_K_M | CPU-only | tg64 | 66.91 | 1.59 | 463.0 | 440 | 431 |
| cpu-thread-12 | Q4_K_M | CPU-only | pp256 | 3232.82 | 429.31 | 463.0 | 440 | 431 |
| cpu-thread-12 | Q4_K_M | CPU-only | tg64 | 65.57 | 1.00 | 463.0 | 440 | 431 |

`pp` 表示 prompt processing，`tg` 表示 token generation。同一 run 同时产生 pp/tg 记录，因此两行共享整次进程的显存峰值，并非阶段级峰值。`llama-bench` 不包含 tokenization 和 sampling 时间，因此在线指标需看下一节。

## 在线流式服务

| 并发 | 请求数 | TTFT p50 ms | TTFT p95 ms | TPOT p95 ms | 总延迟 p95 ms | 单请求平均 TPS | 聚合 TPS | 峰值 VRAM MiB | 服务增量 MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 30 | 23.63 | 36.13 | 3.53 | 199.40 | 293.27 | 262.10 | 562 | 553 |
| 2 | 60 | 30.30 | 45.84 | 4.54 | 253.74 | 234.63 | 415.27 | 562 | 553 |
| 4 | 120 | 33.44 | 53.21 | 7.28 | 380.24 | 150.96 | 543.23 | 562 | 553 |

## C++ KV Cache 调度 A/B

| 淘汰惩罚 | trials | 当前请求 ms | 后续长会话 ms | 序列总延迟 ms | 重复 prefill tokens | 选择淘汰 tokens |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 5 | 29.38 | 1408.42 | 1436.76 | 415 | 405 |
| 0.50 | 5 | 98.94 | 56.01 | 155.34 | 35 | 10 |

惩罚 0 等价于上游最长公共前缀选择；正惩罚使用本项目新增的净收益评分。当前请求可能少复用 token，但能避免破坏更有价值的长会话缓存，因此必须比较请求序列而非单请求。

## 真实 CUDA Tail COW

| 方法 | Median E2E ms | P95 E2E ms | Bytes/op | Launch/op | Extra device bytes |
|---|---:|---:|---:|---:|---:|
| tail_block_cow | 0.0082 | 0.0289 | 196608 | 1 | 196608 |
| whole_sequence_copy | 0.7629 | 0.9220 | 12582912 | 128 | 12582912 |

3 个 fresh process、每个方法 300 samples；Tail COW P95 改善 96.86%。

## Adaptive Prefill

| Backend | Mode | Median wall ms | P95 wall ms | Effective chunk median |
|---|---|---:|---:|---:|
| cpu | greedy | 12485.56 | 12568.22 | 0 |
| cpu | fixed64 | 13471.84 | 13548.43 | 64 |
| cpu | fixed256 | 13085.14 | 13106.44 | 256 |
| cpu | adaptive | 13126.96 | 13179.55 | 31 |
| cuda | greedy | 639.78 | 640.00 | 0 |
| cuda | fixed64 | 739.86 | 765.94 | 64 |
| cuda | fixed256 | 651.76 | 666.29 | 256 |
| cuda | adaptive | 692.56 | 712.94 | 80 |

Adaptive 能避开错误 fixed-64，但当前 CUDA trace 未稳定击败 greedy/fixed-256；该负结果保留。

## Adaptive Speculation

| Backend | Mode | Median wall ms | P95 wall ms | Acceptance |
|---|---|---:|---:|---:|
| cpu | none | 4269.71 | 4657.51 | 0.0% |
| cpu | fixed | 3540.91 | 3564.45 | 85.4% |
| cpu | adaptive | 3420.58 | 3461.88 | 97.2% |
| cuda | none | 409.72 | 417.21 | 0.0% |
| cuda | fixed | 222.55 | 226.88 | 85.4% |
| cuda | adaptive | 218.50 | 231.35 | 97.2% |

## Mixed Prefill / Decode

| Backend | Policy | Trials | TTFT median / P95 ms | TPOT P95 ms | Latency P95 ms | Aggregate output TPS |
|---|---|---:|---:|---:|---:|---:|
| cpu | upstream | 3 | 1556.69 / 7693.09 | 673.07 | 11729.82 | 13.08 |
| cpu | cacheflow | 3 | 1566.96 / 3011.56 | 281.17 | 6933.29 | 19.94 |
| cuda | upstream | 3 | 55.19 / 153.14 | 14.76 | 514.31 | 284.98 |
| cuda | cacheflow | 3 | 62.92 / 83.70 | 12.42 | 542.41 | 275.33 |

CPU 本轮 latency P95 -40.9%、aggregate TPS +52.5%；CUDA 本轮 latency P95 +5.5%、aggregate TPS -3.4%。这是当前轮结果，不外推为稳定收益。

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

| Backend | Mode | Trials | Objective median ms | CacheFlow decisions | Exploration | Positive lower bound |
|---|---|---:|---:|---:|---:|---:|
| cpu | upstream | 10 | 4542.93 | 0 | 0 | 0 |
| cpu | always | 10 | 3608.49 | 195 | 0 | 0 |
| cpu | rule | 10 | 3743.66 | 60 | 0 | 0 |
| cpu | learned | 10 | 4042.18 | 16 | 16 | 0 |
| cpu | oracle | 10 | 3559.28 | 0 | 0 | 0 |
| cuda | upstream | 10 | 207.40 | 0 | 0 | 0 |
| cuda | always | 10 | 214.94 | 162 | 0 | 0 |
| cuda | rule | 10 | 211.76 | 61 | 0 | 0 |
| cuda | learned | 10 | 208.06 | 40 | 40 | 0 |
| cuda | oracle | 10 | 194.31 | 0 | 0 | 0 |

相对 upstream 的 learned objective 变化：CPU -11.02%；CUDA +0.32%。
真实短 trace 的 learned 路径共出现 0 次 positive-lower-bound 决策；当前结果证明的是受限探索和 fail-closed 护栏，稳定优势下的置信触发由确定性 native replay 覆盖，不能据此宣称线上长周期收敛。

## Production Engine Phase Trace

| Phase | Duration us | Share |
|---|---:|---:|
| execute | 9485332 | 99.9058% |
| commit | 7402 | 0.0780% |
| plan | 1293 | 0.0136% |
| prepare | 253 | 0.0027% |

共 246 个 production Engine spans。该数据来自 `in-process Chrome trace; WPR sampled stacks require elevated SeSystemProfilePrivilege`；它是 phase-duration trace，不是 sampled-stack CPU flame graph。

## 最小质量护栏

| 量化 | 通过题数 | 总题数 | 规则准确率 |
|---|---:|---:|---:|
| Q4_K_M | 2 | 5 | 40% |
| Q8_0 | 1 | 5 | 20% |
| F16 | 1 | 5 | 20% |

## 自动计算观察

- Q4_K_M 权重大小是 F16 的 38.5%，CUDA decode 速度是 F16 的 2.07 倍。
- 同一 Q4_K_M 配置下，CUDA decode 速度是 12 线程 CPU-only 的 4.84 倍。
- 并发从 1 增至 4 时，聚合输出吞吐提高到 2.07 倍；同时应结合 TTFT/TPOT 尾延迟判断交互体验。
- Cache-aware 调度在冲突序列中减少 91.6% 的重复 prefill token、减少 97.5% 的缓存淘汰，中位序列延迟提升 9.25 倍。
- 3 种精度在 5 题 smoke set 上的规则准确率为 20%–40%；样本过少，不能据此比较量化精度。

## 解释边界

- 量化对照使用同一 Qwen2.5-0.5B-Instruct GGUF 仓库和固定 revision。
- 结论不可直接外推到更大模型、其他 GPU 或不同上下文分布。
- 5 道规则题只是 smoke-level 质量护栏，不能替代标准 benchmark 或 perplexity。
- VRAM 为 nvidia-smi 对整块 GPU 的轮询值；增量以 case/服务启动前为基线，可能受其他 GPU 进程及采样间隔影响。
- TTFT 按首个非空 SSE content 事件计时，是首 token 延迟的服务端接口近似；TPOT 使用服务返回的 completion token 数。
- 原始输出位于 `results/raw/`，重新运行会覆盖汇总文件但保留固定配置。
