# llama.cpp 可复现基线报告

> 该报告由实验脚本从原始 JSON/CSV 自动生成；速度只代表当前机器、固定版本和固定配置。

## 环境

- 平台：`Windows-11-10.0.22631-SP0`
- Python：`3.13.13`
- GPU：`NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB, 561.19`
- llama.cpp：`b9632` / `acd79d603`

## 离线算子基线

| Case | 量化 | 后端 | 测试 | tokens/s | 标准差 | 模型 MiB | Run 峰值 VRAM MiB | Run 增量 VRAM MiB |
|---|---|---|---|---:|---:|---:|---:|---:|
| gpu-quantization | Q4_K_M | CUDA | pp256 | 14754.06 | 3237.99 | 463.0 | 625 | 625 |
| gpu-quantization | Q4_K_M | CUDA | tg64 | 311.57 | 10.60 | 463.0 | 625 | 625 |
| gpu-quantization | Q8_0 | CUDA | pp256 | 17446.60 | 2879.43 | 638.7 | 753 | 753 |
| gpu-quantization | Q8_0 | CUDA | tg64 | 256.79 | 8.76 | 638.7 | 753 | 753 |
| gpu-quantization | F16 | CUDA | pp256 | 13770.64 | 3856.47 | 1202.1 | 1241 | 1241 |
| gpu-quantization | F16 | CUDA | tg64 | 152.50 | 1.12 | 1202.1 | 1241 | 1241 |
| cpu-thread-6 | Q4_K_M | CPU-only | pp256 | 2633.14 | 616.53 | 463.0 | 431 | 431 |
| cpu-thread-6 | Q4_K_M | CPU-only | tg64 | 56.32 | 8.36 | 463.0 | 431 | 431 |
| cpu-thread-12 | Q4_K_M | CPU-only | pp256 | 3470.45 | 685.78 | 463.0 | 431 | 431 |
| cpu-thread-12 | Q4_K_M | CPU-only | tg64 | 60.73 | 6.19 | 463.0 | 431 | 431 |

`pp` 表示 prompt processing，`tg` 表示 token generation。同一 run 同时产生 pp/tg 记录，因此两行共享整次进程的显存峰值，并非阶段级峰值。`llama-bench` 不包含 tokenization 和 sampling 时间，因此在线指标需看下一节。

## 在线流式服务

| 并发 | 请求数 | TTFT p50 ms | TTFT p95 ms | TPOT p95 ms | 总延迟 p95 ms | 单请求平均 TPS | 聚合 TPS | 峰值 VRAM MiB | 服务增量 MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 30 | 15.96 | 39.45 | 3.60 | 205.48 | 286.35 | 257.10 | 553 | 553 |
| 2 | 60 | 26.75 | 44.22 | 5.15 | 270.26 | 220.62 | 394.68 | 553 | 553 |
| 4 | 120 | 35.08 | 54.12 | 8.51 | 435.24 | 141.35 | 510.94 | 553 | 553 |

## C++ KV Cache 调度 A/B

| 淘汰惩罚 | trials | 当前请求 ms | 后续长会话 ms | 序列总延迟 ms | 重复 prefill tokens | 选择淘汰 tokens |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 5 | 27.92 | 1348.56 | 1379.33 | 415 | 405 |
| 0.50 | 5 | 92.37 | 51.95 | 146.67 | 35 | 10 |

惩罚 0 等价于上游最长公共前缀选择；正惩罚使用本项目新增的净收益评分。当前请求可能少复用 token，但能避免破坏更有价值的长会话缓存，因此必须比较请求序列而非单请求。

## 最小质量护栏

| 量化 | 通过题数 | 总题数 | 规则准确率 |
|---|---:|---:|---:|
| Q4_K_M | 2 | 5 | 40% |
| Q8_0 | 1 | 5 | 20% |
| F16 | 1 | 5 | 20% |

## 自动计算观察

- Q4_K_M 权重大小是 F16 的 38.5%，CUDA decode 速度是 F16 的 2.04 倍。
- 同一 Q4_K_M 配置下，CUDA decode 速度是 12 线程 CPU-only 的 5.13 倍。
- 并发从 1 增至 4 时，聚合输出吞吐提高到 1.99 倍；同时应结合 TTFT/TPOT 尾延迟判断交互体验。
- Cache-aware 调度在冲突序列中减少 91.6% 的重复 prefill token、减少 97.5% 的缓存淘汰，中位序列延迟提升 9.40 倍。
- 3 种精度在 5 题 smoke set 上的规则准确率为 20%–40%；样本过少，不能据此比较量化精度。

## 解释边界

- 量化对照使用同一 Qwen2.5-0.5B-Instruct GGUF 仓库和固定 revision。
- 结论不可直接外推到更大模型、其他 GPU 或不同上下文分布。
- 5 道规则题只是 smoke-level 质量护栏，不能替代标准 benchmark 或 perplexity。
- VRAM 为 nvidia-smi 对整块 GPU 的轮询值；增量以 case/服务启动前为基线，可能受其他 GPU 进程及采样间隔影响。
- TTFT 按首个非空 SSE content 事件计时，是首 token 延迟的服务端接口近似；TPOT 使用服务返回的 completion token 数。
- 原始输出位于 `results/raw/`，重新运行会覆盖汇总文件但保留固定配置。
