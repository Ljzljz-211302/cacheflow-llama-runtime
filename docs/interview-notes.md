# 面试追问卡

## 30 秒项目介绍

我修改了 llama.cpp 的 C++ slot scheduler，实现带缓存淘汰成本的 KV-cache-aware 调度。上游只最大化当前 prompt 的前缀复用，可能破坏更有价值的长会话缓存；我的策略用“复用 token−惩罚×淘汰 token”评分，并增加引擎原生 KV、内存和 cache-hit metrics。5 次冲突 workload 中，重复 prefill 减少 91.6%，两请求序列中位延迟提升 9.88 倍。外围实验台负责可复现构建、CUDA/CPU/量化对照和结果校验。

## 为什么这不是只包装 llama.cpp

个人 C++ patch 修改了任务调度的决策结果，包含新的 CLI 参数、纯函数策略模块、原生 C++ 测试和 Prometheus 指标。`vendor/` 是未计入个人代码量的固定上游；`patches/` 才是可审查的本人引擎贡献。Python 只负责构造 workload 和分析结果，不决定 slot 选择。

## 调度目标函数与取舍

上游最长公共前缀策略等价于淘汰惩罚为 0。新策略的正惩罚保护长缓存，但可能让当前请求少复用 token，所以不能只看当前 TTFT。实验必须加入后续会话请求并统计累计 prefill 与序列延迟。惩罚系数不是普适常数，应根据会话回访概率、SLO 和缓存压力调参；生产系统可进一步在线学习该参数。

## 正确性与复杂度

对每个空闲 slot 计算一次 LCP，上游已经承担这部分开销；新增评分是 O(number_of_slots) 的常数运算。零惩罚保持原行为，负收益时回退 LRU，相同得分由最久未使用 slot 打破平局。测试覆盖兼容模式、保护长缓存、阈值/负收益回退和 LRU tie-break。

## 为什么 Q4 更小但不保证快四倍

权重位宽下降直接减少文件和权重访存，但运行时仍有反量化、激活、KV Cache、调度和非矩阵算子。decode 常受内存带宽影响，因此更小权重可能明显加速；prefill 的矩阵计算占比更高，收益形态不同。最终速度还取决于 kernel、batch、GPU 利用率和量化 block 格式。

## Prefill 与 decode

- Prefill 一次处理整个 prompt，矩阵较大、并行度高，更偏计算密集。
- Decode 每轮只增加一个 token，却要读取模型权重并访问逐渐增长的 KV Cache，常更偏带宽和调度瓶颈。
- `llama-bench` 的 `pp`/`tg` 分开测量；在线 TTFT 还包含 tokenization、排队、采样和网络栈。

## KV Cache 公式与边界

当前估算使用 `2 × L × H_kv × D_head × context × slots × bytes`。两个因子来自 K 和 V。GQA/MQA 通过减少 KV heads 降低 Cache；量化 KV 可以继续降内存，但可能影响精度或 kernel 支持。公式没有包含 allocator 对齐、计算图、临时 buffer 和模型权重，所以推荐器保留 runtime 与安全余量。

## 为什么并发提高吞吐却恶化延迟

continuous batching 让多个请求共享 GPU 执行，提高利用率和聚合吞吐；但单请求会等待调度并与其他序列共享计算、带宽和 KV 容量。因此必须同时报告 aggregate TPS、TTFT、TPOT 和 p95，而不能只报平均吞吐。

## 如何保证比较公平

1. 三种量化来自同一官方仓库和固定 revision。
2. 固定机器、电源模式、prompt/generation tokens、线程和重复次数。
3. 区分 CPU-only 与仅“加载了 CUDA backend”。
4. 保留原始 JSON、均值和标准差。
5. 性能比较之外保留固定质量题；若形成论文结果，应换标准数据集并增加随机种子和置信区间。

## 当前结果中最值得讲的失败

0.5B 模型能高速生成，但会把简单算术 `17+25` 回答成错误结果，也会给二分查找附加错误条件。最初只检查必要关键词会误把“包含正确词但同时含错误陈述”的回答判为通过，因此质量评分增加了 forbidden patterns。这说明评测器本身也需要错误分析与反例驱动迭代。

## 如果继续两周

1. 用标准语料运行 perplexity，对比 Q4/Q8/F16，而不是扩大自制题库。
2. 加入固定功耗/温度和 VRAM 时间序列，解释 laptop GPU 的频率波动。
3. 对 batch、context、GPU offload 做 Pareto 前沿，而不是只找单点最快参数。
4. 用真实多轮会话 trace 学习 eviction penalty，并与 LRU、LCP、Belady oracle 比较命中率和 SLO。
5. 安装 CUDA Toolkit，构建同一 patch 的 CUDA server，验证 CPU 结论是否在 GPU prefill/continuous batching 下保持。
