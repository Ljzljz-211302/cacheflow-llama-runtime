# 面试追问卡

## 30 秒项目介绍

我基于固定版本的 llama.cpp 和同一 Qwen2.5-0.5B 模型构建了可复现推理实验台，对 Q4、Q8、F16 的内存、prefill/decode、CPU/GPU 和在线并发进行受控比较。我没有只搭服务页面，而是实现了 SSE 级 TTFT/TPOT、结构化实验、KV Cache 预算推荐、正负规则质量护栏和 SHA-256 制品复现。实验显示量化显著改善速度与体积，但小模型事实错误仍是主要瓶颈。

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
4. 阅读 `llama-bench`、slot scheduler 和 KV Cache 代码，选择一个上游可接受的小改动并补 C++ 测试。
