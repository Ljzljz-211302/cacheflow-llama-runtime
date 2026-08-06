# ADR-0001: Adopt copy-aware Paged KV execution as the research direction

- **Status:** Accepted
- **Date:** 2026-08-06

## Context

CacheFlow Runtime 已经具备事务式迭代、Paged KV 生命周期、Prefix/COW/Swap、CUDA KV Remap 与在线 Benefit Gate，但这些能力仍容易被理解为若干独立工程模块。项目需要一个可证伪的研究问题，将模型侧 Decode 算法、CUDA 算子、运行时调度与性能证据连接起来，同时控制在单人可完成、可复现的范围内。

## Decision

项目研究主线确定为：**面向消费级单 GPU LLM Serving 的 Copy-aware Paged KV 管理与自适应执行策略**。

研究围绕三个问题展开：

1. KV 搬运在何种请求长度、共享比例、块布局与显存压力下成为 Decode 瓶颈？
2. 受限范围的 Paged Decode Attention 何时优于 Direct 或 KV Remap，代价来自访存、索引、归约还是 kernel launch？
3. 能否用可解释的在线代价模型，在 Direct、KV Remap、Paged Decode Attention、Swap 和 Recompute 之间安全选择，并优于固定阈值？

首个 Paged Decode Attention 研究切片限定为 Qwen2.5、FP16 KV、GQA、单 token decode、单张消费级 NVIDIA GPU。任何未覆盖模型、dtype、prefill 或多 GPU 场景必须进入回退路径，不得作为已支持能力宣传。

## Consequences

- 优点：形成从算法、CUDA kernel、运行时策略到实验评估的完整因果链；现有模块成为研究基线而不是无关功能堆叠。
- 优点：研究结论可以出现负结果；即使 Paged Decode Attention 并非全区间占优，动作边界和原因仍是有效产出。
- 代价：必须维护参考实现、正确性 oracle、统计协议与设备状态记录，开发速度会慢于只追求演示。
- 代价：早期结论只适用于受限模型和硬件，不能外推为通用 Serving 结论。
- 放弃：本阶段不实现多 GPU/分布式推理、通用 prefill PagedAttention、训练/微调系统或覆盖所有 llama.cpp 模型的 kernel。

## Revisit when

- 受限实现无法在任何预注册工作负载上提供可测价值，且 profiling 证明瓶颈不在 KV 搬运；
- 新硬件或上游接口使 Direct/KV Remap/Paged 的成本结构发生根本变化；
- 项目获得多 GPU 实验资源并明确扩大研究问题。
