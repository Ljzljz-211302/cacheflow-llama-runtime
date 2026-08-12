# CacheFlow Runtime 面试学习资源

## Knowledge

- [Paper: Attention Is All You Need — NeurIPS 2017](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf)
  Transformer、自注意力和自回归解码的原始依据。用于学习 Q/K/V、causal attention 与 prefill/decode 差异。
- [Paper: Orca — OSDI 2022](https://www.usenix.org/system/files/osdi22-yu.pdf)
  iteration-level scheduling 与 selective batching 的代表工作。用于理解为何 serving 不能按完整请求批处理。
- [Paper: Efficient Memory Management for LLM Serving with PagedAttention — SOSP 2023](https://doi.org/10.1145/3600006.3613165)
  分块 KV 管理、逻辑块到物理块映射与共享的主要学术参照。用于对照本项目的 Block Table、Prefix 与 COW。
- [Paper: FlashAttention — NeurIPS 2022](https://papers.neurips.cc/paper_files/paper/2022/file/67d57c32e20fd0a7a302cb81d36e40d5-Paper-Conference.pdf)
  从 HBM/SRAM 数据移动出发解释 IO-aware tiling 与 online softmax。用于区分 FlashAttention 的 IO 算法目标和 PagedAttention 的 KV 地址/内存管理目标。
- [Paper: Sarathi-Serve — OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/agrawal)
  Chunked Prefill 和吞吐—延迟权衡的一手资料。用于理解本项目 Adaptive Prefill 的动机和边界。
- [Paper: Megatron-LM — 2019](https://arxiv.org/abs/1909.08053)
  Transformer 层内张量并行的代表工作。用于理解 Tensor Parallelism 的矩阵拆分、集合通信与模型容量权衡。
- [Paper: ZeRO — SC 2020](https://arxiv.org/abs/1910.02054)
  通过分片优化器状态、梯度和参数减少数据并行显存冗余。用于理解训练显存组成与 ZeRO 各阶段的通信—容量取舍。
- [NVIDIA NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/index.html)
  NVIDIA 多 GPU/多节点集合通信的官方资料。用于核对 All-Reduce、All-Gather、Reduce-Scatter 与通信 API 语义。
- [Paper: SmoothQuant — ICML 2023](https://proceedings.mlr.press/v202/xiao23c/xiao23c.pdf)
  W8A8 训练后量化及激活异常值迁移的一手资料。用于区分 weight-only、weight-activation 与 KV 量化的目标和风险。
- [Paper: Fast Inference from Transformers via Speculative Decoding — ICML 2023](https://proceedings.mlr.press/v202/leviathan23a.html)
  推测解码的正确性与加速来源。用于区分论文中的 draft-model decoding 和本项目 N-gram speculation controller。
- [Paper: A Contextual-Bandit Approach to Personalized News Article Recommendation — WWW 2010](https://arxiv.org/abs/1003.0146)
  LinUCB 的经典应用。用于理解按 action 分开的在线 Ridge、置信半径和探索—利用权衡。
- [CUDA Programming Guide: Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)
  Host/device、SM、kernel、线程层次与异构执行的官方说明。
- [CUDA Programming Guide: Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)
  Stream、Event、异步拷贝和计时语义。用于理解项目的 CUDA Event 与异步生命周期。
- [NVIDIA Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)
  memcheck、racecheck、initcheck、synccheck 的官方定义和能力边界。
- [Prometheus Metric Types](https://prometheus.io/docs/concepts/metric_types/)
  Counter、Gauge、Histogram、Summary 的官方语义。用于解释项目为何对决策计数用 counter，对终态置信度用 gauge。
- [llama.cpp repository](https://github.com/ggml-org/llama.cpp)
  固定上游的代码语境。用于区分上游 GGUF/GGML/采样/HTTP 与个人改动。

## Wisdom (Communities)

- [vLLM GitHub Discussions](https://github.com/vllm-project/vllm/discussions)
  观察真实 serving 的 KV、调度和性能问题；适合检验设计是否只在实验环境成立。
- [NVIDIA Developer Forums — CUDA Programming](https://forums.developer.nvidia.com/c/accelerated-computing/cuda/206)
  CUDA 异步错误、同步与性能诊断的高质量实践讨论。
- [llama.cpp GitHub Discussions](https://github.com/ggml-org/llama.cpp/discussions)
  理解上游约束、模型兼容和实际部署问题；提问前需准备最小复现和硬件信息。

## Gaps

- 当前机器已用 Nsight Systems 完成 PID/request/kernel 机制关联；Nsight Compute 已尝试，但受 driver/tool compatibility 与 `ERR_NVGPUCTRPERM` 限制，当前不作 occupancy/roofline/DRAM-bound 归因。
- 当前实验是单机单 GPU、小模型；分布式 tensor/pipeline parallel 只作为延伸问题，不作为项目已实现内容。
