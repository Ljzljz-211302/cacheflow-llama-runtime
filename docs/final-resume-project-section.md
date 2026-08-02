# 项目经历

## CacheFlow Runtime：单机大模型推理调度与 CUDA KV 优化

**个人项目｜2026.07–至今**  
**技术栈：** C++17、CUDA、llama.cpp、GGML/GGUF、Python、CMake、Prometheus、SQLite

- 基于固定 llama.cpp 上游版本重构真实 `llama-server → llama_decode → KV Cache → CUDA` 推理链路，设计 Prefill/Decode 分离的缓存感知调度器，实现 Continuous Batching、Aging 防饥饿、请求背压、Deadline、取消传播及故障回退。
- 设计 KV Block Manager 与 Prefix Index，实现逻辑/物理块映射、引用计数、Prefix 共享、partial-tail Copy-on-Write、Pinned Memory 异步 Swap 和检查点恢复；在真实 Qwen2.5 CUDA 请求中共享 21 个 Prefix KV Block，COW 后输出与 cold deterministic decode 保持一致。
- 实现 descriptor-driven CUDA KV Remap 算子，以 `uint4` 完成 128-bit Gather/Scatter；支持重叠映射的 snapshot 语义、非对齐及尾部标量回退、非法 grid 前置拒绝，并将成功指标接入原生 Prometheus；Compute Sanitizer memcheck 0 error、racecheck 0 hazard。
- 在 RTX 4050 Laptop GPU 上完成 20 组配对且交替执行顺序的微基准：相对标量实现，1/4/16/32 Block 的 GPU 中位耗时分别改善 **53.33% / 48.89% / 3.13% / 1.87%**；开发推免面试学习助手作为真实负载，覆盖带引用 SSE 回答、SQLite 会话恢复、并发限流与客户端中断，应用旅程累计执行 **5.60M** 向量化 KV Remap 字节。
