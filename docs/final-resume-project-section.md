# 项目经历

## CacheFlow Runtime：单机大模型推理调度与 CUDA KV 优化

**个人项目｜2026.07–至今**  
**技术栈：** C++17、CUDA、llama.cpp、GGML/GGUF、Python、CMake、Prometheus、SQLite

- 基于固定 llama.cpp 上游版本重构真实 `llama-server → llama_decode → KV Cache → CUDA` 推理链路，设计 Prefill/Decode 分离的缓存感知调度器，实现 Continuous Batching、Aging 防饥饿、请求背压、Deadline、取消传播及故障回退。
- 设计 KV Block Manager 与 Prefix Index，实现逻辑/物理块映射、引用计数、Prefix 共享、partial-tail Copy-on-Write、Pinned Memory 异步 Swap 和检查点恢复；在真实 Qwen2.5 CUDA 请求中共享 21 个 Prefix KV Block，COW 后输出与 cold deterministic decode 保持一致。
- 实现 descriptor-driven CUDA KV Remap 算子，以 `uint4` 完成 128-bit Gather/Scatter；支持重叠映射的 snapshot 语义、非对齐及尾部标量回退、非法 grid 前置拒绝，并将成功指标接入原生 Prometheus；Compute Sanitizer memcheck 0 error、racecheck 0 hazard。
- 建立无 profiler 主结果与 Nsight Systems 机制证据分离的 CUDA 因果链：aligned 1-block CUDA-event 改善 57.10%（95% CI 38.51%–57.14%）；scalar/vector 在方法隔离 trace 中均为 10/10 launches，定位收益来自向量访存路径而非减少 kernel 数。
- 实现 Qwen2.5 Paged Decode K1–K4 CUDA kernel 与生产 dispatch：从 H10 的 +50.35% 长上下文回退出发，将算子重构为 256-thread CTA 覆盖完整 7:1 GQA 组、`half2` K/V 向量访问、FP32 online-softmax 与设备端 64/128-token 自适应分区；24 个 CPU/CUDA oracle 边界用例覆盖 1–2048 token。构建 3 类真实文档、18 个精确长度 workload、12 个严格平衡匹配进程块的 432-cell 客观矩阵，将 512–2048 token 主中位回退降至 3.98%（95% CI 2.50%–5.38%，P95 13.34%），因 CI 上界仍超 +5% 而保持 fail-closed，并定位下一步为 upstream vector attention 的页表 accessor 融合。
- 设计 Direct/Remap/Paged/Swap/Recompute 统一候选接口及 capability/resource gate；以 40 条隔离 trace、200 组真实观测比较 H0/A1/T1/L1，本次 matched-workload replay 中 T1/H0 累计 oracle regret 为 5.506/15.543ms、harmful decision 均为 0，paired trace-cluster CI 为 −0.5285～−0.0511ms；完成 500 万次零分配 chooser，最差 P99 1μs。
- 设计 backend-local 在线 Ridge 收益门控与置信下界、有限探索/漂移回退；以 16-trial 联合 Williams 实验平衡 8 个 `backend×policy` treatment 的执行位置和一阶前驱效应，CPU/CUDA paired oracle regret 为 5.04%/10.52%，生产决策路径最坏 trial P99 为 2/5μs（预算 50μs）。
- 从一手论文、作者代码与 NVIDIA 官方文档审计 FlashInfer/vLLM/PagedAttention 等相关工作，区分本机可运行 baseline 与 related work；预注册研究问题、证伪条件、配对/bootstrap 统计和负结果规则，禁止挪用外部性能数字或事后放宽门槛。
- 在 RTX 4050 Laptop GPU 上完成 20 组配对且交替执行顺序的微基准：相对标量实现，1/4/16/32 Block 的 GPU 中位耗时分别改善 **53.33% / 48.89% / 3.13% / 1.87%**；开发推免面试学习助手作为真实负载，覆盖带引用 SSE 回答、SQLite 会话恢复、并发限流与客户端中断，应用旅程累计执行 **5.60M** 向量化 KV Remap 字节。
