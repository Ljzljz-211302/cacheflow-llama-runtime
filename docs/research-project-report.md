# CacheFlow Runtime：科研型项目报告

## 1. 经历性质判断

本工作适合放在简历的“项目经历”“科研实践”或“独立研究项目”，不建议在没有导师、实验室、立项、论文或投稿事实的情况下写成正式“科研经历”。它具备明确问题、假设、系统实现、对照实验和负结果边界，因此具有科研型项目属性；但项目属性不能替代真实学术组织关系。

推荐名称：**CacheFlow Runtime：面向单机 LLM Serving 的缓存感知调度与 CUDA KV 优化**。

## 2. 研究问题

消费级 GPU 显存有限，多轮及并发请求会重复执行 Prefill，并产生 KV Cache 复制、共享尾块写时复制和抢占恢复开销。本项目研究三个问题：

1. 如何在 llama.cpp 的真实请求槽位和 Decode 热路径中，以 Prefix 命中、等待时间和显存压力联合调度请求；
2. 如何用块级引用计数、COW、Swap 与持久化检查点管理 KV 生命周期；
3. 对真实模型 KV tensor 的 Gather/Scatter，128-bit 向量化是否能在保持 snapshot 语义和安全回退的前提下降低 CUDA 执行时间。

## 3. 系统设计与个人实现

项目固定 llama.cpp `acd79d603cb2e1c84c0886137b80f1ad649b6857` 为上游基线，以可逆 patch 管理个人修改。当前相对基线包含约 8,708 行新增、99 行删除，覆盖 61 个上游文件；不把未修改的 vendor 代码计为个人工作。

主要模块包括：

- 缓存感知 Scheduler：区分 Prefill/Decode，加入 Aging、防饥饿、背压、Deadline 和取消；
- KV Block Runtime：逻辑块到物理块映射、Prefix 索引、引用计数、partial-tail COW、抢占和恢复；
- CUDA KV 后端：Pinned Memory、异步 Stream/Event、Gather/Scatter、Swap 与错误回退；
- 在线收益门控：按 backend/context bucket 维护 EWMA、不确定性、探索和漂移回退；
- 真实应用消费者：推免面试学习助手，通过本地资料检索、SQLite 会话和 SSE 调用真实 CUDA llama-server。

## 4. 向量化 KV Remap 算子

### 4.1 算子语义

算子输入是一组 descriptor：`source`、`destination`、`staging_offset` 和元素数。执行分为 Gather 与 Scatter 两阶段：先将所有源区间快照到 staging，再统一写回目的区间，因此源/目的重叠和循环映射仍具有确定性。

### 4.2 CUDA 实现

- 对源、目的和 staging 均满足 16-byte 对齐的完整区间，使用 `uint4` 完成一次 128-bit、8 个 FP16 元素的加载或存储；
- 对非对齐地址和不足 8 个 FP16 的尾部，在同一公开接口中自动执行标量回退；
- 将工作单元从“每线程 1 个 FP16”调整为“每线程最多 8 个 FP16”，降低 descriptor-driven remap 的线程和指令开销；
- 对超过 CUDA `grid.y` 上限的 descriptor 数在 launch 前返回 `cudaErrorInvalidValue`；
- 暴露 vectorized/scalar remap byte 指标，避免只凭源码宣称算子进入生产路径。

### 4.3 正确性验证

- 独立 CPU oracle 验证重叠块交换的 snapshot 语义；
- 非对齐 19 元素尾部验证标量回退和前后守卫区；
- 非法 grid shape 验证 launch 前拒绝；
- Compute Sanitizer memcheck：0 error；racecheck：0 hazard；
- 真实 Qwen2.5-0.5B CUDA 服务共享 21 个 Prefix KV block，partial-tail COW 后输出与 cold deterministic decode 一致；
- 真实面试应用旅程记录 5,603,330 个向量化 KV remap 字节。

## 5. 实验设计与结果

硬件为 NVIDIA GeForce RTX 4050 Laptop GPU，CUDA 架构 sm_89。布局为 32 层、8 个 KV Head、Head Dimension 128、Block Size 16、FP16 K/V。每个规模执行 20 组 scalar/vectorized 配对试验，奇偶 trial 交替执行顺序，避免固定先后顺序偏差。

| Remap blocks | Scalar GPU median | Vectorized GPU median | 配对中位改善 |
|---:|---:|---:|---:|
| 1 | 0.029696 ms | 0.013312 ms | 53.33% |
| 4 | 0.117712 ms | 0.059776 ms | 48.89% |
| 16 | 0.812544 ms | 0.787920 ms | 3.13% |
| 32 | 1.600864 ms | 1.571344 ms | 1.87% |

验收门槛要求任一规模的配对中位回归不得超过 3%，且至少一个规模改善达到 10%；本轮全部通过。结果表明向量化对小批 descriptor remap 收益明显，随着传输规模增大，内存带宽逐渐成为主要瓶颈，收益收敛。

## 6. 结论边界

- 上表是 KV Remap 微基准，不等价于 TTFT、TPS 或端到端延迟提升；
- 实验只覆盖单台 RTX 4050 Laptop GPU 和指定 FP16 KV 布局，不能外推到 A100/H100 或多 GPU；
- 当前实现包含 KV 数据移动算子与受限 Qwen2.5-0.5B 单 token Paged Decode K1/K2；K2 只在 D64/GQA7/context≤17 生产 envelope 晋级，它不是通用 FlashAttention、GEMM、prefill PagedAttention 或任意模型实现；
- 真实应用指标证明算子被调用，不能证明已有外部用户或线上采用率。
- 本次增量的确定性功能、Sanitizer、真实模型与 Remap 性能门禁均通过；2026-08-08 严格 Full 也原生退出 0。此前两次时序敏感失败保留为历史反例；一次 Full 通过只说明该冻结提交和环境满足门槛，不声称未来所有运行都不会受统计波动影响。

在上述边界内，本项目可以支撑 AI Infra、LLM Serving、CUDA 推理优化方向的项目面试和科研实践陈述。
