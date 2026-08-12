# 简历项目经历：CacheFlow Runtime

> 最终可直接投递版本见 [`final-resume-project-section.md`](final-resume-project-section.md)。本文其余内容保留为不同篇幅版本和面试表述边界。

## 推荐放置

优先放在“项目经历”或“科研实践”，不建议在没有导师、实验室或论文事实时放入正式“科研经历”。如果简历只有“科研/项目经历”合并栏目，可以标注“个人科研型项目”。

## 一页简历版本

### CacheFlow Runtime：单机大模型推理调度与 CUDA KV 优化｜个人科研型项目｜2026.07–至今

技术栈：C++17、CUDA、llama.cpp、GGML/GGUF、Python、CMake、Prometheus、SQLite

- 基于固定 llama.cpp 上游提交重构真实 `llama-server → llama_decode → KV Cache → CUDA` 路径，实现 Prefill/Decode 分离的缓存感知调度、Aging 防饥饿、背压、取消、Deadline 及多请求 Continuous Batching。
- 设计 KV Block Manager 与 Prefix 索引，实现引用计数、partial-tail Copy-on-Write、Pinned Memory 异步 Swap、检查点恢复和故障回退，并以真实 Qwen CUDA 请求验证共享 21 个 Prefix KV Block 后输出与 cold decode 一致。
- 实现 descriptor-driven CUDA KV Remap 算子：使用 `uint4` 进行 128-bit Gather/Scatter，支持重叠映射 snapshot 语义、非对齐/尾部标量回退和非法 grid 拒绝；Compute Sanitizer memcheck/racecheck 均为 0 error。
- 分离无 profiler 主结果与 Nsight Systems 机制回放：aligned 1-block CUDA-event 改善 57.10%（95% CI 38.51%–57.14%），misaligned 反例回退 137.94%；scalar/vector launch 数均为 10/10，排除“减少 launch”解释，并明确 NCU 同时受 driver/tool 不兼容与 performance-counter 权限限制。
- 实现受限 Qwen2.5 Paged Decode K1/K2 CUDA kernel 与生产 dispatch；K2 以双 warp tile 将每 KV head 的 K/V 装载从 7 次降为 4 次，并实现转置 shared K 与 warp stable-softmax；同进程 30 组×每臂 16 次请求中端到端 median/P95 保持在 0.55%/1.52% 回退内（median 回退 95% 上界 2.86%），NSYS 相同 480 次 kernel 总时长下降 50.44%，通过预注册生产替换门槛；Paged 相对 Direct 的旧负结果仍保留并保持 opt-in。
- 设计 Direct/Remap/Paged/Swap/Recompute 统一候选接口、capability/resource gate 与可解释代价模型；Issue #6 正式 replay 仅覆盖 Direct、device/host Swap、Recompute 的 200 条观测，另完成 500 万次零分配 chooser 开销微基准，保留 Remap/Paged 当时被 mask 及服务器状态无法完全克隆的边界。
- 设计 backend-local 在线 Ridge 收益门控、置信下界、有限探索和漂移回退；16-trial 联合 Williams 实验平衡 8 个 `backend×policy` treatment 的位置与一阶前驱效应，CPU/CUDA paired oracle regret 为 5.04%/10.52%，生产 chooser 最坏 trial P99 为 2/5μs（预算 50μs）。
- 审计一手论文、作者实现和 NVIDIA 官方文档，区分本机可复现 baseline 与 related work；预注册研究问题、配对/bootstrap 统计、证伪门槛和负结果规则，避免外部数字挪用与事后调参。
- 在 RTX 4050 Laptop GPU 上完成 20 组配对、交替顺序微基准；相对标量实现在 1/4/16/32 Block 上的 GPU 中位耗时分别改善 53.33%/48.89%/3.13%/1.87%，所有规模无回归；通过原生 Prometheus 指标确认真实应用累计执行 5.60M 向量化 KV Remap 字节。
- 开发推免面试学习助手作为真实用户负载，覆盖本地资料检索、带引用 SSE 回答、SQLite 会话恢复、并发限流与客户端中断，并以独立应用进程和真实 CUDA 模型完成端到端验收。

## 三条精简版本

- 深度修改 llama.cpp 推理热路径，实现缓存感知调度、KV Block/COW/Swap、五动作统一代价模型、在线收益门控及真实 CUDA Serving 观测链路。
- 编写 128-bit 向量化 CUDA KV Remap 算子，支持重叠 snapshot、非对齐尾部回退；Compute Sanitizer memcheck/racecheck 0 error。
- 以配对实验、bootstrap CI、NSYS 因果链、200 条动作观测和 500 万次 chooser 微基准验证机制与开销边界；真实 Qwen 应用路径累计命中 5.60M 向量化字节。

## 面试口述版本

“这个项目不是在 llama.cpp 外面套一层 Python。我的主要工作进入了真实 llama-server 的调度和 KV 热路径：上层用缓存命中、等待时间和显存压力决定请求顺序；中层维护 KV Block、引用计数、COW 和 Swap；底层实现了 descriptor-driven CUDA KV Remap。新算子对齐时使用 uint4 做 128-bit Gather/Scatter，不对齐和尾部自动回退标量。它通过 CPU oracle、Compute Sanitizer、真实 Qwen Prefix 共享和应用原生指标验证。微基准说明小批 remap 提升约 52%–55%，大批量收益收敛到约 2%–4%，所以我不会把它包装成端到端推理提速。”

## 禁止使用的夸大表述

- “实现了通用 FlashAttention/PagedAttention”——当前仅实现并接入受限 Qwen2.5-0.5B Paged Decode K1/K2，split-K2 的 host 正确性能力门覆盖 D64/GQA7/context≤2048；v2.10 的 K2/K1 替换只覆盖 page16/context17，H10 长上下文 Paged/Direct 仍为负结果，不能外推通用模型或 prefill；
- “端到端推理加速 53.33%”——该数字仅属于 KV Remap 微基准；
- “支持生产级公网多租户”——当前定位为单机可信环境；
- “科研成果/论文成果”——除非后续确有导师、立项、论文或投稿事实。
