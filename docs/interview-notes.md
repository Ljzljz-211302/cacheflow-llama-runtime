# CacheFlow Runtime 面试深挖手册

> 本文保留早期深挖笔记。2026-08-06 之后的统一事实、完整基础讲解和问题答案请以 [`lessons/cacheflow-runtime-complete-interview-handbook.html`](../lessons/cacheflow-runtime-complete-interview-handbook.html) 为准。

## 30 秒版本

我基于固定版本 llama.cpp 做了一个单机 LLM Serving / AI Infra fork，重构了真实推理热路径，而不是在外面套 Python。核心包括 transaction-based Engine loop、token-level scheduler、paged prefix KV 的 Block Table/引用计数/尾块 COW、Host/File/CUDA 事务 Swap，以及真实 K/V Tensor 上的 CUDA Gather/Scatter/COW。项目还实现了 adaptive prefill/speculation、故障注入、原生 metrics 和 CPU/CUDA 多 trial 验证。最重要的结果是 Tail COW 把最终矩阵的 P95 从 0.998 ms 降到 0.026 ms；但我也保留了 CUDA mixed workload 指标跨轮反号的负结果，并据此说明策略需要 backend-aware gating。

## 三分钟版本

上游 llama.cpp 的 server 很成熟，但调度、slot 状态、KV 生命周期和 decode 调用集中在一个大循环中，策略和机制容易互相污染。我先把一次 iteration 定义成 prepare、plan、execute、commit 四阶段：Scheduler 只产生不可变计划，Runtime Adapter 执行，只有 execute 成功才 commit；失败则 abort，不发布半完成状态。`server_inference_engine` 是控制面所有者，`server_context` 只保留组合、HTTP 适配和调用 `llama_decode` 的 callback。

KV 侧不是简单加一个命中率计数器。我实现了固定 token block 的 Block Table、PrefixIndex、Reservation 和 refcount。多个 sequence 可以共享包括部分 tail 在内的 exact prefix；第一次 append 前自动 COW，避免修改共享块。逻辑块通过 llama memory capability 接到真实 Attention K/V Tensor。CUDA 路径使用 descriptor-driven gather、staging、scatter 保证重复/重叠映射的 snapshot 语义；disjoint mapping 则保留更快的 `cudaMemcpyAsync`。Swap 使用 pinned host、独立 stream/event；Host/File store 有 budget、checksum、原子 rename 和失败回滚。

算法侧包含 decode priority + aging、公平 token budget、基于缓存价值的 victim、backend/context/concurrency bucket 的在线 prefill cost model，以及接受率 EWMA/证据门槛/迟滞/KV 压力共同控制的 speculative draft length。所有策略都有 upstream/fixed 开关。

验证上，我不仅测微基准。测试分为纯算法、随机状态性质测试、CPU/CUDA 逐元素对照、Compute Sanitizer、真实 Qwen server smoke、OpenAI SSE/取消/deadline/背压、故障注入、模型矩阵和 fresh-process A/B。结果不是全正：CPU mixed workload 的 P95/吞吐明显改善，但 CUDA 下 latency P95 和吞吐退化。这说明 control-plane 决策成本和 GPU batching 形态必须分 backend 建模，也避免把一个漂亮数字包装成普适结论。

## 项目边界：哪些是上游，哪些是个人工作

上游提供：GGUF 解析、模型 graph、GGML 通用 CPU/CUDA backend、绝大多数模型算子、HTTP 基础设施、sampling 和已有 KV memory 抽象。

个人实现：

- Engine ownership、iteration transaction 和 sequence phase 状态机；
- Scheduler、KV capacity/admission/victim、Block Manager、PrefixIndex 和 KV Runtime；
- Host/File transactional Swap Store；
- llama Attention KV 的 CacheFlow capability/adapter；
- CUDA block copy、snapshot-safe Gather/Scatter、partial-tail COW、pinned swap/stream/event；
- adaptive prefill/speculation 控制器；
- 原生 metrics、fault injection、真实服务测试和可复现实验。

不要说“我重写了 llama.cpp”，也不要把 vendor 总行数当个人代码量。应说“我固定上游 commit，在其真实 hot path 上形成 56-file、约 +7.5K/-0.1K 的可重放差异；最终数字以当前 patch 为准”。

## 架构与状态所有权

```text
OpenAI HTTP/SSE
      |
server_context（组合根/协议适配/llama_decode callback）
      |
server_inference_engine（唯一控制面所有者）
      +-- scheduler / capacity planner / speculation controller
      +-- kv runtime / swap store / runtime adapter
      +-- iteration transaction / sequence phase state
      |
llama memory capability
      +-- unsupported memory: upstream fallback
      +-- attention KV: paged CUDA adapter
             +-- direct async copy（disjoint）
             +-- gather -> staging -> scatter（overlap/snapshot）
             +-- partial-tail COW / D2H-H2D swap
```

关键不变量：

1. Scheduler 不直接修改 Runtime，只返回计划。
2. execute 失败不能 publish plan，也不能释放仍被引用的块。
3. Block Manager 是逻辑 block/refcount/reservation 的唯一所有者。
4. 共享 partial tail 在 append 前必须 COW。
5. CUDA event 完成前，source/destination/staging/pinned buffer lease 均不能释放。
6. 不支持 block capability 的 memory 回退上游路径，不能假装支持。

## 为什么需要 iteration transaction

如果“先改 slot 状态，再调用 decode”，OOM 或 CUDA error 会留下半提交状态：token 位置已前进但 K/V 未写完，reservation 已释放但 block 仍被引用，或者 HTTP 已发出不可撤销的 chunk。事务把决策与提交分开：

```text
prepare -> immutable plan -> execute(runtime callback) -> commit
                                      |
                                      +-- failure -> abort/rollback
```

这里不是数据库 ACID 的完整实现。它保证的是单 iteration 内的原子可见性和资源守恒；网络已经发送的数据仍不能回滚，因此 HTTP publish 必须位于成功提交之后。

## Scheduler 算法

目标不是单独最大化当前 prefix hit，而是在 token budget、KV capacity、deadline 和公平性下选择本轮动作。高层可写成：

```text
utility = reused_tokens
        - eviction_penalty * victim_recompute_tokens
        - deadline_risk
        + aging_bonus
```

decode 通常优先，因为每延迟一轮都会直接增加 inter-token latency；prefill 用 aging 防止饥饿，并受本轮 chunk/token budget 限制。选择 slot 时对候选计算 LCP/复用与 victim 代价，当前 slot 数较小时是 O(S)；PrefixIndex 查找按 block token 哈希推进，近似 O(P/B)，其中 P 是 prompt token 数、B 是 block size。

追问“为什么不用最优全局调度”：精确求解需要未来到达、生成长度和回访概率，在线不可得且求解开销不适合每 token iteration。当前策略是可解释启发式，并通过 upstream 开关和 trial 验证。

## Block Table、PrefixIndex、refcount 与 COW

每个 sequence 的逻辑 token 范围映射到物理 blocks。PrefixIndex 只发布已经写入且可共享的 exact token prefix，refcount 决定块何时回收。Reservation 在 admission 时预留最坏/计划容量，避免执行中才发现不足。

过去只共享完整块会浪费 1 到 B-1 个 token 的 prefix。本实现允许 exact partial tail 共享，但它不能原地 append：若 refcount > 1，先申请新块，复制有效 tail token，更新当前 sequence 的映射，再减少旧块 refcount。申请或复制失败时原映射保持不变。

COW 复杂度从复制整个 sequence 的 O(L) 降为 O(B)。当前 benchmark 中 64 blocks 整序列方案复制 12,582,912 B/128 launches，而 tail COW 复制 196,608 B/1 launch。

## CUDA 侧为什么算 AI Infra

AI Infra 不等于“写了一个 CUDA demo”。这里的 kernel 直接操作 llama Attention 的生产 K/V tensors，且必须处理：

- K/V layout、element size、head/layer stride 和非整除 tail；
- 非连续、重复和重叠 block mapping；
- snapshot 语义，防止 in-place source 被提前覆盖；
- stream/event 顺序、异步 buffer 生命周期和错误传播；
- pinned host budget、D2H/H2D restore 和 allocation failure；
- 与调度、refcount、COW、preemption 的一致性。

Gather 将 source descriptors 对应的 token ranges 写入 staging；Scatter 再写 destination，因此重叠映射读取的是同一快照。对完全 disjoint 的映射，两次 kernel 加 staging 反而比多个异步 memcpy 慢，所以生产 adapter 分流。这是机制驱动的优化，不是“CUDA 一定快”。

## Host/File/CUDA Swap 的事务语义

Swap payload 不只含 K/V bytes，还含 opaque serialized llama sequence state，否则 restore 后逻辑位置与物理 K/V 可能不一致。

- save 成功：快照具备 checksum，file store 写临时文件后原子 rename，再允许释放 resident KV。
- save 失败：resident KV 保持，不得丢请求。
- restore 成功：先验证 budget/checksum/state，再安装映射。
- restore 失败：丢弃无效快照，回到完整 recompute；不能安装半份 K/V。
- CUDA copy：event 完成后才能复用 pinned/device buffer。

故障注入覆盖 next-save、next-restore、KV OOM、compute failure 和 CUDA allocation failure。

## Adaptive Prefill

过小 chunk 增加 iteration/kernel launch 和调度开销，过大 chunk 会阻塞 decode、恶化 TTFT/TPOT。控制器按 backend、context、concurrency bucket 记录候选 chunk 的 wall time，估计 cost，在 greedy 和固定候选之间在线选择。

面试时必须主动讲负结果：历史版本的 CPU 在线 chunk 曾劣于 greedy 和所有 fixed 候选。最终控制器按 backend bucket 选择动作，CPU 在证据不足时回退候选集合中的 `chunk=0` greedy，CUDA 才继续在线调 chunk；A/B 脚本有 2% 回归硬门槛。正确结论是“控制器带安全动作”，不是“自适应总是最优”。下一步可用 contextual bandit 或加 SLO penalty，但必须控制探索风险。

## Adaptive Speculation

草稿长度不是越长越好。接受率低时，draft model 成本和额外 KV 压力超过 target token 节省。控制器维护 acceptance EWMA，同时使用最小证据量防止冷启动抖动，用迟滞防止长度来回切换，并在 KV pressure 高时缩短或关闭 draft。

估算净收益：

```text
net_saved_ms ~= accepted_tokens * target_ms_per_token - measured_draft_wall_ms
```

这是在线估计，不是 profiler 的 causal attribution。当前 A/B 使用 N-gram speculation；不能把它描述成完整 draft Transformer 联合部署。

## 性能结果怎么讲

Mixed workload 是 8 个并发 long-prefill/short-decode 请求、4 workers、共享 prefix，每个组合 3 个 fresh process。

- CPU 最终轮：TTFT P95 7883.35 -> 3073.61 ms，latency P95 12779.54 -> 7043.05 ms，TPS 13.94 -> 17.65；median TTFT 1603.06 -> 1625.82 ms 略退化。
- CUDA：历史两轮 latency/throughput 结论反号；最终轮 TTFT P95 155.31 -> 93.71 ms、TPOT P95 20.30 -> 12.76 ms、TPS 235.26 -> 264.90，但 latency P95 543.48 -> 618.81 ms。现有样本不足以声称端到端收益稳定。

因此项目的价值不靠“所有指标全赢”，而是：hot path 真接入、指标可解释、重复实验能暴露方差，并形成下一版 gating 设计。若面试官问生产是否默认开启，答案应是：当前不应对所有 CUDA workload 默认开启，应按 backend/workload guard，并保留 upstream fallback。

## Profiler 结果怎么讲

生产 `--engine-trace` 生成 Chrome/Perfetto complete events。最终 CPU mixed workload 中 execute 占 99.9145%，plan 0.0148%，说明 Engine 拆分没有把控制面变成主要 wall-time 热点。

这不是 sampled-stack flame graph。WPR 需要当前会话没有的 `SeSystemProfilePrivilege`，所以材料里明确写限制。图能回答 phase attribution，不能回答 execute 内部具体 CPU 函数栈；后者需管理员权限 WPR、Nsight Systems/Compute 或 Linux perf 再验证。

## 正确性与测试金字塔

1. 纯 C++：scheduler、transaction、engine state、capacity、block、runtime、store、spec controller。
2. 性质测试：随机 allocate/share/append/COW/release/preempt，持续检查总量、reservation、refcount、prefix parent 和 residency。
3. CPU reference vs CUDA：block size 8/16/32/64、随机/重复/重叠 mapping、tail、in-place protection、多 stream。
4. Compute Sanitizer：memcheck 和 racecheck；它是最终全量入口硬门槛。
5. 真实模型：Prefix/COW/Swap、OpenAI、取消/deadline/backpressure、fault injection、Q4/Q8/F16 CPU/CUDA matrix。
6. 性能/质量：fresh process trial、原始数据、上游输出 hash、质量题护栏。

“测试很多”不是最终论据；最关键的是每个失败模式对应一个不变量，并且 production smoke 能证明测试 seam 与真实路径一致。

## 常见追问与回答

### 为什么 block size 默认 16？

它权衡 prefix 粒度、metadata/refcount 开销、copy launch 和 tail waste。16 是当前硬件/模型实验点，不是理论最优；需要按 KV bytes/token、并发和 kernel 吞吐调优。

### 为什么 partial tail 也共享？

真实 prompt 很少正好 block 对齐。只共享完整块会系统性损失尾部命中；COW 把写入安全代价限制在一个 block。

### 如何避免 ABA 或异步 UAF？

物理 block 在 refcount/lease 为零前不回收到 free list；CUDA submission 持有 buffer/block lease，event 完成后统一释放。当前单 Engine thread 简化了控制面并发；若扩展多线程，需要 generation id 或 handle version。

### File swap 崩溃一致性如何做？

写 temp、flush/close、校验后 atomic rename；旧文件在 commit 前仍有效。当前是单进程快照，不是分布式 WAL，也未承诺断电下目录项持久性。

### 为什么不用 vLLM？

项目目标是理解并修改 C++/CUDA inference runtime，而不是最快搭服务。llama.cpp 让同一 fork 覆盖 CPU、CUDA、量化与 HTTP，适合验证跨 backend 策略；生产选型仍应按吞吐、生态和部署约束比较 vLLM/TGI/TensorRT-LLM。

### 最大技术债是什么？

CUDA mixed workload 缺少 backend-aware policy guard；Hybrid/Recurrent 只有 fallback；WPR sampled-stack 未完成；0.5B 单 GPU 不能代表大模型/多 GPU。回答技术债比假装完成更可信。

### 如果再做两周？

当前已完成 prefill 级 Conservative Benefit Gating、53-wave 长驻收敛、CUDA Event/Engine/TTFT 服务因果链，以及 4-regime Nsight Systems KV kernel timeline。下一步是在管理员启用 GPU performance counter、并匹配 driver/NCU 版本后补 DRAM/L2/occupancy/roofline；在此之前只讲 NSYS launch/copy/sync 与 paired latency，不假装已有硬件计数器。之后再把动作空间扩展到 slot placement/KV admission/speculation，而不是继续堆外围功能。

### Benefit Gating 怎么讲？

先讲负结果：固定adaptive chunk在不同backend/workload上结论反号。然后画出同一生产Seam的两条shadow plan，说明learned policy只在悲观CacheFlow cost仍低于乐观upstream cost加安全margin时启用。重点追问是置信半径、SLO reward、drift与探索预算。最后展示16-trial联合 `backend×mode` Williams验收：每个treatment×位置和有向前驱各2次，trial边界有washout，并报告paired oracle regret；不声称这是所有模型和硬件上的全局最优。

最终在真实 socket send seam 固定并发发送顺序的短程fresh-process实验中，positive-lower-bound次数为0，验证backend-local冷启动风险边界。生产 `choose()` 用C++ `steady_clock` 原位计时并导出Prometheus histogram：CPU/CUDA learned共139/142次样本，最坏trial P99为2/5 us，低于50 us预算；固定CPU upstream绕过chooser时明确记录0样本，而learned缺样本会直接失败。原始trial还保存bucket/count/sum，重算P99并核对决策counter。CUDA零probe意味着该短trace不能识别CacheFlow因果效果，不能拿跨进程延迟差冒充算法收益。随后53-wave单进程CUDA trace在 `beta=1.0` 下产生26次探索和125次positive-lower-bound，覆盖36 waves、最长及终端连续均为35 waves；最后一次上下文gauge为8.926/4.463 ms，但面试中要主动说明last-value gauge无法代表同一wave中的所有特征向量，硬证据是逐wave动作counter。分布切换后0次错误启用、3次安全回退；验证器还能从逐请求TTFT和counter重算phase/acceptance并拒绝篡改。因此可以声称当前0.5B/CUDA/workload上观察到持续在线利用，但不能外推其他模型与硬件。

### CUDA 因果链怎么讲？

先说明干预不是相关性截图：每个 trial 配对运行 upstream/always，并做 Latin 顺序轮换。然后沿四层证据讲：动作计数变化 → prefill token/chunk 变化 → KV copy byte/CUDA Event 与 GPU activity 变化 → Engine execute/TTFT 变化。本轮强制启用增加 23 个 chunk，同时减少 145 个 prefill token、2.52 MB KV copy 和 0.260 ms CUDA Event，但 Engine execute 增加 31.2 ms、TTFT P95 增加 44.8 ms；说明 kernel/copy 局部改善不等于服务尾延迟改善，分块与排队结构同样重要。最后展示可提交 evidence JSON，并主动限定这不是 Nsight 全 kernel census。

如果追问算子机制，切到 H2 artifact：20-pair 无 profiler 主表与 5-pair NSYS 机制 trace 分开。aligned 1 block 的 GPU 改善 +57.10% [38.51%, 57.14%]，16/32 blocks 只有 +5.24%/+3.93%，misaligned 1 block 则回退 137.94%。四个 regime 的 scalar/vector NSYS launch 均为 10/10，所以不是减少 launch 数；错位时 vector grid 中每个线程走 8-lane scalar fallback，说明 alignment 必须进入 action gate。NCU 报 `ERR_NVGPUCTRPERM`，所以不能回答 occupancy/DRAM-bound，只能说明需要哪组 metric 才能继续证伪。

## 三分钟现场演示顺序

1. 打开 `server-inference-engine.*` 和 iteration transaction，指出状态所有权。
2. 运行 `test-kv-block-manager`，展示 partial-tail share -> append COW 和容量守恒。
3. 打开 `llama-kv-cache-paged.cu`，说明真实 tensor gather/scatter/COW 与 direct-copy 分流。
4. 运行 CUDA tensor adapter smoke，展示 shared blocks、COW 和相同输出。
5. 展示短 trace、53-wave 收敛表和 CUDA 因果链：先区分探索/利用，再解释为什么 always 在该 trace 上增加数据移动并恶化 TTFT。

## 面试中的禁区

- 不说“重写 llama.cpp”或把上游行数算个人贡献；
- 不把 N-gram speculation 说成 draft Transformer；
- 不把受限 Qwen2.5-0.5B Paged Decode 说成所有 memory 类型、模型、context 或 prefill 都支持；
- 不把 phase trace 说成 sampled-stack profiler；
- 不说 CacheFlow 在 CUDA workload 普遍更快；
- 不用单次 trial 或微基准代替端到端结论；
- 最终 `verify.ps1 -Full` 未通过前，不说项目已严格验收。
