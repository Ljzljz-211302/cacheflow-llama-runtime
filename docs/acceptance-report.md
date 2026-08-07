# CacheFlow Runtime 严格验收报告

验收基线：llama.cpp `acd79d603cb2e1c84c0886137b80f1ad649b6857`

个人 fork：`vendor/llama.cpp` 当前分支 `codex/cacheflow-runtime`

硬件：RTX 4050 Laptop 6 GiB（sm_89）、i5-13500H、Windows 11 WDDM

模型：Qwen2.5-0.5B-Instruct Q4_K_M / Q8_0 / F16

## 当前结论

2026-08-01 基线版本曾通过唯一入口 `scripts/verify.ps1 -Full`，完整运行 1395.4 秒并以 0 退出。本次向量化 KV Remap 增量已通过架构/patch 门禁、CPU/CUDA 构建、全部单元测试、Compute Sanitizer、真实 Qwen tensor/COW、模型矩阵、用户应用旅程和 20-trial 配对微基准；但两次重新执行 Full 均被既有统计型性能门禁拦截：第一次为长驻实验末态收益 7.86 ms 小于不确定性 9.00 ms，第二次为 CPU learned oracle regret 26.1% 和 CUDA paired regression 3.7% 越过 20%/3% 门槛。相同长驻场景随后单独复现通过（17.82 ms > 8.49 ms）。因此本增量的功能/内存安全/算子性能门禁通过，但不能声称当前全量统计套件稳定绿灯。

“存在代码”“单元测试通过”和“生产路径通过”是三个不同层级。本报告只把有生产 smoke 或真实模型证据的条目标为生产接入。

## 架构验收矩阵

| 验收项 | 实现证据 | 自动化证据 | 本轮状态 |
|---|---|---|---|
| Engine 所有权边界 | `server-inference-engine.*` 统一拥有 Scheduler、Capacity、Spec、KV Runtime、Swap、Adapter、iteration transaction | `test-inference-engine`、真实 server smokes | 通过 |
| Engine 固定执行顺序 | `server_inference_engine::step()` 统一调用 prepare、plan/execute、commit；context 只提供适配 callback | Engine committed/aborted step 测试、production trace | 通过 |
| prepare/plan/execute/commit | `server-inference-iteration.*`；`update_slots()` 只进入 Engine step | `test-inference-iteration`、Engine trace | 通过 |
| 状态迁移唯一入口 | `server_sequence_state` 由 Engine 持有；非法 phase transition 拒绝 | `test-inference-engine` | 通过 |
| 深模块与依赖方向 | Scheduler、KV Block/Runtime/Store、Speculation 均为小接口；低层不依赖 HTTP | 架构门禁、CMake 分目标编译、原生测试 | 通过 |
| 生产/测试同一 Seam | `server_runtime_adapter` 被 production Engine 与 deterministic test 共用 | `test-runtime-adapter`、`test-inference-engine` | 通过 |

## 功能与正确性矩阵

| 验收项 | 生产证据 | 失败语义/正确性证据 | 本轮状态 |
|---|---|---|---|
| Token-level Continuous Batching | OpenAI 并发服务与 mixed prefill/decode workload | Scheduler 公平性、chunk budget、starvation 测试 | 通过 |
| Prefix Block 分享 | 真实 CPU/CUDA 服务第二请求只 prefill 未命中尾部 | Block Table/PrefixIndex 随机状态容量守恒 | 通过 |
| 部分 Tail COW | 真实 CUDA 日志和 `cuda_kv_copy_on_write_total` | append 自动 COW；CPU/CUDA 逐元素对照 | 通过 |
| KV 准入/抢占/恢复 | 真实 preempt smoke；Host/File/CUDA restore | save 失败保留 resident；restore 失败丢弃并重算 | 通过 |
| Host/File 事务 Swap | server CLI 选择 memory/path store；真实序列 state 序列化 | budget、checksum、temp+rename、next-save/restore failpoint | 通过 |
| Adaptive Prefill | CPU/CUDA 真实 A/B | backend-aware safety action；2% 错误固定点回归硬门槛 | 通过 |
| Conservative Benefit Gating | 生产 shadow upstream/CacheFlow plan；CPU/CUDA 独立模型 | 置信下界、有限探索、高压/漂移回退、deterministic replay、每端 10 trials | 通过 |
| Adaptive Speculation | CPU/CUDA N-gram 真实 A/B | EWMA、证据门槛、迟滞、KV 压力反馈 | 通过 |
| CUDA Gather/Scatter | 真实 llama K/V tensor 的 descriptor-driven gather -> staging -> scatter | 随机/重复/重叠/尾部映射逐元素一致 | 通过 |
| CUDA 异步 Swap | 真实 Qwen K/V D2H/H2D | pinned pool、stream/event lease、allocation failpoint | 通过 |
| OpenAI 协议 | 非流式 JSON、SSE chunk、`[DONE]` | 断流取消、deadline、背压后服务恢复 | 通过 |
| 上游兼容 | 同 MSVC、相同模型/seed 的 upstream policy | 5/5 输出 SHA-256 一致 | 通过 |
| 模型/负载矩阵 | Q4/Q8/F16 × CPU/CUDA × 并发/上下文 | 14 个真实 server case | 通过 |
| CUDA 内存/竞态 | Compute Sanitizer memcheck/racecheck | memcheck 0 errors；racecheck 0 hazards/0 errors/0 warnings | 通过 |

Hybrid/Recurrent memory 若不提供 block capability，`llama_memory_cacheflow_set_block_size` 返回 false，Engine 保持上游 memory 行为；本项目不把这种回退宣称为物理 paged-KV 支持。

## 性能证据

### Mixed prefill/decode（每轮 3 次 fresh process，每组 24 请求）

| Backend / Policy | TTFT median / P95 ms | TPOT P95 ms | Latency P95 ms | Aggregate output TPS | 结论 |
|---|---:|---:|---:|---:|---|
| CPU upstream | 1603.06 / 7883.35 | 437.20 | 12779.54 | 13.94 | 基线 |
| CPU cacheflow | 1625.82 / 3073.61 | 277.82 | 7043.05 | 17.65 | 尾延迟/吞吐改善，中位 TTFT 略退化 |
| CUDA upstream | 68.66 / 155.31 | 20.30 | 543.48 | 235.26 | 基线 |
| CUDA cacheflow | 74.45 / 93.71 | 12.76 | 618.81 | 264.90 | TTFT/TPOT P95 与吞吐改善，latency P95 退化 |

表格是最终轮证据。此前两轮 CUDA cacheflow 的 latency/吞吐出现反号；最终轮又同时出现 latency P95 退化和吞吐改善。当前轮原始数据为 `results/mixed_workload_trials.csv`，跨轮数据为 `results/mixed_workload_repeated_runs.csv`。因此现有 3-trial 证据不足以声称 CUDA 端到端收益稳定，不能用 CPU 收益外推 GPU。

### Tail COW（真实 RTX 4050，3 个 fresh process × 300 samples）

| 方法 | Median / P95 E2E ms | Bytes/op | Launch/op | Extra device bytes |
|---|---:|---:|---:|---:|
| Tail Block COW | 0.0089 / 0.0258 | 196,608 | 1 | 196,608 |
| Whole Sequence Copy | 0.7786 / 0.9980 | 12,582,912 | 128 | 12,582,912 |

P95 改善 97.41%。这证明被隔离的 COW hot path，不等价于整个 serving workload 都改善。

### CUDA Transport 负结果

完全不重叠映射中，per-block `cudaMemcpyAsync` 比 staging Gather/Scatter 更快。因此 production 对 disjoint mapping 使用直接异步复制，仅在重叠/重复映射需要 snapshot 语义时走 staging kernel。原始 trial 与 kernel/端到端/显存数据保存在 `results/raw/` 和 `results/cuda-kv-transport-summary.json`。

### Adaptive 策略边界

- Adaptive Speculation 在当前 CPU/CUDA trace 中略优于 fixed 候选，但只限定当前 N-gram workload。
- Adaptive Prefill 避免 fixed-64 错误点，却没有稳定击败 CUDA greedy/fixed-256；因此只声称在线控制可规避明显坏点。

### Conservative Benefit Gating

新控制器不再按 backend 硬编码开关，而是在真实 prefill 决策点生成 upstream greedy 与 CacheFlow shadow plan，按 backend-local contextual confidence model 保守选择。最终 10-trial Latin 结果：CPU learned objective median 4177.39 ms、paired upstream regression -18.11%、paired-oracle regret 15.35%；CUDA learned 190.70 ms、paired regression -2.82%、paired-oracle regret 1.02%。两端均通过原 3% fresh-process paired regression、20% paired upstream/always/rule oracle regret 和 harmful-trace wrong-enable 门槛。此前“两个独立中位数之比”的错误统计口径已移除，阈值未放宽。

短生命周期风险预算按 backend 隔离：CPU learned 产生 19 次 `safe_exploration`；已知 always 有害的 CUDA fresh process 提高最小样本门槛，在该 trace 内 0 次 probe 并 fail closed；`positive_lower_bound_decisions` 均为 0。CUDA positive-lower-bound 的生产实证由下一段长驻实验提供。

随后新增的单进程长驻门禁补齐了这一限制：CUDA server 连续运行 53 waves，`confidence_beta=1.0`、每动作最少 12 个样本。冷启动 CacheFlow/positive 均为 0；稳定阶段 18 次有限探索后产生 142 次 positive-lower-bound，覆盖 33 waves、最长连续 13 waves，终态收益 21.29 ms 大于 8.82 ms 不确定性；切换后 CacheFlow/positive 均为 0、安全回退为 3。门禁要求持续启用且终态仍保持置信，不取最大置信快照。

### CUDA profiling 因果链

3 组 paired Latin upstream/always 干预通过。强制 CacheFlow 中位造成决策 +13、prefill chunk +23、prefill token -354、自研 KV kernel launch +2、KV copy +20,066,300 B、CUDA Event +0.808 ms、GPU busy 与最大 idle gap中位差不变，Engine execute 汇总 -11,446 us，但 TTFT P95 +85.61 ms。完整 GPU samples、Engine events 和相关 Prometheus 快照保存在 `results/cuda_causal_profile_evidence.json`；门禁拒绝仅有采样噪声而没有 material 请求级结果的 trace。

H2 进一步完成 4 个预注册 KV regime、160 条无 profiler paired trials 与 4 份真实 Nsight Systems trace。aligned 1 block 的 CUDA-event 改善为 +57.10% [95% CI +38.51%, +57.14%]，aligned 16/32 blocks 只有 +5.24%/+3.93%，按 10% 门槛判为 neutral；misaligned 1 block 保留为反例，CUDA-event/end-to-end 分别回退 137.94%/113.02%。NSYS 中 scalar/vector 每个 regime 均为 10/10 launches，否定“减少 launch 数”解释。正式 artifact 在 `results/research/h2-kv-profile-v1.0.0/`。

Nsight Compute 已实际执行，但当前 driver/tool compatibility 检查和 `ERR_NVGPUCTRPERM` 阻止硬件计数器采集；失败命令与日志均保留。因此验收只通过 NSYS + no-profiler effect 的限定主张，明确不声称 memory-bound、roofline、achieved occupancy 或 hardware DRAM bytes。

### 受限 Paged Decode Attention（Issue #5）

K1 CUDA 原型直接按 block table 从非连续物理页读取 K/V 并输出 attention，不在 timed path materialize contiguous KV。独立 CPU FP32 oracle 与相同数学的 contiguous CUDA 路径共同覆盖 `14/2/D64`、`28/4/D128`、ratio-7 GQA、context `1/15/16/17/31/32`、ragged batch、fragmented pages、unused-page poison、output guards 和 invalid shape/page table fail closed。

正式实验保存 9 个 regime × 20 pairs × 2 methods = 360 条无 profiler observations；每个 regime 计时前的独立 CPU FP32 oracle 最大绝对误差为 `3.6e-8`。Paged K1 的 GPU-event 中位数在全部 regime 均比 contiguous 对照慢 6.46%–13.07%；0.5B shape 的 context 1024/batch 1 回退 13.07%（95% CI 12.91%–13.32%），7B shape 回退 11.54%（11.32%–11.68%）。4 份 NSYS trace 各精确包含 5 次对应方法 launch。batch-1 与 batch-4 的长 context 回退差只有 1.38/1.69 个百分点，未触发 K3 split-K；因长 context 回退超过 3%，预注册规则选择 K2 GQA reuse 作为下一待验证假设。

本验收只证明受限算法正确、负结果可复现并能给出下一内核决策，不证明生产可用或端到端加速。NCU 仍报告 driver incompatibility 与 `ERR_NVGPUCTRPERM`，因此 memory-bound、occupancy 和 hardware DRAM-byte 解释继续禁止。完整 hash-bound artifact 位于 `results/research/h3-paged-decode-v1.0.0/`。

服务级链路也已补齐：3 个 no-profiler upstream/always pairs 给出主要请求效应，相同 seed/config 的 NSYS 重放用 6 个 server PID 和 72 个 request ID 连接 scheduler、KV action、CUDA timeline 与 TTFT；profiler replay 的 42 次自研 KV launch 均与逐进程运行时 counter 完全相等。本轮 no-profiler 中 decision +20、prefill chunk +30、prefill token -86、KV launch +0、copy bytes +10,518,500、CUDA Event -1.040 ms；paired median TTFT P95 +126.932 ms、Engine execute +146,493 us，两者同向恶化。因此报告明确写“未复现旧的 execute/TTFT 反号”，不挑选更好看的历史 run 替代。

exact output hash 被保留为审计指标而非并发性能硬门槛：batch composition 会改变近似相等 logits 与 EOS 位置；HTTP/SSE、上游兼容和语义质量分别由专门 gate 验证。

## Profiler / Flame 证据

`--engine-trace PATH` 在真实 production Engine 中记录 prepare/plan/execute/commit Chrome/Perfetto complete events。最终一次 CPU mixed workload 共 242 spans、9.553 s：execute 99.9145%，commit 0.0674%，plan 0.0148%，prepare 0.0033%。原始 trace 为 `results/raw/engine-trace-cpu-cacheflow-trial-1.json`，渲染结果为 `results/engine-flame.svg`。

这是一张 phase-duration flame chart，不是 sampled-stack CPU flame graph。WPR 因当前会话缺少 `SeSystemProfilePrivilege` 无法采集 sampled stacks；该限制保留为事实，不把失败伪装成通过。

## 可观测性

- Request：TTFT、TPOT、latency、queue latency Histogram；
- Scheduler：iterations、decode/prefill token、chunk、batch token/sequence、starvation；
- KV：used/free/shared、prefix hit、COW、eviction、swap/restore、admission failure；
- Speculation：draft/accepted、acceptance、draft length、disabled reason、估算 net saved；
- CUDA：kernel/blocks/bytes、copy/swap Event 时间、bandwidth、waited event、pinned current/peak、errors。

## 复现与硬门槛

```powershell
cd D:\llama
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\verify.ps1 -Full
```

完整入口包含 CPU/upstream/CUDA 构建、全部 C++/Python 测试、真实 server smokes、故障注入、兼容/模型矩阵、Compute Sanitizer、3-trial CPU/CUDA 性能、production trace、质量护栏和报告再生成。任何一步非零退出即整体验收失败。

## 个人贡献边界

固定上游提供 GGUF/模型 graph、GGML 通用算子、既有 backend、HTTP 基础设施和 sampling。个人差异是 Engine 拆分、Scheduler、KV 资源/块/事务 Swap、真实 CUDA KV Adapter 与 kernels、自适应/收益门控、在线策略持久化、metrics、fault injection 和复现实验。当前相对固定上游为 61 files、+8708/-99。代码量只统计该 patch；不得把 `vendor/llama.cpp` 原有代码算作个人实现，也不得声称“重写了 llama.cpp”。

### 生产重启与状态损坏

真实 CPU `llama-server` 三进程 smoke 已通过：第一进程产生 10 次 benefit observation 并完成 10 次后台原子保存；强制终止后第二进程报告 `restored=1` 且恢复全部 10 次观测；第三进程读取人为截断的 checkpoint 时报告 `failed=1`、保持 0 次模型观测并继续成功处理请求。证据位于 `results/benefit-checkpoint-smoke.json`。该结果证明单机单 writer 边界内的在线模型可恢复，不外推为跨副本模型同步。

早期 Python 控制面已归档到 `prototypes/cacheflow`，只保留设计演进与原型测试证据；生产 `llama-server` 不导入或调用它。

## 已知限制

- 单机单 GPU、0.5B 模型结果不能外推多 GPU 或大模型；
- Hybrid/Recurrent 仅 capability fallback，未实现物理 block backend；
- WPR sampled-stack profile 因权限未完成；现有图只回答 Engine phase 时间归属；
- CUDA cacheflow 当前 mixed workload 的吞吐和 latency P95 退化，需要后续加入 backend-specific policy gating；
- Benefit Gating 当前仅控制 prefill plan，尚未扩展到 slot placement、KV admission、swap 与 speculation 的联合动作空间；
- 当前 10-trial fresh-process 短 trace 没有 positive-lower-bound；在线收敛只由新增 53-wave 单进程 CUDA trace 支持，不能外推其他 GPU、模型或负载；
- 同工具链输出兼容不等于跨编译器逐字节兼容；
- 换 GPU、驱动、CUDA 或模型后必须重跑 Sanitizer 和矩阵。

## 用户应用验收

推免面试学习助手已通过真实 CUDA 用户旅程，而非单请求 smoke：两个 fresh application subprocess 覆盖应用重启；三个持久化会话覆盖机器学习续聊、408 独立会话和浏览器取消；两个用户请求并发执行，超出应用生成槽时返回 429。断流后数据库只保留 user 消息，不存在半条 assistant 答案。

本轮原生证据为：456 个 cached prompt token、19 个 prefill chunk、6 次 CUDA KV kernel、5,603,330 个向量化 Remap byte、2 次 CUDA benefit decision、2 次在线策略 checkpoint，平均 busy slot/decode 为 1.19837；llama 日志同时出现 CUDA0、CacheFlow policy、shared prefix 和真实模型 token 后的原生 cancel task。Chromium 浏览器实际完成 B+ 树提问、SSE 回答与资料卡片显示，首条引用命中数据库学习文档 `7.1 B+ 树`。这些数据证明用户应用流量进入 CacheFlow/CUDA 链路，不声称已有外部用户或线上采用率。

## 向量化 KV Remap 验收

- CUDA public seam：对齐重叠交换、非对齐 19 元素尾部与非法 grid 均通过；
- Compute Sanitizer：新算子 memcheck 0 error，racecheck 0 hazard；
- 真实 Qwen：共享 21 个 Prefix KV Block，partial-tail COW 后与 cold deterministic decode 输出一致；
- 真实用户应用：原生指标累计 5,603,330 个 vectorized remap byte；
- 20 组配对交替顺序微基准：1/4/16/32 Block 的 GPU 中位改善为 53.33%/48.89%/3.13%/1.87%，最差规模仍为正向，不宣称等比例端到端收益。
