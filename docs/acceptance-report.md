# CacheFlow Runtime 严格验收报告

验收基线：llama.cpp `acd79d603cb2e1c84c0886137b80f1ad649b6857`

个人 fork：`vendor/llama.cpp` 当前分支 `codex/cacheflow-runtime`

硬件：RTX 4050 Laptop 6 GiB（sm_89）、i5-13500H、Windows 11 WDDM

模型：Qwen2.5-0.5B-Instruct Q4_K_M / Q8_0 / F16

## 当前结论

2026-08-08 在外层提交 `bf6f8b1`、vendor提交 `130bd22` 上，唯一严格入口 `scripts/verify.ps1 -Full` 曾原生以0退出。随后Standards复审指出当时的Williams设计没有覆盖真实执行流的row/backend边界，长驻结果也缺少从逐wave原始证据独立重算的验证器。当前提交已改为16-trial联合 `backend×mode` Williams设计、trial washout和可篡改检测的长驻验证器，定向实验及131项快速测试通过；在新的完整Full再次以0退出前，本段不把旧Full冒充当前HEAD验收。阈值未放宽，Full产生的非正式易波动输出不替换已提交的正式研究 evidence。

“存在代码”“单元测试通过”和“生产路径通过”是三个不同层级。本报告只把有生产 smoke 或真实模型证据的条目标为生产接入。

## 已执行 Issue 覆盖

| Issue | 状态 | 本仓库中的可核查成果 |
|---|---|---|
| #2 | 已关闭 | `research-charter.md` 锁定研究问题、假设、机制、证伪条件和负结果规则。 |
| #3 | 已关闭 | `research_protocol.json` 与人类可读协议冻结变量、配对统计、bootstrap、主指标和 pass/fail gate。 |
| #4 | 已关闭 | H2 KV/profile 与真实服务 NSYS 因果链，保留 native report、SQLite、无 profiler 主结果及 NCU 不可用边界。 |
| #5 | 已关闭 | 受限 Qwen Paged Decode K1 CUDA kernel、FP32 online softmax、CPU oracle、跨页/非连续页和 fail-closed 测试。 |
| #6 | 已关闭 | Direct/Remap/Paged/Swap/Recompute 统一动作接口、H0/A1/T1/L1、200 组原始观测、500 万次 replay 及可篡改检测 artifact。 |
| #7 | 实现完成，Issue 尚未关闭 | 生产 Remap/Paged dispatch、原子故障回退、v1.1 负结果、联合 Williams 实验、在线 Ridge 与 chooser P99 证据。 |
| #10 | 已关闭 | 一手论文、官方实现和本地可复现基线审计；外部系统只作 related work，不挪用其性能数字。 |

Issue #1 是总路线图；#8 的消融/鲁棒性/外部有效性与 #9 的最终可复现发布、论文式报告仍未完成，不能列为既成成果。

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
| Conservative Benefit Gating | 生产 shadow upstream/CacheFlow plan；CPU/CUDA 独立模型 | 置信下界、有限探索、高压/漂移回退、deterministic replay、16-trial联合 Williams blocks | 通过 |
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

新控制器不再按backend硬编码开关，而是在真实prefill决策点生成upstream greedy与CacheFlow shadow plan，按backend-local contextual confidence model保守选择。正式协议把 `backend×mode` 组成8个联合treatment，执行16 trial的两个完整Williams blocks；64个treatment×process-position单元及56条有向直接前驱均各覆盖2次，trial间有1秒显式washout。原始行保存完整 `treatment_order/process_position/previous_treatment/trial_washout_ms`，并在 `HTTPConnection.request()` 的真实connect/body-send seam固定每波 `0..5`；128/128 trial rows的两波observed order均为 `0..5`。CPU learned objective median 4205.21 ms、paired upstream regression -23.80%、paired-oracle regret 5.04%；CUDA learned 280.01 ms、paired regression +2.16%、paired-oracle regret 10.52%。两端均通过原3% fresh-process paired regression、20% paired oracle regret和harmful-trace wrong-enable门槛；CUDA 8个harmful trials中非探索错误启用为0。生产chooser的Prometheus histogram分别记录139/142次CPU/CUDA learned决策，最坏trial P99为2/5 us，低于预注册50 us预算；原始trial行保存完整累积bucket、count与sum，P99由验证器重算且bucket总数必须等于决策counter。零干预时不得把跨进程wall-clock噪声解释为CacheFlow因果效果；只有实际干预仍受原3%门槛约束。此前两个独立中位数之比、局部row前驱平衡等过强口径均已移除，阈值未放宽。

短生命周期风险预算按 backend 隔离：CPU learned 产生 32 次 `safe_exploration`；已知 always 有害的 CUDA fresh process 提高最小样本门槛，在该 trace 内 0 次 probe 并 fail closed；`positive_lower_bound_decisions` 均为 0。CUDA positive-lower-bound 的生产实证由下一段长驻实验提供。

随后新增的单进程长驻门禁补齐了这一限制：CUDA server 连续运行 53 waves，`confidence_beta=1.0`、每动作最少 12 个样本。冷启动 CacheFlow/positive 均为 0；稳定阶段 26 次有限探索后产生 125 次 positive-lower-bound，覆盖 36 waves、最长及终端连续均为 35 waves；最后一次上下文 gauge 的收益 8.926 ms 大于 4.463 ms 不确定性，但该last-value只作诊断。切换后 CacheFlow/positive 均为 0、安全回退为 3。硬门禁使用逐wave动作counter，要求持续启用且稳定阶段最后至少3个wave仍有positive action，不用全程最大值，也不让另一个异构请求的last-value gauge抹掉已发生的上下文决策。逐wave请求TTFT与counter可独立重算全部phase/acceptance，篡改raw或copied acceptance均由普通测试拒绝。

### CUDA profiling 因果链

3 组 paired Latin upstream/always 干预通过。强制 CacheFlow 中位造成决策 +13、prefill chunk +23、prefill token -354、自研 KV kernel launch +2、KV copy +20,066,300 B、CUDA Event +0.808 ms、GPU busy 与最大 idle gap中位差不变，Engine execute 汇总 -11,446 us，但 TTFT P95 +85.61 ms。完整 GPU samples、Engine events 和相关 Prometheus 快照保存在 `results/cuda_causal_profile_evidence.json`；门禁拒绝仅有采样噪声而没有 material 请求级结果的 trace。

H2 进一步完成 4 个预注册 KV regime、160 条无 profiler paired trials 与 4 份真实 Nsight Systems trace。aligned 1 block 的 CUDA-event 改善为 +57.10% [95% CI +38.51%, +57.14%]，aligned 16/32 blocks 只有 +5.24%/+3.93%，按 10% 门槛判为 neutral；misaligned 1 block 保留为反例，CUDA-event/end-to-end 分别回退 137.94%/113.02%。NSYS 中 scalar/vector 每个 regime 均为 10/10 launches，否定“减少 launch 数”解释。正式 artifact 在 `results/research/h2-kv-profile-v1.0.0/`。

Nsight Compute 已实际执行，但当前 driver/tool compatibility 检查和 `ERR_NVGPUCTRPERM` 阻止硬件计数器采集；失败命令与日志均保留。因此验收只通过 NSYS + no-profiler effect 的限定主张，明确不声称 memory-bound、roofline、achieved occupancy 或 hardware DRAM bytes。

### 受限 Paged Decode Attention（Issue #5）

K1 CUDA 原型直接按 block table 从非连续物理页读取 K/V 并输出 attention，不在 timed path materialize contiguous KV。独立 CPU FP32 oracle 与相同数学的 contiguous CUDA 路径共同覆盖 `14/2/D64`、`28/4/D128`、ratio-7 GQA、context `1/15/16/17/31/32`、ragged batch、fragmented pages、unused-page poison、output guards 和 invalid shape/page table fail closed。

正式实验保存 9 个 regime × 20 pairs × 2 methods = 360 条无 profiler observations；每个 regime 计时前的独立 CPU FP32 oracle 最大绝对误差为 `3.6e-8`。0.5B 的 16-token case 为 neutral；17-token case 的中位改善为 +22.87%，但 95% CI 为 0.00%–47.08%，不能据此宣称稳定材料性获益。所有 medium/long regime 回退 10.41%–13.05%；0.5B shape 的 context 1024/batch 1 回退 13.05%（95% CI 12.91%–13.12%），7B shape 回退 11.59%（11.37%–11.78%）。4 份 NSYS trace 各精确包含 5 次对应方法 launch。batch-1 与 batch-4 的长 context 回退差只有 0.84/1.18 个百分点，未触发 K3 split-K；因长 context 回退超过 3%，预注册规则选择 K2 GQA reuse 作为下一待验证假设。环境记录为 RTX 4050 `sm_89`、6141 MiB 显存，最大 benchmark device allocation 16,864,272 B，全部 resource gate 通过。

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

固定上游提供 GGUF/模型 graph、GGML 通用算子、既有 backend、HTTP 基础设施和 sampling。个人差异是 Engine 拆分、Scheduler、KV 资源/块/事务 Swap、真实 CUDA KV Adapter 与 kernels、自适应/收益门控、在线策略持久化、metrics、fault injection 和复现实验。当前相对固定上游为 69 files、+11030/-99。代码量只统计该 patch；不得把 `vendor/llama.cpp` 原有代码算作个人实现，也不得声称“重写了 llama.cpp”。

### 生产重启与状态损坏

真实 CPU `llama-server` 三进程 smoke 已通过：第一进程产生 10 次 benefit observation 并完成 10 次后台原子保存；强制终止后第二进程报告 `restored=1` 且恢复全部 10 次观测；第三进程读取人为截断的 checkpoint 时报告 `failed=1`、保持 0 次模型观测并继续成功处理请求。证据位于 `results/benefit-checkpoint-smoke.json`。该结果证明单机单 writer 边界内的在线模型可恢复，不外推为跨副本模型同步。

早期 Python 控制面已归档到 `prototypes/cacheflow`，只保留设计演进与原型测试证据；生产 `llama-server` 不导入或调用它。

## 已知限制

- 单机单 GPU、0.5B 模型结果不能外推多 GPU 或大模型；
- Hybrid/Recurrent 仅 capability fallback，未实现物理 block backend；
- WPR sampled-stack profile 因权限未完成；现有图只回答 Engine phase 时间归属；
- CUDA cacheflow 当前 mixed workload 的吞吐和 latency P95 退化，需要后续加入 backend-specific policy gating；
- Benefit Gating 当前仅控制 prefill plan，尚未扩展到 slot placement、KV admission、swap 与 speculation 的联合动作空间；
- 当前16-trial联合fresh-process短trace没有positive-lower-bound；在线收敛只由53-wave单进程CUDA trace支持，不能外推其他GPU、模型或负载；
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

## Unified KV Action Policy 验收

- 生产动作：Direct、真实 CUDA Remap、受限 opt-in Paged、CUDA-managed Swap、事务型 host Swap、Recompute 均走真实服务路径；selected/observed counter 与实际动作一致，`{action,reason}` 联合计数可解释每个动作为何被选中，restore/CUDA 失败按原动作记录失败并清理状态。
- 能力门禁：Remap 只在真实 donor prefix 可执行时开放；Paged 只支持 Qwen2.5-0.5B、FP16 KV、page 16、D64、单 token/batch 1、context ≤ 17、完整 GPU offload，默认关闭。超出 envelope 的受控用户请求回退 Recompute，不把 H3 microbenchmark 当作生产证据。
- 验收入口：`scripts/run_issue7_acceptance.ps1` 串行执行 Direct/Paged differential、真实 Remap、KV pressure fallback、晚到 CUDA failure/recovery；CUDA build gate 另执行 `test-backend-ops` 的 Paged Flash Attention CPU/CUDA 交叉验证、独立 F16 CUDA oracle、策略与 block-table 单测。
- 正式 Paged 服务证据：v1.1 在干净提交 `9182882` 上用 17-token 跨页请求完成 10 组 AB/BA，均逐字输出 `,`，Paged 进入真实 production graph 10 次、fallback 0；机制隔离 NSYS replay 恰有 24 个 `cacheflow_paged_decode_fattn_k1<64>` layer launch。Paged client P95 29.210 ms 对 Direct 27.354 ms，回退 6.78%；配对 Paged-Direct 中位差 +2.705 ms，bootstrap 95% 区间 [-1.185, +12.019] ms。因此 +5% promotion gate 失败，保留 opt-in/默认不选，不把正确性通过写成性能通过。完整 hash-bound 工件位于 `results/research/h7-production-paged-v1.1.0/`；未跨页的 v1.0 仅保留为 superseded 审计记录。
- 算法：H0 固定安全规则、A1 解析模型、T1 分桶查表、L1 置信约束 Ridge 使用相同 snapshot 和完整动作边界；无效/OOD/冷启动/不确定性均 fail closed。
- 数据隔离：20 个 trace group 按时间顺序切为 12 train / 8 evaluation，session、prefix family 与 trace 均无交叉；16 个 held-out resident/preempted snapshot 采用 paired action observation。
- v1.0.0 因错误使用 HTTP 往返计时且 L1 合同与生产不一致而被否决，仅保留在 `results/research/superseded/`。v1.1.0 从干净提交 `32c8fe7` 重采集：40 trace 按时间切为 20 train/20 evaluation，resident/preempted 各 20 个 held-out pair。
- v1.3.0 正式 CUDA matched-workload replay 保存并语义校验 200 组原始 Prometheus/响应 observation，独立采集两个 regime 的 Recompute，按真实 observation 顺序训练并强制完整 trace cluster。H0/A1/L1 的 median/P95/cumulative regret 均为 0/1.953/15.543 ms、harmful 0；L1 未切换。离线 T1 为 0/1.516/5.506 ms、harmful 0，paired trace-cluster mean-regret delta 95% CI 为 [-0.5285, -0.0511] ms。动作服务器的状态特征存在已报告偏差，故只称本次 matched-workload replay 中低于 H0，不作克隆状态因果或生产在线收益声明。v1.0.0 至 v1.2.0 均仅作否决审计记录。
- 热路径：5 个 regime、500 万次 choose、零 allocation，最差 p99 1.000 us，decision/action ratio p99 0.0644%；源码哈希绑定审计未发现直接 CUDA 同步符号或后端 include，但不证明运行时间接同步为零。raw Windows maximum 1997.800 us 保留为 report-only 抢占诊断。
- 证据所有权：artifact 精确绑定协议/模型哈希、外层实现提交、固定上游及 replay patch、完整文件树；validator 同时验证干净开发提交树或 bootstrap 已应用 patch 树，拒绝额外 vendor 改动，并从 raw rows 重算 H0/A1/T1/L1 与 overhead。
