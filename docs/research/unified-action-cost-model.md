# Direct / Remap / Paged / Swap / Recompute 统一成本模型：一手资料与设计约束

> 状态：GitHub Issue #6 的研究输入，2026-08-08。本文只采用原始论文、作者维护的源代码和厂商官方文档。文中的阈值与候选公式是本项目待预注册、待证伪的设计，不是已经取得的性能结果。

## 1. 结论先行

Issue #6 不应实现成“给五个动作各拍一个权重，然后取最小值”。可靠的决策顺序应是：

1. 从一次不可变的 scheduler snapshot 生成全部特征；
2. 先用正确性、资源和设备能力约束删除不可执行动作；
3. 对剩余动作预测从“现在”到“恢复下一次有效 decode”的完整系统成本；
4. 只有候选相对安全基线的优势大于不确定性和切换 margin 时才采用；
5. 缺少标定、输入越界、预测非有限、资源不足或运行时失败时，立即回到确定性的安全策略。

推荐同时比较三个模型：透明的确定性启发式 `H0`、按环境与 regime 分桶的查表模型 `T1`、每动作独立的正则化线性模型 `L1`。分析式模型 `A1` 作为所有模型共享的结构先验和冷启动估计。最终进入真实调度路径的优先候选是 `L1 + H0 fallback`，而不是树模型或神经网络：它能够复用本仓库已有的在线 ridge、置信半径、漂移和 cooldown 机制，且可用固定大小数组实现有界、无分配的热路径。

H3 实验已经证明受限 Paged K1 在中长 context 上比相同数学路径的 contiguous CUDA comparator 慢约 10.41%--13.05%。后续 K2 只在 Qwen2.5-0.5B、D64/GQA7、context 17--24 的生产 Paged 内部通过了 K1 替换门槛；它没有提供中长 context 或 Paged 相对 Direct 的正证据。因此 Issue #6 中的 Paged 仍是 **evidence-gated experimental action**，不能因分析式模型预测“省去了 remap bytes”而自动进入用户请求路径。

## 2. 动作语义必须先固定

统一模型只有在动作边界相同的情况下才有意义。主指标的计时边界统一定义为：

```text
scheduler snapshot ready
  -> decision
  -> required mapping / transfer / recomputation
  -> downstream attention or restore dependency satisfied
  -> next useful decode token becomes runnable
```

单独报告 decision CPU time、CUDA event time、暴露给 foreground 的 stall 和完整 action time，但优化目标使用完整 action time；不得拿 Paged 的单 kernel latency 与 Remap 的 remap-plus-attention 时间混比。

| 动作 | 本项目中的精确定义 | 必须满足的合法性条件 | 必须计入的成本 |
|---|---|---|---|
| `DirectReuse` | 现有 K/V 所有权、位置、布局和下游读取合同均已满足，不移动数据 | exact token identity、生命周期有效、无待完成写、布局可被当前 attention 读取 | page/table lookup、依赖等待、下游 attention；搬运字节为 0 |
| `CudaRemap` | 用 descriptor + staging + Gather/Scatter 建立 snapshot 语义并产生目标布局 | remap kernel 支持 dtype/alignment/tail，staging 与目标显存有足够 headroom | descriptor、两次 kernel launch、读/写/staging bytes、同步与后续 contiguous attention |
| `PagedDecode` | 不物化连续 K/V，attention 直接按 block table 读取物理页 | 仅支持已验证 shape、dtype、page size、mask、设备；生产 adapter 与结果证据均通过 | page lookup、碎片/地址间接、paged attention 每步成本，以及省去 remap 的收益 |
| `StoreSwap` | 通过 transactional Host/File store 保存和恢复完整、可校验的 sequence state | store budget、checksum/schema、exact prompt compatibility、restore handle 有效 | serialize/copy/I/O、save + restore、排队、恢复依赖和持久化容量价格 |
| `CudaSwap` | 当前实现把真实设备 K/V 通过独立 stream 搬到 pinned host，再搬回设备 | pinned pool、D2H/H2D 能力、事件和容量均可用 | D2H + H2D bytes、分段/launch、不能被计算隐藏的传输、pinned-memory 压力 |
| `Recompute` | 丢弃不可保留状态，随后按原 token 序列重新 prefill | token/state 可重放、模型与采样语义兼容、prefill budget 可容纳 | 被重算 token 的 profiled prefill、对并发 decode 的 stall、排队和临时 KV |

这里特意区分 `StoreSwap` 和 `CudaSwap`。当前 `CudaSwap` 仍以 host pinned memory 为后备层，并不是“把 KV 永久换到另一张 GPU”。若未来加入真正 D2D/peer tier，应新增独立 capability、带宽和 action ID，不能复用当前分布。NVIDIA 官方说明 pinned memory 的 H2D/D2H 带宽更高，但分配代价大且属于稀缺资源；异步 copy 要用 pinned memory，copy/compute 重叠还要求设备支持并使用不同的非默认 stream。[CUDA C++ Best Practices Guide 12.6](https://docs.nvidia.com/cuda/archive/12.6.3/pdf/CUDA_C_Best_Practices_Guide.pdf)

## 3. 一手资料给出的可测变量

### 3.1 页面、碎片与复用

PagedAttention 用 logical block table 映射非连续 physical KV blocks，并通过 block sharing/COW 降低碎片和重复存储；但论文也报告其当时的 attention kernel 因 block-table lookup、分支和可变长度处理，比对应 FasterTransformer kernel 慢约 20%--26%。这说明“少浪费显存”和“单次 kernel 更快”是两个不同目标。[PagedAttention / vLLM, SOSP 2023](https://arxiv.org/abs/2309.06180)

因此最少记录：

- logical page 数、physical page 数、page size；
- 最后一页有效 token 比例与内部碎片字节；
- contiguous run 数、重复 source、重叠/循环 mapping；
- 当前引用计数、COW 必要性和页表版本；
- context、batch、Q/KV heads、head dimension、dtype；
- 已测的 contiguous/paged attention regime 与标定年龄。

复用不能只用全局命中率表示。SGLang 的作者维护实现为 radix node 保存 `last_access_time` 和 `hit_count`，匹配/插入时更新，并用 eviction priority heap 选择可驱逐叶子；这为 reuse distance、频率和 protected reference 作为特征提供了直接系统先例。[SGLang RadixCache 固定快照](https://github.com/sgl-project/sglang/blob/6c7498113f19c2cac9c4c0b2c20f4498b25f6bba/python/sglang/srt/mem_cache/radix_cache.py)

### 3.2 搬运、带宽、launch 与 overlap

对每个候选记录：

- K/V、descriptor、page table、staging 的实际 bytes，而非逻辑 token 数的粗略替代；
- D2D、D2H、H2D、host/file 的分别标定带宽；
- 连续区间数和 kernel/copy launch 数；
- foreground iteration 是否能与 transfer 重叠，以及未被隐藏的 exposed time；
- GPU free/reserved bytes、pinned-host free bytes、store budget 和 in-flight reservation；
- CUDA stream/event 依赖与 async engine capability。

INFERCEPT 直接把 swap latency 写成 token 数到搬运延迟的映射，并指出分页 KV 分散在多个物理区域会增加 kernel launch；其 swap waste 将 swap-in/out 和同时运行请求受到的等待都计入，而不是只看 copy latency。它又通过离线标定 `T_fwd` 和带宽，使每轮可隐藏的 swap 量满足 `T_swap(N_i)=T_fwd(B_i)`，并同时约束 CPU、GPU 容量与 swap budget。[INFERCEPT, §3.2--§4.3](https://arxiv.org/pdf/2402.01869)

FlexGen 的分析模型同样把不同方向 I/O 写成 `bytes / measured_bandwidth`，在假设完全 overlap 时取 transfer 与 compute 各阶段耗时的最大值，并在 GPU/CPU/disk 容量约束下做 placement search。论文同时承认带宽随负载变化、碎片难以精确建模，策略可能 OOM，需要校正。这支持“分析式冷启动 + 实测查表/在线残差”，不支持把理论峰值带宽写死。[FlexGen, §4.3 与附录 A.3](https://arxiv.org/pdf/2303.06865)

### 3.3 重计算与系统机会成本

INFERCEPT 对 recompute 使用按 token 数标定的 `T_fwd(C)`，并把重计算自身占用与它延长 iteration 后对其他请求造成的等待都计入。其 chunked recomputation 用离线 profile 得到 GPU saturation point，并只把当前 iteration 剩余的 query-token capacity 用于重算。[INFERCEPT, §4.2](https://arxiv.org/pdf/2402.01869)

vLLM 当前官方优化文档则明确：显存不足时会 preempt 并 recompute，V1 默认选择 recompute 而非 swap，因为在其 V1 架构中 recompute overhead 更低；文档也提醒 preemption 会损害端到端延迟。[vLLM 官方 Preemption 文档](https://github.com/vllm-project/vllm/blob/7e85d3a42cc180d8b8fa85ca815c3e90bf2cb970/docs/configuration/optimization.md)

二者合起来给出的约束是：Swap 和 Recompute 没有全局优胜者，选择必须包含 context token 数、每 token KV bytes、标定 prefill cost、transfer bandwidth、并发 batch work、可隐藏比例和容量压力。

## 4. 统一特征快照

第一版使用固定 POD，不允许候选模型自己读取可变 runtime state：

```text
identity: backend, model_hash, model_shape, action_capability_mask
layout:   cached_tokens, kv_bytes, logical_pages, physical_pages,
          last_page_fill, contiguous_runs, snapshot_required
reuse:    reuse_distance_iterations, reuse_count, predicted_next_use,
          request_family_id
work:     decode_tokens, prefill_tokens, active_sequences,
          expected_remaining_decode_tokens, queue_wait_ms
memory:   gpu_free_bytes, gpu_reserved_bytes, kv_pressure,
          pinned_free_bytes, store_free_bytes
profile:  d2d/h2d/d2h/store_bandwidth, launch_intercept,
          prefill_ms_by_tokens, attention_ms_by_shape,
          calibration_age, ood_distance
overlap:  async_engine_available, transfer_stream_available,
          foreground_work_ms, dependency_ready
```

所有 bytes 和容量用整数；所有浮点输入在进入模型前检查 finite 并 clamp 到预注册范围。`request_family_id` 只用于数据切分/分层，不能作为线上模型偷记具体 trace 的高基数特征。

## 5. 候选模型

### 5.1 H0：强制保留的简单启发式

```text
mask every infeasible action
if valid exact state is already directly consumable: DirectReuse
else if a compatible CudaSwap/StoreSwap handle already exists: restore it
else if snapshot mapping is required and CudaRemap is supported with headroom: CudaRemap
else: Recompute
```

Paged 在当前 H3 负结果下不进入 H0。保存阶段若 swap 失败，H0 释放部分状态并选择 Recompute；不得用半成功 swap 继续。H0 是安全底线和强制 baseline，不是故意做弱的对照。

### 5.2 A1：分析式完整动作成本

对 snapshot `x` 与可行动作 `a`：

\[
\hat C_a(x)=T_{decision}+T_{exposed,a}+T_{downstream,a}
 + \lambda_{slo}P_{slo,a}+\lambda_{mem}R_{mem,a}.
\]

其中 transport 原语先用校准的仿射模型：

\[
T_{move}=n_{launch}L_{launch}+\frac{B_{read}+B_{write}}{BW_{direction}},
\qquad
T_{exposed}=\max(0,T_{move}-T_{overlap}).
\]

各动作使用同一 horizon：

- `DirectReuse`：`T_dependency + T_attention_contiguous`；
- `CudaRemap`：`T_gather + T_scatter + T_attention_contiguous`；
- `PagedDecode`：`T_attention_paged(context, batch, shape, fragmentation)`，若页面布局预计跨多步保留，则按实际 horizon 累加，不得无限外推；
- `CudaSwap`：`T_D2H_save + wait_until_reuse + T_H2D_restore + restore_dependency`，目标成本只计 exposed 部分，但 raw transfer 另报；
- `StoreSwap`：相同结构，另含 serialize/checksum/file 或 host-store latency；
- `Recompute`：`T_queue + T_prefill(recomputed_tokens, concurrent_decode) + restore_dependency`。

`P_slo` 是预计越过 TTFT/TBT deadline 的惩罚，`R_mem` 是该动作在 horizon 内占用稀缺 GPU/pinned/store 容量的 byte-time。INFERCEPT 的 Preserve/Swap/Discard waste 公式说明 byte-time 与其他请求 stall 必须进入系统成本，而非只比较本请求完成时间。[INFERCEPT 原始论文](https://arxiv.org/pdf/2402.01869)

A1 必须把 `max` overlap 公式与 `sum` 串行公式都保留；只有 stream、依赖与实际 concurrent copy/compute 证据存在时才选 `max`。否则 fail closed 使用串行上界。

### 5.3 T1：分桶查表模型

Clockwork 为每个 model/worker/batch 维护近期 action-duration profile，并结合 memory state 与 pending actions预测候选何时完成；论文实现用最近 10 次动作的时长更新 profile。[Clockwork, OSDI 2020, §4.5](https://www.usenix.org/system/files/osdi20-gujarati.pdf)

本项目的 T1 键为：

```text
backend × model_shape × action × context_bucket × batch_bucket
× page_run_bucket × kv_pressure_bucket × overlap_flag
```

每格保存 count、median、MAD、p95 和更新时间。缺格不做跨 backend 外推，回 A1/H0；插值只能在同 backend、shape、action 内相邻 bucket 进行。表的更新与决策必须是常数上界，持久化异步完成。

### 5.4 L1：受约束的每动作线性模型

每个 action 独立拟合非负 latency/cost：

\[
\hat C_a(x)=\max(0,\theta_a^T\phi(x)),\qquad
u_a(x)=\beta\sqrt{\phi(x)^T A_a^{-1}\phi(x)}.
\]

选择条件不是单纯 `argmin C`，而是候选相对 H0 的保守优势：

```text
upper_cost(candidate) + switch_margin < lower_cost(H0)
```

否则执行 H0。保守 contextual bandit 的原始工作把“始终不低于 baseline 的固定比例”作为安全约束，并在无法证明安全时执行 baseline；这里借鉴的是这一接口原则，不宣称当前有限样本获得其理论保证。[Conservative Contextual Linear Bandits, NeurIPS 2017](https://papers.nips.cc/paper_files/paper/2017/hash/bdc4626aa1d1df8e14d80d345b2a442d-Abstract.html)

训练时 loss 使用 cost regression；最终模型选择以 held-out regret 与 tail harm 为主，而不是 R²。树模型可以作为离线研究上界，但在证明序列化、OOD、最坏执行时间和真实路径开销以前不得成为生产候选。

## 6. 防止 trace 泄漏

随机拆 decision rows 是无效评估：同一 session、prefix family、到达 burst 和设备热状态会产生大量相关行。scikit-learn 官方文档说明 `GroupKFold` 保证同一 group 不同时出现在训练和测试中；其时间序列指南也指出相邻观测存在自相关，普通随机 K-fold 会产生不合理的训练/测试相关性，应该用只训练过去、评估未来的 `TimeSeriesSplit`。[GroupKFold 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)、[Cross-validation 官方指南](https://scikit-learn.org/stable/modules/cross_validation.html)

本项目固定两层切分：

1. 外层 final test 按完整 `trace_id` 分组；同一 session、prefix family 和其所有 action replay 必须落在同一 split；
2. 内层只在 training traces 中按时间顺序做 tuning/calibration，validation 永远晚于对应 training window；
3. normalizer、带宽/profile 表、bucket 边界、模型参数、margin 和 OOD 阈值都只从 training 得到；
4. final test 只运行一次确认性分析；发现失败不能改阈值后重跑同一 test；
5. manifest 保存 split seed、trace ID/hash、时间范围和零 group-overlap 断言。

若同一真实 trace 为每个动作做反事实 replay，所有动作必须共享同一个 pre-action snapshot、输入和 seed，并在隔离副本或 fresh process 上运行。线上只能观测被选择动作时，不得伪造未执行动作的 cost；oracle regret 只来自完整 paired replay 数据。

## 7. 误判成本与报告

对第 `i` 个 paired snapshot，可行动作集合为 `F_i`，主 regret 定义为：

\[
r_i=C_i(a_i)-\min_{a\in F_i}C_i(a),\qquad r_i\ge 0.
\]

这比 action accuracy 更重要：把 0.02 ms 的第二名选成第一名，和导致一次长 context recompute 的错误，不能算作同等的一次分类错误。至少报告：

- mean / median / p95 / p99 / max / cumulative regret；
- cost ratio `C(chosen)/C(oracle)`；
- harmful-decision rate：相对 H0 回退超过预注册 3% 或绝对 SLO 预算的比例；
- 每个真实 action 与预测 action 的 cost-weighted confusion matrix；
- OOD、低置信、资源 gate、运行时失败各自的 fallback 次数与额外成本；
- 各 trace family、context、pressure、backend 的 worst subgroup；
- H0、A1、T1、L1 在完全相同 paired snapshots 上的 bootstrap confidence interval。

## 8. 安全回退与 capability gate

动作在评分前被屏蔽，不能先让不可行动作“获胜”后再临时补救：

- `DirectReuse`：token identity、layout、ownership、refcount、event/lifetime 全部有效；
- `CudaRemap`：kernel capability、descriptor/grid、staging + output headroom、stream 可用；
- `PagedDecode`：生产 adapter、shape/dtype/page/mask、correctness、设备和该 regime 的正证据全部有效；
- `CudaSwap`：pinned buffer、host/GPU headroom、async copy capability、save/restore event 可用；
- `StoreSwap`：完整事务容量、schema/checksum、handle compatibility 和 I/O budget 可用；
- `Recompute`：tokens/sequence state 可重放，模型身份一致且 prefill budget 可容纳。

统一 fallback 触发器：空/非有限输入，未标定或标定过期，OOD 超阈值，候选优势不超过 uncertainty + margin，预测成本违反 deadline/capacity，运行时 launch/transfer/store 失败，近期 regret/harm alarm 或 drift cooldown。

fallback 必须选择当前 phase 下 H0 的**可行**动作，而不是固定写死 Direct；例如 swap save 失败时选 Recompute，已有 exact compatible swap handle 的 restore 失败时清理 handle 后 Recompute，Paged gate 失败且目标 layout 需要 snapshot 时选 CudaRemap。每次回退记录 `request_id/action/reason/predicted_cost/observed_cost`，fast-path 成功率不得混入 fallback 样本。

## 9. 真实调度路径的有界开销

chosen model 必须满足：固定动作数组、固定维度特征、无 heap allocation、无文件 I/O、无 GPU synchronize、无 blocking lock；checkpoint、profile persistence 和复杂统计均在 inference iteration 外异步完成。

预注册的本机工程门禁建议为：

```text
p99 choose() CPU time <= 50 us
raw wall-clock max is retained as a report-only Windows preemption diagnostic
p99 decision / corresponding scheduler-iteration CPU time <= 1%
zero allocation and zero CUDA synchronization in choose()
```

这些数值是本项目待验证的 gate，不是论文常数。测试覆盖 1/2/4/6 个候选动作、冷/热 cache、OOD/fallback、最大 feature 值和 100 万次稳定调用；报告 p50/p95/p99/max。随后必须在真实 scheduler path 做 policy-off/H0/A1/T1/L1 配对 A/B，检查 TTFT、TBT、吞吐、scheduler CPU、锁等待和错误率。Clockwork 表明小型 profile/内存状态可以支撑实时 action prediction，但并不自动证明本实现开销可忽略。[Clockwork 原始论文](https://www.usenix.org/system/files/osdi20-gujarati.pdf)

主开销数字来自无 profiler 的稳态测量；NSYS 只用于确认没有隐藏同步、意外 memcpy 或额外 kernel，不能把 trace 过的时长当主结论。

## 10. 同一协议下的最小实验矩阵

| 维度 | 固定覆盖 |
|---|---|
| model | 当前真实 Qwen2.5-0.5B；7B 只有资源允许且端到端加载成功才单列 |
| backend | CPU correctness；CUDA performance，绝不混合训练/结论 |
| context | page boundary、short、medium、long |
| batch | 1、4；若服务并发更高另加真实 regime |
| placement | identity、contiguous runs、seeded fragmented、snapshot overlap |
| pressure | low、near admission threshold、forced preemption |
| reuse | immediate、short/long reuse distance、never reused |
| actions | 每个 snapshot 的全部可行动作；不可行原因保留 |
| models | H0、A1、T1、L1 使用同一 split、特征和 cost boundary |
| statistics | paired randomized order、至少 20 pairs/regime、bootstrap CI、负结果保留 |

确认性实验前锁定：protocol hash、模型/二进制 hash、action implementation commit、trace split、feature schema、cost weights、H0、candidate hyperparameters、margin、OOD、sample count、失效和停止规则。Paged K1 的既有负结果作为先验 gate 保留，不能在统一模型中删除或重新定义 comparator。

## 11. 对实现的直接约束

1. 新增一个深模块拥有 `snapshot -> feasible actions -> score -> decision -> observation`，scheduler 不解释模型系数；
2. capability mask 与 reason enum 是公共合同，任何模型都不能绕过；
3. action scorer 只消费不可变 POD，禁止读 allocator/store 的可变内部状态；
4. cost observation 明确标注 `decision/action/exposed/downstream/total` 五段时间；
5. H0 永久可选并且始终可独立回放；
6. 模型状态按 backend、model hash、shape、feature schema 和 protocol version 隔离；
7. 反事实训练数据来自同 snapshot 的完整 paired replay，线上 chosen-only feedback 不冒充 oracle；
8. production 默认先 shadow：计算决策但执行 H0，直到 held-out regret、harm、fallback 和 overhead 全部过门；
9. canary 后仍保留 per-action kill switch、regret alarm、drift cooldown 和 H0 fallback；
10. 报告明确区分 kernel frontier、完整 action frontier 与用户请求 SLO，禁止跨层宣称加速。

## 12. 来源到设计约束映射

| 一手来源 | 可采用的约束 | 不允许外推的内容 |
|---|---|---|
| [PagedAttention, SOSP 2023](https://arxiv.org/abs/2309.06180) | block/page/fragmentation 与 kernel overhead 同时进入特征；capacity 与 kernel latency 分报 | 其 A100 系统吞吐不等于本机 Paged K1 会更快 |
| [INFERCEPT](https://arxiv.org/pdf/2402.01869) | swap/recompute 用 bytes、带宽、launch、并发机会成本、overlap 和容量联合评分 | API interception 分布不等于普通聊天复用分布 |
| [FlexGen](https://arxiv.org/pdf/2303.06865) | 分方向 `bytes/BW`、compute/transfer overlap、容量约束、硬件标定 | latency-insensitive 大 batch offload 结果不外推在线 serving |
| [Clockwork, OSDI 2020](https://www.usenix.org/system/files/osdi20-gujarati.pdf) | 分桶 action profile、memory/pending state、deadline-aware rejection | DNN model load profile 不等于 KV action profile |
| [vLLM V1 官方优化文档](https://github.com/vllm-project/vllm/blob/7e85d3a42cc180d8b8fa85ca815c3e90bf2cb970/docs/configuration/optimization.md) | recompute 是合法安全 fallback，并须观测 preemption | “V1 recompute 更低开销”只属于其架构，仍需本机比较 |
| [SGLang RadixCache 源码](https://github.com/sgl-project/sglang/blob/6c7498113f19c2cac9c4c0b2c20f4498b25f6bba/python/sglang/srt/mem_cache/radix_cache.py) | last access、hit count、lock/ref 与 eviction priority 可作为复用特征 | LRU/LFU 不是统一动作成本最优性的证明 |
| [CUDA Best Practices 12.6](https://docs.nvidia.com/cuda/archive/12.6.3/pdf/CUDA_C_Best_Practices_Guide.pdf) | pinned memory、stream、async engine、transfer overlap 均需能力门控 | 理论/示例带宽不能代替本机标定 |
| [GroupKFold / TimeSeriesSplit 官方指南](https://scikit-learn.org/stable/modules/cross_validation.html) | trace/session 分组，训练在过去、测试在未来 | 普通 row shuffle 不能证明跨 trace 泛化 |
| [Conservative Contextual Linear Bandits](https://papers.nips.cc/paper_files/paper/2017/hash/bdc4626aa1d1df8e14d80d345b2a442d-Abstract.html) | 不确定时回到已知 baseline 的接口原则 | 本项目未完成形式化假设验证前，不声称理论 safety/regret bound |

## 13. 本研究仍未证明什么

- 没有证明统一模型已经优于 H0；
- 没有证明 Paged 已普遍优于 Direct；K2 仅在受限短 context envelope 内替换 K1，现有中长 context 证据仍支持禁用该路径；
- 没有证明 Swap 比 Recompute 普遍更快，或反之；
- 没有证明 50 us/1% 开销门槛已经通过；
- 没有用离线 replay 替代真实 scheduler path；
- 没有把不同 backend、硬件、模型 shape 的数据混成一个可泛化模型。

Issue #6 只有在同协议比较 H0/A1/T1/L1、trace-safe split、paired oracle regret、结构性 fallback、真实调度接入和上述 overhead gate 全部通过后，才可关闭。
