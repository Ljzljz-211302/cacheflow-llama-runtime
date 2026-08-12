# Copy-aware Paged KV 研究章程

- **版本：** 1.0.0
- **日期：** 2026-08-06
- **状态：** Issue #2 的预注册研究问题；实验协议由后续 Issue #3 单独冻结
- **机器可校验原件：** [`config/research_claims.json`](../../config/research_claims.json)
- **基线注册表：** [`config/research_baselines.json`](../../config/research_baselines.json)
- **一手资料审计：** [前人工作与可复现基线](primary-source-foundations.md)

## 1. 总问题与主张边界

本项目研究：在消费级单 GPU llama.cpp Serving 中，何时应显式复制 KV、何时应直接分页读取，以及能否用可解释且有安全回退的策略在这些动作间选择。

章程不预设 Paged Decode Attention 一定更快。原始 PagedAttention 工作已经给出反例：分页 attention kernel 可能因索引、分支和变长处理而更慢，系统仍可能因减少 KV 浪费、容纳更大 batch 而获益。因此所有后续结论必须分开回答：

1. 算子是否正确、kernel 是否更快；
2. 峰值 KV 与可驻留请求数是否改善；
3. TTFT、TPOT/TBT、请求尾延迟和吞吐是否改善。

当前已实现 Direct/Scalar Remap/Vector Remap 与受限 Paged Decode K1–K4；K4 正确性覆盖 1–2048 token，并在 H13 中将旧 K2 的主中位回退从 +50.35% 降至 +3.98%，但置信上界 +5.38% 未过 +5% 晋级门。只有通过对应 falsification gate 和预注册协议的范围可以写入简历；通用 Paged、Paged 优于 Direct 或跨设备结论不得宣传为既有能力。

## 2. 研究问题总览

| ID | 研究问题 | 状态 | 核心主张 | 首要证伪条件 |
|---|---|---|---|---|
| H1 | 128-bit 向量化何时降低 snapshot-preserving Remap 成本？ | existing-evidence | 锁定环境中 1/4/16/32 blocks 的配对改善非递增 | 趋势不再非递增、任一规模回归超过 3%、没有规模改善至少 10%，或正确性失败 |
| H2 | KV 搬运何时成为请求级瓶颈？ | limited evidence | 对齐小搬运中向量化材料性获益，规模增大后收益低于门槛；错位 layout 明确反转 | 所有预注册高搬运场景中 KV 均不 material，或没有 CUDA mediator |
| H3 | Paged Decode Attention 何时优于 Direct/Remap？ | limited evidence | K1 中长 context 相对 contiguous comparator 回退；K2 在受限短 context 内通过 K1 替换门槛 | 所有场景既无延迟 non-inferiority 也无容量收益，或数值不正确 |
| H4 | 可解释代价模型能否安全优于固定规则？ | limited evidence | 当前 L1 安全回退到 H0；尚未证明 learned 收益 | regret/回归超门槛、开销吃掉收益、分布切换后仍持续错误启用 |
| H5 | 调度能否改变 CUDA 工作却在 execute 汇总下降时恶化 TTFT？ | existing-evidence | 请求排队结构可使 phase aggregate 与请求 SLO 反向 | 决策、CUDA mediator、请求结果三段因果链任一不存在 |

完整字段——自变量、因变量、混杂因素、基线、机制、证伪条件、证据和边界——以 JSON 原件为准，并由单元测试拒绝缺字段、无证伪条件、伪造 observed result 或引用未注册基线。

## 3. H1：向量化 Remap

### 变量与基线

- 自变量：Scalar/`uint4`、1/4/16/32 blocks、对齐及 scalar-tail 比例。
- 因变量：paired CUDA-event 时间、operation 端到端时间、vector/scalar bytes、正确性和 Sanitizer。
- 基线：`scalar-remap`、`vector-remap`；descriptor、mapping、staging、stream 和同步点必须相同。
- 混杂：WDDM、频率/温度、固定先后顺序、不同 mapping 或 warm-up。

### 当前证据

20 组配对且交替顺序的结果为 53.33% / 48.89% / 3.13% / 1.87%，在锁定的四个规模上呈非递增趋势。这不支持“端到端推理加速 53.33%”。减少指令、launch、cache 或 bandwidth 中哪一项造成趋势仍是 prospective explanation，必须由 H2 profiling 区分；真实应用累计 5,603,330 vectorized bytes 只证明生产链路调用。

### 负结果处理

必须按规模、对齐和 fallback 比例保留回归，不允许把不利区间平均进一个正向 headline。

## 4. H2：KV 搬运瓶颈

### 变量、指标与机制

- 自变量：context、moved bytes、fragmentation/overlap、prefix share/COW、batch/arrival、显存压力。
- 指标：KV kernel/copy time、launch、有效带宽、TTFT、TPOT/TBT、queue/request P95、吞吐、resident capacity。
- 基线：`upstream`、`direct-copy`、`scalar-remap`、`vector-remap`。
- 机制：显式搬运消耗 bandwidth 与 launch；snapshot 合法性限制 Direct；调度改变排队后可能掩盖或放大该成本。

Issue #4 现有 4 个预注册 regime、160 条无 profiler paired observations 与 4 份 NSYS native/SQLite trace。aligned-small 的 CUDA-event 改善中位数为 57.10%（95% CI 38.51%–57.14%），aligned 16/32 blocks 均低于 10% 材料性门槛；misaligned-small 反而回退 137.94%（95% CI 137.84%–140.55% regression），对应 end-to-end 回退 113.02%。NSYS 中每个 regime 的 scalar/vector 都各有 10 次 launch，因此收益并非来自减少 launch 数；错位反例则证明 layout 合法性必须进入 action gate。

这仍没有证明 memory-bound：本机 NCU 同时报 driver incompatibility 与 `ERR_NVGPUCTRPERM`，因此 DRAM throughput、L2 和 occupancy 未采到。`effective_payload_gbps` 只允许解释为逻辑 payload / CUDA-event time，不能冒充硬件带宽或 roofline。当前 H2 结论只覆盖 KV 算子机制；原有 3-pair 服务因果链仍负责说明 scheduler/CUDA mediator 与 TTFT 可能反向，不能把 +57.10% 写成端到端推理加速。

服务级 NSYS 复现实验进一步把 3 个 no-profiler pairs 与相同 seed/config 的 3 个 profiler pairs 分开：6 个 trial/mode server PID、72 个确定性 request ID 均连接到 benefit decision、prefill shape、KV action、PID-filtered NSYS timeline 和 TTFT。profiler replay 共记录 42 次自研 KV launch，每个进程的 runtime counter 与 NSYS 完全相等。本轮 no-profiler 中 decision +20、chunk +30、prefill token -86、KV launch +0、copy bytes +10,518,500、CUDA-event -1.040 ms、TTFT P95 +126.932 ms、Engine execute +146,493 us；它关闭了用户请求因果链接，但没有复现旧实验的 execute/TTFT 反号，因此两种结果都保留。

## 5. H3：Paged Decode Attention frontier

### 变量、指标与机制

- 自变量：Direct/Remap/Paged、context、page size、physical fragmentation、batch/GQA、prefix reuse。
- 正确性：与连续 attention oracle 比较 masking、GQA、online softmax 和输出容差；unsupported shape 必须 fail closed 或 fallback。
- 性能：kernel time/device bytes、peak KV、admissible batch、TTFT/TPOT/P95/throughput。
- 混杂：layout conversion、不同数学/精度、graph capture、allocator/scheduler 变化。

Paged 路径省去 materialization，却增加 page-table lookup、不规则访存、mask、softmax 和归约。它可能只改善容量、不改善 batch-1 latency；这仍是有效结果，但必须明确属于哪一层收益。若所有预注册场景既没有延迟 non-inferiority 也没有容量收益，则否定当前受限实现，而不是更换 workload 追正结果。

Issue #5 的 H3 v1.0.0 已得到受限混合结果：K1 直接分页 kernel 在每个 regime 计时前通过独立 CPU FP32 oracle，最大绝对误差 `3.6e-8`。0.5B 的 16-token case 为 neutral，17-token case 有高噪声的 +22.87% 中位改善但区间跨过材料性门槛；所有 medium/long regime 回退 10.41%–13.05%。0.5B/7B-shape 在 context 1024、batch 1 分别回退 13.05% 与 11.59%；对应 batch-1/batch-4 差只有 0.84/1.18 个百分点，不支持优先做 split-K。该预注册规则选择了 K2 GQA KV reuse，后续 K2 以两 query-head warp tile、每 KV head 7→4 次 K/V tile 装载、转置 shared K 和 warp stable-softmax 实现。正式 v2.10 在受限 0.5B/D64/GQA7/page16/context17 重复 cached-request envelope 内，每 variant 保留 480 条请求计时与 600 次真实 Paged graph entry：请求级 median/P95 回退仅 0.55%/1.52%，median 回退 bootstrap 95% 上界为 2.86%，相同 480 次目标 kernel 总时长降低 50.44%，通过预注册 bounded-replacement 门槛；这不是中长 context、端到端普适加速或 Paged-vs-Direct 的正结论。NCU 不完整，仍禁止 memory/occupancy 与硬件 DRAM-byte 归因。

## 6. H4：自适应动作策略

比较对象必须包含固定规则 `H0`，而不是只比较一个故意较差的 always action。候选模型使用 moved bytes、fragmentation、reuse distance、memory pressure、launch/transfer cost 和 decode work；指标包括 paired oracle regret、upstream regression、wrong-enable、fallback、decision overhead 和请求级 SLO。

训练/调参 trace 与评估 trace 必须隔离；exploration 不得计作收敛。若固定规则在全部 held-out regime 内与模型无显著差异或更安全，则保留固定规则并报告复杂模型没有价值。

Issue #6 的 v1.0.0 至 v1.2.0 因证据合同问题否决，v1.4/v1.5 的 D1/D2 负结果也完整保留。v1.6.0 使用新的 collection/bootstrap seed、120 个全新隔离 trace、600 组原始观测和 60 fit/20 calibration/40 evaluation 时间拆分验证风险预算 D3。D3 24 次切换，累计 regret 2.082 ms，相对 H0 42.590 ms 下降 95.1%；P95 0 对 3.169 ms；matched-workload trace-cluster CI [-0.7232, -0.3069] ms；harmful 按全部 eligible decisions 为 1/80（1.25%），按实际 switches 为 1/24（4.17%），累计 gain/harm 为 41.480/0.972 ms，全部门禁通过。harmful gate 使用本次 replay 的经验比例而非比例置信上界，不作总体风险低于 5% 的人口推断。动作服务器并非同一热进程或克隆状态，action×process 混杂未消除；artifact 的 `paired_trace_cluster...` 只是不再更名的机器字段，不是 Trial Pair。故只称 matched-workload replay 中预测选择有效并授权同进程 monitored canary，不作动作因果、生产在线或跨环境收益声明。

Issue #6 的正式 replay 动作仍只包含 Direct、CUDA-managed Swap、transactional host Swap 与 Recompute；它不被事后改写。Issue #7 另行增加真实 CUDA Remap 和默认关闭的受限 Paged production adapter：Paged 只对 Qwen2.5-0.5B、FP16 KV、page 16、D64、单 token/batch 1、context ≤ 2048、完整 GPU offload 开放，超出能力范围在 KV mutation 前回退。v2.10 的 K2/K1 替换证据限定 context17；H10 另以来源绑定语料验证 64–2048 token 的 Paged/Direct 路径且得到未晋级负结果。Issue #7 v1.1 的 10 组 17-token 跨页 AB/BA 配对证明正确性与生产图接入，但 Paged client P95 相对 Direct 回退 6.78%，配对差中位数 +2.705 ms，bootstrap 95% 区间 [-1.185, +12.019] ms，故 +5% promotion gate 失败并保持 opt-in；未跨页的 v1.0 降级为 superseded。Issue #6 的 500 万次 choose 结果与 hash-bound artifact 仍位于 `results/research/h4-kv-action-v1.3.0/`，不冒充 Issue #7 在线收益。

## 7. H5：已观察到的反向因果链

强制 CacheFlow 相对 upstream 的 3 组 paired Latin 干预记录了：决策 +13、prefill chunk +23、prefill token -354、KV launch +2、copied bytes +20,066,300、CUDA Event +0.808 ms；Engine execute 汇总 -11,446 us，但 TTFT P95 +85.61 ms。

这是一条受限 workload 上的反例：更少 aggregate execute time 不保证更好请求尾延迟。它不证明普遍因果关系，且 100 ms GPU sampling 不能替代 Nsight。后续实验若不能同时观察“策略干预 → scheduler/action → CUDA mediator → request outcome”，不得宣称策略造成端到端变化。

## 8. 统一负结果规则

- 预注册之后不因结果不利而改 hypothesis、阈值、主指标或 workload；必要变更必须升版本并保留旧版本。
- 正确性失败优先于性能结果；fallback 计数必须可见，不能把 fallback 时间算作新动作成功。
- 报告 paired raw trials、effect size、不确定性和反号复验，不只报告最快一次或独立中位数之比。
- `existing-evidence` 与 `prospective` 严格分离。当前 JSON 校验禁止 prospective claim 填入 observed result。
- 单张 RTX 4050 的结论不外推到 A100/H100、多 GPU、prefill PagedAttention 或任意模型/dtype。

## 9. 与后续 Issues 的边界

- Issue #3 冻结实验协议、统计方法和确切 pass/fail thresholds。
- Issue #4 用 profiling 验证或否定 H2。
- Issue #5 规定并原型验证 H3 的受限 Paged Decode Attention。
- Issue #6 设计 H4 的统一代价模型。
- Issue #7 才允许把通过门禁的 Paged/Policy 接入生产路径。
- Issue #7 已完成受限生产接入，但性能门禁否决默认推广；“可执行”与“值得默认选择”是两个独立状态。
- Issue #8/#9 负责消融、外部有效性、制品和论文式报告。
