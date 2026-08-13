# CacheFlow Runtime Domain Glossary

## CacheFlow Runtime

面向消费级单 GPU 大模型推理服务的实验型运行时。它在保持正确性与可回退性的前提下，研究请求调度、Paged KV 生命周期、KV 数据搬运和 Decode 执行方法之间的联合决策。

_Avoid_: 把它称为通用分布式推理框架、完整替代 llama.cpp 或已经完成的 PagedAttention 系统。

## Serving Iteration

调度器收集当前活动请求、生成一次不可变执行计划、执行模型与 KV 操作并提交状态的最小事务边界。失败时，本轮未提交的状态不得泄漏到下一轮。

_Avoid_: batch step（无法体现计划、执行、提交的事务语义）。

## Iteration Plan

Serving Iteration 在执行前冻结的决策快照，包含被选请求、token 预算、KV 操作、执行方法及回退路径。执行组件消费计划，而不在执行过程中隐式改变调度决策。

## KV Block

Paged KV 管理的固定容量逻辑分配单元。它具有逻辑身份、物理驻留位置、引用计数和生命周期状态；逻辑块不等同于连续显存地址。

_Avoid_: page（仅在明确指代操作系统或 CUDA 内存页时使用）。

## Block Table

从请求的逻辑 token 区间到 KV Block 物理位置的有序映射。Decode 算子通过它读取非连续 KV，而不是假设每个请求拥有一段连续 KV 缓冲区。

## Prefix Share

多个请求复用相同、只读的完整前缀 KV Block。共享只表示物理存储复用，不允许一个请求的后续写入改变其他请求可见的数据。

## Partial-tail COW

当共享前缀的最后一个 KV Block 尚未填满且请求即将写入时，先复制该块的有效区间，再让写请求获得独占尾块的 Copy-on-Write 操作。

## KV Remap

在模型算子执行前，依据 Block Table 将离散 KV 数据显式重排为下游算子所需连续布局的执行方法。当前 CUDA 实现包含标量与 `uint4` 向量化路径。

_Avoid_: PagedAttention（KV Remap 仍然发生了显式搬运）。

## Paged Decode Attention

Decode Attention 直接通过 Block Table 读取离散 KV Block、完成在线 softmax 与输出归约的执行方法，不先构造完整连续 KV。“单 token decode”指每个活跃序列本轮各产生一个 query token，不等于整个 ubatch 只能有一个序列；生产布局、CPU reference 与 CUDA K1–K4 均具有 sequence 维，已验证 batch 1/2/4/8。当前仍只承诺 Qwen2.5-0.5B、FP16 KV、GQA7、单 GPU，并保留正确性回退。

_Avoid_: 泛指所有分页 KV 管理；宣称支持 prefill、任意模型或多 GPU。

## KV Execution Action

一次 Serving Iteration 针对某个 KV 工作负载选择的具体执行方法。研究动作空间包括 Direct、KV Remap、Paged Decode Attention、Swap 与 Recompute；未实现的动作必须标记为候选而非可用能力。

## Copy-aware Policy

显式估算 KV 搬运字节、计算量、延迟和内存压力，并据此选择 KV Execution Action 的策略。它的目标不是永远避免复制，而是在观测条件下选择预期代价最低且满足安全约束的方法。

## Benefit Gate

基于在线观测特征和收益标签更新的 Ridge 回归门控器，用于决定某项优化是否值得启用。其预测必须受最小样本数、数值稳定性、置信约束和确定性回退保护。

_Avoid_: 强化学习策略（当前实现不是 RL）。

## Paired-Delta Policy

从同一 matched-workload trace 的候选 KV Execution Action 与 H0 观测中学习完整动作成本差，并且仅在风险调整后仍有收益时推荐切换的阶段条件化策略。当前 H4 的各动作由独立固定模式服务进程采集，因此这里的 `paired` 是历史实现名，不构成上文定义的 Trial Pair，也不支持克隆状态因果解释；它在通过同进程 canary 前只属于离线候选。

_Avoid_: 绝对成本模型；在线强化学习；把跨进程 matched workload 称为 Trial Pair；已上线学习策略。

## Piecewise Paired-Delta Policy

在 Paired-Delta Policy 的同一带符号目标、安全上界和 H0 回退之上，按预注册上下文区间分别拟合与校准的 D2 策略。它避免把短上下文中 Recompute 的收益与长上下文中的损失平均成一个无效的全局线性关系。

_Avoid_: 在确认性 evaluation 数据上选择分界点；取消置信门禁；把离线通过写成已在线启用。

## Risk-Budgeted Paired-Delta Policy

D3 对低概率负优化显式设置上限、同时要求统计显著净收益的离线策略。它不再要求候选 harmful rate 等于相对自身必为零的 H0，而是联合约束配对 regret 置信区间、P95、5% harmful-rate budget，以及累计收益必须覆盖累计损失。

_Avoid_: 忽略单次负优化；只报告平均值；把 5% 预算解释成允许生产路径无监控退化。

## Evidence Gate

把项目能力声明与可复现实验证据绑定的验收规则。只有在预注册工作负载、正确性对照、统计方法和原始结果都满足门槛后，能力才能从“候选/原型”升级为“已验证”。

## Research Baseline

研究实验中预先固定的对照实现或机制，必须记录源码版本、许可、模型与 KV 布局、硬件假设、构建/运行入口和可比性边界。`quantitative` 仅表示可在当前锁定环境中进行单变量定量比较；`conditional-kernel` 需要先满足布局和正确性条件；`related-work` 只用于解释设计，不进入本项目的性能主表。

_Avoid_: 把不同模型格式、运行时、调度器或硬件上的论文数字称为 baseline；把尚未在本机通过正确性门禁的外部实现称为 quantitative baseline。

## Research Claim

研究章程中可被实验推翻的主张。每条 Research Claim 必须显式记录自变量、因变量、混杂因素、Research Baseline、预期机制、证伪条件、证据来源、适用边界和负结果处理，并区分 `prospective`、`limited-evidence` 与 `existing-evidence`；`limited-evidence` 表示已有可复现观测，但仍有关键机制指标或适用范围未闭合。

_Avoid_: objective（只表达目标而不要求可证伪）；把 prospective hypothesis 写成已经获得的 result。

## Research Charter

按版本冻结的一组 Research Claim 及其统一范围、证据状态和负结果规则。机器可读注册表负责拒绝无证伪条件、伪造 observed result 或引用未知 Research Baseline 的主张；人类可读文档解释研究问题之间的关系。

_Avoid_: roadmap（只描述工作顺序）；experiment protocol（后者负责冻结采样、统计与门槛，而不是研究主张本身）。

## Experiment Protocol

在观察确认性结果前按版本冻结的实验执行与分析契约，包含 workload、warm-up、Trial Pair、随机种子、计时边界、固定样本量、失效规则、效应量、置信区间、验收门槛和制品字段。

_Avoid_: benchmark script（只执行代码但不冻结分析决策）；根据已观察结果修改同一版本的门槛。

## Trial Pair

同一 configuration 和 nuisance block 内，各执行一次 baseline 与 candidate 的最小配对分析单位。两侧共享输入、layout、bytes、进程热状态和测量边界，执行顺序由预注册 seed 随机化并记录。

_Avoid_: 把两个独立运行集合的中位数事后相除；只删除 pair 中较慢的一侧。

## Profiler Mechanism Evidence

与无 profiler Trial Pair 分开采集、只用于解释机制的 NSYS/NCU 证据。NSYS 拥有 CUDA API、kernel、memcpy 与 synchronization 的时间线；NCU 在指标实际存在时拥有 DRAM/L2/occupancy 等硬件计数器解释。Profiler replay 或 tracing 下的 wall time 不进入性能主表。

_Avoid_: 把 profiler 插桩耗时当成自然延迟；只有 NSYS timeline 时声称 memory-bound、roofline 或 achieved occupancy。

## Effective Payload Throughput

逻辑 KV payload bytes 除以无 profiler CUDA-event 时间，用于描述同一语义工作量的有效处理速率。它包含实现效率但不等于实际 DRAM traffic 或显存硬件带宽。

_Avoid_: hardware bandwidth；没有 NCU DRAM counters 时把该值称为 bandwidth utilization。
