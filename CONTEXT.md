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

Decode Attention 直接通过 Block Table 读取离散 KV Block、完成在线 softmax 与输出归约的执行方法，不先构造完整连续 KV。首个研究实现只承诺 Qwen2.5、FP16 KV、GQA、单 token decode、单 GPU，并保留正确性回退。

_Avoid_: 泛指所有分页 KV 管理；宣称支持 prefill、任意模型或多 GPU。

## KV Execution Action

一次 Serving Iteration 针对某个 KV 工作负载选择的具体执行方法。研究动作空间包括 Direct、KV Remap、Paged Decode Attention、Swap 与 Recompute；未实现的动作必须标记为候选而非可用能力。

## Copy-aware Policy

显式估算 KV 搬运字节、计算量、延迟和内存压力，并据此选择 KV Execution Action 的策略。它的目标不是永远避免复制，而是在观测条件下选择预期代价最低且满足安全约束的方法。

## Benefit Gate

基于在线观测特征和收益标签更新的 Ridge 回归门控器，用于决定某项优化是否值得启用。其预测必须受最小样本数、数值稳定性、置信约束和确定性回退保护。

_Avoid_: 强化学习策略（当前实现不是 RL）。

## Evidence Gate

把项目能力声明与可复现实验证据绑定的验收规则。只有在预注册工作负载、正确性对照、统计方法和原始结果都满足门槛后，能力才能从“候选/原型”升级为“已验证”。

## Research Baseline

研究实验中预先固定的对照实现或机制，必须记录源码版本、许可、模型与 KV 布局、硬件假设、构建/运行入口和可比性边界。`quantitative` 仅表示可在当前锁定环境中进行单变量定量比较；`conditional-kernel` 需要先满足布局和正确性条件；`related-work` 只用于解释设计，不进入本项目的性能主表。

_Avoid_: 把不同模型格式、运行时、调度器或硬件上的论文数字称为 baseline；把尚未在本机通过正确性门禁的外部实现称为 quantitative baseline。
