# ADR 0003：Paged 动作采用布局感知的混合执行

- 状态：Accepted
- 日期：2026-08-14

## 背景

H19 已证明自定义 K4 能在真实 non-unified KV 布局下执行 batch 1/2/4/8，但不能证明它优于 upstream Direct。batch 8 的吞吐中位变化为 -3.22%，原始 wave P95 回退 16.00%。NSYS 进一步把差距定位到有效注意力计算：context 1024、batch 8 时，upstream MMA 主 kernel 约 26.08 us，而 K4 主 kernel 约 40.54 us；merge 只有约 1.62 us，因此继续删除 host wrapper 或空 kernel launch 不能解决主要差距。

同时，实际 block table 存在两类布局：

1. 每个逻辑页在物理地址上连续排列。此时页间接寻址没有必要，自定义 K4 只会替换已经更快的 upstream MMA kernel。
2. 逻辑页映射到不连续物理块。此时 upstream contiguous kernel 不具备等价的页表语义，必须使用 K4 或 fail closed。

把两类布局都强制交给同一 kernel，不是 Paged Attention 的算法要求，而是执行路由缺少布局信息。

## 决策

Paged 保持一个高层动作，但在 KV mutation 前按布局选择执行路径：

- `physically_contiguous=true`：复用 upstream contiguous Flash Attention/MMA 图；
- `physically_contiguous=false`：进入自定义 Paged CUDA K4；
- capability、模型或页内连续性不满足：保持既有 fail-closed 回退。

`llama_paged_decode_layout` 在构造 block table 时计算 `physically_contiguous`。判断要求每个 live row 的相邻 block base 恰好相差 page size；短行 padding 不参与判断。连续页 fast path 默认启用，可用 `LLAMA_CACHEFLOW_PAGED_CONTIGUOUS_FASTPATH=0` 强制关闭，以便独立验收碎片化 K4。

混合执行必须保持可观测性分层：

- `paged_decode_contiguous_fastpath_calls_total/sequences_total` 只统计 upstream fast path；
- `paged_decode_calls_total` 与设备端 CUDA dispatch/sequences 只统计自定义 Paged 图；
- fast path 不得增加自定义 CUDA dispatch 计数，也不得被写成“Paged kernel 加速”。

## 算子诊断与否定结果

K4 的长上下文分区改为批次级统一几何：最大 block-table 容量不超过 512 token 时使用 64-token partition，否则使用 128-token partition。producer 与 merge 使用同一 partition size，修复异构 `[512,513]` batch 的潜在语义分裂，并在 context 1024 时把 grid 与 partial scratch 从 16 个 partition slot 减到 8 个。NSYS 显示空 CTA 删除没有降低有效 K4 时间，因此该改动只主张减少 grid/scratch，不主张时延加速。

实验性 K5 用 WMMA 计算 `QK^T` 和 `PV`，32 个 production CPU/CUDA oracle case 全部通过；但 context 1024、batch 8 的主 kernel 为约 46.69 us（256-token partition 为 48.69 us），慢于 K4 的约 40.54 us，因此 K5 不成为默认实现，只保留为可复现的否定候选。

## 验证

- layout 单元测试覆盖连续、反序页、页内非法、异长 batch 与乱序输入；
- K1-K5 各在独立进程中运行 32 个 production CPU/CUDA oracle case，覆盖 context 1–2048、batch 1/2/4/8、unified 与 non-unified/stream 布局；
- 强制 `FASTPATH=0` 的真实服务验收继续要求一个 batch-8 graph、24 个逐层 K4 dispatch、192 个 sequence-layer execution 与零 fallback；
- fast-path 实验必须要求 fast-path calls/sequences 完整、自定义 Paged CUDA 计数为零，并比较 token、cache length 与 top-64 分布。

## 结论边界

该决策消除“物理连续页仍强制使用较慢 K4”的生产负优化，但不证明碎片化 K4 严格优于 Direct，也不证明容量收益。正式晋级必须另行预注册混合路由非劣实验；碎片化场景的系统价值则应与需要 materialize/remap 的等价基线比较，不能用连续 KV 的 Direct 结果替代。
