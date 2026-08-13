# ADR 0002：Paged Decode 使用序列感知的批量执行模型

- 状态：Accepted
- 日期：2026-08-13

## 背景

旧实现把“单 token decode”错误地等同于“整个 batch 只有一个 token”：布局对象只有一张 block table 和一个 context length，CPU oracle 固定读取第 0 行，CUDA grid/scratch 没有 sequence 维，服务端也只在 `n_tokens == 1` 时选择 Paged。这个限制不是 Paged Attention 的算法要求，而是早期实现为了先验证单序列数学正确性留下的架构简化。它使系统无法利用连续批处理把多个用户当前各自的一个 decode token 合进同一次图执行，也无法验证分页 KV 的主要系统价值——在显存更紧凑时维持更大的并发集合。

## 决策

Paged Decode 的基本工作单元改为“每个活跃序列一个 query token”，一次 ubatch 可以包含多个不同序列：

1. 布局使用二维逻辑模型。`block_table[sequence][logical_block]` 以行主序展平保存，每行用 `0` 补齐；`context_lengths[sequence]` 独立记录各序列的有效历史长度，因此补齐项永远不参与寻址，也不能再用第 0 个序列的长度裁剪其他行。
2. KV cache builder 从 ubatch 的每个唯一 sequence 读取实际 cell 元数据与 pending destination，逐序列检查逻辑位置连续、页内物理 cell 连续和 context 上限。任一序列不满足条件时，整个图在 KV mutation 前 fail closed，避免同一 attention 图混用 Direct/Paged 语义。
3. GGML CPU reference 用 `query_index + queries_per_stream * stream` 选择 context length 和 block-table row；多 query 的 Paged reference 禁止落入只适用于共享连续 KV 的 tiled 快路。这样同时覆盖 unified KV 的 `[D,B,H,1]` 和服务默认 non-unified KV 的 `[D,1,H,B]`，不会把 stream 维中的序列全部错误映射到第 0 行。
4. CUDA K1–K4 的 grid 均加入 sequence 维。Q/output 根据实际 batch 所在维选择 stride，non-unified K/V 额外使用 stream stride；每个 kernel 从自己的 table row 与 context length 读取 KV。K3/K4 的 partition scratch 也加入 sequence 维，防止不同请求覆盖同一 online-softmax 状态。
5. 服务只在 ubatch 满足“一 token/唯一 sequence”且每个 token 都有待执行 Paged 决策时标记 Paged topology。prefill、同一序列多 token、混合动作或 unsupported shape 保持原有回退，不用局部 relabel 冒充批量 Paged。
6. 累计指标除 graph call/fallback 外，新增 processed sequences 与 observed max batch。这样 `8` 次 Paged 决策不能仅凭 `8` 次单序列 graph call 冒充批量执行。

## 不变量

- `context_lengths.size()` 等于 query batch，block-table 行数与之相同。
- 每个序列的逻辑 block 必须完整覆盖自己的 context，行尾 padding 永远不参与寻址。
- CUDA 输出保持 GGML 的两种合法布局：unified 为 `[head_dim, query_head, sequence, 1]`，non-unified 为 `[head_dim, query_head, 1, sequence]`；实现分别选择 `nb2`/`nb3` 作为 sequence stride，sequence 不是 head 的附加索引。
- partition scratch 的地址包含 sequence、query head 和 partition 三个坐标。
- Direct 与 Paged 对相同 prompt、seed、采样参数逐字输出一致；Paged arm 所有序列均被计数、零 fallback，且 `max_batch >= 2`、graph calls 小于请求数。

## 验证

- 纯布局测试覆盖异长序列、不同物理页、行 padding、非连续页内 cell 与非法布局。
- production `GGML_OP_FLASH_ATTN_EXT` 以 CPU reference 对照 CUDA K1/K2/K3/K4；每个 variant 均覆盖既有 24 个 batch-1 边界 context，并分别新增 unified 与 non-unified 的 batch 2/4/8 异长、跨页、反序物理页用例，共 30 个 case。全验收为每个 variant 启动独立进程，避免进程静态 selector 污染。
- Compute Sanitizer 对 batch 8 执行 memcheck、对 batch 4 执行 racecheck。
- `scripts/run_paged_batch_acceptance.py` 在默认 non-unified 的真实 `llama-server -np 8` 中预热 8 个不同 resident prefix，经 barrier 同时发出请求。除 Direct/Paged 输出和 cache length 逐项相等、零 fallback 外，还要求 CUDA 后端实际产生 24 个逐层 Paged dispatch、192 个 sequence-layer dispatch，并比较每请求 top-64 分布（至少 48 个共同 token，最大 logprob 误差不超过 1.0；本次为 55 与 0.6900）。graph admission 指标与 CUDA dispatch 指标分开，不能再以前者冒充 kernel 执行。工件绑定完整运行时 DLL/EXE、模型哈希、vendor revision、工作树 overlay 与原始日志。

## 边界与后果

该决策消除了 batch=1 的架构限制，但不把功能验收解释为性能晋级。当前正式 H7/H13 Direct/Paged 性能实验仍是 batch=1 的历史结论；多序列 batch 是否因容量、更高 occupancy 或调度摊销而优于 Direct，必须使用新的并发/显存/SLO 协议测量。已验证的服务功能规模为 8 个并发序列；prefill、多 token/sequence、任意模型、多 GPU和跨请求 continuous-batching 性能上界不在本 ADR 的已证明范围内。
