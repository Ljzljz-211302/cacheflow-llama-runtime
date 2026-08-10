# 受限 Paged Decode Attention：一手资料审计与 CUDA 原型设计

## 1. 结论先行

Issue #5 要实现的不是通用 PagedAttention，也不是把现有 KV Remap 换个名字，而是一个可证伪的、
decode-only CUDA kernel：输入单 token query 和非连续物理页上的 K/V，按逻辑页表直接计算

\[
O_h=\operatorname{softmax}(Q_hK_{g(h)}^T/\sqrt{D})V_{g(h)},
\]

其中 `g(h) = floor(h / group_size)`。第一版在 RTX 4050 Laptop (`sm_89`) 上验证，支持 Q/K/V FP16、
output 与累加 FP32、head dimension 64/128、page size 16、GQA、单 token autoregressive decode、完整 causal
history、单 GPU。它必须从页中直接读取并产生真实 attention output；不得先把 K/V 搬成连续 cache。

推荐的第一版 kernel 是 **one-CTA-per-query-head + FP32 online softmax**。它牺牲同一 GQA 组内的
K/V 复用，换取 batch=1 时足够的 CTA 数、实现边界清楚和较小的归约范围。只有 profiling 显示
GQA 重复读取是下一瓶颈后，才实现 one-CTA-per-KV-head 的协作版本；只有长 context 显示单 CTA
串行扫描不足，才实现 split-KV + state merge。这个顺序使每次优化都有可拒绝的机制假设。

最重要的模型边界是：**当前仓库真实服务模型 Qwen2.5-0.5B 不是 head_dim 128**。官方固定配置为
hidden size 896、14 Q heads、2 KV heads，因而 head_dim 为 64；Qwen2.5-7B 的官方固定配置为
hidden size 3584、28 Q heads、4 KV heads，因而 head_dim 为 128、GQA group size 为 7。
因此原型同时实现两条显式 specialization：当前本地模型忠实的 `14/2/64/16`，以及仅用于内核形状
研究的 Qwen2.5-7B `28/4/128/16`；后者不构成 7B 端到端服务证据。
现有 `32 layers × 8 KV heads × 128` 只能称为 Qwen 风格的 KV movement 合成 workload，不能称为
Qwen2.5-0.5B 模型形状。上述 head dimension 是由官方配置中的 `hidden_size / num_attention_heads`
推得的。[0.5B 固定配置](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/config.json)、
[7B 固定配置](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/a09a35458c702b33eeacc393d103063234e8bc28/config.json)
这里的 FP16 限定是 KV/Q/O 计算格式，不是说 7B 权重也必须 FP16；6 GiB 本机能否承载量化 7B 权重、
指定 context 和全部 runtime workspace 是后续真实接入必须单独通过的显存 gate。

## 2. 一手资料及其使用边界

| 来源 | 本项目采用什么 | 不从它推出什么 |
|---|---|---|
| [vLLM SOSP 2023 论文](https://arxiv.org/abs/2309.06180) | 逻辑连续 token 到非连续物理 block 的页表抽象；按需分配和共享 KV block 的系统动机 | 不搬用论文 GPU 上的吞吐数字，不假设分页 kernel 天然更快 |
| [vLLM 固定 revision 的 Paged Attention 设计文档](https://github.com/vllm-project/vllm/blob/62a86318de3655f970baf7c2ff89c81a72c1a1b3/docs/design/paged_attention.md) | 历史 kernel 的 K/V 分离布局、每个 `(sequence, head)` CTA、16-byte vec、mask、FP32 max/sum/value reduction 作为设计参照 | 该文档自己标注为 historical，固定 revision 已不再包含文中所述当前 CUDA kernel；不能把它描述成该 revision 的生产实现 |
| [FlashInfer 固定 layout 文档](https://github.com/flashinfer-ai/flashinfer/blob/3e96dfa35ee3f3b69f8f7ea04c4f44e1ca49703e/docs/tutorials/kv_layout.rst) 与 [paged KV 类型](https://github.com/flashinfer-ai/flashinfer/blob/3e96dfa35ee3f3b69f8f7ea04c4f44e1ca49703e/include/flashinfer/page.cuh) | NHD/HND 定义、CSR-like `indptr/indices/last_page_len`、页长公式和物理地址计算 | FlashInfer 不管理 page allocation；不能把其 allocator/scheduler 能力算成本项目能力 |
| [FlashInfer 固定 decode kernel](https://github.com/flashinfer-ai/flashinfer/blob/3e96dfa35ee3f3b69f8f7ea04c4f44e1ca49703e/include/flashinfer/attention/decode.cuh) 与 [online state](https://github.com/flashinfer-ai/flashinfer/blob/3e96dfa35ee3f3b69f8f7ea04c4f44e1ca49703e/include/flashinfer/attention/state.cuh) | GQA dispatch、16-byte 向量宽度、shared-memory staging、online state、partition-KV 和 merge 是可行设计 oracle | 不复制其性能结论；其 Linux/PyTorch/JIT 栈不是当前 Windows llama.cpp 的端到端基线 |
| [FlashInfer 固定 differential tests](https://github.com/flashinfer-ai/flashinfer/blob/3e96dfa35ee3f3b69f8f7ea04c4f44e1ca49703e/tests/attention/test_batch_decode_kernels.py) | page size 16、FP16、head_dim 128、GQA、partial last page 的参考构造，以及 `rtol=atol=1e-3` 的上游比较先例 | 上游 tolerance 不是本项目正确性的替代品；本项目仍须保留自己的逐 case 原始误差 |
| [FlashAttention 论文](https://arxiv.org/abs/2205.14135) 与 [FlashAttention-2](https://arxiv.org/abs/2307.08691) | tiled exact attention、safe online softmax、减少 HBM 中间量；FA2 对 work partition/非 matmul FLOP 的分析 | A100/H100 dense prefill 结果不能外推到 Ada batch-1 paged decode |
| [Qwen2.5 技术报告](https://arxiv.org/abs/2412.15115) 与上述两个官方 model config | Qwen2.5 使用 GQA；具体模型 shape 由固定配置决定 | “Qwen2.5”不是一个唯一 shape，不能混用 0.5B 和 7B 的 geometry |
| [Transformers 固定 Qwen2 attention source](https://github.com/huggingface/transformers/blob/5d6dff88d711635e143a5f27060758e3a066c730/src/transformers/models/qwen2/modeling_qwen2.py#L138-L193) | Q/KV head repeat mapping、`head_dim^-0.5` scale、causal mask 和 FP32 softmax 是模型语义参照 | Python eager reference 不代表本项目 CUDA kernel 的性能或 layout |
| [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) | 合并访存、shared memory、occupancy 只是待 profile 的实现因素 | 没有 NCU counter 时不能宣称 memory-bound、occupancy 受限或达到 roofline |

审计 revision 必须进入实验 manifest：vLLM `62a86318...`、FlashInfer `3e96dfa3...`、
FlashAttention `d7e4dba3...`、Transformers `5d6dff88...`、Qwen2.5-0.5B `7ae55760...`、
Qwen2.5-7B `a09a3545...`。

固定 FlashInfer 的 CUDA-core `DISPATCH_GQA_GROUP_SIZE` 只实例化 `{1,2,3,4,8}`，不含 Qwen2.5
0.5B 和 7B 都使用的 ratio 7。因此 stock CUDA-core path 不能作为 `28/4` 的 drop-in binary oracle；
只能借鉴算法/布局，或为 ratio 7 明确新增受审计 specialization。若把 shape 改成 ratio 4/8 来跑
FlashInfer，它只能成为 synthetic comparator，不能冒充 Qwen2.5 model-faithful 对照。

## 3. 原型支持合同

### 3.1 受限 fast-path shape

| 项 | 硬约束 |
|---|---|
| device | 本机 RTX 4050 Laptop，compute capability 8.9；其他设备不拥有本研究的性能结论 |
| phase | `q_len = 1` decode；不支持 prefill、chunked prefill 或 speculative multi-token verify |
| model-faithful shapes | 本地 0.5B：`14/2/64`；7B kernel shape：`28/4/128`；两者 `group_size=7` |
| dtype | Q/K/V 为 IEEE FP16；output、QK、softmax state 和 PV 均用 FP32 |
| paging | page size 固定 16 token；K/V 分离物理平面；页表索引为 `uint32/int32` |
| position | Q 和 cached K 已在模型 attention 层中完成 RoPE；kernel 内 `pos_encoding=NONE` |
| mask | 全历史 causal decode；无 ALiBi、sliding window、custom mask、logit soft-cap 或 attention sink |
| scale | 精确传入 `1/sqrt(head_dim)` 的 FP32 值；不得重复做 Q/K quantization scale |
| residency | Q/K/V、page table、length、output 均在同一 CUDA device；执行期间页表和页不可变 |
| output | `[batch, query_heads, head_dim]` FP32；一次调用每个 request 恰有一个 output token |

两条 shape 必须分栏报告。`14/2/64/16` 是当前 0.5B 的模型忠实内核几何；`28/4/128/16` 只说明
7B 模型几何下的 kernel 行为，不能冒充本机已加载 7B 权重或完成端到端服务。

### 3.2 物理数据布局

原型以单个 attention layer 为调用边界，使用独立 K/V plane 和 physical-page-outer NHD；生产接入时
由现有全层 block backend 提供目标层的 plane 偏移，而不是在 kernel 内重新拥有 allocator：

```text
K, V: fp16 [capacity_pages][16][kv_heads][head_dim]
Q:    fp16 [batch][query_heads][head_dim]
O:    fp32 [batch][query_heads][head_dim]

offset(page, token, kv_head, dim) =
    page  * (16 * kv_heads * head_dim)
  + token * (kv_heads * head_dim)
  + kv_head * head_dim
  + dim
```

这是“physical page outermost、page 内 NHD”。它保留现有整 block Copy/COW/Swap 的连续性，同时使
固定 layer 内相邻 FP16 head_dim 元素连续。K1 每个线程使用标量 FP16 load，不声明 16-byte vector
fast path，也不把额外 alignment 作为支持条件；向量化或 shared staging 必须作为后续独立候选测量。
7B shape 每层每页 K+V 为 `2*16*4*128*2 = 32768` bytes；0.5B shape 为 4096 bytes。原型只分配
单层输入；生产接入必须另行记录实际层数和总分配，防止把单层 footprint 当成端到端显存需求。

原型采用矩形页表与显式 context length，不能从未初始化页槽推断长度：

```text
page_table:      uint32 [batch][max_pages_per_sequence]
context_lengths:uint32 [batch], 1 <= value <= 16 * max_pages_per_sequence
physical_page = page_table[b][logical_token / 16]
token_in_page = logical_token % 16
```

只有 `ceil(context_lengths[b]/16)` 个已使用槽参与验证；这些槽的 physical page id 必须小于 capacity，
未使用槽可以保留 sentinel。不同 request 可指向同一只读 prefix page。空 context、超过矩形容量的
context 与已使用的越界页必须在 host plan 创建时拒绝，不能让 kernel “保护性地”读 page 0 后继续算。
CSR-like `indptr/indices/last_page_len` 是 FlashInfer 的参考布局，不是本原型已实现接口。

### 3.3 causal 与位置语义

调用时 `context_len` **包含当前 decode token 已追加的 K/V**。该 query 对逻辑 token
`[0, context_len)` 全可见，`[context_len, allocated_pages*16)` 全不可见。对 q_len=1 且 query 位于
最后一个 KV 位置，这等价于 FlashInfer decode 路径使用 non-causal mask：不存在未来的有效 KV，
但 last-page padding 仍必须被 length mask 排除。FlashInfer 固定 wrapper 也只在 planned q_len > 1
时选择 causal mask；单 token decode 选择 non-causal。

为了让 off-by-one 错误必然暴露，boundary tests 要把最后页的无效槽填成 NaN/Inf/大幅值，而不是 0。
输出出现非有限值即失败。第一版不在 kernel 内做 RoPE：Qwen 已旋转的 Q 与 cached K 是输入合同。
如果调用方传入未旋转 tensor、启用 sliding-window 或要求 ALiBi/custom mask，必须回退。

## 4. 数学与数值合同

对 query head `h`，`kv_head = h / 7`（整数除法，query heads 按连续组映射），对每个有效 token：

```text
s_i = float_dot(fp16(Q[h]), fp16(K[i, kv_head])) * float(1 / sqrt(head_dim))
```

不得用 FP16 reduction；不得在第一版使用 `--use_fast_math`。每个 tile 保持 FP32 online state
`(m, l, o[128])`。把当前 state 与新 tile state 合并时：

\[
\begin{aligned}
m' &= \max(m,m_t),\\
l' &= e^{m-m'}l + e^{m_t-m'}l_t,\\
o' &= e^{m-m'}o + e^{m_t-m'}o_t.
\end{aligned}
\]

最终 `O = o/l` 并保留为 FP32 output；后续生产 adapter 是否转换 dtype 属于独立接口决策。该可合并 state 直接对应 FlashAttention 的 safe online
softmax，也对应 FlashInfer `state_t::merge`；因此未来 split-KV 只能合并 `(m,l,o)`，不能先独立
normalize 各 partition 后平均。因为 fast path 保证 `context_len>0` 且输入有限，`l>0`，第一版不加
`1e-6` epsilon。epsilon 会改变数学语义并可能隐藏空输入 bug。

正确性 oracle 固定如下：

1. CPU 按逻辑页顺序读取同一逻辑 K/V，仅用于测试；dot、`exp`、sum、PV 与 CUDA 合同一致使用 FP32，
   expected output 保留 FP32。
2. 每个元素要求 `abs(actual-expected) <= 1e-3 + 1e-3*abs(expected)`，且两边均有限。这个门槛与
   FlashInfer 固定 FP16 paged-vs-contiguous tests 的 `atol=rtol=1e-3` 对齐，但本项目仍输出
   `max_abs`, `max_rel`, failing index 和 seed，不能只保存 pass/fail。
3. 同一逻辑 K/V 在 identity、reverse 和 seeded-random physical page placement 下的输出都要分别
   对 oracle 通过；不要求不同 CUDA reduction order 的 FP16 output bitwise 相同。
4. 另做 invariant：全零 Q 应得到有效 V 的逐 token 均值；单 token context 应精确等于 V（FP16
   round-trip）；相同 K logits 应产生均匀权重；页外 poison 不能影响输出。

如果 `1e-3/1e-3` 在预注册随机域中失败，不能看到结果后放宽。先将失败保留为负结果并定位是
FP16 cast、reduction order、近零相对误差还是索引错误；任何新 tolerance 必须升 protocol version。

## 5. 第一版 kernel 及后续可证伪选项

### K1：one CTA per `(request, query_head)`（第一实现）

```text
grid = (batch, query_heads)
block = head_dim threads（D64 为 64，D128 为 128）
```

- CTA 一次加载一条 D64 或 D128 Q；由 warps 轮转 logical pages/tokens，经 page table 读取对应 `kv_head`。
- FP16 数据以 16-byte vector load 进入寄存器或小型 shared tile；QK/PV 和 state 全为 FP32。
- warp 先形成局部 `(m,l,o)`，再通过 shared memory 合并 warp states，CTA 最后写一个 FP32 head。
- 优点：batch=1 有 28 个 CTA，能在本机多个 SM 上并行；索引和归约边界最简单。
- 代价：同一 KV head 的 7 个 query heads 各自读取相同 K/V，GQA reuse 为零。

K1 是推荐起点，因为它同时满足 Issue #5 的“直接读非连续页”和“真实 attention output”，并把后续
优化问题变成可观测选择，而不是一次写成无法归因的大 kernel。

### K2：one CTA per `(request, kv_head)`，组内复用 K/V

一个 CTA 同时处理 7 个 query heads，K/V tile 只从 global memory 取一次，然后被 7 组 QK/PV 使用。
它可能降低 bytes，但 batch=1 只有 4 个 CTA，且 7 组 online output state 会增加 registers/shared
memory 和归约工作。只有 K1 的 NCU（若可用）或受控 byte/latency slope 支持“重复 KV load 是主因”
时才实现 K2。不能仅凭 GQA 理论上可复用就宣称更快。

#### 已落地的 K2-T2

最终生产实现没有直接采用“一个 CTA 吞下全部 7 个 query heads”的草图，因为 batch 1 时每层只剩 2 个 CTA，并行度过低。K2-T2 选择折中：每个 CTA 含两个 warp、各负责一个 query head；GQA7 每个 KV head 切成 4 个 tile，最后一个 tile 只有一个活跃 warp。每个 CTA 先把当前 KV head 的 K/V 从 global memory 装入 shared memory 一次，两个 query heads 复用这份数据；因此每 KV head 的装载由 K1 的 7 次降为 4 次，而不是降为 1 次。K 写成 `[dimension][token]`，使同一 warp 的 lane 读取不同 token 时落入不同 bank；V 保持 `[token][dimension]`，便于每个 lane 累加两个 output dimension。

logit 计算把一个 token 分给一个 lane；lane 串行遍历 D64 做点积，随后用 `__shfl_down_sync` 求 warp max 与 sum，得到数值稳定的 softmax 权重。最后每个 lane 从其他 token lane 广播权重并累加 V。相比 K1 的逐 token CTA barrier 与 14 个 query-head CTA，K2-T2 同时减少重复 KV 装载、CTA 数和 block-wide synchronization。这里的“复用”由代码干预与 K1/K2 trace 支持；由于 NCU counter 不可用，不能进一步声称硬件 DRAM bytes、occupancy 或 memory-bound 原因已被直接测量。

正式 v2.10 在 Qwen2.5-0.5B `14Q/2KV/D64`、page16、context17 的重复 cached 请求中完成 30 组同进程随机配对：每 arm 保留 16 条原始请求计时，每 variant 共 480 条；输出一致、累计 600 次请求级 Paged graph entry/variant、0 fallback。请求级 median 为 6.900/6.938 ms（K2 回退 0.55%，配对簇 bootstrap 回退 95% 上界 2.86%），P95 为 30.102/30.559 ms（回退 1.52%），相同 480 次目标 kernel 总时长由 8.174 降至 4.051 ms（-50.44%）。该门槛只允许 K2 替换同一 Paged 路径的 K1；它不证明 Paged 已优于 Direct，也不允许把 context17 性能结论外推到 host 已验证但未纳入该试验的 context18--32，更不外推到 K3 所针对的长 context。

### K3：split-KV + merge state

把长 context 分为若干连续逻辑 token partition；第一 kernel 每个 partition 输出 FP32 `(m,l,o)`，
第二 kernel 用第 4 节公式合并。FlashInfer 固定 decode source 同样在长序列/临时 workspace 可用时
选择 partition-KV 并 merge states。候选 partition sizes 为 `{256, 512, 1024}` token，但阈值必须由
本机 paired experiment 决定。K3 增加一次 kernel launch、临时显存和 state traffic，因此必须保留
短 context 反例。

### K0：仅用于诊断的两遍 logits 版本

若 K1 correctness 无法迅速定位，可保留一个非性能基线：kernel A 将有效 FP32 logits 写到 scratch，
kernel B 做 stable softmax × V。它仍直接读取 paged K/V，因此是合法 correctness prototype；但它会
产生 O(context) scratch 和额外 HBM traffic，不能成为主性能候选，也不能称为 FlashAttention-style。

## 6. fail-closed 与 fallback

原型的 `llama_paged_decode_supported` 返回布尔值与静态 reason 文本；`plan_create`/launch 返回 CUDA
error。当前明确拒绝的条件为：

```text
zero_batch_or_page_capacity
unsupported_head_dim
unsupported_page_size
non_divisible_gqa_shape
invalid_attention_scale
invalid_context_length
invalid_used_page_id
null_plan_or_tensor
```

设备、dtype、phase、RoPE/mask mode 属于此 C++ seam 的静态调用合同，而不是可在裸指针 API 中动态识别
的字段；生产接入若需要可观测 fallback counter，必须在拥有这些语义的 runtime adapter 中实现。

原型 benchmark 中默认行为是 **fail closed**：返回 non-zero，不产生未初始化 output。未来服务接入才
允许 fallback，且 fallback 必须是现有已验证的 contiguous attention 路径：必要时先按页表 Remap，
再执行上游 attention。fallback latency、copy bytes、reason 和 request id 必须单独记录；不得把
fallback 样本计入 paged fast-path 成功或 kernel latency。CUDA launch 后立即检查 launch error，
在测试/benchmark 边界同步检查执行错误；一旦错误，该 pair 整体 invalid，不只删除出错 arm。

页共享只读是允许的；任何 COW/append 与 attention 并发必须由 runtime event/stream dependency 保证。
kernel 本身不增加 reference count，也不拥有 page lifetime。page table 或 page 在 kernel 运行期间
被释放/改写属于调用方违约，不得靠 defensive offset 掩盖。

## 7. differential test matrix

必须在性能实验前全部通过：

| 维度 | 固定 cases |
|---|---|
| context boundary | `1, 15, 16, 17, 31, 32`，覆盖页前、页尾和跨页首 token |
| page placement | identity 与 seeded fragmented；显式 `{3,1}` 非连续两页案例 |
| last page | 未使用物理页和页尾槽以大幅值 poison，输出不得受影响 |
| value invariants | single-token、zero-Q mean、跨页逐 token 均值 |
| random | D64 与 D128 各一个固定 seed，FP16 Q/K/V 含正负值，均对独立 CPU oracle |
| batch | `1, 2` correctness（含 ragged `31/49`）；`1, 4` 性能实验 |
| invalid | head_dim、page size、非整除 GQA、零 context、已使用越界页和 null launch |

测试源码固定保存 seed、logical-to-physical mapping 与 context lengths；失败时打印误差索引和数值。
output 前后 guard 是本 Issue 的越界 gate。Compute Sanitizer 与 shared-prefix 并发属于生产接入前的独立
强化项；未运行时必须明确为 limited，不能用 guard test 冒充 Sanitizer 已通过。

## 8. profiling 与下一设计决策

### 8.1 公平比较对象

本原型的两种 action 分开计时，不能偷换边界：

1. `contiguous-attention`：K/V 已连续，只计上游 contiguous decode attention；是算法下界参照。
2. `paged-attention-K1`：直接从页读到 output；不含 allocator/scheduler，但包含页表读取。

`remap+contiguous-attention` 是后续生产接入所需的端到端 action baseline，本 Issue 不用单层 microbenchmark
冒充已有上游完整 attention 路径。

所有 arm 使用逐元素相同的 FP16 Q/K/V、相同 logical token order、两条固定 shape 和相同 stream，先过
oracle。K/V physical placement 固定为 identity/seeded fragmented；每个 pair 内执行顺序
seeded random，至少 20 pairs，warm-up 不进样本。主效应使用无 profiler CUDA event；host enqueue
和同步后的 end-to-end 分别报告。Profiler replay 的 duration 只解释机制，不进入主性能结论。

### 8.2 预注册 regimes

```text
batch:          1, 4
context:        16, 17, 256, 1024
placement:      identity, seeded-fragmented
implemented:    K1-D64-64threads, K1-D128-128threads
implemented:    K2-T2-D64-GQA7-short-context
future only:    K3-split{256,512,1024}
```

6 GiB 设备上先计算并记录 input/output/workspace/allocator peak；无法满足显存 gate 的 case 作为
`resource-limited` 保留，不缩小 context 后冒充同一 regime。性能主张只限本机 `sm_89`。

### 8.3 NSYS/NCU 决策表

沿用 [H2 profiling 协议](cuda-profiling-causal-chain.md)：NSYS capture 只包正式 region，记录 launch、
memcpy、同步和 duration；NCU 若仍因 `ERR_NVGPUCTRPERM` 或 driver incompatibility 不完整，就禁止
memory-bound、DRAM bandwidth、L2 hit、occupancy 和 roofline 主张。

| 观测 | 下一步 | 可以说什么 |
|---|---|---|
| K1 随 context 近线性增长，K2 在相同输出下稳定降低 kernel time，且 NCU 证实 DRAM bytes/sector 降低 | 保留 K2，继续检查 registers/occupancy | GQA 组内 KV reuse 在该 regime 有因果支持 |
| K2 bytes 降但 kernel time 不降或回退 | 保留负结果，不接入 K2 | 复用收益被并行度/归约/资源代价抵消 |
| 长 context K1 明显变慢，K3 降低 event time且 merge launch 代价已包含 | 为长 context 设候选阈值，再做独立确认 | split-KV 在已测边界有效 |
| K3 短 context 回退 | 必须保留，支持阈值策略 | split 并非普遍优化 |
| random placement 比 identity 慢 | 只能称 placement-associated effect；有 NCU L2 指标后才讨论 cache mechanism | 页布局可能参与性能边界 |
| paged kernel 快但 `remap+contiguous` 端到端更快 | Paged 不进入该 regime 的 action policy | 单 kernel 优势不足以支持动作选择 |

如果 NCU 不可用，仍可根据无 profiler paired latency、NSYS launch 数和人工控制的 intervention 决定
“下一个应实现 K2 还是 K3”，但结论必须写成受限关联/干预结果，不能补写硬件 counter 故事。

## 9. Issue #5 的严格交付门槛

- 非连续、reverse 和 random physical pages 上真实计算 attention output；源码路径中不存在先 gather
  全量 K/V 到 contiguous buffer 的步骤。
- 第 7 节全部 differential/invalid/poison cases 通过，原始 seed/mapping/error 可审计；主 tolerance
  固定 `atol=rtol=1e-3`。
- 所有不支持 shape/mode fail closed；若服务层启用 fallback，reason/cost/count 可见且不混入 fast path。
- 分别覆盖 D64/D128、边界/中/长 context、batch 1/4 和 identity/fragmented placement；必须报告
  neutral 与 loss，不要求 universal speedup。
- 每个优化陈述都链接：raw inputs/page table → correctness record → no-profiler paired result → 对应
  NSYS/NCU trace（若 NCU 不完整则明确 limited）。
- 报告明确区分：已实现的 Qwen2.5-7B 模型几何 kernel shape、已实现的本地 0.5B 模型几何，以及尚未
  完成的生产服务 dispatch；在真实请求实验前，不得宣称已经加速现有 0.5B 用户路径。
- 本 Issue 只完成原型和 design decision。生产 dispatch、在线 policy 和用户请求接入仍由后续 Issue
  gate；不得因 microbenchmark 通过就默认启用。

## 10. 给实现者的最小顺序

1. 先把第 3 节合同变成 POD descriptor、host validator 和独立 CPU FP32 oracle。
2. 先写 invalid/boundary/poison/random differential tests，确认它们在没有 kernel 时失败。
3. 实现 K1 online-softmax，分别实例化 D64/64 threads 与 D128/128 threads；不启用 fast math，不做 K2/K3。
4. correctness、guards、Sanitizer gate 通过后，才运行第 8 节无 profiler paired benchmark。
5. 用 NSYS（以及条件允许时 NCU）选择 K2 或 K3；一次只改变一个机制。
6. 保留失败候选和反例，形成 Issue #5 报告；不要在本 Issue 接入默认服务路径。
