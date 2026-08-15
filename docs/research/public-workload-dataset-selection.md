# llama.cpp / AI Infra 公开数据集选型与实验边界

> 状态：数据协议设计稿  
> 目标：用公认、可追溯的公开数据替换“项目内文档拼 prompt”作为主要外部证据，同时保留少量合成张量只做算子正确性测试。  
> 适用范围：llama.cpp 服务性能、Paged Attention、KV Cache、continuous batching、调度策略和模型质量回归。  
> 不适用范围：模型训练数据质量、训练后能力提升或真实商业用户采用率。

## 1. 结论先行

可以，而且应当引用公开数据集。但不存在一个同时公开真实 prompt、真实到达时间、真实输出长度和质量标签的单一 LLM serving 数据集。最稳妥的方案是让不同数据集各自回答一个它真正能回答的问题：

1. **算子边界层**：保留程序生成的 Q/K/V、页表和形状，仅用于 correctness，不作为 workload 真实性证据；
2. **公开内容层**：用 LMSYS-Chat-1M 提供真实对话内容，用 LongBench/LongBench-E 提供长上下文内容和质量标签；
3. **公开流量层**：用 BurstGPT v2.0 作为主要 arrival/concurrency trace，用 Azure LLM Inference Trace 2024 做第二来源复核。

推荐的主组合是：

| 证据问题 | 主数据集 | 备用/复核 | 能宣称什么 |
|---|---|---|---|
| 普通对话内容是否来自真实用户 | LMSYS-Chat-1M | ShareGPT 仅作兼容基线 | 内容与多轮结构来自公开真实对话 |
| 长上下文优化是否损伤答案质量 | LongBench-E | LongBench v2（硬件允许时） | 在标准任务和标准标签上的质量差值 |
| 到达率、突发性、并发和长度压力是否真实 | BurstGPT v2.0 | Azure LLM Inference Trace 2024 | 到达和 token 长度分布来自公开生产 trace |
| CUDA 页表和边界语义是否正确 | 确定性合成张量 | 随机属性测试 | 在覆盖形状内通过数值 oracle；不能称真实数据集表现 |

**最重要的口径限制**：把 BurstGPT 的时间戳与 LMSYS 的文本按长度匹配，得到的是“公开 trace 驱动、公开语料填充的合成 replay”，不是某个真实线上系统的端到端请求记录，也没有恢复内容与到达时间的真实联合分布。

## 2. 为什么 Infra 项目仍然需要数据集

监督学习经常把数据写成 `(输入, 标签)`；推理基础设施的数据单位则通常是一条请求事件：

```text
(arrival_time, prompt/content, input_tokens, requested_output_tokens,
 session_id, model, request_type)
    ->
(success, actual_output_tokens, TTFT, TPOT/ITL, E2E latency,
 queue_time, KV occupancy, route, numerical/model output)
```

因此 Infra 不是“不需要数据集”，而是数据集同时承担三种职责：

- **语义内容**决定 tokenizer 后的真实 token 序列、语言、轮次和前缀结构；
- **流量过程**决定请求何时到达、同时存在多少请求、是否突发以及 KV 压力如何变化；
- **标签或基线输出**决定系统优化有没有破坏数值正确性或模型任务质量。

单一来源通常只覆盖其中一部分，所以实验必须把“来源真实”与“组合后仍是合成实验”同时写清。

## 3. 数据集逐项核对

### 3.1 BurstGPT v2.0：主要生产流量 trace

**数据所有者与一手来源**

- 官方仓库：[HPMLL/BurstGPT](https://github.com/HPMLL/BurstGPT)
- 固定版本：[Release v2.0](https://github.com/HPMLL/BurstGPT/releases/tag/v2.0)
- 论文：[BurstGPT: A Real-World Workload Dataset to Optimize LLM Serving Systems, KDD 2025](https://doi.org/10.1145/3711896.3737413)
- 许可：[CC BY 4.0](https://github.com/HPMLL/BurstGPT/blob/main/LICENSE)

官方 README 将其描述为由 Microsoft Azure 支撑的真实 LLM serving workload。`BurstGPT_1` 与 `BurstGPT_2` 覆盖连续 121 天、约 529 万行；`BurstGPT_3` 另覆盖连续 110 天、约 534 万行。正式实验应绑定 v2.0 release 和下载文件 SHA-256，不能只写“用了 BurstGPT”。

**官方字段**

| 字段 | 含义 | 本项目用法 |
|---|---|---|
| `Timestamp` | 从第一天 00:00:00 起计算的请求提交秒数 | 保留相邻事件间隔，重放 arrival 和 burst |
| `Session ID` | 会话 ID；只在 conversation mode 中存在 | 保留会话内先后顺序；不可据此恢复 prompt |
| `Elapsed time` | 从提交到完整响应结束的时间，不是 TTFT | 仅描述源系统现象；不能直接当本地 llama.cpp 对照延迟 |
| `Model` | ChatGPT/GPT-3.5 或 GPT-4 | 分层统计；不假设与本地 Qwen 成本相同 |
| `Request tokens` | 请求 token 数 | 本地 tokenizer 下构造/筛选长度相近的输入 |
| `Response tokens` | 响应 token 数；失败记录可能为 0 | 作为目标 generation length 或分层变量 |
| `Total tokens` | 请求与响应 token 总数 | 完整性校验，不应独立参与模型拟合以免共线 |
| `Log Type` | `Conversation log` 或 `API log` | 分层报告会话/API workload |

**明确没有的内容**

- 没有 prompt 或 response 正文；
- 没有正确答案、任务标签或模型质量分数；
- `Elapsed time` 不是 TTFT，也不是在本项目硬件上的 latency；
- token 数来自源服务 tokenizer，不能保证等于 Qwen/llama.cpp tokenizer 的 token 数。

**适合评价**

- exact-arrival 或按预注册比例缩放后的 request replay；
- RPS、并发、队列长度、突发性和周期性；
- TTFT、TPOT/ITL、E2E latency 的 P50/P95/P99；
- request/output/total token throughput、成功率、OOM/429；
- KV 占用、页分配、碎片、eviction/preemption 和 batch size 随时间变化；
- 同一 session 内的请求顺序，但不能凭 Session ID 宣称真实 prefix reuse。

**局限与实验口径**

源 trace 中的 GPT-3.5/GPT-4 与本地 Qwen2.5-0.5B 的模型结构、tokenizer、部署规模和响应策略不同。它能验证“算法在真实 arrival/length 形状下怎样表现”，不能证明“复现了 Azure 的绝对延迟或资源规模”。如果缩放时间戳，应报告比例，例如 `time_scale=0.1` 表示把间隔压缩为原来的 10%，并将原始 replay 与缩放压力实验分开。

### 3.2 Azure LLM Inference Trace 2024：独立生产 trace 复核

**数据所有者与一手来源**

- 官方说明：[Azure LLM inference trace 2024](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md)
- 官方文件：[Code trace](https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_code_1week.csv)、[Conversation trace](https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_conv_1week.csv)
- 配套论文：[DynamoLLM, HPCA 2025](https://arxiv.org/abs/2408.00741)
- 许可：[CC BY](https://github.com/Azure/AzurePublicDataset/blob/master/LICENSE)

该数据是多个 Azure LLM inference service 在 2024 年 5 月 10 日至 19 日 trace 的样本，官方分别提供 code 和 conversation 一周 CSV。

**官方字段**

| 字段 | 含义 | 本项目用法 |
|---|---|---|
| `TIMESTAMP` | invocation time | 复现到达间隔和并发过程 |
| `ContextTokens` | 上下文 token 数 | 输入长度分层 |
| `GeneratedTokens` | 生成 token 数 | 输出长度分层或目标输出长度 |

**明确没有的内容**

- 因客户隐私要求，官方明确不提供 prompt 正文；
- 没有 session ID、模型名、质量标签和逐请求 TTFT/TPOT 字段；
- 它比 BurstGPT 字段更少，因此更适合做跨来源 robustness check，而不是替代 BurstGPT 的全部分析。

**适合评价**

用与 BurstGPT 相同的 runner、同一硬件和同一指标，在 code/conversation 两个 trace 上分别重放。如果某项调度收益只在 BurstGPT 成立而在 Azure 两类 trace 中消失，应报告 workload-sensitive，而不是挑选有利 trace。

### 3.3 LMSYS-Chat-1M：主要真实对话内容来源

**数据所有者与一手来源**

- 官方数据卡：[lmsys/lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)
- 论文：[LMSYS-Chat-1M, ICLR 2024](https://arxiv.org/abs/2309.11998)
- 访问方式：Hugging Face gated dataset；登录后提交姓名、邮箱、单位和国家并接受专用协议。

官方数据卡称该数据包含 100 万条真实对话，来自 2023 年 4 月至 8 月 Vicuna demo 与 Chatbot Arena，涉及 25 个模型、约 21 万个独立 IP 和 154 种语言。平均每个样本 2.0 turns，平均 prompt/response token 数分别为 69.5/214.5；这些统计使用数据作者口径，正式 replay 仍应使用本地目标 tokenizer 重新计数。

**数据卡字段**

| 字段 | 含义 | 本项目用法 |
|---|---|---|
| `conversation_id` | 对话标识 | group split，避免同一对话跨 train/eval |
| `model` | 原回答模型 | 描述性分层，不当作本地模型标签 |
| `conversation` | OpenAI API 风格 `role/content` 消息序列 | 真实文本 prompt、多轮结构和前缀内容 |
| `turn` | turn 数 | 单轮/多轮分层 |
| `language` | 检测语言 | 中英文及多语言分层 |
| `openai_moderation` | moderation 类别、分数和 flagged | 合规过滤；不是任务正确性标签 |
| `redacted` | 是否经过姓名脱敏 | 敏感性和脱敏影响分层 |

**许可与发布限制**

它不是普通 MIT/Apache 数据下载。专用协议允许在合规条件下为研究或商业研发使用，但禁止把数据集向第三方分发、复制、披露、再授权、托管或嵌入，并要求不得尝试识别个人；数据所有者还保留要求删除数据的权利。因此：

- 原始文件和抽取后的正文不得提交到本项目 Git；
- 只提交下载说明、转换代码、row-id/hash 清单和不含正文的统计；
- 公开报告引用数据卡与论文，不能把处理后的对话附件打包发布；
- 使用前执行 moderation/PII 过滤，并记录过滤规则与数量。

**明确没有的内容**

- 没有 request arrival timestamp；
- 没有真实服务 latency、TTFT/TPOT 或 GPU 指标；
- assistant response 是历史模型输出，不是保证正确的 ground truth；
- 数据卡明确说明没有做 benchmark decontamination，可能含公开基准题；
- PII 脱敏可能误改文本。

**适合评价**

- 真实 prompt token 长度、语言、轮次和内容多样性；
- tokenizer/content-sensitive 的 prefill 与 decode correctness；
- 多轮会话的确定性 replay；
- Direct 与 Paged 在同一文本、seed 和采样参数下的 token/logprob 一致性；
- 不适合单独验证 arrival、concurrency 或答案准确率。

### 3.4 LongBench / LongBench-E：长上下文内容与质量标签

**数据所有者与一手来源**

- 官方仓库：[THUDM/LongBench](https://github.com/THUDM/LongBench)
- v1 数据说明：[LongBench README](https://github.com/THUDM/LongBench/blob/main/LongBench/README.md)
- 论文：[LongBench, ACL 2024](https://aclanthology.org/2024.acl-long.172/)
- 仓库许可：[MIT](https://github.com/THUDM/LongBench/blob/main/LICENSE)

LongBench v1 是中英双语、多任务长上下文 benchmark。标准化样本包含：

| 字段 | 含义 |
|---|---|
| `input` | 问题或任务指令 |
| `context` | 长文档、多文档、代码或 few-shot 上下文 |
| `answers` | 可接受标准答案列表 |
| `length` | 中文按字符、英文按词统计的前三项总长度；不是本地 tokenizer token 数 |
| `dataset` | 子任务名 |
| `language` | 语言 |
| `all_classes` | 分类任务类别；其他任务为空 |
| `_id` | 样本 ID |

官方评测脚本不是统一使用“准确率”，而是按任务选择 QA F1、中文/英文 ROUGE、分类分数、检索分数、计数分数或代码相似度；LongBench-E 还分别报告 `0–4k`、`4–8k` 和 `8k+` 长度桶。官方 `dataset2maxlen.json` 为不同任务规定最大生成长度。

**适合评价**

- 长上下文 prefill 和 decode 性能；
- Direct 与 Paged 的任务分数差值，而不只是 token 是否一致；
- 按语言、任务和上下文长度分层的质量保持；
- 长度增长下 TTFT、峰值 KV 内存、页数、碎片与吞吐变化。

**正确用法**

1. 使用同一 prompt template、同一解码参数分别运行 Direct/Paged；
2. 先运行官方评测脚本得到两臂任务分数；
3. 报告 `score_paged - score_direct` 及按 `_id` 配对 bootstrap 区间；
4. 同时保存逐样本输出一致率、最大 logprob 差、TTFT/TPOT 和峰值显存；
5. 任何截断都记录“原始长度、Qwen tokenizer 长度、实际保留长度、截断策略”；
6. 超过模型或硬件能力的样本不得静默删除，应列入 excluded manifest 和覆盖率分母。

**局限**

- 没有到达时间、并发和真实服务会话；
- `length` 不是 Qwen token 数；
- benchmark 内容分布不等于生产聊天流量；
- 仓库 MIT 许可不应自动被解释为重写每个上游子数据集的原始许可，发布派生数据前仍需核对具体子任务来源；
- 如果系统优化不修改模型权重、量化或数值路径，并且两臂生成逐 token 完全相同，任务分数理论上也应相同；LongBench 的价值在于形成标准、可审计的质量回归门禁。

**LongBench v2 的位置**

LongBench v2 官方仓库给出 503 道多项选择题、8k 至 2M words 的 context，字段含 context、question、A–D 选项和 ground-truth answer，适合用 accuracy 做更强长上下文质量回归。但本项目当前模型/显存不应为了“覆盖 v2”而静默截断到失去题意。现阶段以 LongBench-E 可支持长度桶为主；只有预注册了最大上下文、截断规则和样本覆盖率后，才把 v2 加为扩展实验。

### 3.5 ShareGPT：常用 serving 兼容基线，不作为主数据来源

ShareGPT 值得保留，是因为 [vLLM 官方 benchmark 文档](https://github.com/vllm-project/vllm/blob/main/docs/benchmarking/cli.md) 仍把 `ShareGPT_V3_unfiltered_cleaned_split.json` 列为 online/offline serving benchmark 支持数据，很多论文和工程结果沿用这一口径，便于横向比较输入/输出长度和 throughput。

但它不应成为本项目“公认开源数据集”的主证据：

- [FastChat 官方 README](https://github.com/lm-sys/FastChat/blob/main/README.md#data) 明确写着不会发布 ShareGPT dataset；
- vLLM 文档下载的是第三方 Hugging Face mirror，而非 ShareGPT 或 LMSYS 的正式数据发布；
- 常用 mirror 主要含 conversation ID 与 human/GPT turns，没有 arrival timestamp、服务 latency 和质量标签；
- mirror 页面标注 Apache-2.0，但这不等于原始对话每一层来源和再发布授权已经由正式数据所有者完整说明；
- 常用 cleaned split 做过语言、内容和 2048-token 切分过滤，已不是原始线上分布。

因此本项目的顺序应是：

1. LMSYS-Chat-1M 作为有官方论文、数据卡、访问协议和字段说明的主要真实对话数据；
2. ShareGPT 仅跑一个与 vLLM 常用脚本一致的 compatibility slice；
3. 报告中把 ShareGPT 标记为 `third_party_mirror / compatibility_only`，不称它为生产流量 trace，也不使用它支撑真实 arrival 结论。

## 4. 可执行的三层数据协议

### 4.1 Layer A：算子边界合成数据，只做 correctness

继续保留程序生成数据，因为公开文本数据不能可靠覆盖：

- context 为 0、1、页边界前后和 partition 边界前后；
- batch 1/2/4/8 与不等长 sequence；
- GQA head 映射；
- 连续、逆序、带洞、重复引用和非法 block table；
- FP16 输入、FP32 oracle、极端 logits 和 online softmax 稳定性。

这一层只允许报告：

- 最大绝对/相对误差；
- exact/close case 通过数；
- sanitizer、越界、fallback/dispatch 计数；
- 覆盖的 shape/page/layout 集合。

不得把这一层称为“真实数据集准确率”或用它证明生产性能。

### 4.2 Layer B：公开内容与标准标签

建立两个独立 corpus，不混成一个未经说明的数据集：

#### B1. LMSYS 内容 replay

- 接受 gated license 后在本机缓存原始 parquet；
- 先按 `conversation_id` 做稳定 hash 抽样，再做语言/turn/长度分层；
- 使用目标模型 tokenizer 重新计数每轮 prompt 与历史上下文；
- 使用完整会话作为 split 单位，禁止同一 conversation 的 turn 落入不同 split；
- 预注册 redacted/flagged 样本过滤规则；
- 输出 manifest 只含 row ID、分层字段、token 数、源文件 hash 和转换 hash，不含正文。

主指标：Direct/Paged 输出 token 一致率、top-k logprob 差、TTFT、TPOT、E2E、吞吐、峰值 KV 内存；按语言、turn、input/output token 桶分层。

#### B2. LongBench-E 质量 replay

- 保留官方 `_id/dataset/language/answers`；
- 用官方 prompt template 和 max output length；
- 另存本地 tokenizer 后的精确 input token 数；
- 按 `_id` 对 Direct/Paged 配对，运行官方 task metric；
- 按 0–4k、4–8k、8k+ 以及实际本地 token 桶分别报告；
- 如果模型上下文不足，报告 coverage rate，不把截断样本与完整样本混成一个均值。

主指标：官方 task score 及两臂差值；辅指标为输出一致、TTFT、TPOT、throughput、KV bytes/page count。

### 4.3 Layer C：公开生产 trace 重放 arrival/concurrency

#### C1. BurstGPT 主确认实验

- 固定 v2.0 的一个或多个连续时间窗口；
- 保留原始 event order、inter-arrival time、session ID、request/response token count；
- 分别运行 exact-time replay 和预注册 time-scale 压力曲线；
- 原始失败行保留在数据审计中，性能主分析可并列报告 all rows 与 without-fails；
- `Elapsed time` 只作为源 trace 描述，不与本地 latency 做 paired comparison。

#### C2. Azure robustness 实验

- code 与 conversation trace 独立运行；
- 使用相同的服务端、runner、指标和统计脚本；
- 结果按来源分开，不先池化；
- 只有两个来源方向一致时，才写“跨两个公开生产 trace 稳健”。

主指标：在固定成功率/错误率门下的 request throughput、output token throughput、TTFT/TPOT/E2E P50/P95/P99、queue time、active batch、KV occupancy、fragmentation、eviction/preemption、OOM/429。

## 5. 内容与流量如何组合而不冒充真实联合分布

### 5.1 可以做的组合

BurstGPT/Azure 没有正文，实际请求需要 tokenizer-valid prompt。可按以下方式填充：

1. 对每个 trace event 读取目标 `input_tokens`；
2. 在 LMSYS 内容池中按“本地 tokenizer 后 token 数”寻找最近长度桶；
3. 在 event 预先绑定的确定性 seed 下选取文本；
4. 过短时选择另一个公开样本或使用明确标记的 padding template，过长时按预注册规则截断；
5. 保存 `trace_row_id -> corpus_row_id -> actual_local_tokens` 映射；
6. 以 trace timestamp 调度请求，记录本地实际输出 token 数。

此实验应命名为：

> `BurstGPT-arrival + LMSYS-content length-matched synthetic replay`

它能说明：在 BurstGPT 的 arrival/length 边际分布与 LMSYS 内容边际分布组成的压力测试中，系统表现如何。

### 5.2 绝对不能写的结论

不能写：

- “这是 BurstGPT 的真实 prompt”；
- “复现了 Azure 用户的真实完整请求”；
- “保留了 prompt 主题与高峰时段、输出长度、模型之间的真实相关性”；
- “LMSYS 文本对应 BurstGPT 的同一用户/session”；
- “组合后是生产联合分布”。

原因是两个独立数据集只分别给出边际观测：

```text
BurstGPT: P(arrival, input_len, output_len, session, model)
LMSYS:    P(content, turns, language, historical_model)
```

长度匹配构造的是研究者指定的 coupling，不能由数据证明它等于真实世界中的：

```text
P(arrival, content, input_len, output_len, session, model)
```

### 5.3 结果必须拆成三张表

1. **Content-only table**：LMSYS/LongBench 的受控 closed-loop 或固定并发结果；
2. **Trace-only table**：BurstGPT/Azure arrival-driven replay，正文填充策略写入标题；
3. **Cross-product stress table**：内容桶 × arrival window × rate scale 的敏感性分析。

不得把三类观察直接池化成一个“总体提升 X%”。跨层结论只能写成“在内容有效性、流量有效性和算子正确性三个独立切面均通过门禁”。

## 6. 统一 manifest 与可复现要求

每个正式数据 artifact 至少保存：

```json
{
  "source_name": "BurstGPT",
  "source_uri": "https://github.com/HPMLL/BurstGPT/releases/tag/v2.0",
  "source_version": "v2.0",
  "source_file_sha256": "...",
  "license": "CC-BY-4.0",
  "downloaded_at_utc": "...",
  "selection_rule": "contiguous-window",
  "selection_seed": 0,
  "selected_row_ids_sha256": "...",
  "tokenizer_model_sha256": "...",
  "local_tokenizer_name": "Qwen tokenizer through llama.cpp",
  "transform_version": "...",
  "transform_source_sha256": "...",
  "excluded_rows": {"reason": 0},
  "content_fill_source": "LMSYS-Chat-1M",
  "joint_distribution_claim": false
}
```

不同数据集还要补充：

- LMSYS：license acceptance date、过滤规则、conversation ID hash、不得再发布标志；
- LongBench：仓库 commit、子任务清单、官方 prompt/maxlen/eval 脚本 hash、完整/截断覆盖率；
- BurstGPT：release、文件名、all/without-fails、原始窗口、time-scale；
- Azure：code/conversation、原始窗口、文件 hash；
- ShareGPT compatibility：mirror URI、文件 hash、`compatibility_only=true` 和 provenance warning。

## 7. 针对本项目的正式指标矩阵

| 层次 | 正确性/质量 | 性能 | 资源与机制 |
|---|---|---|---|
| CUDA operator | CPU FP32 oracle、max abs/rel error、边界 case pass | kernel time 仅作微基准 | load/store、occupancy、page lookup、fallback |
| LMSYS content | exact tokens、top-k overlap、max logprob delta | TTFT、TPOT、E2E、tokens/s | KV pages、prefix reuse、batch、显存 |
| LongBench-E | 官方 QA F1/ROUGE/classification/retrieval/count/code metric；Paged-Direct 配对差 | 同样本配对 TTFT/TPOT | 长度桶、截断率、KV bytes |
| BurstGPT/Azure | Direct/Paged 同输入同 seed 一致；失败率 | req/s、tok/s、P50/P95/P99、queue | active batch、KV occupancy/fragmentation、eviction/OOM |

对于不修改权重、量化或近似语义的 Paged Attention，正式 promotion gate 至少同时要求：

1. 算子 oracle 在预注册容差内；
2. serving 输出和 logprob 与 Direct 基线满足一致性门；
3. LongBench 官方质量分数不劣于 Direct 的预注册容差；
4. BurstGPT 主 trace 与 Azure robustness trace 的错误率不增加；
5. 主要性能指标的置信区间通过预注册 superiority 或 non-inferiority 门；
6. 所有 dataset、模型、二进制、脚本与选择规则有 hash 绑定。

## 8. 最小落地顺序

### Phase 1：替换项目自构正文的主证据

1. 接入 LongBench-E，先完成质量和长上下文 paired replay；
2. 接受 LMSYS license，在本地建立 deterministic content slice；
3. 原 3 份项目文档语料降级为开发 smoke，不再作为最终外部有效性证据。

### Phase 2：真实 arrival replay

1. 下载 BurstGPT v2.0，选择连续、不可结果后更换的固定窗口；
2. 用 LMSYS 文本做长度匹配填充，报告 `synthetic replay`；
3. 跑原始 rate 与多档预注册 rate scale；
4. 接入 Azure code/conversation 两个 trace 做跨来源复核。

### Phase 3：标准兼容与消融

1. 额外跑 vLLM 常用 ShareGPT cleaned split，仅作 compatibility row；
2. 对比固定合成 token、LMSYS 内容、LongBench 内容，判断正文内容是否改变性能结论；
3. 对比 constant/Poisson、BurstGPT、Azure arrival，判断收益是否依赖 arrival process；
4. 报告每个来源独立结果和最差子组，不合并追求一个更好看的均值。

## 9. 面试中的准确表述

可以这样回答：

> 我把数据分成三个独立证据层。CUDA 算子层仍用确定性合成张量，因为只有它能完整覆盖页边界、非连续页表和 GQA 形状，但它只承担数值正确性。真实文本层使用官方 gated 的 LMSYS-Chat-1M，并用 LongBench-E 的标准答案和官方指标做长上下文质量回归。服务流量层使用 BurstGPT v2.0 的真实 arrival、session 和输入输出 token 长度，再用 Azure 2024 code/conversation trace 做跨来源复核。由于生产 trace 因隐私不含 prompt，我会用 LMSYS 文本按本地 tokenizer 长度匹配填充，但明确把它称为 trace-driven synthetic replay；我不会声称两个独立数据集拼接后恢复了真实内容与到达时间的联合分布。

如果被问“准确率是多少”，应分开回答：

- LongBench/LongBench v2：报告官方 task score 或多选 accuracy，以及 Paged 相对 Direct 的配对差；
- LMSYS：没有正确答案标签，报告输出一致、logprob 差和性能，不称 accuracy；
- BurstGPT/Azure：没有正文和正确答案，报告服务 SLO、吞吐、失败率和资源指标；
- CUDA 合成数据：报告 oracle 数值误差与 case pass，不称真实数据集准确率。

## 10. 一手来源索引

- [BurstGPT 官方仓库与 schema](https://github.com/HPMLL/BurstGPT)
- [BurstGPT v2.0 release](https://github.com/HPMLL/BurstGPT/releases/tag/v2.0)
- [BurstGPT KDD 2025 论文](https://doi.org/10.1145/3711896.3737413)
- [Azure LLM Inference Trace 2024 官方说明与 schema](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md)
- [DynamoLLM / Azure trace 配套论文](https://arxiv.org/abs/2408.00741)
- [LMSYS-Chat-1M 官方数据卡、字段与专用许可](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)
- [LMSYS-Chat-1M 论文](https://arxiv.org/abs/2309.11998)
- [LongBench 官方仓库](https://github.com/THUDM/LongBench)
- [LongBench v1 数据格式与评测说明](https://github.com/THUDM/LongBench/blob/main/LongBench/README.md)
- [LongBench 官方评测代码](https://github.com/THUDM/LongBench/blob/main/LongBench/eval.py)
- [LongBench ACL 2024 论文](https://aclanthology.org/2024.acl-long.172/)
- [vLLM 官方 serving benchmark 数据集说明](https://github.com/vllm-project/vllm/blob/main/docs/benchmarking/cli.md)
- [FastChat 官方关于 ShareGPT 数据不发布的说明](https://github.com/lm-sys/FastChat/blob/main/README.md#data)
