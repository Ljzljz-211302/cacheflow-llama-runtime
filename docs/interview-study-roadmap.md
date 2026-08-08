# CacheFlow Runtime 推免面试学习路线（零基础 18 天）

> 本路线保留分日训练安排；项目事实、最新实验数字和完整问答统一以 [`lessons/cacheflow-runtime-complete-interview-handbook.html`](../lessons/cacheflow-runtime-complete-interview-handbook.html) 为准。当前增量专项与一次严格 Full 均已通过；一次通过不外推为所有未来运行都无统计波动。

课程入口：[lessons/index.html](../lessons/index.html)；速查：[术语表](../reference/glossary.html)、[公式与代码地图](../reference/formulas-and-code-map.html)。

## 一、你需要掌握到什么程度

项目面试不是“把 README 背出来”。每个核心点要经过五层：

1. 定义：能准确说出概念是什么；
2. 动机：能说明没有它会出现什么具体问题；
3. 算法：能写公式、状态机或不变量；
4. 实现：能定位到本项目类、函数、测试和指标；
5. 评价：能解释负结果、适用边界和替代方案。

面试中最能体现个人工作的五条主线是：

- iteration transaction 将调度决策与不可逆 runtime 副作用隔离；
- Paged Prefix KV + partial-tail COW + refcount/capacity 不变量；
- Host/File/CUDA swap 与异步生命周期、失败原子性；
- backend-local online Ridge + uncertainty + bounded exploration + drift fallback；
- paired intervention 把 policy、scheduler、CUDA、Engine、TTFT 串成因果证据。

## 二、零基础桥接：前 4 天

如果下表里的内容不会，不再要求你“先自行补基础”。课程已提供四节桥接课：

- Day 0A：[一次聊天请求发生了什么](../lessons/0000a-from-chat-to-inference.html)：模型、推理、token、server、CPU/GPU；
- Day 0B：[看懂项目公式的数学](../lessons/0000b-math-without-fear.html)：向量、矩阵、点积、P95、回归和置信；
- Day 0C：[程序、内存与并发](../lessons/0000c-programs-memory-and-concurrency.html)：进程、线程、指针、状态机、事务、COW；
- Day 0D：[GPU 与 CUDA 从零](../lessons/0000d-gpu-from-zero.html)：Host/Device、kernel、thread、stream、event。

每天完成正文、手算/画图任务和末尾小测。四节都能不用术语复述后，再进入原 14 天核心路线。因此完全零基础用 18 天；已有本科基础者可通过小测后从 Day 1 开始。

## 三、进入主课前的诊断表

在开始课程前闭卷回答。任一项不会，先补相应基础，不要直接背项目答案。

| 模块 | 必须会 | 达标任务 |
|---|---|---|
| 线性代数 | 矩阵乘法、逆、正定矩阵、二次型、L2 正则 | 解释 `xᵀA⁻¹x` 为什么非负 |
| 概率统计 | 均值/中位/P95、方差、置信区间、配对实验 | 构造一个“独立中位数误导”的例子 |
| 机器学习 | 线性回归、偏差—方差、在线学习、探索—利用 | 推导 Ridge 正规方程 |
| 操作系统 | 虚拟内存、页表、COW、引用计数、调度、饥饿 | 写出共享页写入前后的映射变化 |
| 数据库/并发 | 事务、原子性、状态机、锁、WAL/rename 思想 | 设计 swap 文件写失败的恢复路径 |
| 计算机组成 | cache/memory bandwidth、SIMD/SIMT、吞吐/延迟 | 区分 compute-bound 与 memory-bound |
| C++ | RAII、所有权、引用/指针、容器失效、异常安全 | 解释嵌套 `finally`/RAII 清理责任 |
| CUDA | grid/block/thread、global/shared memory、stream/event | 画两 stream 的正确同步时间线 |
| Transformer | Q/K/V、causal attention、自回归生成、GQA | 推导 KV cache 容量 |

## 四、14 天核心学习安排

每天建议 2.5–3.5 小时：阅读 45–70 分钟，代码 45 分钟，闭卷输出 30 分钟，间隔复习 20 分钟。

### Day 1：Transformer 推理

- 完成 Lesson 01。
- 手算两个模型的 KV bytes，分别使用 MHA 和 GQA。
- 口述：prefill/decode 的输入形状、并行性和瓶颈。
- 代码定位：`llama-context.cpp`、`llama-kv-cache.*`，只追调用关系，不试图读完整上游。

验收：能在白板写 Attention 和 KV 容量公式，并解释每个变量。

### Day 2：Serving 与指标

- 完成 Lesson 02。
- 阅读 Orca 的 iteration-level scheduling 摘要和核心设计。
- 画项目 prepare → plan → execute → commit/abort 状态机。
- 对 TTFT、TPOT、latency、throughput 分别给出一个“只优化它会作弊”的例子。

验收：能解释为何 request、sequence、slot 必须分离。

### Day 3：Paged KV

- 完成 Lesson 03 前半。
- 阅读 `server-kv-block-manager.*` 和对应测试。
- 用 4-token block 手工模拟 share、append、COW、release。
- 写出 refcount 与 Block Table 的守恒关系。

验收：不看代码画出 partial-tail COW 的正确操作顺序。

### Day 4：Capacity、Preemption、Swap

- 完成 Lesson 03 后半。
- 阅读 capacity planner 和 swap store。
- 为 allocation、file write、CUDA copy、restore capacity 四类失败写安全结果。
- 对比 recompute 与 swap 的成本：计算、Host 内存、磁盘延迟、恢复 TTFT。

验收：回答“为什么 rename 有助于文件 checkpoint 原子性”。

### Day 5：CUDA 基础

- 完成 Lesson 04。
- 在纸上写一维 kernel 索引和边界检查。
- 解释 coalescing、warp divergence、pinned memory。
- 画 stream/event happens-before。

验收：回答“函数返回为何不代表 GPU 完成；如何安全释放 buffer”。

### Day 6：项目 CUDA 路径

- 阅读 `src/llama-kv-cache-paged.cu`、`server-kv-block-cuda.cu`。
- 对 gather、scatter、COW 标注 input/output/mapping。
- 解释 direct-copy 与 staging 的适用条件。
- 阅读对应 CUDA tests/bench 和 sanitizer 脚本。

验收：能说明 sanitizer 通过能证明和不能证明什么。

### Day 7：Ridge 基础

- 完成 Lesson 05 的 1–3 节。
- 从目标函数对 θ 求导，得到 `(XᵀX+λI)θ=Xᵀy`。
- 用二维样本手算一次 A、b、预测。
- 解释 λ 太大/太小的后果。

验收：能说明二次型 `xᵀA⁻¹x` 和数据覆盖的关系。

### Day 8：保守在线门控

- 完成 Lesson 05。
- 阅读 `server-benefit-policy.*` 和 `test-benefit-policy.cpp`。
- 逐分支列出 fixed、rule、cold start、exploration、positive bound、fallback、cooldown。
- 解读最终 53-wave CSV，找出从探索到持续利用的转折。

验收：能准确区分 safe exploration、positive lower bound、SLO miss、drift。

### Day 9：Adaptive Prefill / Speculation

- 完成 Lesson 06。
- 阅读 Sarathi-Serve 与 speculative decoding 论文摘要。
- 写出 chunk 太大/太小、depth 太大/太小的成本。
- 明确项目 N-gram controller 与 draft-model 论文算法的差异。

验收：给定 workload 状态，提出动作并同时说明回退条件。

### Day 10：实验统计

- 完成 Lesson 07 的 1–4 节。
- 解释 fresh process、paired trial、Williams balanced Latin、进程位置/直接前驱平衡和 backend 热状态混杂。
- 说明为什么 `median(learned)/median(upstream)` 不是 paired ratio。
- 设计一个最低 3 trial 的随机顺序表。

验收：指出 material effect 与 statistical significance 的区别。

### Day 11：CUDA 因果链

- 完成 Lesson 07。
- 阅读 summary、trials、evidence JSON。
- 逐层回答：干预是否生效？scheduler mediator？CUDA mediator？系统结果？
- 写出两个竞争解释和需要的 Nsight 证据。

验收：不用“CUDA 更快/更慢”这种过度概括，准确复述最终负结果。

### Day 12：源码串讲

- 按公式与代码地图走完一次请求。
- 每个模块只用一句话说清“拥有哪个状态、暴露哪个窄接口、失败时谁回滚”。
- 随机从 10 个测试中挑 3 个，说明其保护的不变量。

验收：5 分钟内定位 Engine、KV COW、benefit policy、CUDA kernel 四处代码。

### Day 13：答辩组织

- 完成 Lesson 08。
- 分别录制 30 秒、3 分钟、15 分钟版本。
- 删除所有“非常快、完全解决、全面优化”等无证据词。
- 把每个性能结论补全硬件、模型、workload、trial 和限制。

验收：3 分钟版本不超时，并包含一个负结果和个人贡献边界。

### Day 14：压力模拟

- 闭卷完成 Lesson 08 的 15 题，每题按 0–3 分评分。
- 让 AI/同学分别扮演系统、机器学习、CUDA 面试官，各追问 15 分钟。
- 对答错题创建错题卡，安排第 1/3/7 天重答。
- 完成一次现场演示，故意让一个参数错误并解释如何诊断。

验收：总分至少 36/45，无 0 分题；不能回答时能诚实界定并提出验证方法。

## 五、面试回答通用模板

对任何技术点使用“六句法”，避免散乱：

1. 问题：原系统在什么 workload 下出现什么失败？
2. 约束：不能破坏哪些兼容、正确性或资源边界？
3. 方案：核心数据结构/算法是什么？
4. 不变量：如何保证状态和异步生命周期正确？
5. 证据：哪个测试、指标、A/B 或 trace 支持？
6. 边界：当前不支持什么，下一步如何验证？

示例——partial-tail COW：

> 共享完整 block 会浪费公共 prompt 的尾部，所以我允许 partial-tail sharing；但追加 token 会污染其他 sequence。写入前检查 refcount，若大于 1 则先分配并复制有效区域，成功后原子切换写者 Block Table，再减少旧引用。CUDA copy 用 stream/event 约束发布时机，allocation/copy 失败保持旧映射。对应单元测试验证 refcount/容量守恒，真实 CUDA smoke 和 sanitizer 验证 tensor 路径。当前是单 GPU block runtime，未实现跨 GPU KV 迁移。

## 六、必须背熟的数据，但不要只背数据

- 固定上游个人差异：61 files、+8708/−99 C/C++/CUDA；外层实验另计。
- 长驻 CUDA：53 waves；18 exploration；142 positive；33 positive waves；最长连续 13；终态 21.29ms > 8.82ms；shift 后 0 错误启用、3 fallback。
- 短程 gating：每端 12-trial Williams blocks 平衡策略位置、直接前驱与 backend 顺序，真实 socket send seam 下 96/96 rows 的两波 observed order 均为 `0..5`；CPU regression −24.54%、oracle regret 5.25%；CUDA −6.04%、0.13%，CUDA fresh-process 0 probe，8 个 harmful trials 中 0 次错误启用。
- CUDA causal：decision +13、chunk +23、prefill token −354、copy +20.066MB、Event +0.808ms、Engine 汇总 −11.446ms、TTFT P95 +85.61ms。
- 严格入口是 `verify.ps1 -Full`。历史提交 `e01a844` 曾原生退出 0，但后续审查发现其 socket send 与截断 Latin 协议仍不充分；当前已改为真实 send guard、每端 12-trial Williams blocks 和交替 backend order，定向门禁通过。只有新的完整 Full 再次退出 0 后才能更新最终验收主张，阈值始终未放宽。

每个数字必须同时说出“它回答什么”和“它不能证明什么”。

## 七、进一步提高面试竞争力

完成本路线后，优先做以下延伸，而不是继续堆代码量：

1. 用 Nsight Systems 定位 chunk 增加后 Engine/TTFT 恶化的 timeline；
2. 用 Nsight Compute 比较 direct-copy/staging kernel 的 occupancy、memory throughput 和 launch 开销；
3. 把 benefit action 扩展到 KV admission 或 speculation，但先设计受约束联合探索；
4. 在 7B 或更大模型、另一类 GPU 上复现实验，检验外部有效性；
5. 做一次与固定版本 vLLM 的同硬件同模型同 workload 对照，但严格区分架构差异。
