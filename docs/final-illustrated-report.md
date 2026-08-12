# CacheFlow Runtime 图文总结报告

## 1. 成果是什么

项目不是在 llama.cpp 外层套一层接口，而是把缓存感知调度、KV 生命周期管理和 CUDA 算子接入真实 `llama-server → llama_decode → KV memory → CUDA` 路径。最终形成两项可独立交付成果：一套面向单机/可信局域网的推免面试学习助手，以及一套保留正负结果、可由原始 trials 重算的独立科研型成果。

![真实执行链与个人贡献边界](assets/final-system-flow.svg)

## 2. 核心创新点

1. **从请求调度贯通到 GPU 数据移动。** 控制面不只决定请求先后，还显式管理 KV 的驻留、抢占、恢复、换入换出和重算，并把决策计数、CUDA event、kernel launch 与请求延迟关联起来。
2. **统一动作代价模型。** 将 Direct、Remap、Swap、Recompute 与受能力门禁约束的 Paged 视为同一动作空间；D3 用有界 ridge 预测动作相对 H0 的增量代价，证据不足时回退 H0。
3. **自研 CUDA KV Remap/Swap。** 用向量化数据搬运取代标量 gather/scatter，并以 CPU oracle、边界/重叠映射和 sanitizer 验证正确性。
4. **Paged Decode 从 K1 演进到 K2。** K1 是每个 query head 一个 CTA 的正确基线；K2 针对 Qwen2.5-0.5B 的 GQA7 结构复用同一 KV head 的加载与在线 softmax 工作，在不增加 launch 数、不触发 fallback 的条件下降低 kernel 总时长。
5. **证据工程本身是成果的一部分。** 实验先预注册门槛，再生成原始请求、Prometheus、NSYS/SQLite、汇总、图表和 manifest；最终结论由验证器从底层证据重算，负结果不删除。

## 3. 实验数据从哪里来、如何处理

### 3.1 数据来源

- **应用数据**：仓库中的 5 份面试知识文档被切分为 128 个检索块；自动 HTTP/SSE 用户旅程验证后端链路，另有 Codex 内置 Chromium 的交互记录验证浏览器流式完成与会话侧栏显示，再调用固定 Qwen2.5-0.5B 模型。
- **H1 Remap 数据**：CUDA benchmark 在 1/4/16/32 个 block 上分别执行标量与向量化方法；每个规模 20 个 confirmatory 配对，另有 warm-up，正式统计排除 warm-up。
- **H4 策略数据**：同一 trace/session/prefix family 下收集 H0/A1/T1/L1 的动作观测，按 trace 分组并按时间切分，避免同一会话泄漏到训练和留出集。
- **H7/H8 服务数据**：同一进程内交替运行对照臂，固定模型、prompt、输出 token、context 和设备。H7 比较 Direct 与 Paged；H8 在已进入 Paged 路径后比较 K1 与 K2。
- **H9 客观输入矩阵**：prompt 不再嵌入 runner；冻结语料文件包含受控边界、中文数据库/机器学习、英文 Attention、中英混合 CUDA 和 C++ 代码六类输入。30 组随机化匹配进程块中，每个 Direct/Paged 独立进程 arm 都运行全部语料，实际 token 数由模型响应记录而非人工假设；它不是共享热状态 Trial Pair。原始 v2.0.0 协议及哈希保持不动；模型和 vendor-diff 哈希属于实验后的验证修正，只增强可审计性，不冒充运行前预注册或 contemporaneous run binding。

### 3.2 处理过程

原始记录先做协议一致性检查：请求参数、trial/arm 顺序、PID、计数器增量和响应内容必须完整。H1/H8 在共享状态的 trial pair 内计算差值；H9 则按随机化匹配进程块比较两个独立进程 arm，不冒充 Trial Pair。不确定性按相应的 pair、匹配进程块或 trace cluster 重采样 bootstrap，以保留相关样本结构。Prometheus 用于确认 graph entry 与 fallback，NSYS SQLite 用 PID 过滤后统计指定 kernel 的 launch 和 duration。只有正确性、延迟非劣化上界、机制改善和零 fallback 同时满足，K2 才能晋级。

## 4. 实验结果

![正式实验结果总览](../results/final-outcome-summary.svg)

- **H1 正结果**：1/4/16/32 blocks 的 kernel 时间配对中位改善为 54.55%、46.32%、5.08%、2.32%。规模越大收益越小，因此只称算子微基准改善。
- **D3 有条件正结果**：留出集 80 个决策中相对 H0 切换 24 次，累计 regret 2.082 ms，harmful decision 1 次；chooser P99 0.900 μs且热路径零分配。它仍是离线 replay，尚不能写成线上普适收益。
- **Paged-vs-Direct 负结果**：P95 从 27.354 ms 上升到 29.210 ms，回退 6.78%，超过 5% 门槛。因此默认启动器不启用 Paged。
- **K2-vs-K1 正结果**：30 组同进程配对、每臂 480 条测量响应和 600 次 Paged graph entry、0 fallback。请求 median/P95 仅回退 0.55%/1.52%，median 回退 bootstrap 95% 上界 2.86%；相同 480 次 kernel 总时长从 8.174 ms 降至 4.051 ms，降低 50.44%。
- **H9 客观矩阵负结果**：6类冻结输入、30组随机化匹配进程块、360个 workload-arm 观测均实际跨页。总体中位数显示 Paged 改善 7.96%，但 block-workload 回退分布P95为 158.62%，最差workload中位回退 44.66%，分别超过20%和5%门槛，因此不能晋级。

![K2/K1 正式对比图](../results/research/h8-k2-production-v2.10.0/k2-production-comparison.svg)

![客观Prompt矩阵分层结果](../results/research/h9-objective-paged-v2.0.0/comparison.svg)

## 5. 应用结果与最终边界

应用旅程已覆盖 UI、SSE、并发、取消、429 背压、重启恢复和本地知识检索，并记录 456 个缓存 prompt token、6 次自研 CUDA KV launch、5603330 个向量化 remap 字节。它能作为可运行应用项目和有实验链的 AI Infra 研究项目交付。

最终不能声称“Paged 全面优于 Direct”或“端到端加速 50.44%”。准确结论是：**多输入评测显示Paged中位数可能受益，但跨匹配块/工作负载的回退分布尾部和输入敏感性仍不达生产门槛；在受限Paged内部，K2已以预注册实验替换K1。**

## 6. 复现与审计

```powershell
.\scripts\verify_final_outcome.ps1
.\scripts\verify.ps1
```

前者重算 H1 并调用 H4/H7/H8 正式验证器，同时校验本报告、最终 JSON、图表和启动器哈希；后者执行全仓快速测试和架构/工件验收。全部源数据路径与 SHA-256 见 [`results/final-outcome.json`](../results/final-outcome.json)。
