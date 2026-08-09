# D2 分段配对差值策略：v1.5 预注册说明

## 为什么需要 D2

v1.4 保留了一个明确的负结果：全局线性 D1 在 40 个留出 trace 上没有产生任何安全切换；没有安全上界的 T1 虽把累计 regret 从 58.142 ms 降到 34.938 ms，却出现了 1 次 harmful decision。训练与开发证据显示，Recompute 相对 device Swap 的完整动作成本会在约 512 token 的上下文尺度附近改变符号。这个观察只用于设计下一版协议，v1.4 evaluation 不会被冒充为 v1.5 的确认性证据。

## D2 改了什么

D2 的学习目标仍是候选动作相对安全基线 H0 的完整动作成本差：

\[
\Delta(x)=C_{\text{candidate}}(x)-C_{H0}(x).
\]

负值表示候选动作更快，正值表示 H0 更快。D2 不直接学习“哪个动作最好”的分类标签，而是分别在 `context_tokens <= 512` 与 `context_tokens > 512` 两个预注册区间内拟合 Ridge 回归；`resident` 与 `preempted` 仍使用不同模型。这样做是为了表达已观察到的分段成本机制，而不是放宽安全阈值。

每个模型只使用时间更早的 fit trace。随后用独立 calibration trace 计算单侧残差分位数。最终只有满足

\[
\widehat{\Delta}(x)+r_{\text{ridge}}(x)+q_{0.95}^{\text{cal}}+0.75\text{ ms}<0
\]

才允许切换。其中，\(\widehat{\Delta}(x)\) 是候选减 H0 的预测时间，\(r_{\text{ridge}}(x)\) 是 Ridge 参数不确定性半径，\(q_{0.95}^{\text{cal}}\) 是同一 regime、同一上下文区间的单侧校准残差，0.75 ms 是预注册的最小收益余量。缺少模型、校准样本或合法动作时，上界视为无穷大并精确回退 H0。

## 怎样证明有效

v1.4 仅作为开发证据：用锁定后的 D2 回放时，它产生 9 次切换，累计 regret 为 24.697 ms，H0 为 58.142 ms；P95 regret 为 2.013 ms，H0 为 3.969 ms；harmful decision 为 0；配对 trace-cluster mean-regret delta 的 95% bootstrap 区间为 [-0.7011, -0.1745] ms。这说明 D2 值得进入新数据验证，不构成确认性结论。

v1.5 必须重新采集 trace，并同时满足：至少一次留出切换；配对 mean-regret delta 的 95% 区间上界小于 0；P95 regret 不高于 H0；harmful rate 不高于 H0。即使全部通过，也只授权独立的生产 canary，不会自动把离线 D2 上线。

## v1.6 风险预算扩展

v1.5 证明了 D2 的“harmful rate 不高于 H0”与 95% 单侧残差门禁会在观测噪声下拒绝全部动作：H0 和任何非基线策略不同，H0 相对自身不可能被标记 harmful，因此该条件实际要求候选零次负优化。D3 保留 512-token 分段、Ridge 半径和 H0 回退，但用 calibration residual 的中位数修正系统偏差，并显式预注册 5% harmful-rate budget。它仍须同时满足 paired CI 上界小于 0、P95 不高于 H0、累计正收益大于累计负收益。该设计优化的是有风险约束的期望完整动作成本，不声称保证每个请求都加速。

v1.6 的全新留出数据通过了全部预注册门禁：D3 产生 24 次切换，累计 regret 从 42.590 ms 降至 2.082 ms，P95 从 3.169 ms 降至 0，paired trace-cluster 95% CI 为 [-0.7232, -0.3069] ms；1/80 harmful decision 位于 5% 预算内，41.480 ms 累计收益覆盖 0.972 ms 累计损失。该结论的边界是单模型、单 GPU、matched-workload 离线 replay；生产上线仍需独立 canary。
