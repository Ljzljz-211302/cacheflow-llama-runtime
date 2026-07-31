# 实验负结果与边界

本文件是验收证据的一部分，不因最终实现获得正收益而删除中间失败结果。

## Adaptive Prefill

最初仅按 iteration latency EWMA 缩放 chunk 的控制器在 CUDA 上把 chunk 压到 16，P95 从 greedy 的 685.88 ms 恶化到 950.70 ms。加入在线 decode/prefill 成本模型后避免了坍缩，但两个中间版本仍慢于 greedy 或 fixed-256。最终版本在当前 CPU trace 上优于两个固定候选，在 CUDA 上优于错误 fixed 参数的中位数，但仍未击败 greedy 中位数和 fixed-256 P95。因此项目不宣称 adaptive 对所有 workload 最优。

完整迭代数据见 `results/adaptive-prefill-tuning-history.csv`，最终三次 fresh-process 原始汇总见 `results/adaptive_prefill_trials.csv`。

## Adaptive Speculation

早期控制器在仅观察到约 70 个 draft token 时就根据 EWMA 调整，虽然接受率约 97.1%，CUDA P95 仍比 fixed draft 慢 14.4%。最终实现要求至少 24 个累计 draft token 证据后才允许低接受率触发冷却。在固定 trace 上最终 CUDA adaptive 优于 fixed，但 CPU 的 fixed P95 仍略好。

数据见 `results/adaptive-speculation-tuning-history.csv` 与 `results/adaptive_speculation_trials.csv`。

## CUDA Gather/Scatter

在源、目标完全不重叠的映射上，每 block `cudaMemcpyAsync` 的延迟低于需要 staging 的 Gather+Scatter。生产 Backend 因而只在重叠/重复映射需要 snapshot 语义时使用两 kernel staging；无重叠映射使用直接异步复制。这是基于负 benchmark 改变实现，而不是隐藏不利数据。原始 40 个样本与汇总位于 `results/raw/cuda-kv-transport.csv`、`results/cuda-kv-transport-summary.json`。

## 工具链确定性

官方预编译 Clang 基线与本地 MSVC fork 在五个样本之一出现 `data cache` / `data structure` 的单词差异。固定 revision 使用相同 MSVC 19.37 重建后，upstream policy 5/5 输出逐字节一致。因此“上游兼容”结论限定为同模型、同 seed、同工具链；跨编译器浮点路径不承诺逐字节确定性。

## Compute Sanitizer

本机为 Windows WDDM，首次运行 Compute Sanitizer 前需要管理员启用 debugger interface。2026-07-31 启用后，固定 sm_89 CUDA 正确性矩阵在 memcheck 下得到 `ERROR SUMMARY: 0 errors`，在 racecheck 下得到 `0 hazards displayed (0 errors, 0 warnings)`。此外仍保留 GPU K/V canary guard、随机映射矩阵、每轮完整性校验及 CUDA allocation/pinned allocation 故障注入。该结论限定当前驱动、硬件和测试矩阵，不能代替其他平台的复验。
