# 实验负结果与边界

本文件是验收证据的一部分，不因最终实现获得正收益而删除中间失败结果。

## Adaptive Prefill

最初仅按 iteration latency EWMA 缩放 chunk 的控制器在 CUDA 上把 chunk 压到 16，P95 从 greedy 的 685.88 ms 恶化到 950.70 ms。加入在线 decode/prefill 成本模型后避免了坍缩，但中间版本仍会在 CPU 上劣于 greedy 和所有 fixed 候选。最终策略按 backend bucket 决策：CPU 在尚无稳定正收益时选择候选集合中的 `chunk=0` greedy 安全动作，CUDA 继续在线选择 chunk；A/B 脚本在 adaptive 比错误 fixed 候选回归超过 2% 时直接失败。这是 safety fallback，不应描述为 CPU 学出了更优 chunk。

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

## Mixed Prefill / Decode

CPU cacheflow 在两轮独立的 3-trial workload 中都改善 TTFT P95、TPOT P95、latency P95 和 aggregate TPS，但 median TTFT 曾退化。CUDA 结论不稳定：第一轮 latency P95 从 524.16 ms 退化到 610.14 ms、aggregate TPS 从 278.85 降到 249.33；紧接着的复验轮却得到 latency P95 602.19 -> 556.07 ms、TPS 241.95 -> 268.48。两轮反号说明 3 trials 在 laptop GPU 上不足以支持“稳定提升”的强结论，CPU 策略也不能直接外推 GPU batching；当前 CUDA 路径应保留 upstream 默认或增加 backend/workload-aware gating。当前请求级数据位于 `results/mixed_workload_trials.csv`，跨轮汇总保存在 `results/mixed_workload_repeated_runs.csv`。

## Profiler 权限

Windows Performance Recorder 的 sampled-stack profile 因当前非管理员会话缺少 `SeSystemProfilePrivilege` 而失败，VSDiagnostics attach 也未建立可用 session。项目保留该失败边界，并使用 production Engine 自带的 Chrome/Perfetto complete-event trace 生成 `results/engine-flame.svg`。这张图只能归因 prepare/plan/execute/commit phase duration，不能替代 CPU sampled stack、Nsight Systems timeline 或 Nsight Compute kernel 分析。

Issue #4 已补上真实 Nsight Systems CUDA timeline：四个 KV regime 均保存 native `.nsys-rep` 和 SQLite export，能够核对自研 gather/scatter launch、descriptor memcpy 与 CUDA synchronization。它仍不是 CPU sampled-stack profile，也不是全模型 serving trace。Nsight Compute 2025.4.1 已安装并实际尝试四个 regime，但 561.19 driver compatibility 检查与 `ERR_NVGPUCTRPERM` 阻止硬件 performance counters；因此当前不能用 occupancy、DRAM throughput、L2 hit 或 roofline 解释结果。报告中的 effective payload GB/s 只是逻辑 bytes / CUDA-event time，不是硬件带宽。
