# CacheFlow 确认性实验协议 v1.0.0

> 本文件是 Issue #3 的人类可读预注册。机器可执行原件为 [`config/research_protocol.json`](../../config/research_protocol.json)，方法依据见 [experiment-protocol-foundations.md](experiment-protocol-foundations.md)。协议变更必须提高版本并保留旧结果，不得覆盖已有确认性规则。

## 1. 单命令制品入口

CUDA H1 Remap 实验：

```powershell
python scripts/run_research_experiment.py `
  --experiment h1-vector-remap `
  --output-dir results/research/h1-vector-remap-v1
```

命令先运行现有 CUDA benchmark，再生成：

- `manifest.json`：协议/主张/source/binary SHA-256、outer/vendor revision 与 dirty flag、真实命令、实时 GPU/toolchain 环境、统计摘要和验收结果；
- `trials.jsonl`：warm-up 与 confirmatory 的逐 action 原始行，包含 pair、真实执行顺序、bytes、正确性和三种计时口径。

只重新封装已经生成的、字段完整的 CSV 时可加 `--skip-execution --source <path>`；这种操作不会伪装成重新采集。

无 CUDA 时使用：

```powershell
python scripts/run_research_experiment.py `
  --experiment cpu-correctness `
  --output-dir results/research/cpu-correctness-v1
```

CPU fallback 验证调度、Block Table、refcount、COW、Swap 和参考语义，只产生 correctness evidence；manifest 永久写入 `performance_claims_allowed=false` 与 `cuda_claims_allowed=false`。

## 2. H1 workload 与配对设计

- 固定布局：32 layers、8 KV heads、head dimension 128、16-token blocks、FP16 K/V。
- treatment：Scalar Gather/Scatter 与 `uint4` Vector Gather/Scatter。
- configuration：1、4、16、32 remapped blocks。
- 每个 configuration 先执行 1 个 warm-up pair，再执行 20 个 confirmatory pairs。
- 伪随机种子固定为 `20260806`；每个 pair 内 treatment 顺序随机并写入 `order_in_pair`。
- 每对共享 descriptor、mapping、staging、stream、bytes 和进程热状态；分析只使用完整 pair。
- warm-up 不进入统计，但仍写入 raw records；不得根据结果临时追加或删除 warm-up。

## 3. 三种不可互换的计时口径

| 字段 | 起点 | 终点 | 能回答什么 | 不能回答什么 |
|---|---|---|---|---|
| `host_enqueue_ms` | descriptor upload 前 | stop event 入队后、同步前 | 主机提交 descriptor/event/kernel 的成本 | GPU 完成时间 |
| `synchronized_kernel_ms` | descriptor upload 后的同-stream CUDA start event | Gather/Scatter 后的 stop event，完成后读取 | 目标 device interval | descriptor upload、请求排队、业务延迟 |
| `end_to_end_ms` | descriptor upload 前 | stop event 完成后 | upload、enqueue、GPU 执行和等待的完整 operation completion | 用户请求 TTFT/TPOT |

不得把 enqueue、CUDA event、operation completion 或请求级 latency 混称为“CUDA latency”。真实服务指标仍需单独记录 TTFT、TPOT/TBT、queue 和 request P95。

## 4. 统计与验收

配对效应定义为：

```text
improvement_percent = 100 * (scalar_ms - vector_ms) / scalar_ms
```

主估计量是 paired improvement 的中位数。95% CI 使用整 pair 重采样的 deterministic percentile bootstrap，固定 10,000 resamples 和配置 seed；不分别 bootstrap 两个 treatment。

H1 同时通过以下门禁才算 confirmatory pass：

1. 所有有效记录通过 oracle，且 correctness 优先于性能；
2. 每个 block count 至少 20 个完整 pair；
3. 每个 block count 的 paired kernel improvement 95% CI 下界不低于 -3%；
4. 至少一个 block count 的 95% CI 下界达到 10%；
5. 1/4/16/32 blocks 的 point estimate 非递增。

CI 表达当前采样下的不确定性，不证明跨 GPU 普遍成立。P95/P99 请求指标不得套用只有 20 pairs 的 microbenchmark bootstrap。

## 5. 无事后异常值删除

结果大或小都不是 invalid 理由。仅允许在观察 outcome 前可判定的错误：设备/API 失败、oracle 失败、缺失计时字段、harness 外部中断。原始记录必须保留 `valid=false` 和原因；确认性分析排除整个 pair，而不是只删较慢的一侧。温度、功耗或 P-state 的新阈值只能进入下一协议版本，不能回头清理 v1 数据。

## 6. 环境、Profiler 与边界

- 每次 CLI 调用实时捕获 GPU UUID/PCI ID、driver、P-state、温度、功耗、时钟、CUDA/CMake/Python 版本和受控环境变量。
- dirty outer/vendor run 会保留制品但 `protocol_compliant=false`；confirmatory 结果必须来自已提交代码。
- Nsight Systems/Compute 用于 Issue #4 的机制诊断，带 profiler 的延迟不进入无 profiler 主表。
- 当前确认性范围只有 Windows WDDM + RTX 4050 Laptop + 锁定 FP16 KV layout；不得外推 A100/H100、多 GPU、任意模型或 Paged Decode Attention。

## 7. Pilot、偏离与负结果

- pilot 与 confirmatory 使用不同目录和 manifest；pilot 不得同时选择参数又验证参数。
- 不追加样本直到 CI 过线；v1 使用固定 20 pairs/configuration。
- 任一命令、样本数、warm-up、随机化、计时、CI 或 gate 偏离都写入新版本或 amendment。
- 失败、OOM、correctness failure、反号结果和 dirty run 都是需要保存的结果，不允许静默重跑到成功。

