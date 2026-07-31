# CacheFlow Runtime 严格验收报告

验收基线：llama.cpp `acd79d603cb2e1c84c0886137b80f1ad649b6857`  
个人 fork：`codex/cacheflow-runtime`  
硬件：RTX 4050 Laptop 6 GiB（sm_89）、i5-13500H、Windows 11 WDDM  
模型：Qwen2.5-0.5B-Instruct Q4_K_M / Q8_0 / F16

## 结论状态

2026-07-31 从唯一入口执行 `verify.ps1 -Full`，545 秒后以退出码 0 完成。功能、正确性、性能证据和工程复现项均通过自动化入口。启用 WDDM debugger interface 后，额外执行 `build_cuda_kv.ps1 -Sanitize`：memcheck 为 0 errors，racecheck 为 0 hazards、0 errors、0 warnings，脚本退出码 0。

## 架构与个人贡献

| 项目 | 证据 | 状态 |
|---|---|---|
| Scheduler / KV / Spec 独立模块 | `server-inference-scheduler.*`、`server-kv-*`、`server-speculation-controller.*` | 通过 |
| Engine 事务边界 | prepare → immutable plan → execute → commit/abort；半执行禁止提交 | 通过 |
| 生产/测试同一 Seam | `server_runtime_adapter` 与 `KvBlockBackend` Interface | 通过 |
| 第三方边界 | 固定上游 + 单独 patch；不统计 `vendor/` 原仓库 | 通过 |
| 个人 C++/CUDA 差异 | 50 files，+5,798 / -54（相对固定上游） | 通过 |
| 阶段化 Git 历史 | Scheduler、KV、CUDA、控制、故障、事务、指标分 commit | 通过 |

## 功能与正确性

| 验收项 | 自动化证据 | 状态 |
|---|---|---|
| Token-level batching / chunked prefill | `test-inference-scheduler`、`run_adaptive_prefill_ab.py` | 通过 |
| Prefix Block 分享 / COW | `test-kv-block-manager` 随机性质测试、`run_kv_block_smoke.py --mode share` | 通过 |
| Admission / Preemption / Restore | `test-kv-capacity-planner`、preempt smoke、真实 CUDA swap smoke | 通过 |
| Host/File 事务 Swap | checksum、原子 rename、budget、save/restore failpoint | 通过 |
| Adaptive Prefill / Speculation | 独立开关、在线模型、EWMA/证据/迟滞测试及 CPU/CUDA A/B | 通过 |
| CUDA Gather/Scatter/COW/Swap | sm_89 编译、随机逐元素矩阵、真实 Qwen K/V 往返 | 通过 |
| OpenAI 非流式 / SSE | schema、标准 chunk、`[DONE]`、同 prompt cache hit | 通过 |
| 取消 / Deadline / 背压 | 断流取消后恢复、并发 deferred、10 ms deadline 后恢复 | 通过 |
| 故障回滚 | KV OOM、compute failure、host/file save/restore、CUDA allocation | 通过 |
| 上游兼容 | 同 MSVC 工具链、固定 seed，5/5 输出 SHA-256 相同 | 通过 |
| 模型矩阵 | Q4/Q8/F16 × CPU/CUDA；并发 1/2/4/8；128/512/2K/4K | 14/14 通过 |
| Compute Sanitizer | memcheck 0 errors；racecheck 0 hazards / 0 errors / 0 warnings | 通过 |

随机状态测试持续检查 Block 总量、Reservation、Refcount、Prefix Parent、Runtime Residency 与 Block Table 一致性。CUDA 随机测试覆盖 Block Size 8/16/32/64、非连续/重复/重叠映射、非整除尾部、in-place source 保护和多 Stream Event。

## 性能证据

### Tail COW（真实 RTX 4050，3 个 fresh process × 300 samples）

| 方法 | Median E2E ms | P95 E2E ms | Bytes/op | Launch/op | Extra device bytes |
|---|---:|---:|---:|---:|---:|
| Tail Block COW | 0.0083 | 0.0236 | 196,608 | 1 | 196,608 |
| Whole Sequence Copy | 0.7631 | 0.9990 | 12,582,912 | 128 | 12,582,912 |

P95 改善 97.64%。数据：`results/raw/cuda-kv-cow.csv`、`results/cuda-kv-cow-summary.json`。

### Adaptive Speculation（3 次 fresh process）

| Backend | Fixed median / P95 ms | Adaptive median / P95 ms | 结论 |
|---|---:|---:|---|
| CPU | 3197.05 / 3235.78 | 3175.64 / 3215.33 | 当前 trace 改善 |
| CUDA | 202.58 / 204.15 | 196.18 / 200.72 | 当前 trace 改善 |

### Adaptive Prefill（3 次 fresh process）

CUDA adaptive median 650.00 ms，优于错误 fixed-64 的 746.89 ms，但未击败 greedy median 642.61 ms；其 P95 654.95 ms 优于 fixed-256 的 657.32 ms。CPU 也没有对所有候选占优。因此这里只证明在线模型能避开明显错误参数点，不宣称 universally optimal。

### Gather/Scatter 负结果

完全不重叠映射上，staging Gather/Scatter 比 per-block `cudaMemcpyAsync` 慢。生产 Backend 因而对 disjoint mapping 使用直接异步复制，只在重叠/重复映射要求 snapshot 语义时使用 staging kernel。完整数据保留在 `results/cuda-kv-transport-summary.json`。

## 原生可观测性

- Request Histogram：TTFT、TPOT、request latency、queue latency，包含 `_bucket/_sum/_count`。
- Scheduler：iterations、decode/prefill token、chunk、batch token/sequence、prefill starvation。
- KV：used/free/shared、prefix hit、COW、evicted block、swap bytes、restore seconds、admission failure。
- Speculation：draft/accepted、acceptance、draft length、disabled reason、估算 net saved ms。
- CUDA：kernel/blocks/bytes、copy/swap Event 时间、带宽、waited Event、Pinned current/peak、errors。

`run_openai_compat_smoke.py` 会直接解析 `/metrics` 并验证类型声明，而不是只检查字符串存在。

## 复现命令

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\verify.ps1 -Full
```

Sanitizer 单独入口：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_cuda_kv.ps1 -Sanitize
```

Windows WDDM 首次运行前需以管理员身份执行 CUDA Toolkit 的 `EnableDebuggerInterface.bat`；当前验收机器已经启用。

## 解释边界

- 结果只适用于固定小模型、单机和当前硬件，不能外推到多 GPU 或大模型。
- 跨 MSVC/Clang 浮点路径出现过一处单词差异；逐字节兼容结论限定同工具链。
- `speculation_net_saved_ms` 是“接受 token × 当前 target 平均时间 − 实测 draft wall time”的在线估计，不是 profiler 的 causal attribution。
- Sanitizer 结论限定当前 Windows WDDM 驱动、RTX 4050 sm_89 与固定随机测试矩阵；换平台后需要复验。
- 负结果和调参历史保留在 `docs/experiment-limitations.md` 及 tuning-history CSV。
