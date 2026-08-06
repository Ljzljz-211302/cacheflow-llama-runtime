# CUDA profiling 因果链：H2 KV movement

## 1. 要回答的不是“哪个数字更小”

H2 研究问题是：在什么搬运规模和内存布局下，显式 KV Remap 会成为可观测成本，128-bit
向量化又在什么地方获益、失效或反转？证据必须串起：

```text
workload/layout intervention
  -> scalar or vectorized KV gather/scatter
  -> launch / memcpy / synchronization timeline (Nsight Systems)
  -> DRAM / L2 / occupancy counters (Nsight Compute, only when available)
  -> no-profiler CUDA-event effect
  -> no-profiler end-to-end effect
```

Profiler 会改变程序执行，因此带 profiler 的 wall time 不能进入性能主表。主效应来自同一进程内、
固定样本量、固定随机种子的 scalar/vectorized 配对试验；NSYS/NCU 只解释机制。

## 2. 预注册 workload

机器可校验配置在 `config/cuda_profile_protocol.json`。当前固定 Qwen 风格 FP16 KV layout：
32 layers、8 KV heads、head dimension 128、16-token block。四个 regime 为：

| regime | blocks | layout | 目的 |
|---|---:|---|---|
| `aligned-small` | 1 | 16-byte aligned | 观察固定成本占比高的小搬运 |
| `aligned-transition` | 16 | 16-byte aligned | 观察向量化收益衰减区 |
| `aligned-large` | 32 | 16-byte aligned | 寻找低于 10% 材料性门槛的中性区 |
| `misaligned-small` | 1 | source/destination/staging + 1 FP16 | 暴露 vector kernel 的 scalar-lane fallback，作为 layout 反例 |

每个 regime 先运行 20 个无 profiler pair；pair 内顺序由记录在原始行中的 seed 随机化。之后用
同一 binary/config 运行 5 个 profiler pair。固定 10,000 次 paired percentile bootstrap，报告
GPU event 与 end-to-end 改善的 median 和 95% CI。低于 `+10%` 不写成材料性收益；低于
`-3%` 的回退必须保留为负结果。

## 3. 一条命令复现

先构建经过 CUDA oracle 验证的 benchmark：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build_cuda_kv.ps1 -Sanitize
```

设置 CLI 路径后执行正式实验：

```powershell
$env:NSYS_PATH = 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.3.1\target-windows-x64\nsys.exe'
$env:NCU_PATH = 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.4.1\target\windows-desktop-win7-x64\ncu.exe'
python scripts/run_cuda_profile_experiment.py `
  --output-dir results/research/h2-kv-profile-v1.0.0
```

若 NCU 因 Windows 性能计数器权限或 driver/tool compatibility 失败，只允许显式运行：

```powershell
python scripts/run_cuda_profile_experiment.py `
  --output-dir results/research/h2-kv-profile-v1.0.0 `
  --allow-ncu-unavailable
```

这个开关不会伪造通过：manifest 中 `ncu_complete=false`，报告自动禁止 memory-bound、roofline、
achieved occupancy 和硬件 DRAM byte 主张。NSYS 仍必须真实采集成功，否则整次运行失败。

## 4. 实际 capture 边界

Benchmark 的 warm-up 在 `cudaProfilerStart()` 之前；正式 profiler pairs 位于
`cudaProfilerStart/Stop` 之间。Windows NSYS 使用 `cuda,nvtx`，Linux 增加 `osrt`。脚本实际
生成并记录等价于下列命令的 argv：

```text
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none
  --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown
  --output=<regime>/nsys bench-kv-block-cuda --profile ... --method paired

nsys export --type=sqlite --output=<regime>/nsys.sqlite <regime>/nsys.nsys-rep

ncu --profile-from-start=off --kernel-name-base=demangled
  --kernel-name=regex:llama_kv_remap_.* --replay-mode=kernel
  --cache-control=all --clock-control=base --metrics=<locked metrics>
  --page=raw --csv --export=<regime>/ncu bench-kv-block-cuda --profile ...
```

NCU metrics 锁定为 kernel duration、DRAM read/write bytes、DRAM peak-throughput percentage、
L2 sector hit rate 和 achieved occupancy。NCU replay/cache/clock control 会扰动自然执行，故只能与
相同 `regime_id` 的无 profiler 数据关联，不能把 NCU duration 当作服务延迟。

## 5. 解析与防止越界结论

`src/llama_lab/cuda_profile_evidence.py` 直接解析 NSYS SQLite 的 CUPTI kernel、memcpy 和 runtime
synchronization 表，并解析 NCU raw-page CSV。解析器只接收名称匹配自研 KV kernel 的行；NCU
replay 过程中产生的 host elapsed time 不进入结果结构。

下列判定是硬约束：

- scalar/vector 的 launch 数相同，只能否定“减少 launch 数”，不能因此证明 launch overhead 不重要；
- `effective_payload_gbps` 是逻辑 payload bytes / CUDA-event time，不是硬件 DRAM bandwidth；
- 没有 NCU DRAM throughput 就不能写 memory-bound 或 roofline；
- 没有 NCU occupancy metric 就不能用 occupancy 解释输赢；
- layout 结论必须同时包含 aligned 与 misaligned 的无 profiler pair；
- 每个机制判断都必须链接同一 regime 的 raw trace 和无 profiler end-to-end CI；
- misaligned 或 large regime 输掉时必须作为 counterexample 保留。

## 6. Artifact 结构与审计

```text
results/research/h2-kv-profile-v1.0.0/
  manifest.json                 # revision、dirty flag、tool version、commands、hashes
  trials.jsonl                  # 4 regimes x 20 pairs x 2 methods
  report.json                   # 机器可读 effect、CI、NSYS/NCU 机制证据
  report.md                     # 面试/评审可读结论与限制
  raw/<regime>-no-profiler.csv
  profiles/<regime>/nsys.nsys-rep
  profiles/<regime>/nsys.sqlite
  profiles/<regime>/ncu.csv     # 成功指标或可审计失败日志
```

输出目录必须为空；正式运行前 outer/vendor worktree 必须 clean。`limited_claims_protocol_compliant`
只表示“在不声称硬件计数器结论的限定范围内合规”，绝不等价于 `ncu_complete=true`。

## 7. 服务级关联实验

算子 `end_to_end_ms` 只表示 descriptor upload、kernel 与同步完成，不能替代 TTFT/TPOT。服务级链路由另一条确认性命令补齐：

```powershell
python scripts/run_service_nsight_causal_experiment.py `
  --trials 3 `
  --output-dir results/research/h2-service-nsight-causal-v1.0.0
```

该命令先运行 3 组无 profiler upstream/always paired service trials，得到主要的 Engine/TTFT 效应；再用相同模型、请求 seed、trial/mode 配置在 NSYS 下重放。每个请求带确定性 `request_id`，每个 server 记录 PID；SQLite parser 按 NSYS `globalPid` 过滤，要求自研 KV launch 数与同一 server 的 Prometheus counter 完全相等。`causal-links.json` 因而逐 trial/mode 连接：

```text
trial_id + server_pid + request_ids
  -> benefit decision / prefill chunks / prefill tokens
  -> KV kernel launches / copied bytes / CUDA-event time
  -> PID-filtered NSYS kernel/memcpy timeline
  -> request TTFT samples / TTFT P95 / Engine execute duration
```

无 profiler run 仍拥有性能主张，NSYS replay 只拥有机制归因；两者通过冻结的 workload configuration 与 request seeds 关联，不把 profiler 扰动后的请求延迟冒充自然执行结果。

## 8. NVIDIA 一手资料

- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)：CLI capture、
  `cudaProfilerApi` range 与 SQLite export。
- [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)：kernel filter、
  raw CSV、report export 与 metric 选择。
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)：replay、
  cache/clock control、采集开销与可复现性边界。
- [NVIDIA ERR_NVGPUCTRPERM](https://developer.nvidia.com/ERR_NVGPUCTRPERM)：硬件性能计数器权限配置。
