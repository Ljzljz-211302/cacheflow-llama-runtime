# CacheFlow 性能实验预注册：方法学与一手资料基础

> 状态：GitHub Issue #3 的协议输入，2026-08-06。本文不是实验结果，也不是已经冻结的预注册；它只把可执行规则追溯到 NVIDIA、NIST、OSF 与 ACM 的一手资料，并标出由这些资料推导出的项目约束。

## 1. 结论

CacheFlow 后续性能实验应当同时保留三个互不替代的 estimand（待估量）：

1. **device interval**：同一 CUDA stream 上 start/stop event 包围目标 GPU 工作，以 `cudaEventElapsedTime` 读取；
2. **host completion latency**：主机单调时钟包围完整调用路径，结束边界必须等待相应 GPU 工作完成；
3. **host enqueue cost**：只量主机提交工作所花的时间，不等待 GPU 完成，并明确标记为 enqueue 而不是执行延迟。

无 profiler 的运行产生主性能数字；Nsight Systems 用来建立 CPU API、stream、copy、kernel 与同步之间的时间线，Nsight Compute 用来解释 kernel 的内存、指令与占用瓶颈。两类 profiler 均会改变执行，因此其运行不能替代无 profiler 基准。

比较 Direct、Scalar Remap、Vector Remap、Paged Decode 等 action 时，以“相同 workload/configuration 的一次完整 action 轮换”为 block；每个 block 内所有 action 各执行一次且顺序随机，分析配对差值或配对比率。样本数、warm-up、失效条件、主/次指标、置信水平和停止规则必须在看见确认性结果前锁定。

## 2. 主张到来源的最小映射

| 编号 | 可采用的主张 | 一手来源 | 对 CacheFlow 的直接约束 |
|---|---|---|---|
| M1 | CUDA kernel 与带 `Async` 后缀的 copy 通常异步返回；主机计时若要得到完成时间，需要在边界正确同步。 | NVIDIA [CUDA C++ Best Practices：Using CPU Timers](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#using-cpu-timers) | 不得把未同步的 host wall time 写成 kernel 或端到端完成延迟；若测一个隔离的 CUDA 调用序列，先排空既有工作，再在结束处等待完成。 |
| M2 | CUDA event 在其 stream 到达 event 时由 GPU 记录时间戳；`cudaEventElapsedTime` 返回两 event 间 GPU 时间。 | NVIDIA [CUDA C++ Best Practices：Using CUDA GPU Timers](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#using-cuda-gpu-timers)；[CUDA Driver Event API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html) | event time 只解释所定义的 device interval；它不包含 event 入队前的主机准备、请求排队和业务层开销。 |
| M3 | stream/event 同步只保证相应 stream/event 的完成；其他非默认 stream 的工作可能被驱动穿插。 | NVIDIA [CUDA C++ Best Practices：Using CPU Timers](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#using-cpu-timers) | device microbenchmark 要独占设备并固定 stream；若保留并发，就把并发明确定义为 workload，而不能宣称隔离 kernel time。 |
| M4 | lazy module loading 与 PTX JIT 可把首次加载成本带入首次调用；预加载或至少一次 warm-up 可把 steady-state 与首次成本分开。 | NVIDIA [Lazy Loading：Impact on Performance Measurements](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html#impact-on-performance-measurements)；[JIT Compilation](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html#just-in-time-compilation) | cold-start 和 steady-state 分表；warm-up 次数、输入和是否丢弃必须预先写死，不能看到数据后再加热。 |
| M5 | Nsight Systems 可由 `cudaProfilerStart/Stop` 或 NVTX capture range 控制采集边界。 | NVIDIA [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) | 用命名 NVTX range 精确包围确认性 trial，排除模型加载和 warm-up；保存实际 CLI 与 `.nsys-rep`。 |
| M6 | Nsight Compute 的指标采集可能需要多 pass replay、内存保存/恢复、cache/clock 控制并引入开销；被 profiler 包围时，host timer/CUDA event 会包含工具影响。 | NVIDIA [Nsight Compute Profiling Guide：Replay](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#replay)；[Overhead](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#overhead)；[Reproducibility](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#reproducibility) | NCU 仅作机制诊断；主表延迟来自单独的无 profiler 运行。报告 replay mode、section/metrics、cache-control、clock-control 和过滤条件。 |
| M7 | 随机区组设计用 block 吸收已知 nuisance factor，并在 block 内比较 treatment；NIST 的原则是“block what you can, randomize what you cannot”。 | NIST/SEMATECH [Randomized Block Designs](https://www.itl.nist.gov/div898/handbook/pri/section3/pri332.htm) | 同一 block 内轮换全部 action，随机化 action 顺序；时间段、workload shape、进程代次等作为 block/stratum，而不是把各 action 分批跑完。 |
| M8 | 自然配对的数据应先形成逐对差值；均值差的置信区间应基于这些配对差值，而非把两组当独立样本。 | NIST [Analysis of Paired Observations](https://www.itl.nist.gov/div898/handbook/prc/section3/prc311.htm)；[Confidence Intervals for Differences](https://www.itl.nist.gov/div898/handbook/prc/section3/prc312.htm) | 原始记录必须含 `block_id`；主分析报告逐 block 的差值/比率及其区间，不只报告两组各自均值。 |
| M9 | bootstrap 可估计复杂统计量的采样不确定性，但并非适合所有分布与尾部统计量。 | NIST [Bootstrap Plot](https://www.itl.nist.gov/div898/handbook/eda/section3/bootplot.htm)；Efron 1979 [原始论文](https://projecteuclid.org/journals/annals-of-statistics/volume-7/issue-1/Bootstrap-Methods-Another-Look-at-the-Jackknife/10.1214/aos/1176344552.full) | 非参数区间应整 block 重采样以保留配对；固定 bootstrap seed/replicate 数。P95/P99 等尾部结论不得用少量样本的普通 bootstrap 伪装成稳健证据。 |
| M10 | outlier 的定义依赖对“正常过程”的刻画；顺序趋势和过程漂移必须检查。 | NIST [What Are Outliers?](https://www.itl.nist.gov/div898/handbook/prc/section1/prc16.htm)；[Trends in Sequential Data](https://www.itl.nist.gov/div898/handbook/prc/section1/prc17.htm)；[Is the Process Stable?](https://www.itl.nist.gov/div898/handbook/pri/section2/pri22.htm) | 不因数值大/小而事后删样本。只按预注册的“trial 无效”条件排除；原始行、原因和含全部样本的敏感性分析都保留。 |
| M11 | OSF 将 preregistration 定义为数据收集/分析前发布的带时间戳、只读研究计划，并要求提前说明变量、检验、决策标准和排除规则。 | OSF [Registrations & Preregistrations](https://help.osf.io/article/330-welcome-to-registrations) | 研究问题、指标、样本量、排除、停止与替代分析先冻结；偏离计划时新增带时间戳 amendment，不能覆盖旧规则。 |
| M12 | `nvidia-smi` 可查询 GPU identity、driver、P-state、温度、功耗、时钟与降频原因。 | NVIDIA [nvidia-smi Documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html) | 每次实验前后保存原始 GPU 状态；跨启动用 UUID/PCI bus ID 标识设备，不只记录易变的 device index。 |
| M13 | ACM 的 Functional artifact 要求 documented、consistent、complete、exercisable，并包含验证证据；Available 要求可长期检索的公开归档与唯一标识。 | ACM [Artifact Review and Badging](https://www.acm.org/publications/policies/artifact-review-and-badging-current) | 结果包至少包含版本化代码、manifest、原始数据、分析程序、运行说明和校验值；GitHub 工作分支本身不等于永久归档。 |

## 3. 由来源导出的计时口径

### 3.1 Device interval

- start/stop event 与目标工作放在同一明确记录的 stream；stop event 完成后读取 elapsed time。
- event 窗口应只包围预注册的工作集合。例如 Remap 可以分别定义 `gather`、`scatter` 和二者合计三个窗口，但三者不能混称。
- 多 stream 方案必须用显式依赖构造真正的 envelope，或改用 Nsight 时间线说明；单 stream event 不能证明整个设备在该时段只做了这些工作。
- 报告毫秒和实际处理 bytes，并由此计算有效带宽；“memory-bound”仍需 NCU memory/roofline 证据，不能从低 FLOP 直接推定。

### 3.2 Host completion latency

- 若目标是隔离调用序列：计时前排空此前 GPU 工作，启动单调时钟，提交目标工作，在完成边界同步，再停止时钟。
- 若目标是真实请求 latency：起点是请求进入已定义服务边界，终点是结果已可由调用方消费；排队、调度、CPU、传输和同步均属于该 estimand，不得用一次全局同步人为删除。
- CPU/GPU 同步会使 pipeline 停顿，因此只放在 measurement harness 或真实业务完成边界，不提交为正常 fast path 的额外同步。

### 3.3 Host enqueue cost

- 主机计时只包围 API/launch 提交，不同步；名称固定为 `host_enqueue_us`。
- 它可以解释 launch 数量、descriptor 构造和调度开销，但不能与 `device_ms` 或 `request_latency_ms` 互换。

### 3.4 Cold-start 与 steady-state

- cold-start 使用 fresh process，并单独记录 context/module/JIT/model 初始化范围。
- steady-state 在预先固定的 warm-up 后开始；warm-up 行保留但标记 `phase=warmup`，不进入确认性统计。
- 不允许根据“曲线看起来还没稳定”临时增加待丢弃样本；若预注册 warm-up 不足，整轮标为协议偏离并重开新的版本化实验。

## 4. Nsight 的测量边界

### 4.1 Nsight Systems：建立因果时间线

推荐的 range 层级是：

```text
cacheflow.confirmatory_trial
  block/<block_id>
    action/<direct|scalar_remap|vector_remap|paged>
      plan
      descriptor
      h2d_or_d2d
      gather
      scatter
      decode_attention
      synchronize
```

只采集必要的 CUDA、NVTX 和 OS runtime 信息；用 NVTX 或 CUDA Profiler API 启停 capture。保存命令行、工具版本、trace 配置和报告文件。时间线用于验证“谁等待谁、copy 与 kernel 是否重叠、是否出现意外同步/其他 context”，而不是把带追踪开销的 wall time 抄进无 profiler 主表。

### 4.2 Nsight Compute：解释 kernel

- 按 kernel 名称、launch instance 或 NVTX range 限定尽量少的目标；先用最小 section/metric set 回答一个机制问题。
- 固定并记录 replay mode、application determinism、`cache-control`、`clock-control`、section/metric 名单和工具版本。
- cache flush 带来更一致但不完全自然的 cache 初态；保留自然 cache 更接近应用，却依赖先前工作。两种口径必须分组，不能混入同一估计量。
- NCU 的 replay 和插桩会改变执行。NCU 报告的 metric 与无 profiler latency 通过共同的 `trial_config_id` 关联，但不把两者当成同一次观测。

## 5. 配对、随机化和统计规则

### 5.1 Block 与执行顺序

一个 block 固定：模型/张量、KV layout、mapping、batch/context、进程热状态、stream 拓扑、功耗策略和一次 workload 实例。block 内每个 action 恰好运行一次；action 顺序由记录在 manifest 中的 PRNG seed 随机排列。下一 block 再产生新排列。

这样做是从 NIST 随机区组设计到本项目的**方法学推导**，不是 NVIDIA 对 GPU benchmark 的强制规范。它降低随时间温升、boost、后台负载和 cache 漂移与 action 顺序绑定的风险；仍应画 run-order 图并检查残差趋势。

### 5.2 效应量与区间

每个比较预先指定一个 reference action，并至少报告：

- 绝对效应：`candidate - reference`，单位与原指标相同；
- 相对效应：`candidate / reference - 1`，另给易读的 speedup；
- 全部逐 block 配对值、点估计和双侧置信区间；
- correctness、OOM、deadline miss 等不能压缩成平均延迟的结果。

若确认性 estimand 是配对均值差，可采用 NIST 的 paired-t interval，但应检查配对差值的分布与顺序稳定性。若使用 median、geometric ratio 或其他非正态统计量，则预先指定整 block bootstrap、置信水平、replicate 数和 seed。CI 表达估计不确定性；“区间跨过零/一”不能证明两方案等价，等价结论需要预注册实际意义阈值。

主结论不能只给 p-value 或单个 speedup。多 workload/多指标的主张数量、primary endpoint 与是否做多重比较校正应在最终预注册中锁定；本文引用的 NIST 页面没有替 CacheFlow 选定具体校正方法。

### 5.3 样本量与停止

- 最简单且最可审计的规则是每个 configuration 固定 `N` 个完整 block；`N` 在确认性采集前由 pilot 或资源预算确定。
- pilot 与 confirmatory 数据分目录、分 manifest；不得用同一批 pilot 同时选参数又验证该参数。
- 不反复加样本直到 CI 或显著性“过线”。若采用精度驱动的序贯停止，算法、最小/最大 block 数、检查点和覆盖率修正必须预注册；否则采用固定 `N`。
- 失败、OOM、超时和 correctness failure 都是结果。除非配置本身违反预先声明的有效域，否则不能静默重跑到成功。

## 6. 无事后删除的异常值策略

“异常慢”不是无效样本。确认性数据只允许按与 outcome 数值无关、且预先可机器判断的条件标为 invalid，例如：

- CUDA/API 返回错误，输出 correctness gate 失败；
- 实际模型/hash、shape、action 或构建版本与 manifest 不符；
- 同卡出现未允许的其他 compute process；
- telemetry 显示预注册禁止的 thermal/power throttle、温度范围或 P-state；
- profiler 意外开启，或所需 event/NVTX 边界缺失；
- harness 中断导致一个 block 未包含全部 action。

每个 invalid block 仍写入原始表，记录机器可读 `invalid_reason`，并且整个 block 的所有 action 一并从配对确认性分析剔除。主报告同时给：预注册有效集、包含所有可读取观测的敏感性分析、每种失效原因的计数。任何新排除规则只能作为带时间戳 amendment 用于未来实验；对已有数据只能标成 exploratory。

## 7. 环境与可复现制品

### 7.1 每次 run 必须捕获

- Git commit、dirty flag、submodule/vendor revision；编译器/CMake/CUDA Toolkit/driver/Nsight 版本；完整 build flags 与 binary SHA-256；
- 模型、输入 trace、配置与生成器的 SHA-256；seed、命令行、环境变量、工作目录和时间戳；
- OS/build、Windows driver model（如 WDDM/TCC）、CPU、RAM；
- GPU UUID、PCI bus ID、产品名、compute capability、driver、P-state、当前 SM/memory clocks、温度、功耗/limit、clock-event/throttle reasons；不支持字段保留 `N/A`；
- 实验前后 `nvidia-smi` 原始 CSV/XML，以及运行期间固定频率 telemetry；
- trial/block/action/phase、计时口径、stream、warm/cold、实际 bytes、kernel/copy 数、correctness 和 failure 字段。

### 7.2 Artifact 最小目录语义

```text
artifact/
  README.md                 # 从干净环境复现的命令与预计资源
  manifest.json             # 版本、环境、假设、指标、排除和 seed
  checksums.sha256
  raw/                      # 只追加的逐 trial 数据与 telemetry
  processed/                # 由脚本生成的表，不手改
  analysis/                 # 固定依赖、统计脚本与图表脚本
  profiles/nsys/            # CLI、配置、NVTX schema、报告
  profiles/ncu/             # CLI、metrics、replay/cache/clock 设置、报告
  logs/                     # build、correctness、失败与协议偏离
```

原始数据只追加；派生表必须能从 `raw/` 一条命令重建。发布时固定 release/tag 和归档 DOI/唯一标识；README 说明哪些结论可在当前 RTX 4050/Windows 上复现，哪些依赖 WSL/Linux、其他 GPU 或尚未实现的 kernel。依据 ACM 定义，作者自己重跑成功只支持 repeatability/functional artifact，不等于已经由独立团队复现结果。

## 8. 已知限制与不得外推的内容

- NVIDIA 文档定义了工具语义和已知扰动，但没有替本项目决定 action、样本量、实际意义阈值或主指标；这些仍需在 Issue #3 的正式协议中锁定。
- NIST 示例主要面向独立/稳定过程。GPU trial 可能有自相关、异方差、温控与频率漂移；配对区组降低但不消除这些问题。发现残差趋势时应报告、建模或重做预注册实验，不能靠删点隐藏。
- 普通 nonparametric bootstrap 假定作为重采样单位的 block 足以近似独立；对强时间依赖和少量 P95/P99 样本不保证覆盖率。必要时应预注册更保守方法或增加独立进程/会话层重复。
- CUDA event 精确描述其 stream 中的 device 时间，不自动代表服务请求体验；host end-to-end 也不能单独定位 GPU 瓶颈，因此两者必须并列。
- Nsight 的 metric、replay、cache 和 clock 设定会改变自然执行。Profiler 证据只能支持限定配置中的机制解释，不能直接外推到其他 GPU、driver、模型或 production concurrency。
- `nvidia-smi` telemetry 是离散采样，可能漏掉短时 thermal/power 事件；应保留 clock-event counters/原因并承认采样分辨率限制。
- Git commit 有时间戳但仍可改写历史，不等价于 OSF 的只读 preregistration；正式确认性实验应发布不可变快照或可核验 release，并把 amendment 追加而非覆盖。

## 9. Issue #3 应据此冻结的字段

正式协议至少要把以下字段变成机器可校验配置：研究主张与方向、configuration grid、baseline/action、primary/secondary endpoint、三种计时口径、cold/warm 定义、block 构造与随机 seed、固定样本量/停止规则、CI/effect-size 算法、失效条件、正确性门禁、telemetry 阈值、Nsight capture/metric 配置、artifact schema、negative-result 与 amendment 规则。

在这些字段被冻结并通过校验前，任何新采集只能标为 pilot/exploratory，不能用于确认 H1–H5。
