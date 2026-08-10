# Copy-aware Paged KV：前人工作与可复现基线审计

> 状态：Issue #10 的研究输入，2026-08-06。本文只采用原始论文、作者维护的代码仓库和厂商官方文档。论文中的性能数字仅用于解释原工作的适用范围，不作为本项目的预期结果。

## 1. 审计结论

本项目在当前机器上能够形成严格定量结论的基线，应限定在**同一个固定版本 llama.cpp、同一 MSVC/CUDA 构建、同一 GGUF 模型、同一请求轨迹**内：

1. `B0 upstream`：固定 llama.cpp `acd79d603cb2e1c84c0886137b80f1ad649b6857` 的原生调度与 KV 路径；
2. `B1 direct-copy`：无重叠、无重复目标映射时，每个 K/V block 使用 `cudaMemcpyAsync`；
3. `B2 scalar-remap`：需要 snapshot 语义时使用标量 Gather + Scatter；
4. `B3 vector-remap`：相同语义、相同 descriptor 和 staging，改用 `uint4` 的 128-bit Gather + Scatter；
5. `H0 deterministic-rule`：简单规则基线——映射不需要 snapshot 时选 Direct，否则选 Vector Remap；服务调度实验另锁定现有 `rule` 策略（仅当 CacheFlow prefill token 更少、active sequence 大于 1 且 chunk 大于 1 时启用）。

其中 `B0` 是系统级对照，`B1`—`B3` 是机制级对照，`H0` 是必须被学习策略超越的简单启发式。它们均已存在于本仓库、可在 Windows 11 + RTX 4050 Laptop 6 GiB 上原生运行，因而是论文式主表中唯一无条件合格的 quantitative baselines。

[FlashInfer](https://github.com/flashinfer-ai/flashinfer) 可在补齐 Linux/WSL 环境和布局适配后成为**外部 kernel oracle**，但不能与 CacheFlow 做端到端归因。vLLM、Sarathi-Serve、Orca、FastServe、FlexGen 和 vAttention 保留为相关工作或可行性参照，不应把它们在不同模型、硬件和软件栈上的论文数字塞进同一性能表。

## 2. 固定实验环境与可比性等级

当前可用环境来自仓库的已记录事实：Windows 11、RTX 4050 Laptop 6 GiB（Ada，`sm_89`）、i5-13500H、CUDA 12.6、MSVC 19.37；主模型为 Qwen2.5-0.5B-Instruct GGUF，Q4/Q8 权重与 FP16 KV，研究切片布局为 32 层、8 KV heads、head dimension 128、block size 16、单 token decode、单 GPU。

本文采用四级分类：

- **Q0：严格定量基线**——同进程或 fresh-process paired trial，只改变一个机制，模型、布局、工具链、请求序列和统计口径相同。
- **Q1：条件式 kernel comparator**——可在同一 GPU 上用相同 FP16 Q/K/V、page size、head shape 和输出容差比较 kernel；不能据此声称服务级提速。
- **R：概念相关工作**——机制能够解释研究假设，但实现、模型、硬件或目标函数不同，不进入定量主表。
- **X：当前不可复现**——现有 OS、显存、驱动权限或项目许可不足；只能引用论文结论。

为避免 `main` 漂移，审计时读取的实现快照为：llama.cpp `acd79d603cb2e1c84c0886137b80f1ad649b6857`、vLLM `62a86318de3655f970baf7c2ff89c81a72c1a1b3`、FlashInfer `3e96dfa35ee3f3b69f8f7ea04c4f44e1ca49703e`、FlashAttention `d7e4dba3e568106b0f1b6323b07c1272f53679b3`、Sarathi-Serve `96f9911790ecc00af12ee9fae47cb8fa9ba0d199`、vAttention `ef3fff25dbe4e10f5897da8648718c53df6a20ea`、FlexLLMGen `004ffef82b46e8dc8685c55d0cdda650bdaf1269`、FastServe `187a2742bd3eb8514d8609d80ad98e105621feed`、TensorRT-LLM `8e588da4a25cdb3830c93046f9097a6291cfb524`。Orca 没有与论文对应的官方可运行仓库，故只固定论文版本。真正进入 Q1 的实现仍须在实验 manifest 中再次固定 revision、wheel/source hash 和完整依赖。

## 3. 前人工作与实现矩阵

| 工作 | 原始机制与官方材料 | 许可 | 模型/布局及硬件假设 | 单 RTX 4050 评估 | 分类与使用方式 |
|---|---|---|---|---|---|
| llama.cpp | 固定官方 [source](https://github.com/ggml-org/llama.cpp/tree/acd79d603cb2e1c84c0886137b80f1ad649b6857)、[CUDA build guide](https://github.com/ggml-org/llama.cpp/blob/acd79d603cb2e1c84c0886137b80f1ad649b6857/docs/build.md)、[MIT license](https://github.com/ggml-org/llama.cpp/blob/acd79d603cb2e1c84c0886137b80f1ad649b6857/LICENSE)。GGUF 模型执行、跨平台 CPU/CUDA backend；CUDA 官方构建入口为 `-DGGML_CUDA=ON`。 | MIT | GGUF；原生 Windows/CUDA 可构建。项目已固定同模型、同 seed、同 MSVC 的 upstream commit。 | 已原生运行；唯一公平的系统级对照。 | **Q0**。锁定 `B0 upstream`；不得用每日变化的 master 替代固定 commit。 |
| CacheFlow Direct / Scalar / Vector Remap | 本仓库固定 fork 的 `server-kv-block-cuda.cu` 与 `llama-kv-cache-paged.cu`。Direct 对非 snapshot 映射逐 K/V block D2D copy；Remap 使用 descriptor、staging、Gather/Scatter；两种 remap 只有标量/128-bit 搬运宽度不同。 | 与 fork 一致，MIT | 当前真实模型切片为 Qwen2.5、FP16 KV、GQA、block 16；Windows CUDA 12.6、`sm_89`。 | 已有正确性、Sanitizer、真实 tensor 和 paired microbenchmark 路径。 | **Q0**。锁定 `B1`—`B3` 与 `H0`；它们回答“复制何时值得、向量化何时失效”，不是 PagedAttention 的替身。 |
| PagedAttention / vLLM | [SOSP 2023 原始论文](https://arxiv.org/abs/2309.06180)提出 block table、非连续 KV block、共享和 COW；固定[官方实现](https://github.com/vllm-project/vllm/tree/62a86318de3655f970baf7c2ff89c81a72c1a1b3)提供 PagedAttention、continuous batching、chunked prefill 和 prefix caching。 | Apache-2.0（[license](https://github.com/vllm-project/vllm/blob/62a86318de3655f970baf7c2ff89c81a72c1a1b3/LICENSE)） | 当前实现支持 Llama/Qwen 等大量 HF 架构；历史官方 GPU 安装要求为 Linux、compute capability ≥ 7.0（[官方文档](https://docs.vllm.ai/en/v0.7.0/getting_started/installation/gpu/index.html)）。其 HF/PyTorch layout、allocator、scheduler 与 GGUF/ggml 不同。 | `sm_89` 满足 GPU 能力且 0.5B 模型显存可行，但 native Windows 不满足官方 OS；WSL/Linux 才能尝试。 | **R**（端到端）；未来可在相同请求分布做非归因 sanity check，但不得作为 Copy/Remap/Paged 单变量主基线。 |
| FlashInfer | [原始论文](https://arxiv.org/abs/2501.01005)把 KV 存储异构性表达为 block-sparse/composable formats；官方 [paged-KV layout](https://docs.flashinfer.ai/tutorials/kv_layout.html)与 `BatchDecodeWithPagedKVCacheWrapper`直接消费 page indices，官方仓库提供 [reference checking 与 kernel benchmark](https://github.com/flashinfer-ai/flashinfer)。 | Apache-2.0（[license](https://github.com/flashinfer-ai/flashinfer/blob/main/LICENSE)） | NHD/HND paged layouts，FP16 等 dtype；当前官方仓库列出 CUDA 12.6/12.8/13.x。Python/PyTorch + JIT，主要面向 Linux serving stack。 | Ada 能力足够，6 GiB 足够做小型 kernel test；但需 WSL/Linux、PyTorch、固定 FlashInfer revision，并写 layout adapter。 | **Q1（有条件）**。只比较相同 Q/K/V、batch、context、page size、GQA shape 的输出与 CUDA-event kernel time；不得把框架开销或论文 GPU 数字并入主表。 |
| FlashAttention | [NeurIPS 2022 原始论文](https://arxiv.org/abs/2205.14135)证明通过 tiling 减少 HBM 与片上 SRAM 间读写的 exact attention；固定[作者官方实现](https://github.com/Dao-AILab/flash-attention/tree/d7e4dba3e568106b0f1b6323b07c1272f53679b3)。 | BSD-3-Clause（[license](https://github.com/Dao-AILab/flash-attention/blob/d7e4dba3e568106b0f1b6323b07c1272f53679b3/LICENSE)） | 原始贡献是 IO-aware dense exact attention，不等于用户态 paged KV 管理；支持的 GPU/dtype/head dimension 随版本变化。 | 可解释在线 softmax、tiling 与 IO 机制，但不是当前 block-table decode 的同义基线。 | **R**。用于形成“消除 remap 是否抵消额外索引/归约成本”的机制假设，不直接报横向速度。 |
| Sarathi-Serve | [OSDI 2024 原始论文](https://www.usenix.org/conference/osdi24/presentation/agrawal)提出 chunked prefill 与 stall-free batching；固定[官方代码](https://github.com/microsoft/sarathi-serve/tree/96f9911790ecc00af12ee9fae47cb8fa9ba0d199)明确是 vLLM fork/research prototype，并提供论文 figure 脚本。 | Apache-2.0（官方仓库） | 官方仅声明测试 CUDA 12.3、A100/H100；重点模型包括 Mistral-7B、Yi-34B、Falcon-180B，软件环境为 Linux/Python。 | 6 GiB 4050 与论文硬件/模型不匹配，且当前项目已有独立 chunked-prefill 实现。 | **R**。作为 prefill/decode 干扰、TTFT/TBT 权衡的概念先例；不复用其性能数字。 |
| Orca | [OSDI 2022 原始论文](https://www.usenix.org/conference/osdi22/presentation/yu)提出 iteration-level scheduling 与 selective batching。 | 论文开放访问；未发现与论文对应、可锁定的官方开源实现许可证。 | 分布式系统，论文主要评估 GPT-3 175B 和多 GPU 环境。 | 机制与 CacheFlow Serving Iteration 同源，但原实现不能在 4050 上等价复现。 | **R/X**。用于阐明 iteration 边界，不作为代码或定量 baseline。 |
| FastServe | [原始论文](https://arxiv.org/abs/2305.05920)研究 token 粒度抢占、MLFQ 与 GPU/Host state offload；固定[作者组织代码](https://github.com/LLMServe/FastServe/tree/187a2742bd3eb8514d8609d80ad98e105621feed)支持 OPT、LLaMA2，并依赖 SwiftTransformer。 | 截至审计时官方仓库根目录没有 `LICENSE`，不能推定为开源许可。 | 定位为 distributed inference；模型与 CacheFlow 的 Qwen2.5/GGUF 切片不同，构建依赖另一 C++ runtime。 | 既不满足同栈公平比较，也不应在许可不明时复制代码。 | **R/X**。只引用抢占/换入换出机制；不导入源码、不报告横向速度。 |
| FlexGen / FlexLLMGen | [ICML 2023 原始论文](https://arxiv.org/abs/2303.06865)用线性规划在 GPU/CPU/磁盘间放置权重、激活和 KV，并面向 latency-insensitive batched throughput；固定[官方代码](https://github.com/FMInference/FlexLLMGen/tree/004ffef82b46e8dc8685c55d0cdda650bdaf1269)。 | Apache-2.0（[license](https://github.com/FMInference/FlexLLMGen/blob/004ffef82b46e8dc8685c55d0cdda650bdaf1269/LICENSE)） | 官方路径主要支持 OPT；论文的 commodity baseline 为 16 GiB GPU，目标包含远大于显存的模型。 | 4050 可研究小模型 offload，但目标函数、模型格式、执行引擎和显存规模均不同。 | **R**。用于 Swap/Recompute/Offload 代价项，不进入在线低延迟主表。 |
| vAttention | 固定[作者官方仓库](https://github.com/microsoft/vattention/tree/ef3fff25dbe4e10f5897da8648718c53df6a20ea)通过 CUDA virtual memory 将虚拟连续性与物理按需分配解耦，使未修改 attention kernel 可用；仓库包含 PagedAttention/vAttention 对照脚本。 | MIT（[license](https://github.com/microsoft/vattention/blob/ef3fff25dbe4e10f5897da8648718c53df6a20ea/LICENSE)） | 官方测试为 Linux、A100、PyTorch 2.3、CUDA 12.1；Yi/Llama，长上下文；小于 2 MiB 物理页还需替换 NVIDIA UVM driver。 | 当前 Windows/WDDM 且无驱动修改范围，无法公平复现；2 MiB page 也与 16-token logical block 语义不同。 | **R/X**。是“分页索引 vs 虚拟连续”的重要反例，但不作为当前 4050 quantitative baseline。 |

TensorRT-LLM 没有进入首轮基线：虽然固定的[官方项目快照](https://github.com/NVIDIA/TensorRT-LLM/tree/8e588da4a25cdb3830c93046f9097a6291cfb524)包含 paged KV、in-flight batching 和 Qwen 等模型支持，代码主体采用 Apache-2.0，但它引入 TensorRT engine building、不同 kernel/量化/调度栈；在 6 GiB Windows 开发环境下无法满足“只改变 KV execution action”的因果要求。它可作为生产系统相关工作，而不是本研究的主对照。

一个必须保留的反例是：PagedAttention 论文 §7.1 报告其当时的 paged-attention kernel 因 block-table lookup、分支与可变长度处理，相比 FasterTransformer attention 本身慢约 20%–26%；论文的系统级收益来自更少 KV 浪费和更大的可用 batch，而不是“分页 kernel 天生更快”。因此本项目的 Paged Decode 候选若降低显存占用却增加单 token kernel time，仍可能是有效结果；评价必须拆成 capacity/admissible batch、kernel latency 和端到端 SLO 三层。

同样，[FlashAttention-2 原始论文](https://arxiv.org/abs/2307.08691)强调 work partition、occupancy、非 matmul FLOP 与 shared-memory 通信会影响效率。这支持对 tile/head/GQA shape 分层，但其 A100 dense attention 结果不能外推到 Ada batch-1 paged decode。[FlashAttention-3](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7ede97c3e082c6df10a8d6103a2eebd2-Abstract-Conference.html)依赖 Hopper 的 TMA/WGMMA 和 warp specialization，只作架构相关反例，不进入 RTX 4050 基线。

## 4. 锁定的可复现实验基线

### 4.1 系统级：B0 upstream

- 源码：固定 `acd79d603cb2e1c84c0886137b80f1ad649b6857`，不得用最新 llama.cpp master。
- 构建：与 fork 相同的 MSVC、CMake、CUDA 架构、Release flags；使用 `scripts/build_upstream_baseline.ps1`。
- 输入：同一模型 SHA-256、tokenized prompt、seed、采样参数、context、parallel slots 和到达轨迹。
- 输出：先做 token/output correctness gate，再测 TTFT、TPOT/TBT、request P50/P95、aggregate TPS、queue time、峰值 KV 和失败数。
- 解释：它回答“整个 CacheFlow 行为相对固定上游是否有收益”，不能隔离是哪一个 CUDA kernel 导致收益。

### 4.2 机制级：B1/B2/B3

固定 descriptor、source/destination mapping、K/V 初值、block size、dtype、stream 和 warm-up，仅切换执行方法：

| ID | 动作 | 必须覆盖的 mapping | 主要测量 |
|---|---|---|---|
| B1 | per-block `cudaMemcpyAsync` Direct | 无 source/destination overlap、无 duplicate destination | GPU event time、launch/copy 次数、有效 GB/s |
| B2 | Scalar Gather + Scatter | overlap、cycle、duplicate-source、非对齐和 tail | GPU event time、kernel 数、bytes、逐元素 oracle |
| B3 | `uint4` Vector Gather + Scatter | 与 B2 完全相同，另含对齐/非对齐分层 | 同 B2，外加 vectorized/scalar fallback bytes |

Direct 不具备任意 overlap 的 snapshot 语义，故不能把 B1 的非重叠数据与 B2/B3 的重叠数据直接相除。两类数据应分别报告：`non-overlap transport frontier` 与 `snapshot-required frontier`。

### 4.3 简单启发式：H0

H0 不学习参数：

```text
if mapping_requires_snapshot:
    Vector Remap
else:
    Direct Copy
```

若研究动作空间扩展到服务调度，则另设 `H-rule`：KV pressure 超阈值回 upstream；否则仅当 `cacheflow_prefill_tokens < upstream_prefill_tokens && active_sequences > 1 && cacheflow_chunks > 1` 时启用 CacheFlow。任何在线 Ridge/cost model 必须至少对 H0/H-rule 报 paired regret 和错误启用率，不能只与一个故意较差的 fixed action 比。

### 4.4 条件式外部 oracle：FlashInfer

只有满足以下全部条件才把 FlashInfer 升级为 Q1：固定 revision 和依赖 lock；在同一 RTX 4050 上运行；输入使用 FP16、8 KV heads、head dim 128、page size 16、相同 batch/context/block table；禁用会改变数学语义的 positional/quantization 选项；先过逐元素或 `allclose` 正确性，再用相同 warm-up、CUDA event 和执行顺序轮换计时。它只回答“自研 Paged Decode kernel 距成熟 paged-decode kernel 有多远”，不回答两套 server 谁更快。

## 5. 公平性、混杂因素与停止条件

必须记录并控制：

- GPU 型号、`sm`、driver/CUDA 版本、WDDM 状态、功耗模式、温度与时钟；
- 进程是否 fresh、模型加载/graph capture/warm-up 状态、CUDA stream 与同步位置；
- 权重/KV dtype、层数、Q/KV head 数、head dimension、block/page size、context 和 batch；
- mapping overlap、duplicate、对齐、有效 tail、实际搬运 bytes 和 kernel launch 数；
- prompt/output length 分布、arrival process、prefix reuse、并发、取消、deadline 与 KV pressure；
- 同一 trial 内配对并交替执行顺序；报告原始样本、paired difference/ratio、置信区间和负结果，而不是反复运行直到通过。

NVIDIA 官方说明，[Nsight Compute Memory Workload Analysis](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)能观察数据传输、cache hit、memory requests，并用 hierarchical roofline 判断 L1/L2/device-memory ceiling；[Roofline 官方说明](https://docs.nvidia.com/nsight-compute/2020.3/ProfilingGuide/index.html)将 arithmetic intensity 定义为 work 与 memory traffic 之比。因此“memory-bound”只能在 Nsight 指标或可校准带宽证据支持后声明，不能由低 FLOP 直觉代替。

基线审计的停止条件：若某候选要求不同模型语义、不同精度、不同请求轨迹、无法锁定版本/许可，或只能在 A100/H100/多 GPU 运行，则停止强行定量对齐，把它降级为 R/X。若 FlashInfer 适配成本超过自研 kernel 的验证成本，也只保留论文/官方实现作为设计 oracle，不让外部框架安装阻塞主研究。

## 6. 主张到来源的最小映射

| 主张 | 一手证据 | 本项目允许的推论 |
|---|---|---|
| PagedAttention 用 block table 管理非连续 KV，并支持共享/COW。 | [Kwon et al., SOSP 2023](https://arxiv.org/abs/2309.06180) | 直接读取分页 KV 可以避免先构造完整连续 KV；是否更快仍须在 4050 实测。 |
| IO-aware tiling 能减少 HBM↔SRAM 读写，同时保持 exact attention。 | [Dao et al., NeurIPS 2022](https://arxiv.org/abs/2205.14135) | Paged Decode kernel 应统计 IO 和在线 softmax成本；不能直接继承 FlashAttention 的速度数字。 |
| FlashInfer 官方接口能够消费 paged KV layout 做 batched decode。 | [FlashInfer layout docs](https://docs.flashinfer.ai/tutorials/kv_layout.html)、[official source](https://github.com/flashinfer-ai/flashinfer) | 在相同 shape/layout 下可作为条件式 kernel oracle。 |
| Iteration-level scheduling 可在 token 迭代边界改变 batch。 | [Orca, OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/yu) | CacheFlow 的 Serving Iteration 有系统先例；不证明当前策略最优。 |
| Chunked prefill 是 throughput/TBT 权衡，不保证所有硬件和 workload 同向。 | [Sarathi-Serve, OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/agrawal) | 必须同时报告 TTFT、TPOT/TBT、吞吐和尾延迟，不能只报 iteration time。 |
| GPU/CPU/disk 间张量放置可优化受限显存下的吞吐。 | [FlexGen, ICML 2023](https://arxiv.org/abs/2303.06865) | Swap/Recompute 是候选动作；其 latency-insensitive 结果不能外推在线服务。 |
| token 粒度抢占与 state offload 可改善不同长度请求的排队。 | [FastServe](https://arxiv.org/abs/2305.05920) | 抢占代价模型应含搬运与恢复；不复用许可不明的仓库代码。 |
| 虚拟连续 KV 可避免改写 attention kernel，但需要 CUDA VMM/driver 条件。 | [vAttention official artifact](https://github.com/microsoft/vattention) | PagedAttention 不是唯一动态 KV 方案；当前 Windows 4050 环境不支持公平复现论文配置。 |
| Roofline/Memory Workload 指标可区分 compute 与 memory hierarchy ceiling。 | [NVIDIA Nsight Compute](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html) | 只有 profiler/带宽证据才能支撑瓶颈因果声明。 |

当前 vLLM 的实现会继续变化；本次审计只把固定 SHA 的 [paged attention dispatch](https://github.com/vllm-project/vllm/blob/62a86318de3655f970baf7c2ff89c81a72c1a1b3/vllm/v1/attention/ops/paged_attn.py)、[attention backend](https://github.com/vllm-project/vllm/blob/62a86318de3655f970baf7c2ff89c81a72c1a1b3/vllm/v1/attention/backends/flash_attn.py)和 [KV cache manager](https://github.com/vllm-project/vllm/blob/62a86318de3655f970baf7c2ff89c81a72c1a1b3/vllm/v1/core/kv_cache_manager.py)用于接口/布局比较，不把旧版设计文档当作当前 kernel 实现。

## 7. 已知限制

- 本审计确认的是“可公平比较性”，不是已经运行了 vLLM/FlashInfer/Sarathi 的复现实验。
- 当前只有一张 RTX 4050 Laptop 6 GiB，且 Windows WDDM、温度和动态功耗会放大方差；结论不能外推 A100/H100。
- FlashInfer 的 Q1 身份是条件式的；在 Linux/WSL 适配、版本锁定和 correctness gate 完成前仍属于 R。
- FastServe 缺少可识别的根许可证；本文不做法律判断，只据此禁止复制其代码并将其降级为论文相关工作。
- vLLM 和 TensorRT-LLM 可以提供产品级 sanity check，但不同模型格式、kernel、allocator 和 scheduler 使其不具备单变量因果可比性。
- 项目当前实现 Direct/KV Remap 与受限 Qwen2.5-0.5B Paged Decode K1/K2，但不是完整通用 Paged Decode Attention；host 正确性能力门覆盖 D64/GQA7/context≤32，正式 K2 性能晋级只覆盖 page16/context17 的重复 cached 请求。
