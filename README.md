# llama.cpp 量化与推理性能实验台

这是一个面向推免面试的可复现 AI Infra 项目。它在同一台机器、同一模型 revision 和同一输入配置下，对 Q4、Q8、F16 的模型体积、实测 VRAM、离线吞吐、在线 TTFT/TPOT、并发扩展和最小质量护栏进行比较，并提供一个基于权重与 KV Cache 的显存预算推荐器。

项目不是聊天 UI。可展示的个人工作包括：固定制品和校验和、统一实验配置、`llama-bench` 结构化归一化、OpenAI SSE 流式计时、并发负载、质量规则、KV Cache 估算、自动报告和测试。

## 当前复现结果

环境：Windows 11、RTX 4050 Laptop 6GB、Intel i5-13500H、llama.cpp `b9632`、Qwen2.5-0.5B-Instruct GGUF 固定 revision。

- Q4_K_M 权重约为 F16 的 38.5%，实测峰值显存为 625 MiB（Q8 753 MiB、F16 1241 MiB），CUDA decode 速度约为 F16 的 1.97 倍。
- Q4 CUDA decode 速度约为 12 线程 CPU-only 的 4.66 倍。
- 并发从 1 增至 4 时，聚合输出吞吐约提高至 2.37 倍，但单请求 TPOT 和尾延迟变差。
- 0.5B 模型在简单专业题上仍会产生事实错误；量化性能提升不能替代质量评测。

完整数值见 [results/report.md](results/report.md)。结果只代表固定环境，不能直接外推到其他模型和硬件。

## 系统结构

```mermaid
flowchart LR
    A[artifacts.json] --> B[bootstrap.ps1]
    B --> C[固定 llama.cpp 二进制/源码]
    B --> D[同源 Q4/Q8/F16 GGUF]
    C --> E[离线 llama-bench]
    D --> E
    C --> F[llama-server]
    D --> F
    E --> G[baseline.csv]
    F --> H[SSE TTFT/TPOT 与质量护栏]
    G --> I[自动报告]
    H --> I
    D --> J[KV Cache 内存推荐器]
```

## 一键复现

项目首次初始化约下载 3GB 模型和 650MB Windows CUDA 运行包。所有制品都固定 URL、revision、大小和 SHA-256；下载中断可继续，错误文件会被保留为 `.invalid-*`。

```powershell
cd D:\llama
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\verify.ps1 -Full
```

`-Full` 将依次运行：

1. Python 单元测试与语法编译；
2. 制品 SHA-256 和 GPU 后端检查；
3. Q4/Q8/F16、CPU/GPU 离线 benchmark；
4. 并发 1/2/4 的在线流式 benchmark；
5. 五道固定题的最小质量护栏；
6. 自动生成 Markdown 报告。

只验证环境和测试：

```powershell
.\scripts\verify.ps1
```

## 分步运行

```powershell
python .\scripts\run_benchmarks.py
python .\scripts\run_server_benchmark.py
python .\scripts\run_quality.py
python .\scripts\generate_report.py
```

显存预算建议器示例：

```powershell
python .\scripts\memory_advisor.py `
  --model .\models\qwen2.5-0.5b-instruct-q4_k_m.gguf `
  --available-mib 5075 `
  --slots 4
```

## 实验设计

### 离线吞吐

`config/experiment.json` 固定 prompt 256 tokens、generation 64 tokens 和三次重复。GPU 比较 Q4_K_M/Q8_0/F16；CPU-only 比较 6/12 线程。`pp` 是 prompt processing，`tg` 是 token generation。官方 `llama-bench` 不含 tokenization 和 sampling 时间，因此不能用它代替在线延迟。

### 在线流式指标

`config/server_benchmark.json` 启动四个服务 slot，分别施加并发 1/2/4 的固定请求，每档至少 30 个样本。脚本从 SSE 第一个非空 content 事件计算 TTFT（首 token 的接口近似），并以首 token 后的生成 token 计算 TPOT；聚合 TPS 使用整组完成 token 数除以墙钟时间。

### 质量护栏

`config/quality.json` 使用固定温度 0、固定任务和正/负规则。它能发现显然错误，但只有五题，不能代替 perplexity、标准评测集或人工事实核验。每条完整输出保存在 `quality_results.csv`，不能只汇报通过率。

### 内存估算

对 decoder-only Transformer，F16 KV Cache 近似为：

```text
2 × layers × kv_heads × head_dim × context × slots × 2 bytes
```

推荐器再加 GGUF 文件大小、固定 runtime 预留和 15% 安全余量。它是容量规划估算；基准脚本同时轮询 `nvidia-smi` 给出整卡基线、峰值和增量，但最终的精细归因仍应使用 NVML 或 profiler 验证。

Windows 预编译包即使使用 `-ngl 0` 也会加载 CUDA backend 并创建上下文，因此 CPU-only case 可能显示数百 MiB 的 CUDA 运行时占用；`execution=CPU-only` 表示模型层未 offload，不表示进程完全不初始化 CUDA。

## 目录

```text
config/       固定制品、离线、在线和质量配置
docs/         架构与面试追问
models/       下载的 GGUF，Git 忽略
runtime/      官方预编译运行包，Git 忽略
scripts/      初始化、实验、报告和验证入口
src/          指标、SSE、benchmark、质量和推荐器实现
tests/        无外部依赖的单元测试
results/      可提交的汇总结果；raw 日志被忽略
vendor/       固定版本 llama.cpp 源码，Git 忽略
```

## 面试演示建议

三分钟演示顺序：先展示固定版本与实验矩阵，再展示 Q4/F16 和 CPU/GPU 数值，随后把并发从 1 提升到 4，最后展示错误样例并解释为什么性能与质量必须同时报告。不要把“运行了 llama.cpp”说成自己实现了推理框架；应明确个人贡献位于实验基础设施、指标语义、质量护栏和内存规划。

进一步追问与回答框架见 [docs/interview-notes.md](docs/interview-notes.md)。

## 来源与许可证

- [llama.cpp](https://github.com/ggml-org/llama.cpp)：MIT；本项目固定 `b9632/acd79d603cb2e1c84c0886137b80f1ad649b6857`。
- [Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF)：Apache-2.0；固定 revision `9217f5db79a29953eb74d5343926648285ec7e67`。
- 第三方源码、二进制和模型不进入本仓库提交；各自许可证独立生效。
