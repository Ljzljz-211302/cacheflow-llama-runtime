# Mission: 用 CacheFlow Runtime 应对推免面试

## Why
在推免计算机/人工智能方向面试中，能够从模型原理、系统设计、CUDA 实现、在线算法和实验方法五个层次独立讲清 CacheFlow Runtime，证明自己理解并完成了核心工作，而不是只会复述开源仓库。

## Success looks like
- 3 分钟内准确讲出问题、架构、个人贡献、关键结果和限制
- 面对白板追问，能推导 KV Cache 容量、COW 不变量、Ridge/UCB 置信门控与 TTFT/TPOT
- 能沿真实代码解释一次推理迭代和一次 CUDA KV 操作
- 能区分局部 kernel 指标、Engine 时间与端到端服务指标，并解释负结果
- 能现场运行最小演示并诚实界定上游代码、个人实现和未完成事项

## Constraints
- 以当前 Qwen2.5-0.5B、RTX 4050 Laptop 和已提交实验为证据边界
- 从没有接触过大模型推理、系统编程或 CUDA 的学习者出发，术语首次出现必须解释
- 优先准备高频推免追问，兼顾算法、操作系统、计算机体系结构、机器学习和工程实践
- 通过主动回忆、白板推导和代码定位建立长期记忆，不以“读过文档”代替掌握

## Out of scope
- 不把本项目包装成分布式多机推理系统
- 不声称完成 Nsight Compute occupancy/roofline 或全模型 kernel census
- 不把固定上游 llama.cpp 代码计入个人贡献
