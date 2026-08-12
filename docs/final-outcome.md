# CacheFlow Runtime 最终成果说明

## 最终定性

当前仓库已经形成可独立交付的**单机应用成果**与**有边界的科研型项目成果**。它不是已发表论文，也不声称 Paged Attention 已普遍优于 upstream/Direct。最终结论由 `results/final-outcome.json` 从正式工件自动生成并校验。

## 可交付应用成果

- 推免面试学习助手自动 HTTP/SSE 用户旅程通过：UI 资源=True，SSE 契约=True，并发用户=2；另有 Chromium 交互 QA=True。
- 本地知识库包含 5 份文档、128 个检索块；断流可传播到 llama-server，429 背压与重启恢复均通过。
- 真实链路记录 456 个缓存 prompt token、6 次自研 CUDA KV kernel、5603330 个向量化 remap 字节。
- 适用范围：单机或可信局域网；不声称已有外部用户采用。

## 可交付科研成果

1. **Upstream 边界**：固定 `acd79d603cb2e1c84c0886137b80f1ad649b6857`，相同工具链的 upstream policy 在 5 个记录用例中输出逐项一致。
2. **向量化 KV Remap**：1/4/16/32 blocks 的配对中位改善分别为 54.55%、46.32%、5.08%、2.32%；这是算子微基准，不外推为端到端加速。
3. **统一动作策略**：D3 在 80 个留出决策中切换 24 次，累计 regret 2.082 ms，harmful decision 1 次；chooser P99 0.900 us、热路径零分配。该结果仍是离线 replay。
4. **Paged-vs-Direct 负结果**：正确性通过，但 P95 从 27.354 ms 增至 29.210 ms（回退 6.78%），超过 5% 门槛，因此 Paged 保持 opt-in。
5. **K2-vs-K1 正结果**：30 组同进程配对、每 variant 480 条测量响应、600 次 Paged graph、0 fallback；请求 median/P95 回退 0.55%/1.52%，median 回退 95% 上界 2.86%；相同 480 次 kernel 总时长由 8.174 ms 降至 4.051 ms（-50.44%），通过预注册替换门槛。

## 面试与简历允许使用的结论

可以表述为：在 llama.cpp 上实现缓存感知调度、KV 生命周期、CUDA Remap/Swap 和受限 Paged Decode，并以预注册配对实验完成 K2 对 K1 的有界生产替换；同时保留 Paged 相对 Direct 未晋级的负结果。

不可以表述为：Paged Attention 全面优于 llama.cpp、端到端延迟提升 50.44%、已经发表论文、已经获得外部用户采用。

## 数据、图表与原始证据

- 图文总结报告：[`docs/final-illustrated-report.md`](final-illustrated-report.md)
- K2/K1 请求与 kernel 对比图：[`results/research/h8-k2-production-v2.10.0/k2-production-comparison.svg`](../results/research/h8-k2-production-v2.10.0/k2-production-comparison.svg)
- K2 正式报告：[`results/research/h8-k2-production-v2.10.0/report.md`](../results/research/h8-k2-production-v2.10.0/report.md)
- Paged-vs-Direct 正式负结果：[`results/research/h7-production-paged-v1.1.0/report.md`](../results/research/h7-production-paged-v1.1.0/report.md)
- 研究项目总报告：[`docs/research-project-report.md`](research-project-report.md)
- 真实应用旅程：[`results/user-application-journey.json`](../results/user-application-journey.json)
- 每项输入工件的 SHA-256 位于 [`results/final-outcome.json`](../results/final-outcome.json)，防止报告数字与原始结果漂移。

## 复现入口

```powershell
.\scripts\bootstrap.ps1
.\scripts\verify.ps1
.\scripts\verify_final_outcome.ps1
.\scripts\start_production.ps1 -ModelPath .\models\qwen2.5-0.5b-instruct-q4_k_m.gguf -ApiKeyFile .\runtime\api-key.txt
```

`verify_final_outcome.ps1` 不重新挑选实验结果，而是校验本页、机器可读结论与正式工件是否一致。完整 GPU 重跑使用 `verify.ps1 -Full`。
