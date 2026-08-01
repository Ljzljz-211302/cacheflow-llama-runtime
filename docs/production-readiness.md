# CacheFlow Runtime 生产准入与运行手册

## 结论与部署边界

当前项目的可投入应用边界是：**单机、单个 `llama-server` 进程、CPU 或单 GPU、由本机或可信反向代理访问**。在这个边界内，真实 GGUF 模型、OpenAI API、流式输出、背压、取消、Deadline、KV 故障降级、指标和在线收益模型跨重启恢复均有自动化验证。

这不是“任意规模生产就绪”的宣称。多机路由、张量/流水并行、跨副本策略聚合、租户配额和 Kubernetes Operator 仍不在当前范围内。外部访问必须由反向代理完成 TLS、访问日志和限流；本仓库的生产启动器只绑定 `127.0.0.1`，避免误把未配置 TLS 的进程直接暴露到公网。

## 生产入口

先生成只包含 API key 的文件并限制其读取权限，然后以前台进程启动，让 systemd、Windows Service Wrapper 或容器运行时负责重启和日志采集：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_production.ps1 `
  -ModelPath .\models\qwen2.5-0.5b-instruct-q4_k_m.gguf `
  -ApiKeyFile D:\secrets\cacheflow-api-keys.txt `
  -Backend cuda `
  -InstanceId gpu0 `
  -Port 8080
```

先用 `-PrintCommand` 做无副作用的配置检查。启动器会验证模型、至少一个非空 API key 和对应后端二进制，关闭 WebUI，创建独立状态目录，并用“模型 SHA-256 + 主机 + 后端 + context + parallel”生成 checkpoint compatibility key。运行期间它会独占 `benefit-<InstanceId>.json.lock`；每个并行副本仍应使用不同 `InstanceId`，误用相同 ID 时第二个进程会在启动前失败，禁止多个进程写同一个状态文件。

## 在线模型持久化

`server_benefit_policy` 的 ridge 模型不再只存在于内存：

1. 推理线程每隔固定观测数生成一个小型不可变快照，只入队，不做磁盘 I/O；
2. 后台单写线程只有一个 pending 槽位，新快照会合并尚未写入的旧快照，队列不会随流量增长；
3. 文件写入同目录 `.tmp`，执行 `fflush + fsync/_commit` 后原子替换正式文件；
4. 状态包含 schema、特征维数、完整策略配置、compatibility key 和 CRC32；
5. 恢复先写入临时内存对象并校验数值有限性、矩阵维数/对称性/正则化对角线，全部通过后才提交；
6. 缺失状态从冷启动开始；模型或配置不兼容从冷启动开始并计数；损坏或 I/O 错误同样 fail closed，不让部分系数进入决策。

默认每 128 次有效观测提交一次。正常析构会刷新最后一批状态；进程崩溃最多丢失一个 checkpoint interval 内的学习结果，不会损坏上一个已提交版本。

## 健康检查与告警

- 存活/就绪：`GET /health`；只有模型加载完成后才接流量。
- 指标：启动参数固定开启 `GET /metrics`，由反向代理限制为运维网络可见。
- 安全：业务请求必须携带 `Authorization: Bearer ...` 或 `x-api-key`；启动器拒绝空密钥文件。

最低告警集合：

| 指标 | 建议条件 | 含义 |
|---|---:|---|
| `llamacpp:benefit_checkpoint_restore_total{result="failed"}` | `> 0` | 文件损坏或读取失败，策略已冷启动 |
| `llamacpp:benefit_checkpoint_restore_total{result="incompatible"}` | `> 0` | 模型/硬件/关键配置发生变化 |
| `llamacpp:benefit_checkpoint_save_total{result="failed"}` | `> 0` | 状态目录权限、空间或设备故障 |
| `llamacpp:benefit_checkpoint_pending` | 长时间为 `1` | 存储延迟异常 |
| `llamacpp:benefit_safety_fallback_total` | 速率突增 | KV 压力或工作负载离开安全区 |
| `llamacpp:benefit_drift_total` | 速率突增 | 在线成本分布改变，已进入 cooldown |
| `llamacpp:request_latency_seconds` / `request_queue_seconds` | 按业务 SLO | 延迟或排队异常 |

## 发布与回滚

1. 新版本先执行 `scripts/verify.ps1 -Full`；
2. 用 `-PrintCommand` 核对模型、端口、状态文件和 compatibility key；
3. 单实例 canary，确认 `/health`、checkpoint restore 结果和核心延迟；
4. 保留 `--scheduler-policy upstream` 作为行为回滚开关；若新版本 checkpoint schema 不兼容，会自动冷启动而不是加载旧系数；
5. 回滚二进制时使用独立 `InstanceId` 或备份状态文件，避免旧版本解释新 schema。

## 已自动验收

`scripts/run_benefit_checkpoint_smoke.py` 通过真实 CPU `llama-server` 完成三段 fresh-process 验收：先产生并落盘在线观测，强制终止后重启并恢复相同观测，再写入截断文件并确认服务不崩溃、恢复失败指标为 1、模型观测归零且请求仍可成功。结果写入 `results/benefit-checkpoint-smoke.json`。

这一切片解决的是生产 P0 的“在线控制状态不能跨重启”问题。下一优先级是 24 小时 soak、磁盘满/只读目录故障注入、反向代理部署模板和请求级 trace correlation；在这些门槛完成前，不应宣称支持无人值守公网多租户部署。
