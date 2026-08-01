# 推免面试学习助手：真实用户应用边界

## 用户任务

用户不是调用 benchmark，而是在浏览器中完成一条持续工作流：选择历史会话，针对本地学习资料提问，查看被检索到的资料来源，接收真实 `llama-server` 的流式回答，并在应用重启后继续会话。

应用默认读取 `D:\exam\tuimian-monitor\docs\study`，也可配置任意 Markdown、文本或 HTML 资料目录。检索使用标题加权和逆文档频率，避免“为什么、面试追问”等通用措辞压过 B+ 树等稀有技术术语。检索上下文只作为不可信资料提供给模型，不能覆盖系统指令。模型 API key 仅存在于服务端进程，不发送到浏览器。

## 运行边界

- 应用和 CacheFlow Runtime 都只监听 `127.0.0.1`；模型 URL 只接受带显式端口的 loopback IP 并拒绝重定向，防止服务端 API key 被误发到远端。外部访问必须经过可信反向代理。
- 应用使用 SQLite WAL 保存会话和完整回答；每个操作独占短连接，消息与会话更新时间在同一事务提交。流式回答完成后才提交 assistant 消息，断连会关闭上游生成器且不会留下半条答案。
- 每次回答携带实际检索来源；无检索命中或模型不可用时返回明确错误，不生成无依据或伪造兜底答案。
- 当前是单用户或可信局域网应用。账号体系、跨设备同步、公共互联网部署和真实用户量需要独立部署后才能声明。

## 数据流

```text
Browser -> Interview Assistant -> local retrieval -> llama-server /v1/chat/completions
                         |                              |
                         +-> SQLite session history    +-> CacheFlow scheduling/KV/CUDA
```

这条链路使 CacheFlow 的共享前缀、多轮 KV 复用、continuous batching、取消、背压和在线收益策略由实际应用请求触发，而不是只由实验脚本触发。

## 验收证据

`run_user_application_journey.py` 启动两个 fresh application subprocess，而不是在验收进程里直接调用 Service。当前真实 CUDA 旅程观测到 441 个缓存 prompt token、6 次自研 CUDA KV kernel、2 次 CUDA benefit decision、2 次在线策略 checkpoint、19 个 prefill chunk，`n_busy_slots_per_decode=1.0739`；同时验证 429 背压、应用进程重启续聊，以及浏览器在收到真实模型 token 后断流会触发 llama-server 原生 `cancel task` 且不保存半条答案。

真实 Chromium 浏览器还完成了输入、发送、SSE 增量显示、资料卡片和回答完成状态检查；B+ 树问题的首条检索结果为数据库文档的 `7.1 B+ 树`，而非通用面试题。浏览器证据位于 `results/user-application-browser-qa.json`。这仍不等于已有外部用户采用。
