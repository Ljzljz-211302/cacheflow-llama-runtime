# 推免面试学习助手：真实用户应用边界

## 用户任务

用户不是调用 benchmark，而是在浏览器中完成一条持续工作流：选择历史会话，针对本地学习资料提问，查看被检索到的资料来源，接收真实 `llama-server` 的流式回答，并在应用重启后继续会话。

应用默认读取 `D:\exam\tuimian-monitor\docs\study`，也可配置任意 Markdown、文本或 HTML 资料目录。检索上下文只作为不可信资料提供给模型，不能覆盖系统指令。模型 API key 仅存在于服务端进程，不发送到浏览器。

## 运行边界

- 应用和 CacheFlow Runtime 都只监听 `127.0.0.1`；外部访问必须经过可信反向代理。
- 应用使用 SQLite WAL 保存会话和完整回答；流式回答完成后才提交 assistant 消息，断连不会留下半条答案。
- 每次回答携带实际检索来源；模型不可用时返回明确错误，不生成伪造兜底答案。
- 当前是单用户或可信局域网应用。账号体系、跨设备同步、公共互联网部署和真实用户量需要独立部署后才能声明。

## 数据流

```text
Browser -> Interview Assistant -> local retrieval -> llama-server /v1/chat/completions
                         |                              |
                         +-> SQLite session history    +-> CacheFlow scheduling/KV/CUDA
```

这条链路使 CacheFlow 的共享前缀、多轮 KV 复用、continuous batching、取消、背压和在线收益策略由实际应用请求触发，而不是只由实验脚本触发。

