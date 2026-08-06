# Domain Docs

本仓库采用 single-context 领域文档布局。

## Before exploring, read these

- 根目录 `CONTEXT.md`（存在时）；
- `docs/adr/` 下与当前工作相关的架构决策记录；
- 若未来出现 `CONTEXT-MAP.md`，它将覆盖本文件的 single-context 假设，并指向相关子上下文。

文件暂不存在时继续工作，不把缺失本身当作错误。领域术语或重要决策真正形成时，由 domain-modeling 流程按需创建。

## File structure

```text
/
├── CONTEXT.md
├── docs/
│   ├── agents/
│   └── adr/
└── src/
```

## Vocabulary rule

Issue 标题、规格、测试名和实现应使用 `CONTEXT.md` 中定义的领域词汇。若需要的概念尚未定义，应先判断是语言漂移还是确有领域缺口，再通过 domain-modeling 补充。

## ADR conflicts

若新方案与既有 ADR 冲突，必须明确指出冲突的 ADR 和重新开启该决策的理由，不得静默覆盖。
