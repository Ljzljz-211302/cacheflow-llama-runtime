# Issue tracker: GitHub

Issues and PRDs for this repo live as GitHub issues. Use the `gh` CLI for all operations.

## Conventions

- **Create an issue**: `gh issue create --title "..." --body "..."`.
- **Read an issue**: `gh issue view <number> --comments`，同时读取 labels。
- **List issues**: `gh issue list --state open --json number,title,body,labels,comments`，按任务需要增加 label/state 过滤。
- **Comment on an issue**: `gh issue comment <number> --body "..."`。
- **Apply/remove labels**: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`。
- **Close**: `gh issue close <number> --comment "..."`。

仓库由 `git remote -v` 推断；在 clone 内运行时 `gh` 自动选择对应仓库。

## Pull requests as a triage surface

**PRs as a request surface: no.** 外部 PR 不进入本仓库的需求分流状态机。

GitHub 的 issue 和 PR 共用编号空间；遇到不确定编号时先判断对象类型，但 `/triage` 只处理 Issues。

## Skill integration

- 当技能要求“publish to the issue tracker”时，创建 GitHub Issue。
- 当技能要求“fetch the relevant ticket”时，使用 `gh issue view <number> --comments`。

## Wayfinding operations

- **Map**：单个带 `wayfinder:map` 标签的 Issue，保存 Destination、Notes、Decisions-so-far、Not-yet-specified 和 Out-of-scope。
- **Child ticket**：优先使用 GitHub sub-issue，标签为 `wayfinder:research`、`wayfinder:prototype`、`wayfinder:grilling` 或 `wayfinder:task`。若 sub-issue 不可用，在 map task list 中链接，并在 child 顶部写 `Part of #<map>`。
- **Blocking**：优先使用 GitHub 原生 issue dependency。添加边时使用 blocker 的数据库 `id`，不是 issue number 或 node id。若 dependency API 不可用，在 child 顶部写 `Blocked by: #<n>`。
- **Frontier**：map 的 open children 中，没有 open blocker、没有 assignee 的第一项。
- **Claim**：任何任务工作开始前先执行 `gh issue edit <n> --add-assignee @me`。
- **Resolve**：写 resolution comment、关闭 child，然后在 map 的 Decisions-so-far 添加一行带链接的结论摘要。
