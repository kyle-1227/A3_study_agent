# A3 Study Agent 测试说明书

## 1. 目的与证据等级

测试目标是验证需求合同、实现正确性、身份隔离、恢复能力、安全边界和真实用户路径。证据分为四级，低级证据不能替代高级证据：

1. 静态/合同检查：语法、格式、架构边界、类型、secret 模式和配置合同；
2. 单元/集成测试：节点、API、SSE、PostgreSQL、RAG 身份和资源合同；
3. 真实部署 canary：Docker、Provider、浏览器、重启/回放和 artifact；
4. 人工内容/用户验收：事实、教学质量、代码可运行性、视频和学习效果。

## 2. 当前可声明结果

| 项目 | 可声明状态 | 说明 |
| --- | --- | --- |
| 完整后端 pytest | `2880 passed / 7 skipped` | 最终 runtime、fallback、Docker 与 hermetic rollout 合同均已纳入 |
| 完整前端测试 | 36 个文件、`187 passed` | typecheck、lint、显式本地 API URL 的 production build 同时通过 |
| Import Linter | `3/3` contracts kept | 架构边界未破坏 |
| P0 / PG / PR / PGR | 有真实节点 adapter 的离线评估变体 | PGR 是唯一生产 served path；六场景数据仅为 smoke authoring |
| Semgrep | 未安装、未运行 | 不得写成通过 |
| Gitleaks | 未安装、未运行 | 不得写成通过 |
| 真实 Docker/Provider/code-practice canary | 连续两轮 `production_success=true` | 同一 Evidence fingerprint；终态、回放、冲突、下载、刷新和 Context V3 均通过 |
| 最终 revision 镜像与 PostgreSQL-only restart | 发布前继续复验 | 不由热同步候选容器替代 |
| 六场景浏览器 canary | code-practice 单场景重复通过；完整套件未完成 | 不声明六场景最终验收通过 |
| 人工内容/教育效果评估 | 未形成正式 Gold 或统计结论 | 六场景 smoke 不是人工评审通过 |

### 2026-07-17 最终工程门禁与真实单场景复核

| 检查 | 实际结果 | 判定 |
| --- | --- | --- |
| 最终 Markdown：严格 UTF-8、围栏、相对链接、生产身份、旧命令/措辞、secret-like 模式 | 全部通过 | 通过 |
| `git diff --check` | 退出码 0；仅提示现有 Windows LF/CRLF 转换 | 通过 |
| 完整后端 pytest | `2880 passed / 7 skipped / 4 warnings` | 通过 |
| 前端 test/typecheck/lint/build | 36 files、`187 passed`；其余三项通过 | 通过 |
| Import Linter | 分析 351 个文件、2205 个依赖；`3 kept / 0 broken` | 通过 |
| `ruff check .` | 失败，基线存在 38 个 lint 错误 | 未通过；本次文档提交未修改 runtime |
| `ruff format --check .` | 失败，55 个文件会被重排，486 个已格式化 | 未通过；本次文档提交未批量改格式 |
| `bandit -r src -x tests` | 失败，共 46 项：High 15、Medium 7、Low 24 | 未通过；高风险项必须在公共生产声明前逐项复核/修复或给出可审计理由 |
| Semgrep / Gitleaks | 均未安装、未运行 | 缺失，不能计为通过 |
| mypy / Vulture | 已安装，本次纯文档变更不涉及类型或死代码，未运行 | 未运行 |
| 真实 code-practice canary | 连续两轮通过；每轮三类下载、replay、409 drift、刷新恢复、15 项上下文注入 | 通过单一重复场景；不是完整六场景 |

Bandit 的高等级报告主要涉及把 MD5/SHA-1 用作内容/身份摘要；还报告 XML 解析、动态 SQL、subprocess 和吞异常等不同等级问题。部分可能是非安全用途或已有参数化边界，但在逐项确认前不能整体忽略。完整输出属于本次命令记录；本提交不以扩大 runtime diff 的方式顺手修复这些基线项。

### 最终 code-practice 浏览器证据

两份脱敏、机器可读报告执行于 browser-tested runtime baseline
`707d79806364d95fd300b21d0cb93411f592d67a`，并显式记录 expected Evidence fingerprint
`6274c8ac2b0e70828d7e5f64f72ed8f2b9ab36ae8683adcf0b274d60df277b01`。同期
`/health/ready` 单独证明 generation/KG/active 身份；该限域报告本身没有嵌入完整
pre/post served identity，因此不能把两类证据混写为报告独立绑定：

- `artifacts/browser_canary/code-practice-707d798-1-20260717T155617Z/result.json`：
  DOCX 40,901 B、Markdown 9,165 B、Python 2,743 B；
- `artifacts/browser_canary/code-practice-707d798-2-20260717T155922Z/result.json`：
  DOCX 41,975 B、Markdown 11,074 B、Python 2,771 B。

两轮均经过 `planner -> agent -> reviewer -> output`，只有一个权威成功终态，
三类下载均为 HTTP 200 且带 attachment，Last-Event-ID replay 匹配，请求 payload
漂移返回 409，刷新后精确恢复 3 个 artifact 链接，Context V3
`injection_count=15`。每轮浏览器控制台各记录一次新线程创建前的 thread-status
瞬时 404；后续请求与终态成功，因此这是已知时序噪声，不得写成“控制台零错误”。

## 3. 后端门禁

完整集成后执行：

```powershell
python -m compileall -q src tests app.py
ruff check .
ruff format --check .
python -m pytest -q
lint-imports --config .importlinter
bandit -r src -x tests
```

文档或部署合同变更至少聚焦执行：

```powershell
python -m pytest -q tests/test_docker_deployment_contract.py tests/test_backend_startup.py tests/test_security.py
```

如果修改结构化输出、画像、graph、RAG 或安全相关 runtime，还必须按 `AGENTS.md` 路由到对应仓库技能并执行相关测试、mypy/安全工具。工具缺失必须记录为缺失，不能把“命令不存在”计为通过。

## 4. 前端门禁

```powershell
Push-Location frontend
npm run test
npm run typecheck
npm run lint
npm run build
Pop-Location
```

前端测试应覆盖 SSE 增量事件、终态去重、错误显示、刷新恢复、Markdown/资源卡渲染、blocked 资源不出现虚假下载、键盘/可访问性和不同视口。构建成功不代表完整用户路径通过。

## 5. 文档与配置 sanity

纯文档提交需要：

- UTF-8 可读，标题、表格和代码围栏配对；
- 所有相对 Markdown 链接指向存在的仓库文件；
- README、开发、测试、部署和第三方说明中的生产身份一致；
- 不出现已废弃的无参数旧索引构建 quickstart；
- 不把 P0/PG/PR/PGR 写成四条生产流量，也不使用过时 Shadow rollout 叙事；
- `RAG_EMBEDDING_API_KEY`、`RAG_RERANKER_API_KEY` 名称精确；
- `git diff --check` 无空白错误；
- changed diff 的 secret-like 扫描不含 key 值、Bearer token、Authorization、完整数据库 URI、Provider body 或真实 `.env` 内容。

命令示例：

```powershell
git diff --check
git status --short
```

secret 扫描必须只检查待提交 diff 或脱敏副本。不要为了扫描而打印真实环境变量、读取真实 `.env` 或输出 Provider 请求。

## 6. 部署合同测试

部署合同至少验证：

- Compose 明确分离 PostgreSQL、backend 和 frontend；
- `A3_ENV_FILE` 和全部必填变量失败即停止；
- 课程资料与 Parent–Child index 宿主路径显式挂载；
- 密封索引只读，运行时 Chroma 和 artifact 使用独立可写持久卷；
- `PARENT_CHILD_GENERATION_ID`、registry primary 和 manifest 身份一致；
- readiness 报告 PostgreSQL、active PGR、shadow disabled、KG/generation/evidence 身份；
- 课程资料与 index 都是 `read_only: true`、`bind.create_host_path: false`，缺路径 fail-fast；
- Evidence 只允许初始轮加最多 3 次补搜、24 个 search task、72 条 ledger entry，required evidence 必须 complete；
- 请求时不存在 Flat RAG、其他 Provider、其他模型或 partial-success 的静默 fallback。

聚焦合同测试还直接证明 round index 3 可安排，而下一次 repair 以 `repair_round_budget_exceeded` typed error 拒绝；不存在第四次 supplement 的 off-by-one。

## 7. 六场景真实浏览器 canary

按[生产部署运行手册](../runbooks/production_deployment.md)执行：

1. 大数据 MapReduce 复习文档；
2. 计算机数据结构测验；
3. 机器学习架构视频脚本；
4. 数学积分/级数思维导图；
5. Python 代码练习与视频动画；
6. 大数据 + 机器学习学习计划与复习文档，并刷新恢复。

当前只完成了第 5 类中的 code-practice 子场景连续两轮；video animation 与其余五类场景不能据此视为通过。

每个场景保存脱敏、机器可读证据，检查：

- SSE sequence 连续且只有一个权威资源终态；
- `stream_done`、thread status 和 artifact 身份一致；
- 学科、topic、资源类型、generation 和 KG 身份正确；
- Last-Event-ID 只回放后续事件；
- 相同 request ID 的 payload 漂移返回显式冲突；
- blocked 资源不生成虚假下载；
- 最终场景刷新后恢复同一用户/线程状态。

ready 和 evidence-blocked 都可能是严格的合法结果。观察到 blocked 不能改写成通过；观察到 ready 也不能自动代表内容正确。运行报告不得保存生成正文或 Provider body。

## 8. PostgreSQL 与恢复测试

在真实线程完成后重启 PostgreSQL，确认 readiness 恢复、thread status 仍能返回权威终态、SSE journal 可在保留窗口内回放、刷新恢复下载卡，且跨用户/跨线程访问被拒绝。恢复过程不得自动切换 generation 或 Flat RAG。

## 9. 安全与许可验收

- API key、Authorization 和完整数据库 URI 不进入日志、trace、SSE、截图或提交文件；
- 路径、URL、文件类型、重定向、子进程和下载 artifact 经过既有安全校验；
- 生成内容抽检事实、敏感信息、违规内容、题目答案和代码；
- [第三方软件与 AI 工具说明](third_party_notices.md)由许可证负责人复核，特别是 PyMuPDF、Psycopg、课程资料和外部服务；
- 赛题第 49 行有关科大讯飞工具的要求由参赛负责人取得真实合规证据。

## 10. 发布判定

只有低级到高级证据都满足本次发布范围，才能声明比赛演示验收通过。当前仓库记录支持“工程门禁完成、真实 Docker/Provider/code-practice 单场景连续两轮通过”；它不支持“完整六场景已通过”“人工学术质量已验收”或“学习效果已经被正式证明”的表述。
