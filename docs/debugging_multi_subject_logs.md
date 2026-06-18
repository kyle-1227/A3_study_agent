# 多 Subject 检索 A3_TRACE 调试日志

本文档用于开发阶段验证多 subject 检索、Web Research V2、Evidence Judge V2 和生成链路。开启后，后端日志会输出统一格式：

```text
A3_TRACE {"stage":"query_rewrite","request_id":"...","session_id":"...","thread_id":"..."}
```

每条日志都带 `request_id`、`session_id`、`thread_id`，排查多轮对话时可以按这些字段过滤。

## 开关

```env
# Master switch for all structured development trace logs
LOG_A3_TRACE=true

# Fine-grained trace switches
LOG_SUPERVISOR_RESULT=true
LOG_QUERY_REWRITE_RESULT=true
LOG_RETRIEVAL_PLAN=true
LOG_RAG_RESULT=true
LOG_CONTEXT_ASSEMBLY=true
LOG_WEB_SEARCH_RESULT=true
LOG_GENERATION_SUMMARY=true
LOG_PLANNING_INTEL=true
LOG_RETRY_TRACE=true
```

开发阶段可以全部打开；稳定后建议只保留 `LOG_RAG_RESULT=true` 和 `LOG_QUERY_REWRITE_RESULT=true`。

## 测试 1：多 subject 普通答疑

输入：

```text
用 Python 做一个机器学习过拟合检测案例
```

预期：

- `query_rewrite.retrieval_plan_count >= 2`
- subjects 包含 `python` 和 `machine_learning`
- `machine_learning` role 接近 `core_concept`
- `python` role 接近 `implementation_tool`
- `rag_retrieve_plan_item.subject_mismatch_count = 0`
- `context_assembly.subject_doc_distribution` 同时包含 `python` 和 `machine_learning`

## 测试 2：多 subject 思维导图

输入：

```text
给我生成一个 Python 实现机器学习过拟合检测的思维导图
```

预期：

- `supervisor.requested_resource_type = mindmap`
- `retrieval_plan` 包含 `python` / `machine_learning`
- `mindmap_planner.subjects_used` 包含两个 subject

## 测试 3：planning 多 subject

输入：

```text
帮我制定一个 Python + 机器学习 4 周入门计划
```

预期：

- `supervisor.requested_resource_type = study_plan`
- 路径经过 `query_rewrite`
- `planning_study_plan_planner.mode = multi_subject`

## 测试 4：RAG subject filter

输入：

```text
Python 函数 参数 返回值 作用域
```

预期：

- `rag_retrieve_single_subject.subject = python`
- `subject_mismatch_count = 0`
- `top_docs.metadata_subject` 全部为 `python`

## Build Check

```bash
python -m py_compile \
  src/observability/a3_trace.py \
  src/graph/supervisor.py \
  src/graph/academic.py \
  src/graph/mindmap.py \
  src/graph/exercises.py

pytest
```

## Web Research V2 排查

开启：

```env
LOG_WEB_SEARCH_RESULT=true
```

重点查看 `stage=web_research_v2.*` 或外部节点名 `web_search` 的字段：

- `provider`：当前 Web Research executor provider，通常为 Tavily。
- `task_count` / `result_count`：计划任务数与可用结果数。
- `search_query`：当前 Tavily 查询；不要和旧 `web_search_query` 输出字段混用。
- `fetch_status` / `summary_status`：源内容读取和 source summarizer 状态。
- `candidate_count`：生成的 Web evidence candidate 数量。
- `error_type` / `error_message`：脱敏后的失败类型与短错误信息。
- `elapsed_ms`：搜索或总结耗时。

如果 `result_count=0`，通常表示 Tavily 没有返回可用结果、接口限流、网络失败或查询过窄。此类内容不会被当成真实搜索结果写入上下文。

也可以脱离 Agent 直接运行：

```bash
python scripts/debug_web_search.py
```

## Hallucination Evaluation 排查

开启：

```env
LOG_RETRY_TRACE=true
```

重点查看 `stage=hallucination_eval` 的字段：

- `success=true`：最终拿到了可解析的 `HallucinationEvaluation`。
- `success=false` 且 `defaulted_to_valid=true`：primary 和 fallback 都没有得到可解析结果，因此按现有业务规则默认通过。
- `primary_called` / `fallback_called` / `fallback_used`：是否调用 primary、是否尝试 fallback、最终是否采用 fallback 结果。
- `failure_phase`：失败阶段，例如 `structured_parsing_error`、`parsed_none`、`fallback_structured_parsing_error`、`fallback_parsed_none`、`primary_call_failed`。
- `parsing_error`：结构化输出解析错误摘要。
- `raw_preview`：LLM 原始返回的短预览，不包含完整 raw。
- `context_rag_count` / `context_web_count`：评估时使用的上下文来源数量。

诊断日志只写入后端 logger，不会进入 `messages`、用户回答、RAG context 或前端气泡。
