# 多 Subject 检索 A3_TRACE 调试日志

本日志用于开发阶段验证多 subject 检索链路。开启后，后端日志会输出统一格式：

```text
A3_TRACE {"stage":"query_rewrite","request_id":"...","session_id":"...","thread_id":"..."}
```

每条日志都带 `request_id`、`session_id`、`thread_id`，排查多轮对话时可按这些字段过滤。

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

开发阶段可以全部打开。稳定后建议只保留 `LOG_RAG_RESULT=true` 和 `LOG_QUERY_REWRITE_RESULT=true`。

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

- `supervisor.intent = planning`
- 路径经过 `query_rewrite`
- `planning_gather_intel.mode = multi_subject`

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
  src/graph/exercises.py \
  src/graph/planner.py

pytest
```

## Web Search 排查

开启：
```env
LOG_WEB_SEARCH_RESULT=true
```

重点看 `stage=web_search` 的字段：
- `query_source`：本次搜索 query 来自 `rewritten_query`、`search_web_query`、`retrieval_plan_top_priority` 还是原始问题。
- `provider`：当前为 `duckduckgo`。
- `ok`：搜索工具是否认为调用成功。
- `result_count`：最终可用结果数量。
- `raw_type` / `raw_count`：底层工具返回的是 `list`、`str`、`str_empty_or_error` 等。
- `error_type` / `error_message`：错误类型与脱敏后的短错误信息。
- `elapsed_ms`：搜索耗时。

如果 `raw_type=str_empty_or_error` 且 `result_count=0`，通常表示 DuckDuckGo 返回了“无有效结果”或限流/错误文本，这类字符串不会被当成真实搜索结果写入上下文。

也可以脱离 Agent 直接运行：
```bash
python scripts/debug_web_search.py
```

## Hallucination Evaluation 排查

开启：
```env
LOG_RETRY_TRACE=true
```

重点看 `stage=hallucination_eval` 的字段：
- `success=true`：最终拿到了可解析的 `HallucinationEvaluation`。
- `success=false` 且 `defaulted_to_valid=true`：primary 和 fallback 都没有得到可解析结果，因此按现有业务规则默认通过。
- `primary_called` / `fallback_called` / `fallback_used`：是否调用 primary、是否尝试 fallback、最终是否采用 fallback 结果。
- `failure_phase`：失败阶段，例如 `structured_parsing_error`、`parsed_none`、`fallback_structured_parsing_error`、`fallback_parsed_none`、`primary_call_failed`。
- `parsing_error`：结构化输出解析错误摘要。
- `raw_preview`：LLM 原始返回的短预览，不包含完整 raw。
- `context_rag_count` / `context_web_count`：评估时使用的上下文来源数量。

注意：诊断日志只写入后端 logger，不会进入 `messages`、用户回答、RAG context 或前端气泡。
