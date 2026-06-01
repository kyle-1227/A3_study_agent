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
