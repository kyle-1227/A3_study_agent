# REQ-06 Completion Report: BM25 index invalidation

## Summary

Added automatic invalidation to the BM25 singleton in `retriever.py`. The BM25
index now tracks the ChromaDB document count at build time and automatically
rebuilds when the count changes (i.e. documents were added or removed). A
`force_rebuild` parameter is also available for manual trigger.

## Files Changed

| File | Change |
|------|--------|
| `src/rag/retriever.py` | Added `_bm25_doc_count` global; `_build_bm25_index()` now stores doc count via `collection.count()`; `_get_bm25()` accepts `force_rebuild: bool` and checks count mismatch to trigger rebuild |
| `tests/test_rag.py` | Added `TestBM25Invalidation` class (5 tests); updated 2 existing `TestBM25Search` tests to save/restore `_bm25_doc_count` and mock vectorstore |

## Design Details

- **`_bm25_doc_count`**: Module-level int, updated by `_build_bm25_index()` via `collection.count()`.
- **Auto-invalidation in `_get_bm25()`**: When an index already exists, checks `_get_vectorstore()._collection.count()` against `_bm25_doc_count`. Rebuilds on mismatch.
- **`force_rebuild=True`**: Skips the count check and always rebuilds.
- **Error handling**: If the count check fails, logs a warning and skips rebuild (defensive).

## Verification

```bash
conda run -n gaokao_tutor env OTEL_TRACING_ENABLED=false python -m pytest tests/ --ignore=tests/test_integration.py -v --tb=short
```

Result: **340 passed, 1 skipped, 0 failed** (5 new invalidation tests + 2 updated existing tests)

## Suggested Commit

```bash
git add src/rag/retriever.py tests/test_rag.py
git commit -m "feat(rag): BM25 索引自动失效重建 (REQ-06)

- retriever.py: 记录 BM25 构建时的文档数，检索时比对 ChromaDB count 自动重建
- 支持 force_rebuild 参数手动触发重建
- 5 项新测试 + 2 项已有测试更新，共 340 passed"
```
