# REQ-05 Completion Report: Section-aware chunking

## Summary

Created `SectionAwareSplitter` that splits exam papers by Chinese section headers
(e.g. "一、现代文阅读", "四、写作") before applying character-level sub-chunking.
Each chunk carries a `section_title` metadata field, ensuring that writing prompts
are never mixed with reading comprehension content in the same chunk.

## Files Changed

| File | Change |
|------|--------|
| `src/rag/section_splitter.py` | **NEW** — `SectionAwareSplitter` class with `SECTION_PATTERN` regex, `_split_into_sections()`, and `create_documents()` (drop-in compatible with `RecursiveCharacterTextSplitter`) |
| `src/rag/loader.py` | Added optional `splitter` parameter to `load_documents()` — when provided, uses custom splitter instead of default `RecursiveCharacterTextSplitter` |
| `scripts/build_index.py` | Uses `SectionAwareSplitter` for subjects in `EXAM_PAPER_SUBJECTS` set; sets `doc_type="exam_paper"` |
| `tests/test_section_splitter.py` | **NEW** — 19 tests across 5 test classes (pattern matching, section splitting, chunking pipeline, metadata, loader integration) |

## Design Decisions

- **Drop-in interface**: `SectionAwareSplitter.create_documents(texts, metadatas)` matches
  LangChain's text splitter API, so `load_documents()` can use either splitter transparently.
- **ADR-005 compliant**: Independent module, does not modify `retriever.py`.
- **Preamble handling**: Text before the first section header is prepended to the first section's body.
- **Fallback**: Documents without section headers produce a single section with `section_title=""`.
- **Default chunk_size=800** per task spec (vs loader's 1000), configurable via constructor.

## Verification

```bash
conda run -n gaokao_tutor env OTEL_TRACING_ENABLED=false python -m pytest tests/ --ignore=tests/test_integration.py -v --tb=short
```

Result: **335 passed, 1 skipped, 0 failed** (19 new tests for section splitter, all passing)

## Suggested Commit

```bash
git add src/rag/section_splitter.py src/rag/loader.py scripts/build_index.py tests/test_section_splitter.py
git commit -m "feat(rag): 节标题感知分块 SectionAwareSplitter (REQ-05)

- 新增 src/rag/section_splitter.py：按中文节标题拆分试卷后再分块
- loader.py 支持自定义 splitter 参数
- build_index.py 对试卷科目启用节感知分块
- 19 项新测试全部通过（共 335 passed）"
```
