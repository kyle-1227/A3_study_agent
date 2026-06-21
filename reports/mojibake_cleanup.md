# Mojibake Cleanup Report

Date: 2026-06-17

## Scope

- Scanned tracked text files and the encoding regression test across `README*.md`, `docs`, `reports`, `app.py`, `src`, `config`, `tests`, and `frontend`.
- Treated PowerShell code-page display issues as non-issues when the underlying file bytes were valid UTF-8 Chinese.
- Excluded generated or binary artifacts from text replacement: `.git`, `node_modules`, `.pytest_cache`, `__pycache__`, `frontend/tsconfig.tsbuildinfo`, images, PDF files, pyc files, and SQLite binary files.

## Cleaned

- Rewrote damaged user-facing docs:
  - `README.md`
  - `README_en.md`
  - `docs/debugging_multi_subject_logs.md`
  - `docs/architecture/v0.2.0/diagram_design.md`
- Fixed the user-visible review-document fallback title in `app.py`.
- Cleaned obvious mojibake comments and docstrings in source and tests.
- Added `tests/test_encoding_integrity.py` so common mojibake patterns are blocked by tests. The test defines suspicious patterns with Unicode escapes rather than embedding broken glyphs directly.

## Result

- User-visible mojibake in the scanned source/docs/test/frontend text scope: expected to be zero.
- Remaining allowed exclusions:
  - Generated frontend build metadata (`frontend/tsconfig.tsbuildinfo`).
  - Binary files such as SQLite databases.
  - Unicode-escape pattern definitions inside the encoding integrity test.

## Notes

- Normal Chinese UTF-8 content was not rewritten merely because a terminal rendered it incorrectly.
- Historical v0.2 architecture prose was rewritten for readability instead of attempting byte-level reconstruction where the original text was not fully recoverable.
