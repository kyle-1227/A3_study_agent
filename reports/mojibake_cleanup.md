# Mojibake Cleanup Report

Date: 2026-06-15

## Scope

- Scanned `src`, `config`, `frontend`, `tests`, and `app.py` for obvious mojibake markers: `鈥?`, `鈹`, `閳`, `锟`, and the Unicode replacement character.
- Cleaned obvious mojibake comments and docstrings in `app.py`, `src/graph/academic.py`, `src/graph/builder.py`, `config/settings.yaml`, and `tests/test_sse_lifecycle.py`.
- Added query rewrite prompt length guidance without rewriting the existing prompt prose.

## Result

- User-visible mojibake matches: 0.
- Obvious mojibake comment/separator matches: 0.
- Residual suspicious matches in the scanned scope: none found.

## Notes

- The Chinese query rewrite prompt was verified as valid UTF-8 and was not mass-rewritten.
- No raw prompt prose was mechanically cleared beyond the targeted length/repetition guidance required by this change.
