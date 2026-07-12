# Performance Sampling Report - 2026-07-12

## Status

`incomplete/blocked` - no real workload samples were executed in this
environment. This report deliberately contains no inferred performance cause,
P95, critical-path duration, or fabricated provider result.

## Implementation Evidence

- Request, node, LLM, search, retrieval, render, database, and checkpoint
  spans now use monotonic clocks. UTC timestamps are only correlation fields.
- Browser telemetry sends only relative `performance.now()` milestones and is
  associated through request ID, thread ID, and trace ID. Browser and server
  clocks are never subtracted.
- Reports distinguish inclusive span duration, exclusive span duration,
  accumulated operation duration, wall-clock interval union, and critical
  path. A critical path is marked `incomplete` when the span hierarchy is not
  valid.
- The checked test suite validated the contracts and interval calculations.

## Environment Checks

The following checks were recorded without reading or printing secret values:

| Condition | Observed state | Sampling impact |
| --- | --- | --- |
| Backend listener on port 8000 | Not running | No live SSE workload available |
| Browser performance ingestion | Disabled in configuration | No browser milestone capability can be issued |
| Frontend performance HMAC secret | Not configured in the shell or `.env` | Ingestion cannot be enabled safely |
| Checkpointer | Configured as memory | PostgreSQL checkpoint timing is unavailable |
| `ffmpeg` | Not installed | `video_animation` render samples are blocked |
| Provider/search/DB credential entries | Present in `.env`, values not inspected | Usability must be verified only during an explicit live run |

## Sample Matrix

No row below has been executed. `n=0`; median, min, max, and P95 are therefore
not reported.

| Workload | Planned n | Completed n | Result |
| --- | ---: | ---: | --- |
| QA categories | 5 each | 0 | Blocked |
| mindmap, quiz, review_doc, code_practice, video_script | 3 each | 0 | Blocked |
| video_animation, study_plan, multi-resource | 2 each | 0 | Blocked |
| historical restore, interrupt/resume | 2 each | 0 | Blocked |

## Preconditions For A Real Run

1. Configure a 32-byte-or-longer `FRONTEND_PERFORMANCE_HMAC_SECRET` and set
   `observability.performance.frontend_ingestion.enabled` to `true`.
2. Start the backend through `python scripts/run_backend.py --no-reload` so
   reload does not add development noise.
3. Confirm the configured provider, Tavily access, local index, and database
   mode are reachable for the intended workload. Do not substitute mock data
   for real sampling.
4. Install and verify the rendering dependencies before video-animation cases.
5. Record the sample matrix through the content-free trace reports. For
   `n < 20`, label P95 provisional; for `n < 5`, report per-sample and
   median/min/max instead of treating P95 as stable.
