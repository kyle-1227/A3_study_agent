from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.context_engineering.session_memory import SessionContextMemoryLedgerV1
from src.context_engineering.thread_window_v3 import build_thread_context_window_v3

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_only_main_streaming_page_uses_v2_client_without_legacy_parsers():
    main_page = REPO_ROOT / "frontend" / "app" / "page.tsx"
    retired_volunteer_page = REPO_ROOT / "frontend" / "app" / "volunteer" / "page.tsx"

    assert not retired_volunteer_page.exists()
    source = main_page.read_text(encoding="utf-8")
    assert 'from "@/lib/agent-stream-client"' in source
    assert "consumeAgentStreamV2" in source
    assert '.split("\\n\\n")' not in source
    assert ".split('\\n\\n')" not in source
    assert 'case "token"' not in source
    assert 'case "text"' not in source


@pytest.mark.anyio
async def test_active_run_preserves_complete_v3_memory_snapshot():
    from app import get_thread_status_payload
    from src.run_control import finish_active_run, get_active_run, start_active_run

    request_ids = [f"request-{index}" for index in range(60)]
    ledger = SessionContextMemoryLedgerV1(
        thread_id="thread-v3",
        updated_at=datetime.now(timezone.utc),
        request_count=len(request_ids),
        request_ids=request_ids,
    )
    window = build_thread_context_window_v3(
        ledger,
        updating=True,
    ).model_dump(mode="json")
    graph = MagicMock()
    graph.aget_state = AsyncMock(
        side_effect=AssertionError("active run is authoritative")
    )

    start_active_run(
        "thread-v3",
        {
            "schema_version": "run_control_v1",
            "run_status": "running",
            "request_context_window": {},
            "thread_context_window": {},
            "session_context_memory_ledger": ledger.model_dump(mode="json"),
            "thread_context_window_v3": window,
        },
    )
    try:
        active = get_active_run("thread-v3")
        status = await get_thread_status_payload(graph, "thread-v3")
    finally:
        finish_active_run("thread-v3")

    assert active is not None
    assert len(active["session_context_memory_ledger"]["request_ids"]) == 60
    assert active["thread_context_window_v3"]["request_count"] == 60
    assert status.thread_context_window_v3.request_count == 60
    assert status.thread_context_window_v3.updating is True
    graph.aget_state.assert_not_called()
