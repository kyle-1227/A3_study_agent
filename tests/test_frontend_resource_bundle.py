"""Frontend resource_final contract guardrails."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _frontend_dedupe_key(event: dict) -> str:
    resource_id = event.get("resource_id")
    if isinstance(resource_id, str) and resource_id:
        return f"resource_id:{resource_id}"
    resource = event.get("resource") if isinstance(event.get("resource"), dict) else {}
    return ":".join(
        [
            "resource_payload",
            str(event.get("thread_id") or ""),
            str(event.get("request_id") or ""),
            str(event.get("resource_type") or resource.get("kind") or ""),
            str(event.get("payload_hash") or ""),
        ]
    )


def test_resource_final_helper_prefers_normalized_payload_and_stable_dedupe():
    helper_source = (PROJECT_ROOT / "frontend" / "lib" / "resource-final.ts").read_text(
        encoding="utf-8"
    )

    assert "event.resource.payload" in helper_source
    assert "parseResourceFinalEvent" in helper_source
    assert "terminal_status" in helper_source
    assert "resourceFinalOutcome" in helper_source
    assert "resourceFinalDedupeKey" in helper_source
    assert "resource_id" in helper_source
    assert "payload_hash" in helper_source
    assert "resource_type" in helper_source
    assert "mergeResourceFinalIntoMessage" in helper_source


def test_frontend_dedupe_key_does_not_use_resource_type_only():
    first = {
        "type": "resource_final",
        "thread_id": "t1",
        "request_id": "r1",
        "resource_type": "mindmap",
        "payload_hash": "payload:v1:a",
    }
    second = {
        **first,
        "request_id": "r2",
    }
    repeated = {
        **first,
    }

    assert _frontend_dedupe_key(first) == _frontend_dedupe_key(repeated)
    assert _frontend_dedupe_key(first) != _frontend_dedupe_key(second)
    assert _frontend_dedupe_key(first) != "mindmap"


def test_page_uses_resource_final_helper_and_restores_persisted_payload():
    page_source = (PROJECT_ROOT / "frontend" / "app" / "page.tsx").read_text(
        encoding="utf-8"
    )
    helper_source = (PROJECT_ROOT / "frontend" / "lib" / "resource-final.ts").read_text(
        encoding="utf-8"
    )

    assert "mergeResourceFinalIntoMessage" in page_source
    assert "attachResourceFinalToAssistant" in page_source
    assert "resourceFinalDedupeRef" in page_source
    assert "status.last_resource_final_payload?.type" in page_source
    assert "resource_final_diagnostic" in helper_source
    assert 'state: "completed_without_resource"' in page_source
    assert "parseResourceFinalEvent(data)" in page_source
    assert "resourceFinalOutcome(event)" in page_source
    assert 'state: "completed_with_resource"' in helper_source


def test_frontend_profile_interrupt_and_stream_finalization_states_exist():
    page_source = (PROJECT_ROOT / "frontend" / "app" / "page.tsx").read_text(
        encoding="utf-8"
    )
    chat_source = (
        PROJECT_ROOT / "frontend" / "components" / "chat-area.tsx"
    ).read_text(encoding="utf-8")

    assert 'data.interrupt_type === "profile_completion_required"' in page_source
    assert "status.profile_completion_request" in page_source
    assert "ProfileCompletionDialog" in page_source
    assert 'state: "waiting_for_profile_completion"' in page_source
    assert 'state: "interrupted"' in page_source
    assert 'state: "failed"' in page_source
    assert "setIsLoading(false)" in page_source
    assert '"waiting_for_profile_completion"' in chat_source
    assert '"completed_with_resource"' in chat_source
    assert '"completed_without_resource"' in chat_source
