"""Tests for evidence context provider."""

from __future__ import annotations

import logging

import pytest

from src.config.evidence_orchestration_contracts import (
    make_resource_assignment_fingerprint,
)
from src.context_engineering.providers.base import ProviderContext
from src.context_engineering.providers import evidence_provider
from src.context_engineering.providers.evidence_provider import EvidenceContextProvider
from src.context_engineering.providers import registry
from src.context_engineering.schema import ContextProviderError
from src.observability.a3_trace import reset_trace_event_sink, set_trace_event_sink


def _context(state: dict) -> ProviderContext:
    return ProviderContext(
        node_name="node",
        llm_node="llm",
        user_query="query",
        current_user_message_index=None,
        state=state,
        messages=[],
        request_id=None,
        thread_id=None,
        max_items_per_provider=10,
        max_content_chars_per_item=4000,
    )


def test_evidence_provider_objectizes_existing_candidates_without_reranking():
    state = {
        "evidence_candidates": [
            {
                "evidence_id": "local:1",
                "source_type": "local_rag",
                "provider": "local",
                "title": "Binary search notes",
                "url": "",
                "content_preview": "Binary search halves the search interval.",
                "score": 0.8,
            },
            {
                "evidence_id": "web:1",
                "source_type": "web",
                "provider": "tavily",
                "title": "Docs",
                "url": "https://example.test/docs",
                "content_preview": "Reference snippet.",
                "evidence_score": 0.5,
            },
        ]
    }

    items = EvidenceContextProvider().collect(_context(state))

    assert [item.metadata["evidence_id"] for item in items] == ["local:1", "web:1"]
    assert items[0].priority == 85
    assert items[1].priority == 75
    assert items[0].content == "Binary search halves the search interval."
    assert "raw_html" not in items[0].metadata


def test_evidence_provider_prefers_graded_evidence_handoff():
    state = {
        "graded_evidence": [
            {
                "evidence_id": "graded:1",
                "source_type": "local_rag",
                "provider": "chroma_rag",
                "title": "Judged source",
                "content": "LLM-judged evidence text.",
                "evidence_score": 0.82,
                "relevance_score": 0.82,
                "score_source": "evidence_item_grader",
                "score_scale": "0-1",
                "score_type": "task_relevance",
                "score_reason": "Directly supports the requested review document.",
            }
        ],
        "evidence_candidates": [
            {
                "evidence_id": "raw:1",
                "content_preview": "raw candidate without judge score",
            }
        ],
    }

    items = EvidenceContextProvider().collect(_context(state))

    assert [item.metadata["evidence_id"] for item in items] == ["graded:1", "raw:1"]
    assert items[0].relevance_score == pytest.approx(0.82)
    assert items[0].metadata["score_source"] == "evidence_item_grader"
    assert items[0].content == "LLM-judged evidence text."


def test_evidence_provider_collects_workspace_after_current_judged_evidence():
    state = {
        "graded_evidence": [
            {
                "evidence_id": "graded:1",
                "source_type": "local_rag",
                "title": "Fresh judged source",
                "content": "Fresh evidence.",
                "relevance_score": 0.9,
            }
        ],
        "task_workspace": {
            "schema_version": 1,
            "evidence_summaries": [
                {
                    "evidence_id": "evidence:v1:old",
                    "original_evidence_id": "old",
                    "title": "Prior judged source",
                    "summary": "Prior compact evidence.",
                    "source_type": "local_rag",
                    "purpose": "factual_grounding",
                    "relevance_score": 0.9,
                    "thread_id": "thread-1",
                    "request_id": "request-1",
                }
            ],
        },
    }

    items = EvidenceContextProvider().collect(_context(state))

    assert [item.metadata["evidence_id"] for item in items] == [
        "graded:1",
        "evidence:v1:old",
    ]
    assert items[0].priority > items[1].priority
    assert items[1].scope == "session"
    assert items[1].metadata["purpose"] == "factual_grounding"


def test_evidence_provider_collects_workspace_when_current_buckets_empty():
    state = {
        "task_workspace": {
            "schema_version": 1,
            "evidence_summaries": [
                {
                    "evidence_id": "evidence:v1:old",
                    "title": "Prior judged source",
                    "summary": "Prior compact evidence.",
                    "source_type": "web",
                    "purpose": "factual_grounding",
                    "support_score": 0.7,
                    "thread_id": "thread-1",
                    "request_id": "request-1",
                }
            ],
        }
    }

    items = EvidenceContextProvider().collect(_context(state))

    assert len(items) == 1
    assert items[0].metadata["retrieval_mode"] == "task_workspace.evidence_summaries"
    assert items[0].content == "Prior compact evidence."


def test_evidence_provider_skips_corrupt_workspace_evidence():
    state = {
        "task_workspace": {
            "schema_version": 1,
            "evidence_summaries": ["bad", {"evidence_id": "", "summary": "missing"}],
        },
        "evidence_candidates": [
            {"evidence_id": "raw:1", "content_preview": "candidate"}
        ],
    }

    items = EvidenceContextProvider().collect(_context(state))

    assert [item.metadata["evidence_id"] for item in items] == ["raw:1"]


def test_evidence_provider_dedupes_existing_bucket_overlap():
    state = {
        "evidence_candidates": [{"evidence_id": "same", "content_preview": "first"}],
        "local_evidence_candidates": [
            {"evidence_id": "same", "content_preview": "second"}
        ],
    }

    items = EvidenceContextProvider().collect(_context(state))

    assert len(items) == 1
    assert items[0].content == "first"


def _resource_assignment(*evidence_ids: str) -> dict:
    subjects = ("computer_science",)
    requirement_ids = ("requirement:1",)
    return {
        "resource_type": "quiz",
        "subjects": list(subjects),
        "requirement_ids": list(requirement_ids),
        "evidence_ids": list(evidence_ids),
        "assignment_fingerprint": make_resource_assignment_fingerprint(
            resource_type="quiz",
            subjects=subjects,
            requirement_ids=requirement_ids,
            evidence_ids=evidence_ids,
        ),
    }


def test_evidence_provider_enforces_assignment_across_every_bucket():
    state = {
        "resource_evidence_assignment": _resource_assignment(
            "assigned:graded",
            "assigned:workspace",
        ),
        "graded_evidence": [
            {
                "evidence_id": "assigned:graded",
                "content": "Assigned current evidence.",
            },
            {
                "evidence_id": "unassigned:graded",
                "content": "Leaked current evidence.",
            },
        ],
        "task_workspace": {
            "evidence_summaries": [
                {
                    "evidence_id": "assigned:workspace",
                    "summary": "Assigned workspace evidence.",
                    "purpose": "factual_grounding",
                },
                {
                    "evidence_id": "unassigned:workspace",
                    "summary": "Leaked workspace evidence.",
                    "purpose": "factual_grounding",
                },
            ]
        },
        "local_evidence": [
            {
                "evidence_id": "unassigned:local",
                "content": "Leaked local evidence.",
            }
        ],
        "web_evidence_candidates": [
            {
                "evidence_id": "unassigned:web",
                "content": "Leaked web evidence.",
            }
        ],
    }

    items = EvidenceContextProvider().collect(_context(state))

    assert [item.metadata["evidence_id"] for item in items] == [
        "assigned:graded",
        "assigned:workspace",
    ]
    serialized = repr(items)
    assert "Leaked" not in serialized
    assert "unassigned:" not in serialized


def test_evidence_provider_fails_when_assignment_resolves_to_no_candidates():
    state = {
        "resource_evidence_assignment": _resource_assignment("assigned:missing"),
        "graded_evidence": [
            {"evidence_id": "unassigned:graded", "content": "not assigned"}
        ],
    }

    with pytest.raises(
        ContextProviderError, match="resolved to no provider candidates"
    ):
        EvidenceContextProvider().collect(_context(state))


def test_evidence_provider_rejects_invalid_assignment_instead_of_falling_back():
    assignment = _resource_assignment("assigned:1")
    assignment["assignment_fingerprint"] = "0" * 64

    with pytest.raises(ContextProviderError, match="assignment_fingerprint"):
        EvidenceContextProvider().collect(
            _context(
                {
                    "resource_evidence_assignment": assignment,
                    "graded_evidence": [
                        {"evidence_id": "assigned:1", "content": "candidate"}
                    ],
                }
            )
        )


def test_evidence_provider_supports_compatible_content_fields():
    fields = (
        "snippet",
        "excerpt",
        "summary",
        "content",
        "text",
        "page_content",
        "coverage_contribution",
    )
    state = {
        "evidence_candidates": [
            {"evidence_id": key, key: f"{key} value"} for key in fields
        ]
    }

    items = EvidenceContextProvider().collect(_context(state))

    assert [item.content for item in items] == [f"{key} value" for key in fields]


def test_evidence_provider_normalizes_equivalent_score_fields():
    state = {
        "evidence_candidates": [
            {
                "evidence_id": "scaled",
                "content_preview": "scaled support",
                "support_score": 85,
                "support_score_scale": 100,
            },
            {
                "evidence_id": "confidence",
                "content_preview": "confidence support",
                "confidence": 0.7,
                "confidence_type": "support_confidence",
            },
        ]
    }

    items = EvidenceContextProvider().collect(_context(state))

    assert items[0].relevance_score == pytest.approx(0.85)
    assert items[1].relevance_score == pytest.approx(0.7)


def test_evidence_provider_does_not_treat_plain_confidence_as_relevance():
    state = {
        "evidence_candidates": [
            {
                "evidence_id": "model-confidence",
                "content_preview": "plain confidence",
                "confidence": 0.9,
            }
        ]
    }

    items = EvidenceContextProvider().collect(_context(state))

    assert len(items) == 1
    assert items[0].relevance_score is None


def test_evidence_provider_applies_limit_before_item_construction(monkeypatch):
    calls = 0
    original = evidence_provider._candidate_to_item

    def counting_to_item(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    context = _context(
        {
            "evidence_candidates": [
                {"evidence_id": "e1", "content_preview": "one"},
                {"evidence_id": "e2", "content_preview": "two"},
            ]
        }
    )
    context = ProviderContext(
        node_name=context.node_name,
        llm_node=context.llm_node,
        user_query=context.user_query,
        current_user_message_index=context.current_user_message_index,
        state=context.state,
        messages=context.messages,
        request_id=context.request_id,
        thread_id=context.thread_id,
        max_items_per_provider=1,
        max_content_chars_per_item=context.max_content_chars_per_item,
    )
    monkeypatch.setattr(evidence_provider, "_candidate_to_item", counting_to_item)

    items = EvidenceContextProvider().collect(context)

    assert len(items) == 1
    assert calls == 1


def test_evidence_provider_decode_failure_is_not_empty_result():
    with pytest.raises(ContextProviderError, match="evidence_candidates"):
        EvidenceContextProvider().collect(_context({"evidence_candidates": "bad"}))


def test_registry_rejects_evidence_missing_required_relevance_score(monkeypatch):
    monkeypatch.setattr(
        registry,
        "get_setting",
        _fake_provider_settings,
    )
    sink: list[dict] = []
    token = set_trace_event_sink(sink)
    try:
        items = registry.emit_context_items_shadow(
            logging.getLogger("test_evidence"),
            node_name="node",
            llm_node="llm",
            messages=[{"role": "user", "content": "question"}],
            state={
                "request_id": "r1",
                "thread_id": "t1",
                "evidence_candidates": [
                    {"evidence_id": "missing", "content_preview": "missing score"},
                    {
                        "evidence_id": "valid",
                        "content_preview": "valid score",
                        "score": 0.6,
                    },
                ],
            },
        )
    finally:
        reset_trace_event_sink(token)

    assert [item.metadata["evidence_id"] for item in items] == ["valid"]
    collected = next(
        event for event in sink if event["stage"] == "context_items_collected"
    )
    payload = collected
    assert payload["evidence_rejected_count"] == 1
    assert payload["missing_required_relevance_score_count"] == 1
    assert payload["evidence_reject_reasons"] == {"missing_required_relevance_score": 1}
    assert "missing score" not in repr(payload)


def test_registry_rejects_invalid_unscaled_evidence_score(monkeypatch):
    monkeypatch.setattr(
        registry,
        "get_setting",
        _fake_provider_settings,
    )
    sink: list[dict] = []
    token = set_trace_event_sink(sink)
    try:
        items = registry.emit_context_items_shadow(
            logging.getLogger("test_evidence"),
            node_name="node",
            llm_node="llm",
            messages=[{"role": "user", "content": "question"}],
            state={
                "request_id": "r1",
                "thread_id": "t1",
                "evidence_candidates": [
                    {"evidence_id": "invalid", "content_preview": "bad", "score": 75}
                ],
            },
        )
    finally:
        reset_trace_event_sink(token)

    assert items == []
    collected = next(
        event for event in sink if event["stage"] == "context_items_collected"
    )
    payload = collected
    assert payload["evidence_rejected_count"] == 1
    assert payload["invalid_relevance_score_count"] == 1
    assert payload["evidence_reject_reasons"] == {"invalid_relevance_score": 1}


def _fake_provider_settings(key: str, default=None):
    if key == "context_engineering":
        return {
            "enabled": True,
            "providers": {
                "enabled": True,
                "shadow_mode": True,
                "strict": False,
                "enabled_sources": ["evidence"],
                "max_items_per_provider": 10,
                "max_content_chars_per_item": 4000,
                "trace_top_items": 10,
            },
        }
    return default
