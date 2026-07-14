"""CE-3 provider supply tests."""

from __future__ import annotations

from typing import cast

from src.context_engineering.providers.base import ContextProvider, ProviderContext
from src.context_engineering.providers.registry import ContextProviderSettings
from src.context_engineering.providers.supply import (
    collect_context_for_policy,
    plan_provider_supply,
)
from src.context_engineering.schema import ContextItem, ContextSourceType
from src.context_engineering.packing.node_policy import resolve_context_policy


def _settings(*sources: str) -> ContextProviderSettings:
    return ContextProviderSettings(
        enabled=True,
        shadow_mode=False,
        strict=False,
        enabled_sources=tuple(cast(ContextSourceType, source) for source in sources),
        max_items_per_provider=10,
        max_content_chars_per_item=1000,
        trace_top_items=5,
    )


def _item(source_type: str, item_id: str) -> ContextItem:
    return ContextItem(
        id=item_id,
        source_type=source_type,
        title=item_id,
        content=f"{item_id} content",
        token_estimate=5,
        estimated=True,
        tokenizer_mode="estimated_mixed",
        priority=80,
        relevance_score=0.8 if source_type == "evidence" else None,
        recency_score=None,
        confidence=None,
        scope="turn",
        lifetime="turn",
        compressible=True,
        can_drop=True,
        disclosure_level="snippet",
        metadata={},
    )


def _all_source_settings() -> ContextProviderSettings:
    return _settings(
        "message",
        "memory",
        "evidence",
        "artifact",
        "profile",
        "trajectory",
        "rules",
        "curriculum",
    )


def _requested_sources_for_node(node_name: str) -> tuple[ContextSourceType, ...]:
    from src.config import clear_cache

    clear_cache()
    resolved = resolve_context_policy(
        node_name=node_name,
        llm_node=node_name,
        state={},
    )
    policy = resolved.injection_policy
    requested = tuple(
        dict.fromkeys(
            (
                *policy.required_sources,
                *policy.optional_sources,
                *policy.injectable_sources,
            )
        )
    )
    plan = plan_provider_supply(
        requested_sources=requested,
        required_sources=policy.required_sources,
        optional_sources=policy.optional_sources,
        settings=_all_source_settings(),
    )
    return plan.requested_sources


class _FakeProvider:
    def __init__(self, source_type: str, items: list[ContextItem]) -> None:
        self.name = f"{source_type}_provider"
        self.source_type = source_type
        self.items = items
        self.collect_count = 0

    def collect(self, context: ProviderContext) -> list[ContextItem]:
        self.collect_count += 1
        return list(self.items)


def test_provider_supply_plan_reports_not_registered_and_disabled():
    plan = plan_provider_supply(
        requested_sources=(
            cast(ContextSourceType, "unknown"),
            cast(ContextSourceType, "evidence"),
        ),
        required_sources=(cast(ContextSourceType, "evidence"),),
        optional_sources=(),
        settings=_settings("message"),
    )

    assert plan.provider_sources_missing == {"unknown": 1, "evidence": 1}
    assert plan.provider_missing_reasons == {
        "unknown": "provider_not_registered",
        "evidence": "provider_disabled",
    }


def test_provider_supply_reports_empty_existing_state():
    _plan, collection = collect_context_for_policy(
        node_name="review_doc_agent",
        llm_node="review_doc",
        messages=[{"role": "user", "content": "draft"}],
        state={},
        requested_sources=(cast(ContextSourceType, "evidence"),),
        required_sources=(cast(ContextSourceType, "evidence"),),
        optional_sources=(),
        settings=_settings("evidence"),
    )

    assert collection.items == []
    assert collection.provider_sources_missing == {"evidence": 1}
    assert collection.provider_missing_reasons == {"evidence": "provider_empty"}


def test_provider_supply_accepts_graded_evidence_handoff():
    _plan, collection = collect_context_for_policy(
        node_name="review_doc_agent",
        llm_node="review_doc",
        messages=[{"role": "user", "content": "draft"}],
        state={
            "graded_evidence": [
                {
                    "evidence_id": "graded:1",
                    "source_type": "local_rag",
                    "provider": "chroma_rag",
                    "title": "Judged source",
                    "content": "LLM-judged evidence text.",
                    "evidence_score": 0.77,
                    "relevance_score": 0.77,
                    "score_source": "evidence_item_grader",
                    "score_scale": "0-1",
                    "score_type": "task_relevance",
                    "score_reason": "Directly supports the requested task.",
                }
            ]
        },
        requested_sources=(cast(ContextSourceType, "evidence"),),
        required_sources=(cast(ContextSourceType, "evidence"),),
        optional_sources=(),
        settings=_settings("evidence"),
    )

    assert collection.provider_sources_missing == {}
    assert collection.provider_missing_reasons == {}
    assert len(collection.items) == 1
    assert collection.items[0].source_type == "evidence"
    assert collection.items[0].relevance_score == 0.77
    assert collection.items[0].metadata["score_source"] == "evidence_item_grader"


def test_provider_supply_rejects_raw_evidence_candidates_without_llm_score():
    _plan, collection = collect_context_for_policy(
        node_name="review_doc_agent",
        llm_node="review_doc",
        messages=[{"role": "user", "content": "draft"}],
        state={
            "evidence_candidates": [
                {
                    "evidence_id": "raw:1",
                    "content_preview": "raw candidate without judge score",
                }
            ]
        },
        requested_sources=(cast(ContextSourceType, "evidence"),),
        required_sources=(cast(ContextSourceType, "evidence"),),
        optional_sources=(),
        settings=_settings("evidence"),
    )

    assert collection.items == []
    assert collection.provider_sources_missing == {"evidence": 1}
    assert collection.provider_missing_reasons == {"evidence": "provider_empty"}
    assert collection.evidence_stats.evidence_rejected_count == 1
    assert collection.evidence_stats.missing_required_relevance_score_count == 1
    assert collection.evidence_stats.evidence_reject_reasons == {
        "missing_required_relevance_score": 1
    }


def test_provider_supply_collects_only_policy_requested_sources(monkeypatch):
    from src.context_engineering.providers import supply as supply_module

    memory_provider = _FakeProvider("memory", [_item("memory", "memory-1")])
    evidence_provider = _FakeProvider("evidence", [_item("evidence", "evidence-1")])
    monkeypatch.setattr(
        supply_module,
        "get_default_providers",
        lambda _settings: [
            cast(ContextProvider, memory_provider),
            cast(ContextProvider, evidence_provider),
        ],
    )

    _plan, collection = collect_context_for_policy(
        node_name="review_doc_planner",
        llm_node="review_doc",
        messages=[{"role": "user", "content": "plan"}],
        state={},
        requested_sources=(cast(ContextSourceType, "memory"),),
        required_sources=(),
        optional_sources=(cast(ContextSourceType, "memory"),),
        settings=_settings("memory", "evidence"),
    )

    assert [item.id for item in collection.items] == ["memory-1"]
    assert memory_provider.collect_count == 1
    assert evidence_provider.collect_count == 0


def test_agent_provider_supply_does_not_request_excluded_sources():
    expected_absent = {
        "study_plan_agent": {"evidence"},
        "review_doc_agent": {"trajectory"},
    }

    for node_name, absent_sources in expected_absent.items():
        requested = set(_requested_sources_for_node(node_name))
        assert requested.isdisjoint(absent_sources), node_name

    assert set(_requested_sources_for_node("study_plan_agent")) == {
        "profile",
        "rules",
        "trajectory",
        "memory",
        "curriculum",
        "artifact",
    }
    assert set(_requested_sources_for_node("review_doc_agent")) == {
        "evidence",
        "rules",
        "curriculum",
        "memory",
        "profile",
        "artifact",
    }
