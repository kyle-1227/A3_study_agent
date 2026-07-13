"""Phase 3B-2A boundary tests: no Phase 4 behavior leaks in."""

from __future__ import annotations

import io
from pathlib import Path
import tokenize

from src.context_engineering.packing.apply import (
    ContextApplyResult,
    ContextApplySelection,
    ContextInjectionPolicy,
)
from src.context_engineering.packing.apply_trace import (
    build_context_applied_event,
    build_context_apply_selection_event,
    build_context_importance_scored_event,
)
from src.context_engineering.packing.importance import ContextImportanceTelemetry

ROOT = Path(__file__).resolve().parents[1]
APPLY_SOURCE = ROOT / "src" / "context_engineering" / "packing" / "apply.py"
IMPORTANCE_SOURCE = ROOT / "src" / "context_engineering" / "packing" / "importance.py"
APPLY_TRACE_SOURCE = ROOT / "src" / "context_engineering" / "packing" / "apply_trace.py"
GRAPH_LLM_SOURCE = ROOT / "src" / "graph" / "llm.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8").lower()


def _executable_source(text: str) -> str:
    tokens: list[str] = []
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type in {tokenize.COMMENT, tokenize.STRING}:
            continue
        tokens.append(token.string)
    return "".join(tokens).lower()


def _policy() -> ContextInjectionPolicy:
    return ContextInjectionPolicy(
        enabled=True,
        apply_enabled_nodes=("review_doc_agent",),
        fallback_on_error=True,
        allow_structured_output=False,
        role="system",
        position="after_system",
        exclude_message_source=True,
        max_injected_context_tokens=1000,
        injectable_sources=("memory",),
    )


def test_apply_module_does_not_call_llm_or_phase4_tools():
    source = _source(APPLY_SOURCE)

    forbidden = (
        "get_node_llm",
        ".ainvoke(",
        ".invoke(",
        "invoke_plain_llm_fail_fast",
        "chatopenai",
        "summary",
        "summarizer",
        "compaction",
        "compact",
        "write_memory",
        "save_memory",
        "upsert_memory",
        "retrieval",
        "embedding",
        "web_search",
        "tavily",
        "rerank",
    )
    for token in forbidden:
        assert token not in source


def test_importance_module_does_not_call_plain_apply_or_phase4_tools():
    source = _source(IMPORTANCE_SOURCE)

    forbidden = (
        "invoke_plain_llm_fail_fast",
        "build_applied_messages",
        "emit_context_usage_trace",
        "emit_context_packing_shadow",
        "emit_context_items_shadow",
        "summary",
        "summarizer",
        "compaction",
        "compact",
        "write_memory",
        "save_memory",
        "upsert_memory",
        "retrieval",
        "embedding",
        "web_search",
        "tavily",
        "rerank",
    )
    for token in forbidden:
        assert token not in source


def test_trace_builders_do_not_output_forbidden_payload_keys():
    applied = build_context_applied_event(
        node_name="review_doc_agent",
        llm_node="review_doc",
        policy=_policy(),
        result=ContextApplyResult(
            applied=True,
            original_message_count=1,
            final_message_count=2,
            injected_items_count=1,
            skipped_items_count=0,
            injected_context_tokens=20,
            final_messages=[{"role": "system", "content": "secret final messages"}],
            original_estimated_tokens=100,
            final_estimated_tokens=125,
            token_delta=25,
            warnings=["api_key=sk-secret-value"],
        ),
    )
    selection = build_context_apply_selection_event(
        node_name="review_doc_agent",
        llm_node="review_doc",
        selection=ContextApplySelection(
            skip_reason="",
            single_resource_result="matched_single_resource",
            selected_item_count=1,
            injectable_item_count=1,
            skipped_item_count=0,
            quality_filtered_count=0,
            budget_dropped_count=0,
            final_injected_count=1,
            injected_context_tokens=20,
            source_counts_before={"memory": 1},
            source_counts_after={"memory": 1},
            drop_reasons={},
        ),
    )
    importance = build_context_importance_scored_event(
        node_name="review_doc_agent",
        llm_node="review_doc",
        telemetry=ContextImportanceTelemetry(
            source_counts={"memory": 1},
            score_buckets={"0.75-1.00": 1},
            reason_code_counts={"useful_memory": 1},
            candidate_count=1,
            scored_count=1,
            kept_count=1,
            dropped_count=0,
            fallback_to_rule_based=False,
            scoring_elapsed_ms=3.0,
            warnings=["db_uri=postgresql://u:p@h/db"],
        ),
    )

    forbidden_keys = {
        "final_messages",
        "injected_context",
        "content",
        "metadata",
        "scorer_prompt",
        "prompt",
        "raw_response",
        "raw_output",
    }
    for event in (applied, selection, importance):
        assert forbidden_keys.isdisjoint(event)

    serialized = repr([applied, selection, importance]).lower()
    assert "secret final messages" not in serialized
    assert "api_key" not in serialized
    assert "db_uri" not in serialized
    assert "postgresql://" not in serialized


def test_trace_module_does_not_declare_forbidden_payload_fields():
    source = _source(APPLY_TRACE_SOURCE)
    forbidden_exact_fields = (
        '"final_messages"',
        '"injected_context"',
        '"content"',
        '"metadata"',
        '"scorer_prompt"',
        '"prompt"',
        '"raw_response"',
        '"raw_output"',
    )
    for token in forbidden_exact_fields:
        assert token not in source


def test_graph_raw_scorer_bypasses_ce_apply_but_keeps_mandatory_input_accounting():
    source = _source(GRAPH_LLM_SOURCE)
    marker = "async def invoke_context_importance_scorer_raw"
    assert marker in source
    raw_body = source.split(marker, 1)[1].split(
        "async def invoke_plain_llm_fail_fast",
        1,
    )[0]
    raw_code = _executable_source(raw_body)

    forbidden = (
        "invoke_plain_llm_fail_fast(",
        "emit_context_items_shadow",
        "emit_context_packing_shadow",
        "build_applied_messages",
        "prepare_messages_with_context_policy(",
        "plain_llm_output",
    )
    for token in forbidden:
        assert token not in raw_code

    required = (
        "build_llm_input_observation(",
        "emit_context_usage_trace(",
        "invoke_with_provider_transport_retry(",
    )
    for token in required:
        assert token in raw_code
