"""Strict schema tests for Context Engineering telemetry."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.context_engineering.schema import ContextBudget, ContextUsageReport, TokenCount


def test_token_count_forbids_extra_fields():
    with pytest.raises(ValidationError):
        TokenCount(value=1, estimated=True, method="estimated_mixed", extra_field=True)


def test_context_budget_rejects_invalid_relationships():
    with pytest.raises(ValidationError, match="reserved_output_tokens"):
        ContextBudget(
            node_name="node",
            llm_node="llm",
            model="model",
            max_context_tokens=100,
            reserved_output_tokens=100,
            max_input_tokens=0,
            warning_ratio=0.7,
            critical_ratio=0.85,
            compact_ratio=0.9,
        )

    with pytest.raises(ValidationError, match="warning_ratio"):
        ContextBudget(
            node_name="node",
            llm_node="llm",
            model="model",
            max_context_tokens=1000,
            reserved_output_tokens=100,
            max_input_tokens=900,
            warning_ratio=0.9,
            critical_ratio=0.85,
            compact_ratio=0.9,
        )


def test_context_usage_report_rejects_inconsistent_totals():
    with pytest.raises(ValidationError, match="used_tokens"):
        ContextUsageReport(
            node_name="node",
            llm_node="llm",
            provider="provider",
            model="model",
            input_estimated_tokens=10,
            reserved_output_tokens=5,
            used_tokens=14,
            max_context_tokens=1000,
            available_tokens=986,
            used_ratio=0.014,
            warning_level="ok",
            estimated=True,
            tokenizer_mode="estimated_mixed",
            message_count=1,
            breakdown={"input_estimated_tokens": 10, "reserved_output_tokens": 5},
        )


def test_context_usage_report_forbids_extra_fields():
    with pytest.raises(ValidationError):
        ContextUsageReport(
            node_name="node",
            llm_node="llm",
            provider="provider",
            model="model",
            input_estimated_tokens=10,
            reserved_output_tokens=5,
            used_tokens=15,
            max_context_tokens=1000,
            available_tokens=985,
            used_ratio=0.015,
            warning_level="ok",
            estimated=True,
            tokenizer_mode="estimated_mixed",
            message_count=1,
            breakdown={"input_estimated_tokens": 10, "reserved_output_tokens": 5},
            prompt_tokens=10,
        )
