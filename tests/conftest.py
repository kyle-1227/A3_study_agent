"""Shared pytest fixtures for A3 Study Agent unit tests.

All unit tests mock external dependencies (LLM APIs, ChromaDB, web search)
so they run offline without API keys.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util._once import Once

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def human_msg():
    """Factory fixture for creating HumanMessage objects."""
    def _make(content: str = "浣犲ソ") -> HumanMessage:
        return HumanMessage(content=content)
    return _make


@pytest.fixture
def ai_msg():
    """Factory fixture for creating AIMessage objects."""
    def _make(content: str = "浣犲ソ锛屽悓瀛︼紒") -> AIMessage:
        return AIMessage(content=content)
    return _make


@pytest.fixture
def sample_state(human_msg):
    """Minimal LearningState dict for testing."""
    return {
        "messages": [human_msg("How do I use a quadratic discriminant?")],
        "intent": "academic",
        "subject": "math",
        "keypoints": ["quadratic function", "discriminant"],
        "context": [],
        "retry_count": 0,
        "hallucination_detected": False,
        "rewritten_query": "",
        "hallucination_reason": "",
        "study_plan_emotional_intel": "",
        "study_plan_emotional_profile": {},
        "study_plan_outline": "",
        "study_plan_artifact": {},
        "study_plan_markdown": "",
    }


@pytest.fixture
def mock_llm_response():
    """Factory that creates a mock LLM response with given content."""
    def _make(content: str) -> MagicMock:
        resp = MagicMock()
        resp.content = content
        return resp
    return _make


@pytest.fixture
def sample_retrieved_docs():
    """Sample retrieved documents for RAG tests."""
    return [
        {"content": "The discriminant b^2 - 4ac helps classify quadratic roots.", "source": "math_2024.pdf", "score": 0.85, "metadata": {"subject": "math"}},
        {"content": "When the discriminant is positive, a quadratic has two distinct real roots.", "source": "math_2024.pdf", "score": 0.72, "metadata": {"subject": "math"}},
    ]


@pytest.fixture
def sample_search_results():
    """Sample web search results."""
    return [
        {"content": "A machine learning course project plan includes data processing, modeling, and evaluation.", "title": "Course project plan", "url": "https://example.com/1"},
        {"content": "A data science learning path can cover Python, statistics, machine learning, and project practice.", "title": "Learning path", "url": "https://example.com/2"},
    ]

def _reset_trace_provider():
    """Force-reset the global TracerProvider so tests can set their own."""
    trace._TRACER_PROVIDER_SET_ONCE = Once()
    trace._TRACER_PROVIDER = None


@pytest.fixture
def in_memory_exporter():
    """Provide an InMemorySpanExporter for capturing spans in tests."""
    _reset_trace_provider()
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    yield exporter
    exporter.clear()
    provider.shutdown()
    _reset_trace_provider()

