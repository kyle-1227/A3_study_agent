"""Unit tests for LearningState definition."""

from src.graph.state import (
    GENERATED_ARTIFACTS_CLEAR,
    MEMORY_CLEAR,
    LearningState,
    evidence_memory_reducer,
    generated_artifacts_reducer,
    initial_request_reset_transient_state,
    task_workspace_reducer,
)


class TestLearningState:
    def test_state_has_required_keys(self):
        annotations = LearningState.__annotations__
        required = ["messages", "intent", "subject", "keypoints", "context", "plan"]
        for key in required:
            assert key in annotations, f"LearningState missing key: {key}"

    def test_state_instantiation(self):
        state: LearningState = {
            "messages": [],
            "intent": "academic",
            "subject": "math",
            "keypoints": [],
            "context": [],
            "plan": "",
        }
        assert state["intent"] == "academic"
        assert isinstance(state["messages"], list)

    def test_state_accepts_all_intents(self):
        for intent in ("academic", "emotional"):
            state: LearningState = {
                "messages": [],
                "intent": intent,
                "subject": "",
                "keypoints": [],
                "context": [],
                "plan": "",
            }
            assert state["intent"] == intent


class TestEvidenceMemoryReducer:
    def test_memory_clear_sentinel_returns_empty_list(self):
        existing = [{"memory_id": "m1", "created_at": "2026-01-01T00:00:00"}]
        assert evidence_memory_reducer(existing, MEMORY_CLEAR) == []

    def test_dedupes_by_memory_id_latest_wins(self):
        existing = [
            {"memory_id": "m1", "created_at": "2026-01-01T00:00:00", "summary": "old"}
        ]
        update = [
            {"memory_id": "m1", "created_at": "2026-01-02T00:00:00", "summary": "new"}
        ]

        result = evidence_memory_reducer(existing, update)

        assert len(result) == 1
        assert result[0]["summary"] == "new"


class TestTaskWorkspaceReducers:
    def test_initial_request_reset_preserves_task_workspace(self):
        reset = initial_request_reset_transient_state()

        assert "task_workspace" not in reset
        assert "workspace_events" not in reset
        assert "context_influence_ledger" not in reset

    def test_task_workspace_reducer_is_idempotent(self):
        update = {
            "schema_version": 1,
            "workspace_id": "workspace:v1:abc",
            "thread_id": "thread-1",
            "active_subject": "math",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "evidence_summaries": [
                {
                    "evidence_id": "evidence:v1:one",
                    "summary": "one",
                    "created_at": "2026-01-01T00:00:00+00:00",
                }
            ],
        }

        once = task_workspace_reducer({}, update)
        twice = task_workspace_reducer(once, update)

        assert len(twice["evidence_summaries"]) == 1

    def test_generated_artifacts_reducer_dedupes_and_clears(self):
        artifact = {
            "artifact_id": "artifact:v1:one",
            "created_at": "2026-01-01T00:00:00+00:00",
            "title": "One",
        }

        merged = generated_artifacts_reducer([artifact], [artifact])

        assert merged == [artifact]
        assert generated_artifacts_reducer(merged, GENERATED_ARTIFACTS_CLEAR) == []
