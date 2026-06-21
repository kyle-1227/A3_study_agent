"""Unit tests for LearningState definition."""

from src.graph.state import MEMORY_CLEAR, LearningState, evidence_memory_reducer


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
        existing = [{"memory_id": "m1", "created_at": "2026-01-01T00:00:00", "summary": "old"}]
        update = [{"memory_id": "m1", "created_at": "2026-01-02T00:00:00", "summary": "new"}]

        result = evidence_memory_reducer(existing, update)

        assert len(result) == 1
        assert result[0]["summary"] == "new"
