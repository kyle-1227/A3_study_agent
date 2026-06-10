"""Unit tests for LearningState definition."""

from src.graph.state import LearningState


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
        for intent in ("academic", "planning", "emotional"):
            state: LearningState = {
                "messages": [],
                "intent": intent,
                "subject": "",
                "keypoints": [],
                "context": [],
                "plan": "",
            }
            assert state["intent"] == intent
