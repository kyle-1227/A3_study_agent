"""Unit tests for LearningState definition."""

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage

from src.graph.state import (
    ACTIVITY_TIMELINE_CLEAR,
    CONTEXT_WINDOW_EVENT_CHAR_LIMIT,
    CONTEXT_WINDOW_HISTORY_CHAR_LIMIT,
    CONTEXT_USAGE_REPORTS_CLEAR,
    DICT_CLEAR,
    EVIDENCE_MEMORY_CHAR_LIMIT,
    GENERATED_ARTIFACTS_CLEAR,
    GENERATED_ARTIFACT_HISTORY_CHAR_LIMIT,
    MEMORY_CLEAR,
    LearningState,
    THREAD_MESSAGE_HISTORY_CHAR_LIMIT,
    THREAD_MESSAGE_HISTORY_LIMIT,
    activity_timeline_reducer,
    bounded_context_event_reducer,
    bounded_context_window_reducer,
    bounded_messages_reducer,
    context_usage_reports_reducer,
    evidence_memory_reducer,
    generated_artifacts_reducer,
    initial_request_reset_transient_state,
    latest_dict_reducer,
    task_workspace_reducer,
)


class TestLearningState:
    def test_state_has_required_keys(self):
        annotations = LearningState.__annotations__
        required = [
            "messages",
            "intent",
            "response_mode",
            "qa_scope",
            "last_qa_response",
            "subject",
            "keypoints",
            "context",
            "plan",
        ]
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


class TestBoundedMessagesReducer:
    def test_preserves_add_messages_id_replacement(self):
        original = HumanMessage(content="old", id="human-1")
        replacement = HumanMessage(content="new", id="human-1")

        result = bounded_messages_reducer([original], [replacement])

        assert len(result) == 1
        assert result[0].content == "new"

    def test_preserves_add_messages_remove_semantics(self):
        result = bounded_messages_reducer(
            [
                HumanMessage(content="question", id="human-1"),
                AIMessage(content="answer", id="ai-1"),
            ],
            [RemoveMessage(id="human-1")],
        )

        assert [message.id for message in result] == ["ai-1"]

    def test_keeps_recent_messages_and_drops_orphan_tool_prefix(self):
        messages = [
            HumanMessage(content=f"question-{index}", id=f"human-{index}")
            if index % 2 == 0
            else AIMessage(content=f"answer-{index}", id=f"ai-{index}")
            for index in range(THREAD_MESSAGE_HISTORY_LIMIT + 8)
        ]

        result = bounded_messages_reducer([], messages)

        assert len(result) == THREAD_MESSAGE_HISTORY_LIMIT
        assert result[-1].id == messages[-1].id

        with_orphan_tool = bounded_messages_reducer(
            [],
            [
                ToolMessage(content="tool output", tool_call_id="call-1", id="tool-1"),
                HumanMessage(content="current question", id="human-current"),
            ],
        )
        assert [message.id for message in with_orphan_tool] == ["human-current"]

    def test_keeps_a_contiguous_suffix_within_character_budget(self):
        chunk_size = THREAD_MESSAGE_HISTORY_CHAR_LIMIT // 2 + 1_000
        messages = [
            AIMessage(content="x" * chunk_size, id="oldest"),
            AIMessage(content="y" * chunk_size, id="newer"),
            HumanMessage(content="current question", id="current"),
        ]

        result = bounded_messages_reducer([], messages)

        assert [message.id for message in result] == ["newer", "current"]


class TestTaskWorkspaceReducers:
    def test_initial_request_reset_preserves_task_workspace(self):
        reset = initial_request_reset_transient_state()

        assert "task_workspace" not in reset
        assert "workspace_events" not in reset
        assert "context_influence_ledger" not in reset
        assert "context_usage_report" not in reset
        assert "context_usage_reports" not in reset
        assert "activity_timeline" not in reset
        assert "last_qa_response" not in reset
        assert "assessment_checkpoint_resources" not in reset
        assert reset["exercise_resource_v3"] == {}
        assert reset["response_mode"] == ""
        assert reset["qa_scope"] == ""
        assert reset["requires_live_verification"] is False

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

    def test_latest_dict_reducer_replaces_previous_resource_and_clears(self):
        previous = {
            "resource_type": "mindmap",
            "mindmap": {"title": "Previous map"},
        }
        current = {
            "resource_type": "study_plan",
            "study_plan": {"title": "Current plan"},
        }

        assert latest_dict_reducer(previous, current) == current
        assert latest_dict_reducer(current, DICT_CLEAR) == {}

    def test_context_histories_enforce_character_and_item_bounds(self):
        small = {"event_id": "small", "summary": "bounded"}
        oversized_event = {
            "event_id": "oversized",
            "summary": "x" * CONTEXT_WINDOW_EVENT_CHAR_LIMIT,
        }
        oversized_window = {
            "window_id": "oversized",
            "summary": "x" * CONTEXT_WINDOW_HISTORY_CHAR_LIMIT,
        }

        assert bounded_context_event_reducer(
            [],
            [small, oversized_event],
        ) == [small]
        assert bounded_context_window_reducer(
            [],
            [small, oversized_window],
        ) == [small]

    def test_memory_and_artifact_histories_drop_oversized_entries(self):
        memory = {
            "memory_id": "memory:small",
            "created_at": "2026-07-11T00:00:00+00:00",
            "summary": "bounded",
        }
        oversized_memory = {
            "memory_id": "memory:oversized",
            "created_at": "2026-07-11T01:00:00+00:00",
            "summary": "x" * EVIDENCE_MEMORY_CHAR_LIMIT,
        }
        artifact = {
            "artifact_id": "artifact:small",
            "created_at": "2026-07-11T00:00:00+00:00",
            "summary": "bounded",
        }
        oversized_artifact = {
            "artifact_id": "artifact:oversized",
            "created_at": "2026-07-11T01:00:00+00:00",
            "summary": "x" * GENERATED_ARTIFACT_HISTORY_CHAR_LIMIT,
        }

        assert evidence_memory_reducer([], [memory, oversized_memory]) == [memory]
        assert generated_artifacts_reducer([], [artifact, oversized_artifact]) == [
            artifact
        ]

    def test_observability_reducers_are_idempotent_and_support_explicit_clear(self):
        from src.observability.activity import build_activity_event
        from src.observability.llm_input import build_llm_input_observation

        activity = build_activity_event(
            thread_id="thread-1",
            request_id="request-1",
            sequence=1,
            kind="stream",
            status="completed",
            activity_key="stream:request-1",
            title="Completed",
            now="2026-07-10T00:00:00+00:00",
        ).model_dump(mode="json")
        observation = build_llm_input_observation(
            node_name="qa_agent",
            llm_node="qa_agent",
            provider="deepseek_official",
            model="deepseek-v4-pro",
            messages=[{"role": "user", "content": "question"}],
            state={"request_id": "request-1", "thread_id": "thread-1"},
            call_purpose="structured_llm",
        )
        report = observation.context_usage_report
        assert report is not None
        report_payload = report.model_dump(mode="json")

        activities = activity_timeline_reducer([activity], [activity])
        reports = context_usage_reports_reducer(
            [report_payload],
            [report_payload],
        )

        assert len(activities) == 1
        assert len(reports) == 1
        assert activity_timeline_reducer(activities, ACTIVITY_TIMELINE_CLEAR) == []
        assert context_usage_reports_reducer(reports, CONTEXT_USAGE_REPORTS_CLEAR) == []
