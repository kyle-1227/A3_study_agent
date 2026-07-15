"""OpenAPI gates for the recommendation terminal and SSE transport."""

from __future__ import annotations

import app as app_module


def test_openapi_exposes_strict_recommendation_final_on_thread_status() -> None:
    app_module.app.openapi_schema = None
    schema = app_module.app.openapi()

    recommendation = schema["components"]["schemas"]["RecommendationFinalV1"]
    assert recommendation["additionalProperties"] is False
    assert {
        "schema_version",
        "type",
        "thread_id",
        "request_id",
        "terminal_status",
        "recommendations",
        "recommendation_final_id",
        "payload_hash",
    } <= set(recommendation["required"])

    status_field = schema["components"]["schemas"]["ThreadStatusResponse"][
        "properties"
    ]["last_recommendation_final_payload"]
    assert {item.get("$ref") for item in status_field["anyOf"]} == {
        None,
        "#/components/schemas/RecommendationFinalV1",
    }


def test_all_agent_stream_routes_document_only_event_stream_success() -> None:
    app_module.app.openapi_schema = None
    schema = app_module.app.openapi()
    routes = (
        ("/stream", "post"),
        ("/resume", "post"),
        ("/threads/{thread_id}/assessment-attempts", "post"),
        ("/threads/{thread_id}/continue", "post"),
        ("/streams/{stream_id}", "get"),
    )

    for path, method in routes:
        content = schema["paths"][path][method]["responses"]["200"]["content"]
        assert set(content) == {"text/event-stream"}
        assert content["text/event-stream"]["schema"] == {"type": "string"}
