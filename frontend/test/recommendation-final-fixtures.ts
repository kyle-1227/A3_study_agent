export function availableRecommendationFinalWire(): Record<string, unknown> {
  return {
    schema_version: "recommendation_final_v1",
    type: "recommendation_final",
    thread_id: "thread-recommendation-1",
    request_id: "00000000-0000-4000-8000-000000000001",
    terminal_status: "available",
    mode: "explicit_request",
    user_id: "learner-1",
    subject: "python",
    learning_guidance_runtime_fingerprint: "a".repeat(64),
    generated_at: "2026-07-15T08:00:00+00:00",
    recommendations: [
      {
        recommendation_id: "recommendation-python-loops-1",
        resource_id: "python.loops.quiz",
        resource_type: "quiz",
        topic_id: "python.loops",
        title: "Python loops quiz",
        rank: 1,
        score: 0.75,
        reason: "The learner needs more practice with loops.",
      },
    ],
    candidate_snapshot: {
      schema_version: "recommendation_candidate_snapshot_v1",
      source_schema_version: "knowledge_graph_v1",
      source_data_version: "2026-07-15",
      source_fingerprint:
        "4eb1d250c3d2ab4681144665c9ecac4b22c1572d557f9dd4e4178edfda3daa10",
      subject: "python",
      candidate_count: 2,
      inventory_hash:
        "recommendation-inventory:v1:d2ca841a42f93af6ad31e2d22d790256fecaf73d381b662f2866380c6419e994",
      targets: [
        {
          resource_id: "python.loops.quiz",
          resource_type: "quiz",
          subject: "python",
          topic_id: "python.loops",
          title: "Python loops quiz",
        },
      ],
      snapshot_id:
        "recommendation-candidates:v1:7af5a588b874e3cee7311950322dd964b9d73fb1935bc0793f84db007443b3d7",
    },
    unavailable_reason: null,
    summary: "Personalized recommendations available: 1.",
    recommendation_final_id:
      "recommendation-final:v1:f0be41b8e47c3253d53c2be0ab43e72a89d192841504a6bc32f5245bfffad79b",
    payload_hash:
      "recommendation-final-payload:v1:d5898a86ed8d282112486021e49ef6c97202ec636ad45e5cc78991c41518496c",
  }
}

export function unavailableRecommendationFinalWire(): Record<string, unknown> {
  return {
    schema_version: "recommendation_final_v1",
    type: "recommendation_final",
    thread_id: "thread-recommendation-1",
    request_id: "00000000-0000-4000-8000-000000000001",
    terminal_status: "unavailable",
    mode: "explicit_request",
    user_id: "learner-1",
    subject: "python",
    learning_guidance_runtime_fingerprint: "a".repeat(64),
    generated_at: null,
    recommendations: [],
    candidate_snapshot: null,
    unavailable_reason: "no_eligible_candidates",
    summary:
      "Personalized recommendations are unavailable because no catalog candidate met the strict evidence and score thresholds.",
    recommendation_final_id:
      "recommendation-final:v1:4304bc6be21e3e0a6d733330fd0e125b1079c2cf969c74d6450656b535dfd1ce",
    payload_hash:
      "recommendation-final-payload:v1:c88a017ecac512ae267e61a7d1b55deb63371a8b7f9736c0d64d59098c98646a",
  }
}
