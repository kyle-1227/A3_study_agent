# v0.3.0 Architecture Diagram

## Runtime Graph

```mermaid
graph TD
  START([Learner Input]) --> supervisor[Supervisor]
  supervisor --> search_query_rewriter[Search Query Rewriter]
  supervisor -->|emotional| emotional_response[Academic Support]
  supervisor -->|unknown| handle_unknown[Unknown Intent]

  search_query_rewriter --> academic_router[Academic Router]
  academic_router --> rag_retrieve[Local RAG]
  academic_router --> web_search[Tavily Web Search]
  rag_retrieve --> evidence_judge[Evidence Judge]
  web_search --> evidence_judge

  evidence_judge --> generate_answer[Generate Answer]
  evidence_judge --> mindmap_planner[Mindmap Planner]
  evidence_judge --> exercise_planner[Exercise Planner]
  evidence_judge --> review_doc_planner[Review Doc Planner]
  evidence_judge --> study_plan_emotional_intel[Study Plan Emotional Intel]

  generate_answer --> evaluate_hallucination[Hallucination Eval]
  evaluate_hallucination -->|pass| END_A([End])
  evaluate_hallucination -->|retry| rewrite_query[Retry Query Rewrite]
  rewrite_query --> academic_router

  mindmap_planner --> mindmap_agent[Mindmap Agent]
  mindmap_agent --> mindmap_reviewer[Mindmap Reviewer]
  mindmap_reviewer -->|approve| mindmap_output[Mindmap Output]
  mindmap_reviewer -->|reject| mindmap_rewrite[Mindmap Rewrite]
  mindmap_rewrite --> mindmap_agent
  mindmap_output --> END_M([End])

  exercise_planner --> exercise_agent[Exercise Agent]
  exercise_agent --> exercise_reviewer[Exercise Reviewer]
  exercise_reviewer -->|approve| exercise_output[Exercise Output]
  exercise_reviewer -->|reject| exercise_rewrite[Exercise Rewrite]
  exercise_rewrite --> exercise_agent
  exercise_output --> END_X([End])

  review_doc_planner --> review_doc_agent[Review Doc Agent]
  review_doc_agent --> review_doc_reviewer[Review Doc Reviewer]
  review_doc_reviewer -->|approve| review_doc_output[Review Doc Output]
  review_doc_reviewer -->|reject| review_doc_rewrite[Review Doc Rewrite]
  review_doc_rewrite --> review_doc_agent
  review_doc_output --> END_D([End])

  study_plan_emotional_intel --> study_plan_planner[Study Plan Planner]
  study_plan_planner --> study_plan_agent[Study Plan Agent]
  study_plan_agent --> study_plan_reviewer_academic[Academic Reviewer]
  study_plan_agent --> study_plan_reviewer_emotional[Workload Reviewer]
  study_plan_reviewer_academic --> study_plan_consensus[Study Plan Consensus]
  study_plan_reviewer_emotional --> study_plan_consensus
  study_plan_consensus -->|approve| study_plan_output[Study Plan Output]
  study_plan_consensus -->|reject| study_plan_rewrite[Study Plan Rewrite]
  study_plan_rewrite --> study_plan_agent
  study_plan_output --> END_P([End])

  emotional_response --> END_E([End])
  handle_unknown --> END_U([End])
```

## Notes

- `rag_retrieve` and `web_search` are parallel evidence-source nodes.
- `evidence_judge` is a barrier fan-in node. It runs once after both evidence-source nodes finish.
- Resource generation nodes only run after Evidence Judge has assembled judged context.
- `study_plan` is a resource-generation sub-agent, not a standalone planning branch.
- Development mode is fail-fast: planner/agent/reviewer failures raise and stop the graph instead of producing fallback output.
