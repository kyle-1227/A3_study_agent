# v0.3.0 Architecture Diagram

## Runtime Graph

```mermaid
graph TD
  START([Learner Input]) --> supervisor[Supervisor]
  supervisor --> episodic_memory_retriever[Episodic Memory Retriever]
  supervisor -->|emotional| emotional_response[Academic Support]
  supervisor -->|unknown| handle_unknown[Unknown Intent]

  episodic_memory_retriever --> memory_use_decider[Memory Use Decider]
  memory_use_decider --> search_query_rewriter[Search Query Rewriter]
  search_query_rewriter --> academic_router[Academic Router]
  academic_router --> rag_retrieve[Local RAG]
  academic_router --> web_search[Tavily Web Search]
  rag_retrieve --> evidence_judge[Evidence Judge]
  web_search --> evidence_judge

  evidence_judge --> generate_answer[Generate Answer]
  evidence_judge --> resource_orchestrator[Resource Orchestrator]
  evidence_judge --> evidence_summary_output[Evidence Summary Output]

  generate_answer --> evaluate_hallucination[Hallucination Eval]
  evaluate_hallucination -->|pass| END_A([End])
  evaluate_hallucination -->|retry| rewrite_query[Retry Query Rewrite]
  rewrite_query --> academic_router

  resource_orchestrator --> resource_worker[Resource Worker]
  resource_worker --> resource_bundle_output[Resource Bundle Output]
  resource_bundle_output --> END_R([End])
  evidence_summary_output --> END_S([End])

  emotional_response --> END_E([End])
  handle_unknown --> END_U([End])
```

## Notes

- `episodic_memory_retriever` and `memory_use_decider` run before query rewriting so memory use is explicit.
- `rag_retrieve` and `web_search` are parallel evidence-source nodes.
- `evidence_judge` is a barrier fan-in node. It runs once after both evidence-source nodes finish.
- Resource generation only runs after Evidence Judge has assembled judged context.
- Single-resource and multi-resource requests both use `resource_orchestrator -> resource_worker -> resource_bundle_output`.
- Supported formal resource types are `review_doc`, `mindmap`, `quiz`, `code_practice`, `video_script`, `video_animation`, and `study_plan`.
- Development mode is fail-fast: planner/agent/reviewer failures raise and stop the graph instead of producing fallback output.
