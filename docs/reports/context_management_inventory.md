# Context Management Inventory

Date: 2026-07-08

Scope: read-only inventory for the current dirty working tree. This report does
not delete, move, or refactor runtime code. It exists to support a later explicit
delete/rebuild decision.

Important workspace note: this inventory reflects the current working tree, not
only committed code. `src/context_engineering/workspace.py` is currently
untracked but is included because active graph/app code imports and tests it.

Tooling note: `rg` was present but failed with `Access is denied` in this
workspace, so the scan used PowerShell `Get-ChildItem` / `Select-String` plus
Python AST inspection.

## 1. Executive Map

The project currently uses "context" to mean at least five different things:

| Layer | Runtime meaning | Main files | Delete risk |
| --- | --- | --- | --- |
| Legacy memory prompt context | User/profile/conversation/memory snippets inserted into `generate_answer` system prompt | `src/context/*`, `src/graph/academic.py`, `app.py` | Medium-high: can affect answer personalization and memory explanations |
| Context Engineering kernel | Strict `ContextItem` collection, token budget, provider supply, packing, optional injection, usage telemetry | `src/context_engineering/*` | High: broad LLM invocation and trace surface |
| Retrieval/evidence context | The graph state field `context`, mostly RAG/web evidence docs for resource generation and answer prompts | `src/graph/state.py`, `src/graph/academic.py`, resource graph files | Very high: resource generation and factual grounding depend on it |
| Task workspace context | Durable thread/request workspace summaries for evidence, artifacts, gaps, and continuation | `src/context_engineering/workspace.py`, `src/graph/supervisor.py`, `src/graph/resource_generation.py`, `app.py` | High: multi-turn resource continuation and artifact reference depend on it |
| Context telemetry/window | SSE events, request/thread windows, last policy/provider/apply summaries | `app.py`, `src/schemas.py`, tests, limited frontend usage | Medium: mostly observability, but app state/tests depend on it |

The biggest source of confusion is that these layers overlap but are not the
same:

- `LearningState.context` is retrieval evidence, not the packed injected
  context block.
- `src/context` memory context is separate from `ProfileContextProvider` and
  `MemoryContextProvider`.
- Plain LLM calls can use Context Engineering apply/injection; structured output
  currently observes/shadows context but does not mutate provider-bound messages.
- `task_workspace` is not just telemetry. It feeds evidence/artifact providers
  and supervisor continuation.

## 2. Legacy Memory Context Layer

### `src/context/context_builder.py`

Purpose: builds legacy memory-context text for LLM prompts from retrieved memory
documents. This is separate from `src/context_engineering`.

Public functions:

| Function | Inputs | Output | Behavior |
| --- | --- | --- | --- |
| `build_memory_context` | `user_id`, `current_query`, optional `subject`, `profile_context`, `conversation_summary`, `budget`, `top_k_episodic`, `top_k_semantic` | `MemoryContextInjection` | Calls memory retrieval, separates episodic/semantic memories, formats profile/conversation/memory sections, enforces legacy `TokenBudget` character/token budget |
| `build_memory_explanation` | retrieval results, `max_sources` | `str` | Produces a short explanation of memory sources used |
| `format_memory_context_for_llm_node` | `MemoryContextInjection`, `verbose` | `str | None` | Renders memory injection for graph LLM nodes |
| `_match_reason_label` | match reason string | `str` | Converts internal match reasons to display labels |

Current runtime caller:

- `src/graph/academic.py` imports `build_memory_context` inside
  `generate_answer`. If `thread_id` exists, it builds memory context and prepends
  it to the system prompt. Exceptions are caught and logged, then answer
  generation continues.

Risk notes:

- This layer can duplicate `MemoryContextProvider` / `ProfileContextProvider`.
- Failure is currently non-fatal in `generate_answer`; that is a legacy behavior
  surface, not a deletion recommendation.
- File comments/text appear partly mojibake-encoded, which increases maintenance
  cost.

### `src/context/token_manager.py`

Purpose: legacy memory-level token/character budget. It is not the same as
Context Engineering model-window budgeting.

`TokenBudget` fields:

| Field | Type/rule | Meaning |
| --- | --- | --- |
| `system_prompt` | `int >= 0` | legacy allocation for system prompt |
| `user_profile` | `int >= 0` | profile allocation |
| `episodic_memories` | `int >= 0` | episodic memory allocation |
| `semantic_summary` | `int >= 0` | semantic memory summary allocation |
| `current_task` | `int >= 0` | current task allocation |
| `rag_evidence` | `int >= 0` | RAG evidence allocation |
| `conversation_summary` | `int >= 0` | conversation summary allocation |
| `total_budget` | `int > 0` | total legacy budget |
| `buffer` | `int >= 0` | reserved buffer |

`TokenBudget` methods/properties:

- `_validate_budget_relationships`
- `from_settings`
- `available`

Module constants/functions:

- `_TOKEN_BUDGET_FIELDS`
- `estimate_tokens`
- `fit_to_budget`
- `fit_to_budget_soft`
- `_required_non_negative_int`

Config source:

- `config/settings.yaml` has `memory.token_budget.*`.

Risk notes:

- This is a separate budget model from
  `context_engineering.default_reserved_output_tokens`,
  `context_engineering.model_limits`, and packing/apply token limits.

### `src/context/errors.py` and `src/context/__init__.py`

- `ContextConfigError(RuntimeError)` is exported.
- `__init__.py` exports `build_memory_context`,
  `build_memory_explanation`, `ContextConfigError`, `TokenBudget`,
  `estimate_tokens`, and `fit_to_budget`.

## 3. Context Engineering Kernel

### `src/context_engineering/schema.py`

Purpose: strict Pydantic contracts for context items, budget, usage, and
context-related errors.

Type aliases:

- `ContextSourceType`:
  `message`, `memory`, `evidence`, `artifact`, `profile`, `trajectory`,
  `rules`, `curriculum`, `unknown`
- `ContextScope`: `node`, `turn`, `session`, `project`, `global`
- `ContextLifetime`: `ephemeral`, `turn`, `session`, `cross_session`,
  `long_term`
- `ContextDisclosureLevel`: `index`, `summary`, `snippet`, `full`

Constants/functions:

- `_SECRET_PATTERNS`
- `SENSITIVE_METADATA_KEYS`
- `normalize_metadata_key`
- `is_sensitive_metadata_key`
- `sanitize_error_message`

Classes and fields:

| Class | Fields / methods |
| --- | --- |
| `ContextConfigError` | `reason`, `warning` |
| `ContextUsageError` | `reason`, `warning` |
| `ContextProviderError` | `provider`, `source_type`, `stage`, `message`, `original_exception_type` |
| `TokenCount` | `value`, `estimated`, `method` |
| `ContextBudget` | `node_name`, `llm_node`, `model`, `max_context_tokens`, `reserved_output_tokens`, `max_input_tokens`, `warning_ratio`, `critical_ratio`, `compact_ratio`; validator enforces ratio order and token relationship |
| `ContextItem` | `id`, `source_type`, `title`, `content`, `token_estimate`, `estimated`, `tokenizer_mode`, `priority`, `relevance_score`, `recency_score`, `confidence`, `scope`, `lifetime`, `compressible`, `can_drop`, `disclosure_level`, `metadata`; validator rejects empty identity/content and sensitive metadata keys |
| `ContextUsageReport` | `node_name`, `llm_node`, `provider`, `model`, `input_estimated_tokens`, `reserved_output_tokens`, `used_tokens`, `max_context_tokens`, `available_tokens`, `used_ratio`, `warning_level`, `estimated`, `tokenizer_mode`, `message_count`, `schema_size_chars`, `breakdown`; validator checks total consistency |

Boundary note: the kernel models are strict, but many graph/workspace values are
plain dictionaries until providers or reducers normalize them.

### `src/context_engineering/budget.py`

Purpose: model-window budget resolution and usage payload generation.

Functions:

- `get_context_engineering_config`
- `get_model_context_limit`
- `build_context_budget`
- `compute_context_usage`
- `build_context_usage_payload`
- `_warning_level`
- `_non_negative_int`
- `_error_payload`

Config inputs:

- `context_engineering.enabled`
- `context_engineering.model_limits`
- `context_engineering.default_reserved_output_tokens`
- `context_engineering.thresholds.*`

Current config includes `model_limits.deepseek-v4-pro: 1000000`.

### `src/context_engineering/policies.py`

Purpose: shared config policy readers for budget thresholds and output reserves.

Functions:

- `get_thresholds`
- `get_default_reserved_output_tokens`
- `resolve_reserved_output_tokens`
- `_require_enabled_config`
- `_required_ratio`
- `_positive_int`

### `src/context_engineering/tokenizer.py`

Purpose: deterministic mixed CJK/non-CJK token estimation.

Constants/functions:

- `_TOKENIZER_MODE = "estimated_mixed"`
- `count_text_tokens`
- `count_messages_tokens`
- `estimate_text_tokens_mixed`
- `estimate_messages_tokens_mixed`
- `count_schema_chars`
- `message_content_to_text`
- `_tokenizer_settings`
- `_validated_estimated_mixed_mode`
- `_estimate_mixed_token_value`
- `_is_cjk_char`

### `src/context_engineering/itemizer.py`

Purpose: convert raw provider data into sanitized `ContextItem` instances.

Constants:

- `_MAX_METADATA_STRING_CHARS = 300`
- `_MAX_METADATA_LIST_ITEMS = 20`
- `_MAX_METADATA_DEPTH = 2`

Functions:

- `estimate_item_tokens`
- `sanitize_metadata`
- `stable_item_id`
- `make_context_item`
- `_sanitize_metadata_value`

### `src/context_engineering/evidence_normalizer.py`

Purpose: normalize evidence relevance/confidence scores before context supply.

Constants:

- `_PRIMARY_SCORE_KEYS`
- `_CONFIDENCE_KEY`
- `_CONFIDENCE_MEANING_KEYS`
- `_CONFIDENCE_RELEVANCE_MEANINGS`
- `_PERCENT_SCALE_VALUES`

`EvidenceNormalizationStats` fields:

- `evidence_rejected_count`
- `evidence_reject_reasons`
- `missing_required_relevance_score_count`
- `invalid_relevance_score_count`

Methods/functions:

- `EvidenceNormalizationStats.as_event_fields`
- `normalize_evidence_candidate_score`
- `normalize_evidence_item`
- `normalize_evidence_items`
- `_score_from_mapping`
- `_normalize_score`
- `_explicit_scale`
- `_confidence_means_relevance`
- `_coerce_number`

### `src/context_engineering/trace.py`

Purpose: safe trace/SSE event payload builders for context usage, collection,
and provider errors.

Constants:

- `_ALLOWED_USAGE_KEYS`
- `_ALLOWED_ERROR_KEYS`
- `_ALLOWED_BREAKDOWN_KEYS`
- `_ALLOWED_TOP_ITEM_KEYS`

Functions:

- `build_context_usage_event`
- `_safe_breakdown`
- `build_context_usage_error_event`
- `build_context_items_collected_event`
- `build_context_provider_error_event`
- `_safe_top_item`
- `emit_context_usage`
- `emit_context_items_collected`
- `emit_context_provider_error`
- `emit_context_usage_error`

### `src/context_engineering/workspace.py`

Purpose: durable task workspace, evidence/artifact/gap compaction, continuation
metadata, and workspace trace/status payloads.

Status: currently untracked in git, but imported by runtime code/tests.

Constants:

- `WORKSPACE_SCHEMA_VERSION`
- `WORKSPACE_ID_PREFIX`
- `ARTIFACT_ID_PREFIX`
- `EVIDENCE_ID_PREFIX`
- `GAP_ID_PREFIX`
- `WORKSPACE_EVIDENCE_LIMIT`
- `WORKSPACE_GAP_LIMIT`
- `WORKSPACE_ARTIFACT_LIMIT`
- `WORKSPACE_EVENT_LIMIT`
- `WORKSPACE_TEXT_BUDGET`
- `WORKSPACE_ARTIFACT_TEXT_BUDGET`
- `WORKSPACE_EVIDENCE_TEXT_BUDGET`
- `WORKSPACE_GAP_TEXT_BUDGET`
- `TASK_WORKSPACE_CLEAR`
- `_SAFE_REF_PATTERN`
- `_RAW_TEXT_KEYS`
- `_SAFE_URL_SCHEMES`
- `_SAFE_FILENAME_PATTERN`

TypedDict fields:

| Type | Fields |
| --- | --- |
| `WorkspaceEvidenceSummary` | `evidence_id`, `original_evidence_id`, `title`, `summary`, `subject`, `normalized_subject`, `source_type`, `support_score`, `relevance_score`, `purpose`, `created_at`, `request_id`, `thread_id`, `usable_for` |
| `WorkspaceCoverageGap` | `gap_id`, `subject`, `normalized_subject`, `role`, `gap`, `suggested_search_query`, `purpose`, `priority`, `created_at`, `request_id`, `thread_id` |
| `WorkspaceArtifactSummary` | `artifact_id`, `resource_type`, `title`, `summary`, `message_preview`, `subject`, `normalized_subject`, `active_learning_goal`, `active_learning_goal_present`, `normalized_learning_goal`, `purpose`, `thread_id`, `request_id`, `created_at`, `metrics`, `artifact_refs` |
| `WorkspaceUpdate` | `schema_version`, `workspace_id`, `scope`, `thread_id`, `request_id`, `active_subject`, `normalized_subject`, `active_learning_goal`, `normalized_learning_goal`, `evidence_state`, `updated_at`, `updated_sources`, `evidence_summaries`, `coverage_gaps`, `artifacts_by_id`, `latest_artifact_by_resource_type`, `artifacts`, `diagnostics` |
| `WorkspaceTracePayload` | workspace identity, counts, active subject/goal, updated sources, diagnostics |
| `WorkspaceContinuationContext` | `can_continue`, `continuation_applied`, `skip_reason`, `workspace_id`, `thread_id`, `request_id`, `active_subject`, `normalized_subject`, `active_learning_goal`, `normalized_learning_goal`, `resource_types`, `diagnostics` |

Functions:

- `utc_now_iso`
- `stable_workspace_id`
- `stable_artifact_id`
- `stable_evidence_id`
- `stable_gap_id`
- `normalize_learning_goal`
- `workspace_scope_from_state`
- `sanitize_workspace_text`
- `sanitize_workspace_metadata`
- `compact_evidence_item`
- `compact_artifact_result`
- `build_workspace_evidence_update`
- `build_workspace_artifact_update`
- `merge_task_workspace`
- `workspace_trace_payload`
- `workspace_status_payload`
- `workspace_continuation_context`
- `workspace_continuation_trace_payload`
- `_continuation_skip`
- `_requested_resource_types_from_state`
- `_has_explicit_current_subject`
- `_normalize_workspace_subject`
- `_base_workspace_update`
- `_base_workspace_from_scope`
- `_coerce_workspace`
- `_should_rotate_workspace`
- `_goal_overlap`
- `_compact_coverage_gaps`
- `_safe_artifact_refs`
- `_safe_ref_value`
- `_safe_url`
- `_safe_filename_or_relative`
- `_safe_relative_path`
- `_safe_metrics`
- `_merge_list_by_id`
- `_merge_artifact_maps`
- `_latest_artifact_by_type`
- `_bounded_items`
- `_enforce_workspace_text_budget`
- `_safe_str_dict`
- `_bounded_strings`
- `_safe_ratio`
- `_safe_iso`
- `_stable_id`
- `_json_chars`

Active behavior:

- Evidence judge output is compacted into `task_workspace`.
- Resource generation artifacts are indexed into `task_workspace`.
- Supervisor can continue a resource request from workspace subject/goal when
  the new user request lacks an explicit current subject.
- Artifact and evidence context providers read workspace summaries.

## 4. Context Providers

Provider protocol source: `src/context_engineering/providers/base.py`.

`ProviderContext` fields:

- `node_name`
- `llm_node`
- `user_query`
- `current_user_message_index`
- `state`
- `messages`
- `request_id`
- `thread_id`
- `max_items_per_provider`
- `max_content_chars_per_item`

`ContextProvider` protocol:

- `name`
- `source_type`
- `collect(context) -> list[ContextItem]`

### Registry and supply

`src/context_engineering/providers/registry.py`

`ContextProviderSettings` fields:

- `enabled`
- `shadow_mode`
- `strict`
- `enabled_sources`
- `max_items_per_provider`
- `max_content_chars_per_item`
- `trace_top_items`

Default provider order:

1. `MessageContextProvider`
2. `MemoryContextProvider`
3. `EvidenceContextProvider`
4. `ProfileContextProvider`
5. `RulesContextProvider`
6. `ArtifactContextProvider`
7. `TrajectoryContextProvider`
8. `CurriculumContextProvider`

Functions:

- `get_context_provider_settings`
- `get_default_providers`
- `get_registered_provider_sources`
- `collect_context_items`
- `collect_context_items_by_source`
- `emit_context_items_shadow`
- `_collect_with_errors`
- `_settings_from_config`
- `_validate_enabled_sources`
- `_positive_int`
- `_user_query_from_messages`
- `_optional_string`

`src/context_engineering/providers/supply.py`

Constants:

- `MISSING_REASON_PROVIDER_NOT_REGISTERED`
- `MISSING_REASON_PROVIDER_DISABLED`
- `MISSING_REASON_PROVIDER_EMPTY`
- `MISSING_REASON_PROVIDER_ERROR`

`ProviderSupplyPlan` fields:

- `requested_sources`
- `required_sources`
- `optional_sources`
- `enabled_sources`
- `disabled_sources`
- `unregistered_sources`
- `provider_count`
- `provider_sources_missing`
- `provider_missing_reasons`

`ContextCollectionResult` fields:

- `items`
- `provider_count`
- `provider_sources_missing`
- `provider_missing_reasons`
- `errors`
- `evidence_stats`

Functions:

- `plan_provider_supply`
- `collect_context_for_policy`
- `emit_context_provider_supply_plan`
- `emit_context_provider_supply`
- `emit_context_items_collected`
- `emit_context_provider_errors`
- `_collect_provider`
- `_user_query_from_messages`
- `_dedupe_sources`
- `_safe_sources`
- `_source_counts`
- `_safe_int_dict`
- `_safe_reason_dict`
- `_optional_string`

### Provider-by-provider inventory

| Provider | Reads | Emits `ContextItem` source | Notable metadata / policy |
| --- | --- | --- | --- |
| `MessageContextProvider` | Current/recent chat messages | `message` | `role`, `message_index`, `kind`, `content_hash`, `request_id`, `thread_id`; current user query is priority 100 and cannot drop |
| `MemoryContextProvider` | `selected_memory`, `conversation_summary`, `memory_summary`, `memory_summaries`, `episodic_memory_results`, `semantic_memory_results` | `memory` | `memory_id`, `memory_type`, `score`, `created_at`, `match_reason`, `source_bucket`; priorities roughly preference 70, semantic 65, other 60 |
| `EvidenceContextProvider` | `graded_evidence`, `evidence_items`, `task_workspace.evidence_summaries`, `local_evidence`, `web_evidence`, `retrieval_evidence`, `evidence_candidates`, `local_evidence_candidates`, `web_evidence_candidates` | `evidence` | Dedupes evidence/source IDs, normalizes scores, workspace evidence requires purpose `factual_grounding`; workspace scope is session |
| `ProfileContextProvider` | `profile_summary`, `profile_context`, `learner_profile_summary`, `learner_profile`, `preferences`, `weaknesses`, `strengths`, nested `profile` / `user_profile` | `profile` | `profile_source`, `user_id`, `confidence` |
| `RulesContextProvider` | `context_engineering.rules`, `context_rules`, `node_rules`, `runtime_rules`, `node_output_contracts`, `resource_quality_rules`, `reviewer_rubrics` | `rules` | Priority 95, node scope, non-compressible, non-droppable |
| `ArtifactContextProvider` | `task_workspace.artifacts_by_id`, `task_workspace.artifacts`, `resource_artifacts_by_type`, `last_generated_artifacts`, specific artifact fields | `artifact` | Relevance by thread/subject; metadata includes artifact refs, resource/task type, subject, thread/request/workspace IDs |
| `TrajectoryContextProvider` | `trajectory`, `step_summaries`, `learning_state`, `progress`, `mastery_profile`, last quiz results | `trajectory` | `step_index`, `node`, `status` |
| `CurriculumContextProvider` | `curriculum_context`, `subject`, `primary_subject`, `available_subjects`, `learning_path`, `chapter_structure`, `knowledge_structure`, `keypoints` | `curriculum` | `path_id` and curriculum alignment metadata |

Provider helper functions were also scanned. Most helpers are local conversion,
dedupe, scoring, and metadata extraction functions. They should be treated as
part of provider behavior until tests prove otherwise.

## 5. Packing, Node Policy, Apply, and Importance

### `src/context_engineering/packing/schema.py`

Aliases:

- `PackingStrategy = "priority_budget"`
- `PackingReason = "required" | "fits_budget" | "over_budget" | "source_disabled"`

`ContextPackingError` fields:

- `reason`
- `warning`
- `node_name`
- `llm_node`
- `selected_tokens`
- `budget_tokens`
- `original_exception_type`

`PackingDecision` fields:

- `item_id`
- `source_type`
- `title`
- `selected`
- `reason`
- `token_estimate`
- `priority`
- `can_drop`
- `budget_before`
- `budget_after`

`PackedContext` fields:

- `node_name`
- `llm_node`
- `strategy`
- `selected_items`
- `dropped_items`
- `decisions`
- `rendered_context`
- `max_context_block_tokens`
- `selected_tokens`
- `dropped_tokens`
- `required_tokens`
- `optional_tokens`
- `remaining_tokens`
- `overflow`
- `warnings`

### `src/context_engineering/packing/policies.py`

`PackingPolicy` fields:

- `enabled`
- `shadow_mode`
- `apply_to_llm`
- `strategy`
- `max_context_block_tokens`
- `trace_selected_items`
- `trace_dropped_items`
- `enabled_nodes`
- `enabled_sources`

Functions:

- `get_packing_policy`
- `node_enabled`
- `_policy_from_config`
- `_validate_strategy`
- `_positive_int`
- `_optional_string_list`
- `_config_error`

### `src/context_engineering/packing/packer.py`

Functions:

- `pack_context_items`
- `_validate_inputs`
- `_optional_sort_key`
- `_decision`

Behavior:

- Required items (`can_drop=False`) are selected first.
- Optional items are sorted by priority, relevance, confidence, recency, token
  estimate, and ID.
- Over-budget required items produce overflow/warnings instead of silently
  disappearing.

### `src/context_engineering/packing/render.py`

Constants/functions:

- `_MAX_TITLE_CHARS`
- `render_selected_context`
- `_redact_content`

### `src/context_engineering/packing/trace.py`

Trace events/functions:

- `context_packing_plan`
- `context_packed`
- `context_packing_error`
- `emit_context_packing_shadow`
- safe preview helpers for selected/dropped items and warnings

### `src/context_engineering/packing/node_policy.py`

Constants:

- `_VALID_MODES`
- `_VALID_POLICY_SOURCES`
- `_ALLOWED_STALE_POLICIES`
- `_ALLOWED_SOURCES`

Dataclass fields:

| Dataclass | Fields |
| --- | --- |
| `SourceBudgetPolicy` | `source_type`, `max_items`, `max_tokens`, `min_priority`, `min_relevance_score`, `min_trust_level`, `allowed_purposes`, `require_user_match`, `require_thread_match`, `require_subject_match`, `require_task_match`, `strict_match`, `stale_policy` |
| `NodeContextPolicy` | `mode`, `risk_tier`, `max_injected_context_tokens`, `max_items_total`, `min_injectable_items`, `injectable_sources`, `required_sources`, `optional_sources`, `exclude_message_source`, `source_overrides` |
| `ResolvedContextPolicy` | `mode`, `risk_tier`, `policy_source`, `injection_policy`, `source_policies`, `legacy_mode_enabled`, `node_policy_enabled`, `summary` |

Functions:

- `resolve_context_policy`
- `build_context_policy_summary`
- `should_emit_context_policy_summary`
- `_policy_from_node_policy`
- `_raw_node_policy`
- `_parse_node_policy`
- `_source_policies_from_config`
- `_source_policy_from_mapping`
- `_source_budget_from_defaults`
- `_matched_node_group`
- `_resource_type_from_state`
- `_legacy_global_configured`
- `_merge_sequence`
- `_bool_value`
- `_positive_int`
- `_optional_int`
- `_optional_float`
- `_optional_sources`
- `_optional_strings`
- `_valid_mode`
- `_valid_policy_source`

Confusing naming to keep in mind:

- Config has `packer.apply.enabled`.
- Config also has `packer.apply.apply_enabled_nodes`.
- Node groups/node policies can set mode `active`, `observe_only`, or
  `disabled`.
- `packer.apply.apply_enabled_nodes: []` does not by itself mean no active node
  policy exists.

### `src/context_engineering/packing/source_policy.py`

Purpose: filter packed/provider items by source-specific trust, match, purpose,
staleness, and budget rules.

Constants:

- `_USER_ALIASES`
- `_THREAD_ALIASES`
- `_SUBJECT_ALIASES`
- `_TASK_ALIASES`
- `_PURPOSE_ALIASES`
- `_ARTIFACT_ALIASES`

`SourceFilterResult` fields:

- `kept_items`
- `dropped_items`
- `source_counts_before`
- `source_counts_after`
- `source_counts_dropped`
- `source_drop_reasons`
- `budget_drop_reasons`
- `drop_reasons`
- `warnings`

Functions:

- `filter_context_items_by_source_policy`
- `_drop_item`
- `_drop_reason`
- `_source_drop_reason`
- `_budget_drop_reason`
- `_metadata_relevance`
- `_matches_policy`
- `_value_matches`
- `_purpose_allowed`
- `_metadata_values`
- `_state_match_values`
- `_apply_source_budgets`
- `_sort_for_source_budget`
- `_is_stale`
- `_metadata_number`
- `_metadata_texts`
- `_string_values`
- `_normalize_text`

### `src/context_engineering/packing/apply.py`

Purpose: decide whether/how to inject rendered context into LLM messages and
build context-apply traces/errors.

Constants:

- `_ALLOWED_SOURCES`
- `_ALLOWED_DROP_ORDER_KEYS`
- `_INJECTED_CONTEXT_HEADER`
- `_INJECTED_CONTEXT_FOOTER`
- `_CONTEXT_SECRET_PATTERNS`
- `_TRUNCATION_MARKER`

Dataclass fields:

| Dataclass | Fields |
| --- | --- |
| `RouteRolloutPolicy` | `enabled`, `route_name`, `apply_enabled_nodes`, `require_single_resource_request`, `sample_rate`, `min_injectable_items` |
| `ApplyQualityPolicy` | `min_priority`, `min_relevance_score`, `max_items_total`, `max_items_per_source` |
| `ApplyBudgetPolicy` | `graceful_degradation_enabled`, `drop_order`, `fallback_if_empty_after_drop` |
| `ApplyFormatPolicy` | `group_by_source`, `include_untrusted_context_warning`, `include_section_headers`, `max_content_chars_per_item`, `source_order` |
| `ImportanceScoringPolicy` | `enabled`, `shadow_mode`, `mode`, `llm_node`, `max_items_to_score`, `max_content_preview_chars`, `timeout_seconds`, `fallback_to_rule_based`, `emit_shadow_telemetry`, `min_shadow_score_for_analysis`, `enabled_for_observe_only`, `disabled_reason` |
| `ContextInjectionPolicy` | `enabled`, `apply_enabled_nodes`, `fallback_on_error`, `allow_structured_output`, `role`, `position`, `exclude_message_source`, `max_injected_context_tokens`, `injectable_sources`, `required_sources`, `optional_sources`, `mode`, `risk_tier`, `policy_source`, `route_rollout`, `quality`, `budget`, `format`, `importance_scoring` |
| `ContextApplySelection` | `skip_reason`, `single_resource_result`, `selected_item_count`, `injectable_item_count`, `skipped_item_count`, `quality_filtered_count`, `budget_dropped_count`, `final_injected_count`, `injected_context_tokens`, `source_counts_before`, `source_counts_after`, `drop_reasons`, `source_counts_dropped`, `warnings`, `mode`, `risk_tier`, `policy_source`, `source_drop_reasons`, `budget_drop_reasons`, `final_items`, `rendered_context` |
| `ContextApplyResult` | `applied`, `fallback_used`, `original_message_count`, `final_message_count`, `injected_items_count`, `skipped_items_count`, `injected_context_tokens`, `final_messages`, `budget_dropped_count`, `final_injected_count`, `original_estimated_tokens`, `final_estimated_tokens`, `token_delta`, `source_counts_after`, `drop_reasons`, `warnings`, `mode`, `risk_tier`, `policy_source`, `source_drop_reasons`, `budget_drop_reasons` |
| `ContextApplyError` | `reason`, `warning`, `node_name`, `llm_node`, `stage`, `fallback_used`, `error_scope`, `recoverable`, `original_exception_type`, required/optional/provider/filter/source count fields |

Functions:

- `get_context_injection_policy`
- `apply_node_enabled`
- `evaluate_context_apply_route`
- `detect_single_resource_request`
- `make_context_apply_skip_selection`
- `with_context_apply_selection_warnings`
- `prepare_context_apply_selection`
- `filter_injectable_items`
- `sanitize_context_content`
- `render_injected_context`
- `build_applied_messages`
- `build_applied_messages_from_selection`
- `build_applied_messages_from_rendered_context`
- `_policy_from_config`
- `_route_rollout_from_config`
- `_apply_quality_policy`
- `_apply_budget_policy`
- `_apply_format_policy`
- `_importance_scoring_policy`
- `_valid_role`
- `_valid_position`
- `_valid_mode`
- `_valid_risk_tier`
- `_positive_int`
- `_optional_positive_int`
- `_optional_float`
- `_optional_string_list`
- `_source_list`
- `_drop_order`
- `_string_keyed_ints`
- `_sample_rate_allows`
- `_injectable_sources`
- `_item_quality_passes`
- `_fit_items_to_budget`
- `_budget_sort_key`
- `_item_score`
- `_item_recency`
- `_context_warning_prefix`
- `_render_context_block`
- `_render_context_item`
- `_truncate_content`
- `_insert_context_message`
- `_insert_after_system`
- `_append_to_last_user`
- `_message_role`
- `_message_content`
- `_make_message`
- `_content_to_text`
- `_message_copy`
- `_source_counts`
- `_drop_reason_counts`
- `_selection_to_trace`
- `_apply_result_to_trace`
- `_error_payload`
- `_sanitize_warning`
- `_safe_reason`
- `_policy_error`

Risk notes:

- Names like `fallback_on_error`, `fallback_used`,
  `fallback_if_empty_after_drop`, and `fallback_to_rule_based` exist in the
  dataclasses/config surface. Current config sets these false, but their presence
  is a governance risk surface.
- This file is central to actual message mutation for plain LLM calls.

### `src/context_engineering/packing/orchestrator.py`

Purpose: top-level per-node pipeline for policy resolution, provider collection,
packing, filtering, apply selection, message mutation, and trace emission.

`ContextPreparedMessages` fields:

- `messages_for_llm`
- `original_messages`
- `trace_call_id`
- `next_trace_seq`
- `context_apply_applied`
- `context_apply_fallback_used`
- `resolved_policy`
- `selection`
- `apply_result`

`_TraceSequencer` fields:

- `request_id`
- `thread_id`
- `node_name`
- `llm_node`
- `call_id`
- `seq`

Functions:

- `prepare_messages_with_context_policy`
- `_trace_ids_from_state`
- `_collect_for_policy`
- `_pack_items`
- `_raise_if_required_sources_missing`
- `_raise_if_required_sources_filtered_out`
- `_requested_sources`
- `_emit_context_policy_resolved`
- `_emit_context_source_filter`
- `_emit_context_apply_plan`
- `_emit_context_apply_selection`
- `_emit_context_applied`
- `_emit_context_apply_error`
- `_emit_context_policy_summary`
- `_emit_workspace_context_collected`
- `_selection_payload`
- `_apply_result_payload`
- `_safe_counts`
- `_safe_reason_map`
- `_safe_warnings`
- `_safe_text`

### `src/context_engineering/packing/importance.py`

Purpose: optional context-importance scoring contract and telemetry. Current
config has importance scoring disabled.

Constants/classes:

- `_ALLOWED_REASON_CODES`
- `ContextImportanceError`
- `ContextImportanceScore`
- `ContextImportanceScores`
- `ContextImportanceTelemetry`

`ContextImportanceScore` fields:

- `item_id`
- `score`
- `reason_code`

`ContextImportanceTelemetry` fields:

- `node_name`
- `llm_node`
- `mode`
- `source_counts`
- `score_buckets`
- `reason_code_counts`
- `candidate_count`
- `scored_count`
- `kept_count`
- `dropped_count`
- `fallback_to_rule_based`
- `scoring_elapsed_ms`
- `disabled_reason`
- `error_reason`
- `error_type`
- `warnings`

Functions:

- `build_importance_scorer_messages`
- `parse_importance_scorer_output`
- `aggregate_importance_success`
- `aggregate_importance_failure`
- `_score_bucket`
- `_safe_source_counts`
- `_safe_error_reason`
- `_safe_error_type`
- `_safe_warnings`

## 6. Graph State Context Fields

Source: `src/graph/state.py`.

Reducers/sentinels:

| Name | Meaning |
| --- | --- |
| `CONTEXT_CLEAR` | Clears retrieval `context` list |
| `MEMORY_CLEAR` | Clears evidence memory lists |
| `RESOURCE_RESULTS_CLEAR` | Clears resource branch results |
| `TASK_WORKSPACE_CLEAR` | Clears task workspace |
| `DICT_CLEAR` | Clears dict reducer state |
| `GENERATED_ARTIFACTS_CLEAR` | Clears generated artifacts |
| `WORKSPACE_EVENTS_CLEAR` | Clears workspace events |
| `evidence_memory_reducer` | Dedupes by `memory_id`, latest wins, max 20 |
| `bounded_context_window_reducer` | Keeps last 30 request context windows |
| `bounded_context_event_reducer` | Keeps last 120 context/window events |
| `merge_dict_reducer` | Shallow merge, with clear sentinel |
| `latest_dict_reducer` | Last dict wins |
| `latest_string_reducer` | Last string wins |
| `generated_artifacts_reducer` | Dedupe by `artifact_id`, max 30 |
| `task_workspace_reducer` | Delegates to `merge_task_workspace` |
| `context_reducer` | Appends retrieval context or clears |
| `resource_branch_results_reducer` | Dedupe branch results by resource type/order |

Persistent/context-window fields on `LearningState`:

| Field | Meaning |
| --- | --- |
| `conversation_summary` | Long-lived conversation summary |
| `evidence_summary_memory` | Long-lived evidence summaries |
| `evidence_gap_memory` | Long-lived evidence gaps |
| `task_workspace` | Durable workspace summary/index |
| `workspace_events` | Durable workspace event summaries |
| `context_usage` | Last context usage report |
| `context_usage_history` | Bounded usage history |
| `request_context_window` | Latest request context-window status |
| `context_window_events` | Bounded context/window trace summaries |
| `last_context_policy_by_node` | Latest resolved context policy summary by node |
| `last_provider_supply_by_node` | Latest provider supply by node |
| `last_context_selection_by_node` | Latest apply selection by node |
| `last_context_applied_by_node` | Latest applied-context summary by node |
| `last_drop_reasons_by_node` | Latest drop reasons by node |
| `last_resource_subnodes` | Resource branch subnode summaries |
| `resource_artifacts_by_type` | Latest resource artifacts indexed by type |
| `last_generated_artifacts` | Bounded generated artifact list |
| `context` | Retrieval/evidence documents for current request/resource generation |

Reset behavior:

- `initial_request_reset_transient_state` clears routing, selected memory,
  query/evidence scratch fields, retrieval `context`, generated artifacts,
  run-control flags, `context_usage`, and `request_context_window`.
- It does not clear `messages`, `conversation_summary`,
  `evidence_summary_memory`, `evidence_gap_memory`, `task_workspace`, or
  `workspace_events`.

Risk note: the state object mixes durable memory/workspace, current retrieval
docs, context-engineering telemetry, and UI status fields. Deleting by name alone
is unsafe because `context` fields have different meanings.

## 7. Runtime Flow Inventory

### Plain LLM invocation

Source: `src/graph/llm.py`.

Key behavior:

- `invoke_plain_llm_fail_fast` calls
  `prepare_messages_with_context_policy`.
- The returned messages may include injected context if node policy/apply mode
  allows it.
- It emits context usage for the final messages.
- Trace payload includes `context_apply_applied` and
  `context_apply_fallback_used`.
- `invoke_context_importance_scorer_raw` intentionally bypasses plain LLM
  context usage/items/packing/apply/state/memory writes.

Adjacent legacy risk:

- `src/graph/llm.py` still contains older provider/model/fallback naming
  surfaces already documented in prior reports. They are not deleted here.

### Structured output invocation

Source: `src/llm/structured_output.py`.

Key behavior:

- Emits context usage after schema injection.
- Collects context items in shadow mode.
- Emits packing shadow telemetry.
- Emits `structured_context_apply` observe telemetry.
- Does not currently mutate structured-output provider messages with context.

Important distinction:

- Structured output context code is telemetry/observe-only at present, while
  plain LLM context policy can mutate messages.

### Evidence and answer generation

Source: `src/graph/academic.py`.

Key behavior:

- Evidence judge output creates `task_workspace` updates via
  `build_workspace_evidence_update`.
- `generate_answer` reads `state["context"]` as retrieved factual grounding and
  formats it into the answer prompt.
- `generate_answer` separately calls legacy `build_memory_context` and prepends
  memory context to the system prompt when `thread_id` exists.
- The final answer LLM call goes through `invoke_plain_llm_fail_fast`, so Context
  Engineering may add another injected context layer.
- Hallucination evaluation also reads `state["context"]`.

Risk note:

- `generate_answer` has both legacy memory context and Context Engineering
  message preparation in the same path.

### Resource generation

Sources:

- `src/graph/resource_generation.py`
- `src/graph/exercises.py`
- `src/graph/mindmap.py`
- `src/graph/review_doc.py`
- `src/graph/code_practice.py`
- `src/graph/study_plan.py`
- `src/graph/video_script.py`

Key behavior:

- Resource planners/agents generally read `state["context"]` for current
  evidence.
- `resource_generation.py` indexes generated artifacts into `task_workspace`
  via `build_workspace_artifact_update`.
- Some resource nodes contain adjacent non-context fallback/degraded-generation
  logic. Those are risky but not context-specific.

### Supervisor continuation

Source: `src/graph/supervisor.py`.

Key behavior:

- `workspace_continuation_context` lets a resource-only follow-up inherit active
  subject/goal from `task_workspace` when the new request lacks an explicit
  subject.
- Emits workspace continuation trace payloads.

Deletion risk:

- Removing workspace without replacing continuation logic will change multi-turn
  resource requests.

### App/SSE state and API status

Source: `app.py`, `src/schemas.py`.

Context-safe payload helpers:

- `_safe_context_top_items`
- `_safe_packing_preview_items`
- `_safe_int_dict`
- `_safe_context_event_summary`
- `_safe_workspace_event_summary`
- `_context_policy_resolved_payload`
- `_context_provider_supply_plan_payload`
- `_context_provider_supply_payload`
- `_context_source_filter_payload`
- `_context_window_status`
- `_update_context_window_state_from_trace`

SSE/trace stages handled:

- `context_policy_resolved`
- `context_provider_supply_plan`
- `context_provider_supply`
- `context_source_filter`
- `context_window_state_updated`
- `context_window_state_update_failed`
- `context_usage`
- `context_usage_error`
- `context_items_collected`
- `context_provider_error`
- `context_packing_plan`
- `context_packed`
- `context_packing_error`
- `context_apply_plan`
- `context_apply_selection`
- `context_applied`
- `context_apply_policy_resolved_summary`
- `context_apply_error`
- `context_importance_scored`
- workspace stages including `task_workspace.update_planned`,
  `task_workspace.updated`, `task_workspace.update_failed`,
  `task_workspace.continuation_*`, `resource_artifacts.indexed`,
  `workspace_context.collected`

API schema fields:

- `ThreadStatusResponse.context_usage`
- `ThreadStatusResponse.context_usage_history`
- `ThreadStatusResponse.request_context_window`
- `ThreadStatusResponse.thread_context_window`

Additional context injection outside CE:

- `generate_sse` injects profile context as an initial `SystemMessage` when
  `user_id` is available. This is separate from `ProfileContextProvider`.

### Frontend consumption

Source: `frontend/app/page.tsx`.

Found direct consumption:

- `mapContextUsage`
- `mapContextUsageError`
- SSE `context_usage`
- SSE `context_usage_error`
- status field `context_usage`

No direct source-code hit was found for the richer apply/provider/window events
in frontend app/components/lib files outside `node_modules`.

## 8. Configuration Inventory

Source: `config/settings.yaml`.

Active Context Engineering config:

| Key | Current value/meaning |
| --- | --- |
| `context_engineering.enabled` | `true` |
| `context_engineering.strict` | `true` |
| `tokenizer.mode` | `estimated_mixed` |
| `tokenizer.estimated` | `true` |
| `model_limits.deepseek-v4-pro` | `1000000` |
| `thresholds.warning_ratio` | `0.70` |
| `thresholds.critical_ratio` | `0.85` |
| `thresholds.compact_ratio` | `0.90` |
| `default_reserved_output_tokens` | `16000` |
| `providers.enabled` | `true` |
| `providers.shadow_mode` | `false` |
| `providers.strict` | `false` |
| `providers.enabled_sources` | message, memory, evidence, artifact, profile, trajectory, rules, curriculum |
| `providers.max_items_per_provider` | `10` |
| `providers.max_content_chars_per_item` | `4000` |
| `providers.trace_top_items` | `10` |
| `packer.enabled` | `true` |
| `packer.shadow_mode` | `false` |
| `packer.apply_to_llm` | `false` |
| `packer.max_context_block_tokens` | `120000` |
| `packer.apply.enabled` | `true` |
| `packer.apply.fallback_on_error` | `false` |
| `packer.apply.allow_structured_output` | `false` |
| `packer.apply.structured_output_context.enabled` | `true` |
| `packer.apply.structured_output_context.mode` | `observe_only` |
| `packer.apply.mode` | `observe_only` global default |
| `packer.apply.exclude_message_source` | `true` |
| `packer.apply.max_injected_context_tokens` | `80000` |
| `packer.apply.importance_scoring.enabled` | `false` |
| `packer.apply.importance_scoring.fallback_to_rule_based` | `false` |

Node policy config:

- Default policy: observe-only, risk tier 2, max 1200 injected context tokens,
  max 4 items.
- Router nodes: observe-only, risk tier 0.
- Planner nodes: active, risk tier 2, required rules.
- Agent nodes: active, risk tier 1.
- Reviewer nodes: active, risk tier 3, required rules.
- Output nodes: disabled, risk tier 4.
- Several per-node overrides exist for review docs, exercises, mindmaps, code
  practice, video scripts, video animation, study plans, adaptive practice, and
  recommendations.

Legacy/coexisting config:

- `context_budget.*` still exists separately from `context_engineering.*`.
- `memory.token_budget.*` exists separately for legacy memory context.

Risk note:

- The project has at least three budget vocabularies:
  `context_budget`, `context_engineering`, and `memory.token_budget`.

## 9. Tests and Regression Anchors

Context-related tests exist across these groups:

| Test area | Files |
| --- | --- |
| Kernel schema/budget/tokenizer/itemizer/trace | `tests/test_context_schema.py`, `tests/test_context_budget.py`, `tests/test_context_tokenizer.py`, `tests/test_context_itemizer.py`, `tests/test_context_trace.py`, `tests/test_context_usage.py` |
| Providers and supply | `tests/test_context_provider_contract.py`, `tests/test_context_provider_registry.py`, `tests/test_context_provider_supply.py`, `tests/test_evidence_context_provider.py`, `tests/test_artifact_context_provider.py`, `tests/test_profile_context_provider.py`, `tests/test_rules_context_provider.py` |
| Packing/render/policy | `tests/test_context_packer.py`, `tests/test_context_packer_policy.py`, `tests/test_context_packer_render.py`, `tests/test_context_packing_schema.py`, `tests/test_context_packer_shadow.py`, `tests/test_context_packing_trace.py` |
| Apply/node policy | `tests/test_context_apply_messages.py`, `tests/test_context_apply_node_policy.py`, `tests/test_context_apply_policy.py`, `tests/test_context_apply_quality_budget.py`, `tests/test_context_apply_route_rollout.py`, `tests/test_context_apply_structured_guard.py`, `tests/test_context_apply_trace.py`, `tests/test_context_apply_plain_llm.py` |
| Workspace | `tests/test_context_workspace.py` |
| App/SSE/status | `tests/test_context_usage_sse.py`, `tests/test_app.py`, `tests/test_run_control.py`, `tests/test_sse_lifecycle.py`, `tests/test_state.py` |
| Regression boundaries | `tests/test_no_context_phase0_regression.py`, `tests/test_phase2_no_packer_regression.py`, `tests/test_phase3_no_compaction_regression.py`, `tests/test_phase3b_no_phase4_regression.py`, `tests/test_phase3b2a_no_phase4_regression.py` |

Deletion planning implication:

- Any removal of providers, packing, apply, workspace, or context SSE should
  first choose which test guarantees remain valid and which tests should be
  intentionally rewritten.

## 10. Candidate Decision Areas

These are not deletion instructions. They are the main areas where an explicit
owner decision is needed.

### A. Likely simplification candidates

1. Consolidate or remove legacy `src/context` memory injection after deciding
   whether `MemoryContextProvider` / `ProfileContextProvider` fully replace it.
2. Rename or separate `LearningState.context` from Context Engineering naming,
   because it is retrieval evidence, not injected context.
3. Remove one of the coexisting budget config surfaces if no runtime path still
   needs it.
4. Decide whether structured output should stay observe-only or have all
   structured-context observe/shadow code removed.
5. Decide whether backend-only rich context SSE events should remain if frontend
   only consumes usage/error events.

### B. High-risk areas to keep until replacement exists

1. `task_workspace`: active continuation, evidence summaries, artifact summaries,
   provider inputs, status payloads, and tests depend on it.
2. `context_engineering/packing/orchestrator.py`: plain LLM message preparation
   flows through it.
3. `EvidenceContextProvider` and `ArtifactContextProvider`: they bridge
   workspace/retrieval/resource outputs into context items.
4. `app.py` context-window state updates: many SSE/status tests depend on these
   payloads even if frontend usage is currently limited.
5. Resource nodes' use of `state["context"]`: this is factual grounding for
   generated resources.

### C. Fallback/dead-code risk surfaces to review, not delete blindly

1. `fallback_on_error`, `fallback_used`, `fallback_if_empty_after_drop`,
   `fallback_to_rule_based`, and `config_error_fallback` names remain in context
   apply/policy surfaces while current config sets fallbacks false.
2. `generate_answer` swallows legacy memory-context build errors and continues.
3. `generate_sse` swallows profile-context injection load errors and continues.
4. Several resource graph files have degraded/fallback generation paths adjacent
   to context code; those need separate business approval.
5. Prior reports already flag legacy provider/model/fallback defaults in
   non-context-specific LLM code.

## 11. Governance Review

### Spec-first change review

- Objective: produce a read-only inventory report for context-management
  deletion/rebuild planning.
- Scope: add this report under `docs/reports/`.
- Non-goals: no runtime edits, no deletion, no renames, no config changes, no
  prompt changes.
- Acceptance: report covers fields, functions, config, state, runtime flow,
  tests, and candidate decision areas.

### Dead code/diff risk review

- No code was deleted.
- No helpers were moved.
- Old fallback/dead-code-like surfaces are reported above instead of removed.
- `vulture` was attempted before report creation and was not available in this
  environment.
- Existing uncommitted changes were not reverted or overwritten.

### Architecture boundary review

- No runtime imports were added or changed.
- No package dependency direction changed.
- The report identifies boundary overlap between legacy memory context,
  Context Engineering, graph state, workspace, and app/SSE status.

### Structured output contract review

- No Pydantic schema was changed.
- No structured output alias-normalization was added.
- No business validation was bypassed.
- Structured-output context behavior is documented as observe/shadow-only.

### No-fallback/no-hardcode review

- No fallback was added.
- No silent defaults were added.
- No provider/model/base URL/API key was hardcoded.
- Existing fallback/hardcode risk surfaces are reported as candidates for later
  explicit work.

### Type contract review

- Protected runtime areas were not edited.
- Public signatures were not changed.
- Type checker was not required for this docs-only inventory.

### Security/secret review

- No secrets, auth headers, DB URIs, or trace bodies were added.
- The report does not expose raw sensitive payloads.
- Existing redaction/sanitization surfaces are listed only by name and behavior.

## 12. Suggested Deletion Planning Order

Recommended order for your decision pass:

1. Decide the target context architecture: one canonical context model or
   separate retrieval/workspace/telemetry models with clearer names.
2. Decide whether legacy memory prompt injection should be replaced by Context
   Engineering providers.
3. Decide whether `task_workspace` is part of the new architecture or a separate
   durable workspace feature.
4. Decide whether structured-output context observation should be kept.
5. Decide which SSE/status payloads are product-facing versus internal
   observability.
6. Only then approve narrow deletion batches with tests updated per batch.

