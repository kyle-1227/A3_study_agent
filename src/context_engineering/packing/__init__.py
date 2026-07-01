"""ContextPacker shadow mode public API."""

from src.context_engineering.packing.apply import (
    ContextApplyError,
    ContextApplyResult,
    ContextInjectionPolicy,
    apply_node_enabled,
    build_applied_messages,
    filter_injectable_items,
    get_context_injection_policy,
    render_injected_context,
)
from src.context_engineering.packing.apply_trace import (
    build_context_applied_event,
    build_context_apply_error_event,
    build_context_apply_plan_event,
    emit_context_applied,
    emit_context_apply_error,
    emit_context_apply_plan,
)
from src.context_engineering.packing.packer import pack_context_items
from src.context_engineering.packing.policies import (
    PackingPolicy,
    get_packing_policy,
    node_enabled,
)
from src.context_engineering.packing.render import render_selected_context
from src.context_engineering.packing.schema import (
    ContextPackingError,
    PackedContext,
    PackingDecision,
    PackingReason,
    PackingStrategy,
)
from src.context_engineering.packing.trace import (
    build_context_packed_event,
    build_context_packing_error_event,
    build_context_packing_plan_event,
    emit_context_packed,
    emit_context_packing_error,
    emit_context_packing_plan,
    emit_context_packing_shadow,
)

__all__ = [
    "ContextApplyError",
    "ContextApplyResult",
    "ContextInjectionPolicy",
    "ContextPackingError",
    "PackedContext",
    "PackingDecision",
    "PackingPolicy",
    "PackingReason",
    "PackingStrategy",
    "apply_node_enabled",
    "build_applied_messages",
    "build_context_applied_event",
    "build_context_apply_error_event",
    "build_context_apply_plan_event",
    "build_context_packed_event",
    "build_context_packing_error_event",
    "build_context_packing_plan_event",
    "emit_context_applied",
    "emit_context_apply_error",
    "emit_context_apply_plan",
    "emit_context_packed",
    "emit_context_packing_error",
    "emit_context_packing_plan",
    "emit_context_packing_shadow",
    "filter_injectable_items",
    "get_context_injection_policy",
    "get_packing_policy",
    "node_enabled",
    "pack_context_items",
    "render_injected_context",
    "render_selected_context",
]
