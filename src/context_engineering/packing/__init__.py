"""ContextPacker shadow mode public API."""

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
    "ContextPackingError",
    "PackedContext",
    "PackingDecision",
    "PackingPolicy",
    "PackingReason",
    "PackingStrategy",
    "build_context_packed_event",
    "build_context_packing_error_event",
    "build_context_packing_plan_event",
    "emit_context_packed",
    "emit_context_packing_error",
    "emit_context_packing_plan",
    "emit_context_packing_shadow",
    "get_packing_policy",
    "node_enabled",
    "pack_context_items",
    "render_selected_context",
]
