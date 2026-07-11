from src.graph.state import LearningState

__all__ = [
    "LearningState",
    "build_graph",
    "build_parent_child_graph",
    "get_compiled_graph",
    "get_compiled_parent_child_graph",
]


def build_graph(*args, **kwargs):
    from src.graph.builder import build_graph as _build_graph

    return _build_graph(*args, **kwargs)


def get_compiled_graph(*args, **kwargs):
    from src.graph.builder import get_compiled_graph as _get_compiled_graph

    return _get_compiled_graph(*args, **kwargs)


def build_parent_child_graph(*args, **kwargs):
    from src.graph.builder import build_parent_child_graph as _build_parent_child_graph

    return _build_parent_child_graph(*args, **kwargs)


def get_compiled_parent_child_graph(*args, **kwargs):
    from src.graph.builder import (
        get_compiled_parent_child_graph as _get_compiled_parent_child_graph,
    )

    return _get_compiled_parent_child_graph(*args, **kwargs)
