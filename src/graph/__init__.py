from src.graph.state import LearningState

__all__ = [
    "LearningState",
    "build_graph",
    "get_compiled_graph",
]


def build_graph(*args, **kwargs):
    from src.graph.builder import build_graph as _build_graph

    return _build_graph(*args, **kwargs)


def get_compiled_graph(*args, **kwargs):
    from src.graph.builder import get_compiled_graph as _get_compiled_graph

    return _get_compiled_graph(*args, **kwargs)
