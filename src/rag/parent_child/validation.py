"""Explicit business-invariant validation for parent-child contracts."""

from __future__ import annotations

from collections import defaultdict

from src.rag.parent_child.exceptions import ParentChildInvariantError
from src.rag.parent_child.ids import (
    make_child_id,
    make_parent_id,
    make_policy_fingerprint,
    sha1_content,
)
from src.rag.parent_child.loader import page_range_for_span
from src.rag.parent_child.models import (
    ChildDocument,
    CleanedSourceDocument,
    ParentChildBundle,
    ParentRecord,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ParentChildInvariantError(message)


def validate_parent_record(
    parent: ParentRecord,
    source: CleanedSourceDocument,
    policy_id: str,
) -> None:
    """Prove a parent is an exact source slice with deterministic identity."""

    _require(parent.doc_id == source.doc_id, "Parent doc_id mismatch")
    _require(parent.subject == source.subject, "Parent subject mismatch")
    _require(parent.policy_id == policy_id, "Parent policy_id mismatch")
    _require(parent.source_file == source.source_file, "Parent source_file mismatch")
    _require(
        parent.source_relpath == source.source_relpath,
        "Parent source_relpath mismatch",
    )
    _require(
        parent.source_file_sha1 == source.source_file_sha1,
        "Parent source_file_sha1 mismatch",
    )
    _require(parent.doc_type == source.doc_type, "Parent doc_type mismatch")
    _require(
        parent.extraction_method == source.extraction_method,
        "Parent extraction_method mismatch",
    )
    _require(
        parent.cleaning_policy_id == source.cleaning_policy_id,
        "Parent cleaning_policy_id mismatch",
    )
    _require(
        parent.pagination_kind == source.pagination_kind,
        "Parent pagination_kind mismatch",
    )
    _require(parent.end_char <= len(source.content), "Parent span exceeds source")
    _require(
        parent.content == source.content[parent.start_char : parent.end_char],
        "Parent content is not the exact source slice",
    )
    expected_hash = sha1_content(parent.content)
    _require(parent.content_sha1 == expected_hash, "Parent exact content hash mismatch")
    _require(
        parent.parent_id
        == make_parent_id(
            doc_id=parent.doc_id,
            policy_id=policy_id,
            parent_index=parent.parent_index,
            exact_parent_content_sha1=expected_hash,
        ),
        "Parent deterministic ID mismatch",
    )
    _require(
        (parent.page_start, parent.page_end)
        == page_range_for_span(
            source, start_char=parent.start_char, end_char=parent.end_char
        ),
        "Parent page range mismatch",
    )


def validate_child_document(
    child: ChildDocument,
    parent: ParentRecord,
    source: CleanedSourceDocument,
) -> None:
    """Prove a child is an exact in-parent slice and cannot cross its parent."""

    metadata = child.metadata
    _require(metadata.parent_id == parent.parent_id, "Child parent_id mismatch")
    _require(metadata.doc_id == parent.doc_id, "Child doc_id mismatch")
    _require(metadata.subject == parent.subject, "Child subject mismatch")
    _require(
        metadata.generation_id == parent.generation_id,
        "Child generation_id mismatch",
    )
    _require(metadata.policy_id == parent.policy_id, "Child policy_id mismatch")
    _require(
        metadata.child_end_in_parent <= parent.parent_chars,
        "Child span crosses its parent",
    )
    _require(
        metadata.start_char == parent.start_char + metadata.child_start_in_parent,
        "Child absolute start does not match parent-relative start",
    )
    _require(
        metadata.end_char == parent.start_char + metadata.child_end_in_parent,
        "Child absolute end does not match parent-relative end",
    )
    _require(
        child.content
        == parent.content[
            metadata.child_start_in_parent : metadata.child_end_in_parent
        ],
        "Child content is not the exact parent slice",
    )
    _require(
        child.content == source.content[metadata.start_char : metadata.end_char],
        "Child content is not the exact source slice",
    )
    expected_hash = sha1_content(child.content)
    _require(
        metadata.content_sha1 == expected_hash, "Child exact content hash mismatch"
    )
    _require(
        metadata.child_id
        == make_child_id(
            parent_id=parent.parent_id,
            child_index=metadata.child_index,
            exact_child_content_sha1=expected_hash,
        ),
        "Child deterministic ID mismatch",
    )
    _require(
        (metadata.page_start, metadata.page_end)
        == page_range_for_span(
            source, start_char=metadata.start_char, end_char=metadata.end_char
        ),
        "Child page range mismatch",
    )
    matching_fields = (
        "source_file",
        "source_relpath",
        "source_file_sha1",
        "doc_type",
        "section_id",
        "section_title",
        "section_path",
        "pagination_kind",
    )
    for field_name in matching_fields:
        _require(
            getattr(metadata, field_name) == getattr(parent, field_name),
            f"Child {field_name} mismatch",
        )


def validate_parent_child_bundle(bundle: ParentChildBundle) -> None:
    """Validate identity, coverage, ordering, and every cross-record invariant."""

    expected_policy_id = make_policy_fingerprint(bundle.policy)
    _require(
        bundle.policy_id == expected_policy_id, "Bundle policy fingerprint mismatch"
    )
    _require(
        bundle.source.loader_policy_id == bundle.policy.loader_policy_id,
        "Bundle loader policy mismatch",
    )
    _require(
        bundle.source.cleaning_policy_id == bundle.policy.cleaning_policy_id,
        "Bundle cleaning policy mismatch",
    )

    parent_indices = tuple(parent.parent_index for parent in bundle.parents)
    _require(
        parent_indices == tuple(range(len(bundle.parents))),
        "Parent indices must be contiguous and ordered",
    )
    _require(bundle.parents[0].start_char == 0, "Parents must start at source offset 0")
    previous: ParentRecord | None = None
    parent_by_id: dict[str, ParentRecord] = {}
    for parent in bundle.parents:
        _require(
            parent.generation_id == bundle.generation_id,
            "Parent generation_id mismatch",
        )
        _require(
            parent.parent_chars <= bundle.policy.parent_hard_max,
            "Parent exceeds parent_hard_max",
        )
        if previous is not None:
            _require(
                parent.start_char >= previous.start_char,
                "Parent starts must be ordered",
            )
            _require(
                parent.start_char <= previous.end_char,
                "Parent spans must not leave coverage gaps",
            )
        validate_parent_record(parent, bundle.source, bundle.policy_id)
        _require(parent.parent_id not in parent_by_id, "Duplicate parent_id")
        parent_by_id[parent.parent_id] = parent
        previous = parent
    _require(
        bundle.parents[-1].end_char == len(bundle.source.content),
        "Parents must cover the source through its final offset",
    )

    children_by_parent: dict[str, list[ChildDocument]] = defaultdict(list)
    seen_child_ids: set[str] = set()
    for child in bundle.children:
        child_parent = parent_by_id.get(child.metadata.parent_id)
        _require(child_parent is not None, "Child references an unknown parent_id")
        if child_parent is None:
            raise ParentChildInvariantError("Unreachable unknown parent branch")
        _require(
            child.metadata.child_id not in seen_child_ids,
            "Duplicate child_id",
        )
        seen_child_ids.add(child.metadata.child_id)
        _require(
            child.metadata.child_chars <= bundle.policy.child_hard_max,
            "Child exceeds child_hard_max",
        )
        validate_child_document(child, child_parent, bundle.source)
        children_by_parent[child_parent.parent_id].append(child)

    for parent in bundle.parents:
        children = children_by_parent.get(parent.parent_id, [])
        _require(bool(children), "Every parent must have at least one child")
        child_indices = tuple(child.metadata.child_index for child in children)
        _require(
            child_indices == tuple(range(len(children))),
            "Child indices must be contiguous and ordered within each parent",
        )
        _require(
            children[0].metadata.child_start_in_parent == 0,
            "Children must start at parent offset 0",
        )
        previous_end = 0
        previous_start = -1
        for child in children:
            metadata = child.metadata
            _require(
                metadata.child_start_in_parent >= previous_start,
                "Child starts must be ordered within a parent",
            )
            _require(
                metadata.child_start_in_parent <= previous_end,
                "Children must not leave parent coverage gaps",
            )
            previous_start = metadata.child_start_in_parent
            previous_end = metadata.child_end_in_parent
        _require(
            children[-1].metadata.child_end_in_parent == parent.parent_chars,
            "Children must cover the parent through its final offset",
        )
