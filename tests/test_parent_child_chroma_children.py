from __future__ import annotations

import hashlib
from pathlib import Path

import chromadb
from chromadb.config import Settings
import pytest

from src.rag.parent_child.bm25_artifact import digest_identifier_set
from src.rag.parent_child.chroma_children import (
    ChromaEmbeddingContractError,
    ChromaInputContractError,
    ChromaStagingPathError,
    ChromaVerificationError,
    iter_child_chroma_metadata_pages,
    write_child_chroma_artifact,
)
from src.rag.parent_child.embedding_batches import EmbeddingBatchExecutionError
from src.rag.parent_child.models import ChildDocument, ChildMetadata
from src.rag.parent_child.provider_clients import ProviderReportedError


class DeterministicEmbedding:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [
            [float(len(text) + coordinate) for coordinate in range(self.dimension)]
            for text in texts
        ]


class WrongDimensionEmbedding:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]


class FailingProviderEmbedding:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        del texts
        raise self.error


class _PagedMetadataCollection:
    def __init__(self, *, page_size: int) -> None:
        self._page_size = page_size
        self.calls: list[tuple[int, int, tuple[object, ...]]] = []

    def get(
        self, *, limit: int | None = None, offset: int | None = None, include: object
    ) -> dict[str, object]:
        if limit is None or offset is None:
            raise AssertionError("child metadata reads require limit and offset")
        if limit > self._page_size:
            raise AssertionError("child metadata read exceeded declared page size")
        if include != ["metadatas"]:
            raise AssertionError("child metadata read must request only metadata")
        identifiers = ("child_a", "child_b", "child_c", "child_d", "child_e")
        selected = identifiers[offset : offset + limit]
        self.calls.append((limit, offset, tuple(include)))
        return {
            "ids": list(selected),
            "metadatas": [{"child_id": identifier} for identifier in selected],
        }


def _child(
    *,
    child_hex: str,
    content: str,
    generation_id: str = "gen-a",
    child_index: int = 0,
) -> ChildDocument:
    content_sha1 = hashlib.sha1(content.encode("utf-8")).hexdigest()
    return ChildDocument(
        schema_version="child_document_v1",
        content=content,
        metadata=ChildMetadata(
            schema_version="child_metadata_v1",
            child_id="child_" + child_hex * 40,
            parent_id="parent_" + "b" * 40,
            doc_id="doc_" + "c" * 40,
            subject="math",
            generation_id=generation_id,
            policy_id="d" * 64,
            child_index=child_index,
            child_start_in_parent=0,
            child_end_in_parent=len(content),
            start_char=child_index * 100,
            end_char=child_index * 100 + len(content),
            child_chars=len(content),
            content_sha1=content_sha1,
            source_file="notes.md",
            source_relpath="math/notes.md",
            source_file_sha1="e" * 40,
            doc_type="notes",
            section_id="section_" + "f" * 40,
            section_title="Limits",
            section_path=("Limits",),
            pagination_kind="logical",
            page_start=1,
            page_end=1,
        ),
    )


def _staging_root(tmp_path: Path, generation_id: str = "gen-a") -> Path:
    path = tmp_path / "indexes" / ".staging" / generation_id
    path.mkdir(parents=True)
    return path


def test_child_metadata_reader_requires_explicit_bounded_pages() -> None:
    collection = _PagedMetadataCollection(page_size=2)

    pages = tuple(
        iter_child_chroma_metadata_pages(
            collection,  # type: ignore[arg-type]
            expected_count=5,
            page_size=2,
        )
    )

    assert tuple(identifier for ids, _ in pages for identifier in ids) == (
        "child_a",
        "child_b",
        "child_c",
        "child_d",
        "child_e",
    )
    assert collection.calls == [
        (2, 0, ("metadatas",)),
        (2, 2, ("metadatas",)),
        (1, 4, ("metadatas",)),
    ]


def test_chroma_writer_persists_and_verifies_exact_children(tmp_path: Path) -> None:
    staging_root = _staging_root(tmp_path)
    children = (
        _child(child_hex="2", content="beta", child_index=1),
        _child(child_hex="1", content="alpha", child_index=0),
    )
    embedding = DeterministicEmbedding(dimension=3)

    artifact = write_child_chroma_artifact(
        children,
        generation_staging_root=staging_root,
        persist_directory=staging_root / "chroma_children",
        generation_id="gen-a",
        collection_name="children-v1",
        distance_metric="cosine",
        expected_dimension=3,
        batch_size=1,
        max_in_flight_batches=1,
        embedding_provider=embedding,
    )

    assert artifact.child_count == 2
    assert artifact.child_id_set_sha256 == digest_identifier_set(
        [child.metadata.child_id for child in children]
    )
    assert artifact.artifact_relative_path == "chroma_children"
    assert embedding.calls == [["alpha"], ["beta"]]
    with chromadb.PersistentClient(
        path=str(staging_root / "chroma_children"),
        settings=Settings(anonymized_telemetry=False),
    ) as client:
        collection = client.get_collection("children-v1", embedding_function=None)
        result = collection.get(include=["documents", "metadatas", "embeddings"])
        assert set(result["ids"]) == {child.metadata.child_id for child in children}
        assert collection.metadata == {
            "schema_version": "chroma_children_v1",
            "generation_id": "gen-a",
            "expected_dimension": 3,
            "hnsw:space": "cosine",
        }


@pytest.mark.parametrize(
    ("children", "error_match"),
    [
        (
            (
                _child(child_hex="1", content="alpha"),
                _child(child_hex="1", content="alpha"),
            ),
            "duplicate child IDs",
        ),
        (
            (_child(child_hex="1", content="alpha", generation_id="gen-b"),),
            "generation_id differs",
        ),
    ],
)
def test_chroma_writer_rejects_invalid_child_set_before_creation(
    tmp_path: Path,
    children: tuple[ChildDocument, ...],
    error_match: str,
) -> None:
    staging_root = _staging_root(tmp_path)
    persist_directory = staging_root / "chroma_children"

    with pytest.raises(ChromaInputContractError, match=error_match):
        write_child_chroma_artifact(
            children,
            generation_staging_root=staging_root,
            persist_directory=persist_directory,
            generation_id="gen-a",
            collection_name="children-v1",
            distance_metric="cosine",
            expected_dimension=3,
            batch_size=8,
            max_in_flight_batches=1,
            embedding_provider=DeterministicEmbedding(3),
        )
    assert not persist_directory.exists()


def test_chroma_writer_rejects_embedding_dimension_mismatch(tmp_path: Path) -> None:
    staging_root = _staging_root(tmp_path)

    with pytest.raises(ChromaEmbeddingContractError, match="dimension mismatch"):
        write_child_chroma_artifact(
            (_child(child_hex="1", content="alpha"),),
            generation_staging_root=staging_root,
            persist_directory=staging_root / "chroma_children",
            generation_id="gen-a",
            collection_name="children-v1",
            distance_metric="cosine",
            expected_dimension=3,
            batch_size=8,
            max_in_flight_batches=1,
            embedding_provider=WrongDimensionEmbedding(),
        )


def test_chroma_writer_preserves_only_sanitized_embedding_failure(
    tmp_path: Path,
) -> None:
    staging_root = _staging_root(tmp_path)
    secret = "sk-secret-do-not-report"
    provider_error = ProviderReportedError(
        code=429,
        retryable=True,
        attempts_exhausted=True,
    )
    provider_error.response_body = {"authorization": secret}
    provider_error.request_url = f"https://provider.invalid/?key={secret}"

    with pytest.raises(ChromaEmbeddingContractError) as captured:
        write_child_chroma_artifact(
            (_child(child_hex="1", content="alpha"),),
            generation_staging_root=staging_root,
            persist_directory=staging_root / "chroma_children",
            generation_id="gen-a",
            collection_name="children-v1",
            distance_metric="cosine",
            expected_dimension=3,
            batch_size=8,
            max_in_flight_batches=1,
            embedding_provider=FailingProviderEmbedding(provider_error),
        )

    batch_failure = captured.value.__cause__
    assert isinstance(batch_failure, EmbeddingBatchExecutionError)
    assert batch_failure.batch_start == 0
    assert batch_failure.batch_size == 1
    assert batch_failure.cause_type == "ProviderReportedError"
    assert batch_failure.provider_code == 429
    assert batch_failure.retryable is True
    assert batch_failure.attempts_exhausted is True
    assert batch_failure.__cause__ is None
    assert batch_failure.__context__ is None
    assert secret not in str(captured.value)
    assert secret not in repr(vars(batch_failure))


def test_chroma_writer_is_staging_only_and_immutable(tmp_path: Path) -> None:
    final_root = tmp_path / "indexes" / "gen-a"
    final_root.mkdir(parents=True)
    child = _child(child_hex="1", content="alpha")

    with pytest.raises(ChromaStagingPathError, match=r"\.staging"):
        write_child_chroma_artifact(
            (child,),
            generation_staging_root=final_root,
            persist_directory=final_root / "chroma_children",
            generation_id="gen-a",
            collection_name="children-v1",
            distance_metric="cosine",
            expected_dimension=3,
            batch_size=8,
            max_in_flight_batches=1,
            embedding_provider=DeterministicEmbedding(3),
        )

    staging_root = _staging_root(tmp_path)
    persist_directory = staging_root / "chroma_children"
    persist_directory.mkdir()
    with pytest.raises(ChromaStagingPathError, match="must not already exist"):
        write_child_chroma_artifact(
            (child,),
            generation_staging_root=staging_root,
            persist_directory=persist_directory,
            generation_id="gen-a",
            collection_name="children-v1",
            distance_metric="cosine",
            expected_dimension=3,
            batch_size=8,
            max_in_flight_batches=1,
            embedding_provider=DeterministicEmbedding(3),
        )


def test_chroma_writer_detects_document_roundtrip_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging_root = _staging_root(tmp_path)

    class CorruptCollection:
        name = "children-v1"
        metadata = {
            "schema_version": "chroma_children_v1",
            "generation_id": "gen-a",
            "expected_dimension": 3,
            "hnsw:space": "cosine",
        }
        configuration = {
            "hnsw": {"space": "cosine"},
        }

        def add(self, **_values: object) -> None:
            return None

        def count(self) -> int:
            return 1

        def get(self, **_values: object) -> dict[str, object]:
            child_id = "child_" + "1" * 40
            expected = _child(child_hex="1", content="alpha")
            return {
                "ids": [child_id],
                "documents": ["corrupt"],
                "metadatas": [expected.metadata.to_chroma_metadata()],
                "embeddings": [[1.0, 2.0, 3.0]],
            }

    class CorruptClient:
        def __init__(self, path: str, settings: Settings) -> None:
            del settings
            Path(path).mkdir(parents=True)
            self.collection = CorruptCollection()
            self.created = False

        def __enter__(self) -> CorruptClient:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def list_collections(self) -> list[CorruptCollection]:
            return [self.collection] if self.created else []

        def create_collection(self, **_values: object) -> CorruptCollection:
            self.created = True
            return self.collection

    monkeypatch.setattr(
        "src.rag.parent_child.chroma_children.chromadb.PersistentClient",
        CorruptClient,
    )
    with pytest.raises(ChromaVerificationError, match="document mismatch"):
        write_child_chroma_artifact(
            (_child(child_hex="1", content="alpha"),),
            generation_staging_root=staging_root,
            persist_directory=staging_root / "chroma_children",
            generation_id="gen-a",
            collection_name="children-v1",
            distance_metric="cosine",
            expected_dimension=3,
            batch_size=8,
            max_in_flight_batches=1,
            embedding_provider=DeterministicEmbedding(3),
        )
