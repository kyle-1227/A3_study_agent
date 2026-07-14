"""Build one immutable parent-child RAG generation without activating it."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
import sys

PROJECT_ROOT_FROM_SCRIPT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FROM_SCRIPT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FROM_SCRIPT))

from src.config.rag_index_config import (  # noqa: E402
    load_rag_index_config,
    resolve_rag_index_config_paths,
)
from src.rag.parent_child.builder import (  # noqa: E402
    compute_embedding_fingerprint,
    GenerationBuildRequest,
    GenerationBuilder,
)
from src.rag.parent_child.embedding_cache import (  # noqa: E402
    CachingDocumentEmbeddingProvider,
    EMBEDDING_CACHE_SCHEMA_VERSION,
    SqliteEmbeddingCache,
)
from src.rag.parent_child.project_paths import require_project_file  # noqa: E402
from src.rag.parent_child.provider_clients import StrictEmbeddingClient  # noqa: E402
from src.rag.parent_child.registry import (  # noqa: E402
    GenerationRegistry,
    create_generation_registry,
)
from src.rag.parent_child.tokenizer import ConfiguredJiebaTokenizer  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--pipeline", choices=("parent-child",), required=True)
    parser.add_argument("--index-config", type=Path, required=True)
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument(
        "--registry-mode",
        choices=("create", "existing"),
        required=True,
    )
    cache = parser.add_mutually_exclusive_group(required=True)
    cache.add_argument("--embedding-cache", type=Path)
    cache.add_argument("--no-embedding-cache", action="store_true")
    parser.add_argument(
        "--embedding-cache-busy-timeout-seconds",
        type=float,
        required=True,
    )
    return parser


def _contained_file(project_root: Path, value: Path) -> Path:
    candidate = value if value.is_absolute() else project_root / value
    if candidate.is_symlink():
        raise ValueError("index config must not be a symlink")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(project_root) or not resolved.is_file():
        raise ValueError("index config must be a file inside project_root")
    return resolved


def run_build(
    *,
    project_root: Path,
    index_config_path: Path,
    generation_id: str,
    code_revision: str,
    registry_mode: str,
    embedding_cache_path: Path | None,
    embedding_cache_busy_timeout_seconds: float,
) -> str:
    """Run one explicit build and return its manifest digest."""

    root = project_root.resolve(strict=True)
    config_path = _contained_file(root, index_config_path)
    config = resolve_rag_index_config_paths(
        load_rag_index_config(config_path),
        project_root=root,
    )
    registry_path = config.storage.resolved_registry_path()
    if registry_mode == "create":
        create_generation_registry(
            registry_path,
            schema_version=config.storage.registry_schema_version,
            busy_timeout_seconds=config.storage.registry_busy_timeout_seconds,
        )
    elif registry_mode == "existing":
        if not registry_path.is_file():
            raise FileNotFoundError("configured generation registry does not exist")
    else:
        raise ValueError("registry_mode must be 'create' or 'existing'")

    if embedding_cache_busy_timeout_seconds <= 0:
        raise ValueError("embedding cache busy timeout must be positive")
    embedding = StrictEmbeddingClient.production(config=config.embedding)
    cache: SqliteEmbeddingCache | None = None
    try:
        embedding_provider = embedding
        if embedding_cache_path is not None:
            contained_cache = require_project_file(root, embedding_cache_path)
            cache = SqliteEmbeddingCache.open(
                project_root=root,
                cache_path=contained_cache,
                expected_schema_version=EMBEDDING_CACHE_SCHEMA_VERSION,
                expected_embedding_fingerprint=compute_embedding_fingerprint(config),
                expected_dimension=config.embedding.expected_dimension,
                busy_timeout_seconds=embedding_cache_busy_timeout_seconds,
            )
            cache.verify_integrity()
            embedding_provider = CachingDocumentEmbeddingProvider(
                cache=cache,
                upstream=embedding,
            )
        tokenizer = ConfiguredJiebaTokenizer(config=config.bm25)
        with GenerationRegistry.open(
            registry_path,
            index_root=config.storage.index_root,
            expected_schema_version=config.storage.registry_schema_version,
            marker_schema_version=config.storage.owner_marker_schema_version,
            busy_timeout_seconds=config.storage.registry_busy_timeout_seconds,
        ) as registry:
            result = GenerationBuilder(
                config=config,
                registry=registry,
                embedding_provider=embedding_provider,
                bm25_tokenizer=tokenizer,
                build_clock=lambda: datetime.now(UTC),
            ).build(
                GenerationBuildRequest(
                    schema_version="generation_build_request_v1",
                    generation_id=generation_id,
                    code_revision=code_revision,
                )
            )
    finally:
        try:
            if cache is not None:
                cache.close()
        finally:
            embedding.close()
    return result.sealed.manifest_sha256


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest_sha256 = run_build(
        project_root=args.project_root,
        index_config_path=args.index_config,
        generation_id=args.generation_id,
        code_revision=args.code_revision,
        registry_mode=args.registry_mode,
        embedding_cache_path=args.embedding_cache,
        embedding_cache_busy_timeout_seconds=(
            args.embedding_cache_busy_timeout_seconds
        ),
    )
    print(
        "Parent-child generation is READY and not activated: "
        f"generation_id={args.generation_id}, manifest_sha256={manifest_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
