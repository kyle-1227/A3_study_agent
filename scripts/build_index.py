"""Build an explicitly selected legacy flat-baseline Chroma index.

Business scenario:
- University / pre-university personalized learning resource generation
- Course materials: Python, Machine Learning, Big Data, Higher Mathematics, Computer Basics
"""

import argparse
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.rag.index_manifest import (
        build_manifest_from_documents,
        write_build_manifest,
    )
    from src.rag.indexer import (
        COLLECTION_NAME,
        build_index,
    )
    from src.rag.loader import load_documents
except ModuleNotFoundError as e:
    missing = str(e).split("'")[1] if "'" in str(e) else str(e)
    print(f"Missing Python dependency: {missing}")
    print("Run:")
    print(f"  {sys.executable} -m pip install -r requirements.txt")
    print("Then retry:")
    print("  python scripts/build_index.py")
    raise SystemExit(1)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=("flat-baseline",), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--persist-dir", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--subject", action="append", required=True)
    parser.add_argument("--doc-type", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--embedding-base-url", required=True)
    parser.add_argument("--embedding-api-key-env", required=True)
    parser.add_argument("--embedding-timeout-seconds", type=float, required=True)
    parser.add_argument("--embedding-document-input-type", required=True)
    parser.add_argument("--embedding-query-input-type", required=True)
    parser.add_argument("--index-batch-size", type=int, required=True)
    parser.add_argument("--index-max-retries", type=int, required=True)
    return parser


def _contained_path(value: Path, *, must_exist: bool) -> Path:
    candidate = value if value.is_absolute() else project_root / value
    if candidate.is_symlink():
        raise ValueError("baseline paths must not be symlinks")
    resolved = candidate.resolve(strict=must_exist)
    if not resolved.is_relative_to(project_root):
        raise ValueError("baseline paths must remain inside project_root")
    return resolved


def _validate_subjects(values: list[str]) -> tuple[str, ...]:
    subjects = tuple(values)
    if len(subjects) != len(set(subjects)):
        raise ValueError("--subject values must be unique")
    for subject in subjects:
        if (
            not subject
            or subject != subject.strip().casefold()
            or subject.startswith("_")
            or subject.endswith("_")
            or "__" in subject
            or not all(character.isalnum() or character == "_" for character in subject)
        ):
            raise ValueError("--subject values must be normalized identifiers")
    return subjects


def _configure_legacy_embedding(args: argparse.Namespace) -> None:
    if not args.embedding_api_key_env.isidentifier():
        raise ValueError("embedding API-key environment name is invalid")
    secret = os.environ.get(args.embedding_api_key_env)
    if secret is None or not secret.strip():
        raise RuntimeError(
            "required embedding API-key environment variable is missing: "
            + args.embedding_api_key_env
        )
    if not args.embedding_base_url.startswith(("http://", "https://")):
        raise ValueError("embedding-base-url must be an absolute HTTP(S) URL")
    if args.embedding_timeout_seconds <= 0:
        raise ValueError("embedding timeout must be positive")
    if args.index_batch_size <= 0 or args.index_max_retries < 0:
        raise ValueError("index batch/retry values are invalid")
    if not (
        args.embedding_document_input_type.strip()
        and args.embedding_query_input_type.strip()
    ):
        raise ValueError("embedding input types must be nonblank")
    os.environ["EMBEDDING_MODEL"] = args.embedding_model
    os.environ["EMBEDDING_API_KEY_ENV"] = args.embedding_api_key_env
    os.environ["EMBEDDING_BASE_URL"] = args.embedding_base_url
    os.environ["EMBEDDING_TIMEOUT"] = str(args.embedding_timeout_seconds)
    os.environ["EMBEDDING_DOCUMENT_INPUT_TYPE"] = args.embedding_document_input_type
    os.environ["EMBEDDING_QUERY_INPUT_TYPE"] = args.embedding_query_input_type
    os.environ["INDEX_ADD_BATCH_SIZE"] = str(args.index_batch_size)
    os.environ["INDEX_MAX_RETRIES"] = str(args.index_max_retries)


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    data_dir = _contained_path(args.data_root, must_exist=True)
    persist_dir = _contained_path(args.persist_dir, must_exist=False)
    manifest_path = _contained_path(args.manifest_output, must_exist=False)
    subjects = _validate_subjects(args.subject)
    if not data_dir.is_dir():
        raise ValueError("data-root must be a directory")
    if persist_dir.exists():
        raise FileExistsError("flat baseline persist-dir must be new and isolated")
    if not args.doc_type.strip() or not args.embedding_model.strip():
        raise ValueError("doc-type and embedding-model must be nonblank")
    _configure_legacy_embedding(args)

    # Startup diagnostics
    print("=== build_index startup ===")
    print(f"Project root      : {project_root}")
    print(f"Data dir          : {data_dir}")
    print(f"Chroma persist dir: {persist_dir}")
    print(f"Subjects          : {list(subjects)}")
    print()

    all_docs = []

    for subject in subjects:
        directory = data_dir / subject
        if not directory.is_dir() or not any(directory.iterdir()):
            print(f"[SKIP] {subject}: {directory} — empty or missing")
            raise FileNotFoundError(
                f"configured subject is empty or missing: {subject}"
            )

        docs = load_documents(
            directory,
            subject=subject,
            doc_type=args.doc_type,
            splitter=None,
        )

        # Count unique source files from document metadata
        source_files = {doc.metadata.get("source_file", "") for doc in docs}
        file_count = len(source_files - {""})

        print(f"[OK]   {subject} — {file_count} files, {len(docs)} chunks")
        all_docs.extend(docs)

    if not all_docs:
        raise RuntimeError("configured subjects produced no flat baseline documents")

    print(
        f"\nBuilding university course RAG index with {len(all_docs)} total chunks ..."
    )
    vectorstore = build_index(
        all_docs,
        persist_directory=str(persist_dir),
        embedding_model=args.embedding_model,
    )
    count = vectorstore._collection.count()
    print(f"Index built successfully — {count} course-material vectors in ChromaDB.")
    manifest = build_manifest_from_documents(
        all_docs,
        collection_name=COLLECTION_NAME,
        chroma_persist_dir=str(persist_dir),
        embedding_model=args.embedding_model,
    )
    write_build_manifest(manifest, manifest_path)
    print(f"Build manifest saved: {manifest_path}")


if __name__ == "__main__":
    main()
