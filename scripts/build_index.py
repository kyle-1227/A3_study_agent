"""Build ChromaDB index from university course materials in data/.

Business scenario:
- University / pre-university personalized learning resource generation
- Course materials: Python, Machine Learning, Big Data, Higher Mathematics, Computer Basics
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv(project_root / ".env")

try:
    from src.rag.loader import load_documents
    from src.rag.indexer import build_index
except ModuleNotFoundError as e:
    missing = str(e).split("'")[1] if "'" in str(e) else str(e)
    print(f"Missing Python dependency: {missing}")
    print("Run:")
    print(f"  {sys.executable} -m pip install -r requirements.txt")
    print("Then retry:")
    print("  python scripts/build_index.py")
    raise SystemExit(1)

DATA_DIR = project_root / "data"

COURSE_DIRS = {
    "big_data": DATA_DIR / "big_data",
    "computer": DATA_DIR / "computer",
    "machine_learning": DATA_DIR / "machine_learning",
    "math": DATA_DIR / "math",
    "python": DATA_DIR / "python",
}

COURSE_DOC_TYPE = "course_material"


def main() -> None:
    # Startup diagnostics
    print("=== build_index startup ===")
    print(f"Project root      : {project_root}")
    print(f"Data dir          : {DATA_DIR}")
    print(f"Chroma persist dir: {os.getenv('CHROMA_PERSIST_DIR', 'NOT SET')}")
    print(f"Course dirs       : {list(COURSE_DIRS.keys())}")
    print()

    all_docs = []

    for subject, directory in COURSE_DIRS.items():
        if not directory.is_dir() or not any(directory.iterdir()):
            print(f"[SKIP] {subject}: {directory} — empty or missing")
            continue

        docs = load_documents(
            directory,
            subject=subject,
            doc_type=COURSE_DOC_TYPE,
            splitter=None,
        )

        # Count unique source files from document metadata
        source_files = {doc.metadata.get("source_file", "") for doc in docs}
        file_count = len(source_files - {""})

        print(f"[OK]   {subject} — {file_count} files, {len(docs)} chunks")
        all_docs.extend(docs)

    if not all_docs:
        print(
            "\nNo course materials found. "
            "Place PDF/MD/TXT files in data/big_data, data/python, "
            "data/machine_learning, data/math, or data/computer."
        )
        return

    print(f"\nBuilding university course RAG index with {len(all_docs)} total chunks ...")
    vectorstore = build_index(all_docs)
    count = vectorstore._collection.count()
    print(f"Index built successfully — {count} course-material vectors in ChromaDB.")


if __name__ == "__main__":
    main()