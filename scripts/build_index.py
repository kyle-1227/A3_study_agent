"""Offline script to build the ChromaDB index from documents in data/.

Note: This script is for sprint programming purpose.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv(project_root / ".env")

from src.rag.loader import load_documents
from src.rag.indexer import build_index
from src.rag.section_splitter import SectionAwareSplitter

DATA_DIR = project_root / "data"

SUBJECT_DIRS = {
    "math": DATA_DIR / "math",
    "chinese": DATA_DIR / "chinese",
}

# Subjects whose documents are exam papers and benefit from section-aware splitting.
EXAM_PAPER_SUBJECTS = {"math", "chinese"}

_section_splitter = SectionAwareSplitter()


def main() -> None:
    all_docs = []
    for subject, directory in SUBJECT_DIRS.items():
        if not directory.is_dir() or not any(directory.iterdir()):
            print(f"[SKIP] {directory} — empty or missing")
            continue
        splitter = _section_splitter if subject in EXAM_PAPER_SUBJECTS else None
        doc_type = "exam_paper" if subject in EXAM_PAPER_SUBJECTS else "exam"
        docs = load_documents(directory, subject=subject, doc_type=doc_type, splitter=splitter)
        print(f"[OK]   {subject}: loaded {len(docs)} chunks from {directory}")
        all_docs.extend(docs)

    if not all_docs:
        print("\nNo documents found. Place PDF/MD/TXT files in data/math/ or data/chinese/ first.")
        return

    print(f"\nBuilding index with {len(all_docs)} total chunks ...")
    vectorstore = build_index(all_docs)
    count = vectorstore._collection.count()
    print(f"Index built successfully — {count} vectors in ChromaDB.")


if __name__ == "__main__":
    main()
