from pathlib import Path
import sys
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
import statistics

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv(PROJECT_ROOT / ".env")

from src.rag.indexer import load_index

REPORT_PATH = PROJECT_ROOT / "reports" / "inspect_chunks.txt"


def inspect_chunks():
    vs = load_index()
    collection = vs._collection

    data = collection.get(include=["documents", "metadatas"])
    docs = data.get("documents") or []
    metas = data.get("metadatas") or []

    print(f"Total chunks: {len(docs)}")

    lengths = [len(d or "") for d in docs]
    if lengths:
        print("\nLength stats:")
        print(f"min: {min(lengths)}")
        print(f"max: {max(lengths)}")
        print(f"avg: {statistics.mean(lengths):.1f}")
        print(f"median: {statistics.median(lengths):.1f}")

    print("\nSubject distribution:")
    for subject, count in Counter((m or {}).get("subject", "unknown") for m in metas).most_common():
        print(f"{count:5d}  {subject}")

    print("\nDoc type distribution:")
    for doc_type, count in Counter((m or {}).get("doc_type", "unknown") for m in metas).most_common():
        print(f"{count:5d}  {doc_type}")

    print("\nSource file distribution:")
    for source, count in Counter((m or {}).get("source_file", "unknown") for m in metas).most_common():
        print(f"{count:5d}  {source}")

    print("\nSample chunks:")
    for i, (doc, meta) in enumerate(list(zip(docs, metas))[:12], 1):
        print("=" * 100)
        print(f"Sample {i}")
        print("metadata:", meta)
        preview = (doc or "").replace("\n", " ")
        print("content:", preview[:1000])


if __name__ == "__main__":
    buffer = StringIO()
    with redirect_stdout(buffer):
        inspect_chunks()

    output = buffer.getvalue()
    print(output, end="")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(output, encoding="utf-8")
    print(f"\n[OK] Full inspect output written to {REPORT_PATH}")
