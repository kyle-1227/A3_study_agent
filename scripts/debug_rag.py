from __future__ import annotations

from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any
import sys

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
from src.rag.retriever import retrieve

REPORT_PATH = PROJECT_ROOT / "reports" / "debug_rag_bilingual.txt"
TOP_K = 5


TEST_CASES = [
    {
        "subject": "python",
        "label": "Python 中文查询：函数",
        "query": "Python 函数 参数 返回值 作用域",
        "expected_terms": ["function", "parameter", "argument", "return", "scope", "def"],
    },
    {
        "subject": "python",
        "label": "Python 英文查询：function",
        "query": "Python function parameter argument return value scope local variable global variable def",
        "expected_terms": ["function", "parameter", "argument", "return", "scope", "def"],
    },
    {
        "subject": "python",
        "label": "Python 中英混合查询：函数",
        "query": "Python 函数 function 参数 parameter argument 返回值 return value 作用域 scope def local variable global variable",
        "expected_terms": ["function", "parameter", "argument", "return", "scope", "def"],
    },
    {
        "subject": "python",
        "label": "Python 中英混合查询：列表字典循环",
        "query": "Python 列表 list 字典 dictionary dict 循环 loop for while 练习题 exercise practice problem",
        "expected_terms": ["list", "dictionary", "dict", "loop", "for", "while", "exercise"],
    },
    {
        "subject": "machine_learning",
        "label": "机器学习中文基准：过拟合正则化",
        "query": "机器学习 监督学习 过拟合 正则化",
        "expected_terms": ["过拟合", "正则化", "监督学习", "泛化"],
    },
    {
        "subject": "big_data",
        "label": "大数据中文基准：Hadoop",
        "query": "大数据导论 Hadoop HDFS MapReduce",
        "expected_terms": ["Hadoop", "HDFS", "MapReduce", "Hive"],
    },
    {
        "subject": "math",
        "label": "高等数学中文基准：极限导数",
        "query": "高等数学 极限 导数 连续",
        "expected_terms": ["极限", "导数", "连续"],
    },
]


def _subject_of(doc: dict[str, Any]) -> str:
    return ((doc.get("metadata") or {}).get("subject")) or "unknown"


def _source_of(doc: dict[str, Any]) -> str:
    return doc.get("source") or ((doc.get("metadata") or {}).get("source_file")) or "unknown"


def _content_of(doc: dict[str, Any]) -> str:
    return str(doc.get("content") or "")


def calc_subject_match_rate(docs: list[dict[str, Any]], expected_subject: str) -> float:
    if not docs:
        return 0.0
    matched = sum(1 for doc in docs if _subject_of(doc) == expected_subject)
    return matched / len(docs)


def calc_term_hits(docs: list[dict[str, Any]], expected_terms: list[str]) -> dict[str, int]:
    joined = "\n".join(_content_of(doc).lower() for doc in docs)
    return {term: int(term.lower() in joined) for term in expected_terms}


def print_summary(docs: list[dict[str, Any]], subject: str, expected_terms: list[str]) -> None:
    print(f"returned_docs: {len(docs)}")
    print(f"subject_match@{len(docs)}: {calc_subject_match_rate(docs, subject):.2f}")

    subject_dist = Counter(_subject_of(doc) for doc in docs)
    print(f"top_subjects: {dict(subject_dist)}")

    source_dist = Counter(_source_of(doc) for doc in docs)
    print(f"top_sources: {dict(source_dist.most_common(3))}")

    print(f"expected_term_hits: {calc_term_hits(docs, expected_terms)}")


def print_docs(docs: list[dict[str, Any]], expected_subject: str, rank_prefix: str = "Rank") -> None:
    if not docs:
        print("No docs returned.")
        return

    for i, doc in enumerate(docs, 1):
        meta = doc.get("metadata") or {}
        content = _content_of(doc).replace("\n", " ")

        print("-" * 140)
        print(f"{rank_prefix} {i}")
        print(f"subject: {_subject_of(doc)}")
        print(f"source: {_source_of(doc)}")
        print(f"score: {doc.get('score')}")
        print(f"rerank_score: {doc.get('rerank_score')}")
        print(f"metadata: {meta}")

        if _subject_of(doc) != expected_subject:
            print("WARNING: subject mismatch")

        print(f"content_preview: {content[:1000]}")


def run_full_retrieve(subject: str, label: str, query: str, expected_terms: list[str]) -> None:
    print("\n" + "=" * 140)
    print("MODE: full retrieve(vector + BM25 + rerank)")
    print(f"SUBJECT: {subject}")
    print(f"LABEL: {label}")
    print(f"QUERY: {query}")

    result = retrieve(query=query, subject=subject, top_k=TOP_K)
    docs = result.get("docs", []) or []

    print(f"is_hit: {result.get('is_hit')}")
    print_summary(docs, subject, expected_terms)
    print_docs(docs, subject)


def run_vector_only(subject: str, label: str, query: str, expected_terms: list[str]) -> None:
    print("\n" + "=" * 140)
    print("MODE: vector only Chroma similarity_search_with_relevance_scores")
    print(f"SUBJECT: {subject}")
    print(f"LABEL: {label}")
    print(f"QUERY: {query}")

    vs = load_index()
    where_filter = {"subject": {"$eq": subject}}

    results = vs.similarity_search_with_relevance_scores(
        query,
        k=TOP_K,
        filter=where_filter,
    )

    docs = []
    for doc, score in results:
        docs.append(
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source_file", "unknown"),
                "score": round(float(score), 4),
                "metadata": doc.metadata,
            }
        )

    print_summary(docs, subject, expected_terms)
    print_docs(docs, subject, rank_prefix="Vector Rank")


def main() -> None:
    for case in TEST_CASES:
        subject = case["subject"]
        label = case["label"]
        query = case["query"]
        expected_terms = case.get("expected_terms", [])

        run_full_retrieve(subject, label, query, expected_terms)
        run_vector_only(subject, label, query, expected_terms)


if __name__ == "__main__":
    buffer = StringIO()
    with redirect_stdout(buffer):
        main()

    output = buffer.getvalue()
    print(output, end="")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(output, encoding="utf-8")
    print(f"\n[OK] Full debug RAG output written to {REPORT_PATH}")