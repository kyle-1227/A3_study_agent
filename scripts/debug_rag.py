from pathlib import Path
import sys
from contextlib import redirect_stdout
from io import StringIO

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv(PROJECT_ROOT / ".env")

from src.rag.retriever import retrieve

REPORT_PATH = PROJECT_ROOT / "reports" / "debug_rag.txt"


TEST_QUERIES = [
    ("python", "Python 函数 参数 返回值 作用域"),
    ("python", "Python 列表 字典 循环 练习题"),
    ("machine_learning", "机器学习 监督学习 过拟合 正则化"),
    ("machine_learning", "机器学习 决策树 随机森林 模型评估"),
    ("big_data", "大数据导论 Hadoop HDFS MapReduce"),
    ("big_data", "大数据 数据采集 数据清洗 数据仓库"),
    ("math", "高等数学 导数 极限 连续"),
]


def show_result(subject: str, query: str):
    print("\n" + "=" * 120)
    print(f"SUBJECT: {subject}")
    print(f"QUERY: {query}")

    result = retrieve(query=query, subject=subject, top_k=5)
    print(f"is_hit: {result.get('is_hit')}")

    docs = result.get("docs", [])
    if not docs:
        print("No docs returned.")
        return

    for i, doc in enumerate(docs, 1):
        print("-" * 120)
        print(f"Rank {i}")
        print(f"source: {doc.get('source')}")
        print(f"score: {doc.get('score')}")
        if "rerank_score" in doc:
            print(f"rerank_score: {doc.get('rerank_score')}")
        print(f"metadata: {doc.get('metadata')}")
        content = str(doc.get("content", "")).replace("\n", " ")
        print(f"content_preview: {content[:1200]}")


def main():
    for subject, query in TEST_QUERIES:
        show_result(subject, query)


if __name__ == "__main__":
    buffer = StringIO()
    with redirect_stdout(buffer):
        main()

    output = buffer.getvalue()
    print(output, end="")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(output, encoding="utf-8")
    print(f"\n[OK] Full debug RAG output written to {REPORT_PATH}")
