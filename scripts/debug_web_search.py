"""Direct web-search diagnostics for development troubleshooting."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from src.tools.search_tool import search_with_diagnostics  # noqa: E402


TEST_QUERIES = [
    "Python underfitting detection example scikit-learn learning_curve",
    "machine learning underfitting learning curve cross validation",
    "Spark MLlib machine learning model evaluation big data",
]


def main() -> None:
    for query in TEST_QUERIES:
        diagnostics = search_with_diagnostics(query)
        print("=" * 100)
        print(f"query: {query}")
        print(f"provider: {diagnostics.get('provider')}")
        print(f"ok: {diagnostics.get('ok')}")
        print(f"result_count: {diagnostics.get('result_count')}")
        print(f"raw_type: {diagnostics.get('raw_type')}")
        print(f"raw_count: {diagnostics.get('raw_count')}")
        print(f"elapsed_ms: {diagnostics.get('elapsed_ms')}")
        print(f"error_type: {diagnostics.get('error_type')}")
        print(f"error_message: {diagnostics.get('error_message')}")
        for i, result in enumerate(diagnostics.get("results", [])[:3], 1):
            title = result.get("title") or "(no title)"
            url = result.get("url") or "(no url)"
            content = str(result.get("content") or "").replace("\n", " ")[:300]
            print(f"- [{i}] {title}")
            print(f"  url: {url}")
            print(f"  preview: {content}")


if __name__ == "__main__":
    main()
