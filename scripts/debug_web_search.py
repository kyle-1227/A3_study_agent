"""Direct Tavily web-search diagnostics for development troubleshooting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv(PROJECT_ROOT / ".env")

from src.tools.search_tool import search_with_diagnostics  # noqa: E402

DEFAULT_QUERY = "deep learning exercises with solutions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Tavily-only Web Search diagnostic.")
    parser.add_argument("query", nargs="*", help="Search query to execute.")
    parser.add_argument(
        "--original",
        default="",
        help="Original user question for diagnostics; defaults to the query.",
    )
    parser.add_argument("--provider", default="tavily", choices=["tavily"], help="Only tavily is supported.")
    parser.add_argument("--max-results", type=int, default=None, help="Override Tavily max_results.")
    parser.add_argument("--timeout", type=float, default=None, help="Override timeout seconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query = " ".join(args.query).strip() or DEFAULT_QUERY
    original = args.original.strip() or query
    diagnostics = search_with_diagnostics(
        query,
        original_user_query=original,
        max_results=args.max_results,
        timeout_seconds=args.timeout,
    )
    print("=" * 100)
    print(f"provider: {diagnostics.get('provider')}")
    print(f"original_user_query: {diagnostics.get('original_user_query')}")
    print(f"query: {diagnostics.get('query')}")
    print(f"ok: {diagnostics.get('ok')}")
    print(f"result_count: {diagnostics.get('result_count')}")
    print(f"raw_type: {diagnostics.get('raw_type')}")
    print(f"raw_count: {diagnostics.get('raw_count')}")
    print(f"elapsed_ms: {diagnostics.get('elapsed_ms')}")
    print(f"response_time: {diagnostics.get('response_time')}")
    print(f"usage_credits: {diagnostics.get('usage_credits')}")
    print(f"status_code: {diagnostics.get('status_code')}")
    print(f"error_type: {diagnostics.get('error_type')}")
    print(f"error_message: {diagnostics.get('error_message')}")
    print("top_results:")
    for i, result in enumerate(diagnostics.get("results", [])[:3], 1):
        title = result.get("title") or "(no title)"
        url = result.get("url") or "(no url)"
        score = result.get("score")
        favicon = result.get("favicon") or ""
        content = str(result.get("content") or "").replace("\n", " ")[:300]
        print(f"- [{i}] {title}")
        print(f"  url: {url}")
        print(f"  score: {score}")
        if favicon:
            print(f"  favicon: {favicon}")
        print(f"  content_preview: {content}")


if __name__ == "__main__":
    main()
