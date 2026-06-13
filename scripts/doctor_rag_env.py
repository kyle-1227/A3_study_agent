"""RAG environment diagnostic script.

Checks Python environment, dependencies, .env variables, and course data
directories WITHOUT connecting to any embedding API.

Usage:
    python scripts/doctor_rag_env.py
"""

import importlib
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

COURSE_DIRS = [
    "data/big_data",
    "data/computer",
    "data/machine_learning",
    "data/math",
    "data/python",
]

CRITICAL_MODULES = [
    "langchain",
    "langchain_core",
    "langchain_chroma",
    "chromadb",
    "langchain_text_splitters",
    "fitz",
    "rank_bm25",
    "jieba",
    "httpx",
    "yaml",
    "dotenv",
]

CRITICAL_ENV_VARS = [
    "OPENROUTER_API_KEY",
    "EMBEDDING_MODEL",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_API_KEY_ENV",
    "CHROMA_PERSIST_DIR",
]


def check_import(module_name: str) -> tuple[bool, str]:
    """Check whether a module can be imported. Returns (ok, display_name)."""
    try:
        importlib.import_module(module_name)
        return True, module_name
    except ImportError:
        return False, module_name


def main() -> None:
    ok_count = 0
    fail_count = 0
    warn_count = 0

    # ── Section 1: Python Environment ──
    print("=== Python Environment ===")
    print(f"  executable : {sys.executable}")
    print(f"  version    : {sys.version.split()[0]}")
    print(f"  project    : {PROJECT_ROOT}")
    print()

    # ── Section 2: Dependencies ──
    print("=== Dependencies ===")
    for mod in CRITICAL_MODULES:
        ok, name = check_import(mod)
        if ok:
            print(f"[OK]   import {name}")
            ok_count += 1
        else:
            print(f"[FAIL] import {name} — missing package")
            fail_count += 1

    if fail_count > 0:
        print()
        print("One or more packages are missing. Run:")
        print(f"  {sys.executable} -m pip install -r requirements.txt")
    print()

    # ── Section 3: .env and Environment Variables ──
    print("=== Environment Variables ===")
    env_path = PROJECT_ROOT / ".env"
    if env_path.is_file():
        print(f"[OK]   .env file found at {env_path}")
        ok_count += 1
    else:
        print(f"[WARN] .env file not found")
        print(f"       (expected: {env_path})")
        warn_count += 1

    # Load .env so we can check variable values
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path)
    except ImportError:
        pass  # Already reported in Section 2

    for var in CRITICAL_ENV_VARS:
        value = os.getenv(var)
        if value is not None and value != "":
            print(f"[OK]   {var} is set")
            ok_count += 1
        else:
            print(f"[WARN] {var} not found")
            warn_count += 1
    print()

    # ── Section 4: Course Data Directories ──
    print("=== Course Data Directories ===")
    course_ok = 0
    for rel_dir in COURSE_DIRS:
        full_dir = PROJECT_ROOT / rel_dir
        if not full_dir.is_dir():
            print(f"[WARN] {rel_dir} — directory does not exist")
            warn_count += 1
            continue

        # Count by extension, excluding .gitkeep
        pdf_count = len(list(full_dir.glob("*.pdf")))
        md_count = len(list(full_dir.glob("*.md")))
        txt_count = len(list(full_dir.glob("*.txt")))

        if pdf_count + md_count + txt_count == 0:
            print(f"[SKIP] {rel_dir} — empty (no .pdf/.md/.txt files)")
            warn_count += 1
        else:
            print(f"[OK]   {rel_dir} — {pdf_count} pdf, {md_count} md, {txt_count} txt")
            ok_count += 1
            course_ok += 1
    print()

    # ── Section 5: Summary ──
    total_modules = len(CRITICAL_MODULES)
    total_env = len(CRITICAL_ENV_VARS)
    print("=== Summary ===")
    print(f"{fail_count} module(s) FAILED of {total_modules}")
    print(f"{warn_count} warning(s) across env vars and data dirs")
    print(f"{course_ok} course dir(s) with files")

    if fail_count > 0 or warn_count > 0:
        print()
        print("Issues detected. To resolve missing dependencies, run:")
        print(f"  {sys.executable} -m pip install -r requirements.txt")
        sys.exit(1)
    else:
        print("All checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
