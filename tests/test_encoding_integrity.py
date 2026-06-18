from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path


TEXT_ROOTS = (
    "README.md",
    "README_en.md",
    "app.py",
    "docs/",
    "reports/",
    "src/",
    "config/",
    "tests/",
    "frontend/",
)

EXCLUDED_PATH_PARTS = (
    ".git/",
    ".pytest_cache/",
    "__pycache__/",
    "node_modules/",
    "frontend/.next/",
)

EXCLUDED_FILES = {
    "frontend/tsconfig.tsbuildinfo",
}

TEXT_SUFFIXES = {
    ".css",
    ".html",
    ".js",
    ".json",
    ".md",
    ".mjs",
    ".py",
    ".tsx",
    ".ts",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}

MOJIBAKE_PATTERNS = [
    chr(0xFFFD),  # Unicode replacement character.
    chr(0x9225),  # UTF-8/GBK mojibake for punctuation.
    chr(0x922B),
    chr(0x9239),
    chr(0x95B3),
    chr(0x9581),
    chr(0x95BF),
    chr(0x951F),
    chr(0x9983),
    chr(0x9286),
    chr(0x4E63),
    chr(0x4E77),
    chr(0x4E80),
    chr(0x4E81),
    chr(0x6D93),
    chr(0xE045),
    chr(0xE1BC),
    chr(0xE1EE),
    chr(0xE21B),
    chr(0xE63F),
    chr(0x20AC) + "?",
]


def _tracked_files() -> list[Path]:
    output = subprocess.check_output(["git", "ls-files"], text=True)
    paths = [Path(line.strip()) for line in output.splitlines() if line.strip()]
    paths.append(Path(__file__))
    return sorted(set(paths))


def _is_scanned_text_file(path: Path) -> bool:
    normalized = path.as_posix()
    if normalized in EXCLUDED_FILES:
        return False
    if any(part in normalized for part in EXCLUDED_PATH_PARTS):
        return False
    if not any(normalized == root or normalized.startswith(root) for root in TEXT_ROOTS):
        return False
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return False
    return path.exists() and path.is_file()


def _find_text_mojibake(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    offenders: list[str] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        for pattern in MOJIBAKE_PATTERNS:
            if pattern and pattern in line:
                escaped = pattern.encode("unicode_escape").decode("ascii")
                offenders.append(f"{path.as_posix()}:{line_number}:{escaped}")
    return offenders


def test_tracked_text_files_do_not_contain_common_mojibake():
    offenders: list[str] = []
    for path in _tracked_files():
        if _is_scanned_text_file(path):
            offenders.extend(_find_text_mojibake(path))

    assert offenders == []


def test_demo_profile_sqlite_text_fields_do_not_contain_common_mojibake():
    db_path = Path("data/demo_profile.db")
    if not db_path.exists():
        return

    offenders: list[str] = []
    conn = sqlite3.connect(db_path)
    try:
        table_rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        for (table_name,) in table_rows:
            columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            text_columns = [
                column[1]
                for column in columns
                if str(column[2] or "").upper() in {"TEXT", "VARCHAR", "CHAR", "CLOB"}
            ]
            for column_name in text_columns:
                rows = conn.execute(f"SELECT {column_name} FROM {table_name}").fetchall()
                for row_index, (value,) in enumerate(rows, 1):
                    if not isinstance(value, str):
                        continue
                    for pattern in MOJIBAKE_PATTERNS:
                        if pattern and pattern in value:
                            escaped = pattern.encode("unicode_escape").decode("ascii")
                            offenders.append(f"{db_path}:{table_name}.{column_name}[{row_index}]:{escaped}")
    finally:
        conn.close()

    assert offenders == []
