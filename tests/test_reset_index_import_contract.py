from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_reset_index_import_does_not_load_project_environment() -> None:
    project_root = Path(__file__).resolve().parent.parent
    environment = os.environ.copy()
    environment.pop("RAG_SPLITTER_MODE", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; import scripts.reset_index; "
                "print(os.environ.get('RAG_SPLITTER_MODE', '<missing>'))"
            ),
        ],
        cwd=project_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "<missing>"
