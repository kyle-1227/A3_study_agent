"""Cold-import regression for assessment and structured-output boundaries."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_structured_output_imports_in_a_fresh_interpreter_without_cycle():
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from src.llm.structured_output import "
                "_prepare_structured_messages_with_context"
            ),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
