"""UTF-8 and minimal Markdown sanity for the evaluation status report."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STATUS_REPORT = ROOT / "docs" / "reports" / "evidence_rollout_evaluation_status.md"


def test_status_report_is_strict_utf8_without_mojibake() -> None:
    payload = STATUS_REPORT.read_bytes()
    text = payload.decode("utf-8", errors="strict")

    assert "\ufffd" not in text
    for marker in ("\u9225", "\u951f", "\u7039"):
        assert marker not in text
    assert all(ord(character) < 128 for character in text)


def test_status_report_has_renderable_structure_and_artifact_entries() -> None:
    text = STATUS_REPORT.read_text(encoding="utf-8")

    assert text.startswith("# Evidence rollout evaluation control-plane status\n")
    assert text.count("```") % 2 == 0
    assert "- `activation_decision.json`:" in text
    assert "- `safe_report.json`:" in text
    assert "- `safe_report.md`:" in text
    assert "| Prerequisite | Status | Evidence |" in text
