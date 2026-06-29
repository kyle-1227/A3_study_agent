"""Frontend resource bundle resource_final parsing guardrails."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
API_BASE_URL = "http://localhost:8000"


def _absolute_url(value: str | None) -> str:
    if isinstance(value, str) and value.startswith("/"):
        return f"{API_BASE_URL}{value}"
    return value or ""


def _simulate_resource_final_merge(message: dict, data: dict) -> dict:
    final_answer = data.get("answer") if isinstance(data.get("answer"), str) else ""
    mindmap = data.get("mindmap") or None
    review_doc = data.get("review_doc") or None
    review_doc_artifacts = data.get("review_doc_artifacts") if isinstance(data.get("review_doc_artifacts"), list) else []
    exercise_artifact = data.get("exercise_artifact") or None
    study_plan = data.get("study_plan") or None

    review_docs = [
        {
            "subject": artifact.get("subject", ""),
            "title": artifact.get("title", "Review Document"),
            "filename": artifact.get("filename", ""),
            "markdownUrl": _absolute_url(artifact.get("markdown_url")),
            "docxFilename": artifact.get("docx_filename", ""),
            "docxUrl": _absolute_url(artifact.get("docx_url")),
            "markdown": artifact.get("markdown", ""),
        }
        for artifact in review_doc_artifacts
    ]

    merged = dict(message)
    merged["content"] = final_answer or message.get("content", "")
    if review_docs:
        merged["reviewDoc"] = None
        merged["reviewDocs"] = review_docs
    elif review_doc:
        merged["reviewDoc"] = {
            "subject": review_doc.get("subject", ""),
            "title": review_doc.get("title", "Review Document"),
            "filename": review_doc.get("filename", ""),
            "markdownUrl": _absolute_url(review_doc.get("markdown_url")),
            "docxFilename": review_doc.get("docx_filename", ""),
            "docxUrl": _absolute_url(review_doc.get("docx_url")),
            "markdown": review_doc.get("markdown", ""),
        }
    if mindmap:
        merged["mindmap"] = {
            "title": mindmap.get("title", "Knowledge Mindmap"),
            "tree": mindmap.get("tree"),
            "xmindUrl": _absolute_url(mindmap.get("xmind_url")),
        }
    if exercise_artifact:
        merged["exercise"] = {
            "title": exercise_artifact.get("title", "Exercise Resource"),
            "filename": exercise_artifact.get("filename", ""),
            "markdownUrl": _absolute_url(exercise_artifact.get("markdown_url")),
            "docxFilename": exercise_artifact.get("docx_filename", ""),
            "docxUrl": _absolute_url(exercise_artifact.get("docx_url")),
        }
    if study_plan:
        merged["studyPlan"] = {
            "title": study_plan.get("title", "Personalized Study Plan"),
            "filename": study_plan.get("filename", ""),
            "markdownUrl": _absolute_url(study_plan.get("markdown_url")),
            "docxFilename": study_plan.get("docx_filename", ""),
            "docxUrl": _absolute_url(study_plan.get("docx_url")),
            "markdown": study_plan.get("markdown", ""),
        }
    return merged


def test_page_resource_final_parser_uses_field_presence_not_resource_type():
    page_source = (PROJECT_ROOT / "frontend" / "app" / "page.tsx").read_text(encoding="utf-8")

    assert "const mindmap = data.mindmap ?? null" in page_source
    assert "const reviewDoc = data.review_doc ?? null" in page_source
    assert "Array.isArray(data.review_doc_artifacts)" in page_source
    assert "const exerciseArtifact = data.exercise_artifact ?? null" in page_source
    assert "const studyPlan = data.study_plan ?? null" in page_source
    assert 'data.resource_type === "mindmap" ? data.mindmap : null' not in page_source
    assert 'data.resource_type === "review_doc" ? data.review_doc : null' not in page_source
    assert 'data.resource_type === "quiz" ? data.exercise_artifact : null' not in page_source


def test_simulated_resource_bundle_final_attaches_all_resource_cards():
    message = {"id": "a1", "role": "assistant", "content": ""}
    data = {
        "type": "resource_final",
        "resource_type": "bundle",
        "answer": "# 已生成多类学习资源",
        "resource_bundle": {
            "type": "resource_bundle",
            "status": "success",
            "resources": [
                {"resource_type": "review_doc", "title": "Python 复习资料"},
                {"resource_type": "mindmap", "title": "Python 思维导图"},
                {"resource_type": "quiz", "title": "Python 练习题"},
            ],
            "errors": [],
        },
        "review_doc_artifacts": [
            {
                "title": "Python 复习资料",
                "filename": "python.md",
                "docx_filename": "python.docx",
                "markdown_url": "/artifacts/review-docs/r1/python.md",
                "docx_url": "/artifacts/review-docs/r1/python.docx",
                "markdown": "# Python 复习资料",
            }
        ],
        "mindmap": {
            "title": "Python 思维导图",
            "tree": {"title": "Python", "children": []},
            "xmind_url": "/artifacts/mindmaps/m1/python.xmind",
        },
        "exercise_artifact": {
            "title": "Python 练习题",
            "filename": "python.md",
            "docx_filename": "python.docx",
            "markdown_url": "/artifacts/exercises/e1/python.md",
            "docx_url": "/artifacts/exercises/e1/python.docx",
        },
        "study_plan": {
            "title": "Python Study Plan",
            "filename": "python-plan.md",
            "docx_filename": "python-plan.docx",
            "markdown_url": "/artifacts/review-docs/s1/python-plan.md",
            "docx_url": "/artifacts/review-docs/s1/python-plan.docx",
            "markdown": "# Python Study Plan",
        },
    }

    merged = _simulate_resource_final_merge(message, data)

    assert merged["content"] == "# 已生成多类学习资源"
    assert merged["reviewDocs"][0]["title"] == "Python 复习资料"
    assert merged["reviewDocs"][0]["markdownUrl"] == "http://localhost:8000/artifacts/review-docs/r1/python.md"
    assert merged["mindmap"]["title"] == "Python 思维导图"
    assert merged["mindmap"]["xmindUrl"] == "http://localhost:8000/artifacts/mindmaps/m1/python.xmind"
    assert merged["exercise"]["title"] == "Python 练习题"
    assert merged["exercise"]["docxUrl"] == "http://localhost:8000/artifacts/exercises/e1/python.docx"
    assert merged["studyPlan"]["title"] == "Python Study Plan"
    assert merged["studyPlan"]["markdownUrl"] == "http://localhost:8000/artifacts/review-docs/s1/python-plan.md"
    assert merged["studyPlan"]["markdown"] == "# Python Study Plan"
