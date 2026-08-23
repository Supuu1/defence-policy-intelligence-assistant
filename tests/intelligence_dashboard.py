"""Regression checks for lightweight dashboard aggregation and empty states."""

from pathlib import Path
import sys
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard_service import build_dashboard_snapshot  # noqa: E402


def main():
    empty = build_dashboard_snapshot([], set(), 0)
    assert empty["metrics"]["Total Documents"] == 0
    assert empty["health"]["classification"] == "Needs Attention"
    assert not empty["timeline"]["available"]
    assert not empty["conflicts"]["available"]

    documents = [
        {
            "filename": "A.pdf",
            "document_id": "doc-a",
            "status": "Processed",
            "pages": [
                {"page_number": 1, "extraction_method": "Native text", "ocr_error": ""},
                {"page_number": 2, "extraction_method": "Native text", "ocr_error": ""},
            ],
            "chunks": [{"document_id": "doc-a"}, {"document_id": "doc-a"}],
        },
        {
            "filename": "B.pdf",
            "document_id": "doc-b",
            "status": "Processed",
            "pages": [{"page_number": 1, "extraction_method": "OCR", "ocr_error": ""}],
            "chunks": [{"document_id": "doc-b"}],
        },
    ]
    entities = [
        {
            "entity": "Ministry of Defence",
            "category": "Ministries",
            "references": [
                {"document_id": "doc-a"},
                {"document_id": "doc-b"},
            ],
        },
        {
            "entity": "Policy Alpha",
            "category": "Policies",
            "references": [{"document_id": "doc-a"}],
        },
    ]
    timeline = [
        {"document_id": "doc-a", "date": "March 2025", "date_precision": "month"},
        {"document_id": "doc-b", "date": "2026-04-01", "date_precision": "exact"},
        {"document_id": "doc-b", "date": "within 30 days", "date_precision": "relative"},
    ]
    conflicts = [
        {
            "document_a": "A.pdf",
            "document_b": "B.pdf",
            "findings": [
                {"topic": "Eligibility", "relationship": "POTENTIAL CONFLICT"},
                {"topic": "Funding", "relationship": "COMPLEMENTARY"},
            ],
        }
    ]

    with (
        patch("src.ocr_service.extract_text_with_ocr") as ocr,
        patch("src.document_processing.process_pdf") as pdf,
        patch("src.semantic_retrieval.build_vector_index_from_parts") as index,
        patch("src.rag_service.generate_structured_response") as gemini,
    ):
        snapshot = build_dashboard_snapshot(
            documents,
            {"doc-a", "doc-b"},
            3,
            entities=entities,
            timeline_events=timeline,
            conflict_analyses=conflicts,
        )
        assert not ocr.called and not pdf.called and not index.called and not gemini.called

    assert snapshot["metrics"] == {
        "Total Documents": 2,
        "Total Pages": 3,
        "Total Text Chunks": 3,
        "Native Text Pages": 2,
        "OCR Pages": 1,
        "Total Extracted Entities": 2,
        "Timeline Events": 3,
        "Detected Conflicts": 1,
    }
    assert snapshot["document_rows"][0]["Entities"] == 2
    assert snapshot["document_rows"][1]["Timeline Events"] == 2
    assert sum(row["Percentage of Corpus"] for row in snapshot["coverage_rows"]) == 100.0
    assert snapshot["entity_distribution"] == {"Ministries": 1, "Policies": 1}
    assert snapshot["timeline"]["earliest"] == "March 2025"
    assert snapshot["timeline"]["latest"] == "2026-04-01"
    assert snapshot["conflicts"]["categories"] == {"Eligibility": 1}
    assert snapshot["health"]["classification"] == "Good"
    assert snapshot["health"]["index_status"] == "Online"

    print("Dashboard metrics, empty state, optional analyses, health, and no-work guarantees passed.")


if __name__ == "__main__":
    main()
