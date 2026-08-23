"""Regression checks for explicit-date timeline extraction and ordering."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.timeline_service import extract_timeline_events  # noqa: E402


def chunk(document_id, filename, page, chunk_id, text, method="Native text"):
    return {
        "document_id": document_id,
        "source_filename": filename,
        "page_number": page,
        "chunk_id": chunk_id,
        "extraction_method": method,
        "text": text,
    }


def main():
    chunks = [
        chunk(
            "doc-a",
            "Policy-A.pdf",
            1,
            "page-1-chunk-1",
            "The policy was published on 15 January 2025. Funding applies in FY 2025-26.",
        ),
        chunk(
            "doc-a",
            "Policy-A.pdf",
            3,
            "page-3-chunk-1",
            "Implementation becomes effective in March 2026. Submit the report within 30 days.",
            "OCR",
        ),
        chunk(
            "doc-b",
            "Policy-B.pdf",
            8,
            "page-8-chunk-1",
            "Applications close on June 30, 2025. The amendment takes effect on 2027-01-15.",
        ),
    ]

    corpus_events = extract_timeline_events(chunks)
    displayed_dates = [event["date"] for event in corpus_events]
    assert "15 January 2025" in displayed_dates
    assert "FY 2025-26" in displayed_dates
    assert "March 2026" in displayed_dates
    assert "within 30 days" in displayed_dates
    assert "2027-01-15" in displayed_dates
    assert displayed_dates[-1] == "within 30 days"
    assert all(event["date"] != "2026-03-01" for event in corpus_events)
    assert [event["evidence_id"] for event in corpus_events] == [
        f"T{position}" for position in range(1, len(corpus_events) + 1)
    ]
    assert all(event["source_filename"] and event["page_number"] for event in corpus_events)

    document_events = extract_timeline_events(chunks, document_id="doc-a")
    assert document_events and all(event["document_id"] == "doc-a" for event in document_events)
    assert any(event["extraction_method"] == "OCR" for event in document_events)
    assert not extract_timeline_events([], document_id="doc-a")

    print("Timeline date precision, chronology, filtering, metadata, and empty-scope checks passed.")


if __name__ == "__main__":
    main()
