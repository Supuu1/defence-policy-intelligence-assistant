"""Regression checks for citation-driven source resolution."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.source_navigation import chunks_for_page, resolve_source_view  # noqa: E402


def main():
    documents = [
        {
            "filename": "native-policy.pdf",
            "document_id": "doc-native",
            "status": "Processed",
            "pages": [
                {
                    "page_number": 12,
                    "text": "Full native page text containing the cited requirement.",
                    "extraction_method": "Native text",
                }
            ],
            "chunks": [
                {
                    "document_id": "doc-native",
                    "page_number": 12,
                    "chunk_id": "page-12-chunk-1",
                    "text": "The cited requirement.",
                }
            ],
        },
        {
            "filename": "scanned-policy.pdf",
            "document_id": "doc-ocr",
            "status": "Processed",
            "pages": [
                {
                    "page_number": 3,
                    "text": "OCR extracted page text.",
                    "extraction_method": "OCR",
                }
            ],
            "chunks": [],
        },
    ]

    view = resolve_source_view(
        documents,
        "doc-native",
        12,
        chunk_id="page-12-chunk-1",
        evidence_id="E2",
    )
    assert view is not None
    assert view.filename == "native-policy.pdf" and view.page_number == 12
    assert view.extraction_method == "Native text"
    assert view.chunk_id == "page-12-chunk-1" and view.evidence_id == "E2"
    assert view.passage == "The cited requirement."
    assert len(chunks_for_page(documents[0], 12)) == 1

    ocr_view = resolve_source_view(documents, "doc-ocr", 3)
    assert ocr_view is not None and ocr_view.extraction_method == "OCR"
    assert resolve_source_view(documents, "doc-native", 99) is None
    assert resolve_source_view(documents, "missing-document", 1) is None

    print("Source navigation metadata, multi-document, OCR, and missing-page checks passed.")


if __name__ == "__main__":
    main()
