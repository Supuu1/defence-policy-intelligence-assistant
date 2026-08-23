"""Targeted checks for production stabilization findings."""

from pathlib import Path
import os
import sys
from unittest.mock import patch

import fitz
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.corpus_service import deduplicate_documents_by_content  # noqa: E402
from src.document_processing import process_pdf  # noqa: E402
from src.rag_service import (  # noqa: E402
    INSUFFICIENT_EVIDENCE_MESSAGE,
    _friendly_llm_error,
    _resolve_gemini_api_key,
    generate_grounded_answer,
)


class ProviderError(Exception):
    def __init__(self, code=None):
        self.code = code


def main():
    documents = [
        {"filename": "first.pdf", "document_id": "same-hash", "chunks": [1]},
        {"filename": "renamed-copy.pdf", "document_id": "same-hash", "chunks": [1]},
        {"filename": "different.pdf", "document_id": "different-hash", "chunks": [2]},
    ]
    unique, duplicates = deduplicate_documents_by_content(documents)
    assert [item["filename"] for item in unique] == ["first.pdf", "different.pdf"]
    assert [item["filename"] for item in duplicates] == ["renamed-copy.pdf"]
    assert sum(len(item["chunks"]) for item in unique) == 2

    assert "quota or rate limit" in _friendly_llm_error(ProviderError(429)).lower()
    assert "api key" in _friendly_llm_error(ProviderError(401)).lower()
    assert "model" in _friendly_llm_error(ProviderError(404)).lower()
    assert "temporarily unavailable" in _friendly_llm_error(ProviderError(503)).lower()

    timeout_error = type("ReadTimeout", (Exception,), {})()
    network_error = type("ConnectError", (Exception,), {})()
    assert "timed out" in _friendly_llm_error(timeout_error).lower()
    assert "network" in _friendly_llm_error(network_error).lower()

    unsupported = generate_grounded_answer("Unsupported question", [])
    assert not unsupported.sufficient_evidence
    assert unsupported.answer == INSUFFICIENT_EVIDENCE_MESSAGE
    assert unsupported.evidence_ids == []

    # Credential-source checks use placeholders and mock dotenv loading, so the
    # developer's real environment file is never read by this test.
    with (
        patch("src.rag_service.load_dotenv"),
        patch.dict(os.environ, {"GEMINI_API_KEY": "environment-placeholder"}),
    ):
        assert _resolve_gemini_api_key() == "environment-placeholder"
    with (
        patch("src.rag_service.load_dotenv"),
        patch.dict(os.environ, {"GEMINI_API_KEY": ""}),
        patch.object(st, "secrets", {"GEMINI_API_KEY": "cloud-placeholder"}),
    ):
        assert _resolve_gemini_api_key() == "cloud-placeholder"
    with (
        patch("src.rag_service.load_dotenv"),
        patch.dict(os.environ, {"GEMINI_API_KEY": ""}),
        patch.object(st, "secrets", {}),
    ):
        assert _resolve_gemini_api_key() is None

    blank_document = fitz.open()
    blank_document.new_page()
    blank_pdf = blank_document.tobytes()
    blank_document.close()
    with patch("src.document_processing.extract_text_with_ocr", return_value=""):
        blank_pages, _, blank_chunks = process_pdf(
            blank_pdf, "blank.pdf", "blank-document", include_preview=False
        )
    assert len(blank_pages) == 1 and blank_chunks == []
    assert blank_pages[0]["extraction_method"] == "Failed"

    print("Duplicate-corpus exclusion and safe Gemini error classification passed.")


if __name__ == "__main__":
    main()
