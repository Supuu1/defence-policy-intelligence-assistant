"""Regression checks for grounded intelligence features and report export."""

from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import src.intelligence_service as intelligence_service  # noqa: E402
from src.intelligence_service import (  # noqa: E402
    ExecutiveSummarySchema,
    GroundedPoint,
    PolicyComparisonSchema,
    SUMMARY_SECTIONS,
    COMPARISON_SECTIONS,
    _validated_result,
    generate_executive_summary_from_chunks,
    prepare_evidence,
)
from src.report_service import DISCLAIMER, build_markdown_report  # noqa: E402


def evidence(document_id, filename, page, chunk, score):
    return {
        "text": f"Supported policy text from {filename} page {page}.",
        "source_filename": filename,
        "document_id": document_id,
        "page_number": page,
        "chunk_id": chunk,
        "extraction_method": "Native text",
        "similarity": score,
    }


def main():
    raw = [
        evidence("doc-a", "A.pdf", 3, "a-1", 0.71),
        evidence("doc-a", "A.pdf", 3, "a-1", 0.71),
        evidence("doc-b", "B.pdf", 8, "b-1", 0.62),
    ]
    summary_evidence = prepare_evidence(raw, "S")
    assert [item["evidence_id"] for item in summary_evidence] == ["S1", "S2"]

    summary_schema = ExecutiveSummarySchema(
        purpose_scope=[GroundedPoint(text="Supported purpose.", evidence_ids=["S1"])],
        restrictions_limitations=[
            GroundedPoint(text="Invented without evidence.", evidence_ids=["X9"])
        ],
    )
    summary = _validated_result(summary_schema, summary_evidence, SUMMARY_SECTIONS)
    assert "Purpose / Scope" in summary.sections
    assert "Restrictions / Limitations" not in summary.sections
    assert summary.cited_evidence[0]["source_filename"] == "A.pdf"

    comparison_evidence = prepare_evidence([raw[0]], "A") + prepare_evidence([raw[2]], "B")
    comparison_schema = PolicyComparisonSchema(
        similarities=[GroundedPoint(text="Supported similarity.", evidence_ids=["A1", "B1"])],
        unique_a=[GroundedPoint(text="Supported A-only point.", evidence_ids=["A1"])],
        unique_b=[GroundedPoint(text="Unsupported B point.", evidence_ids=["A9"])],
    )
    comparison = _validated_result(
        comparison_schema, comparison_evidence, COMPARISON_SECTIONS
    )
    assert "Common Provisions / Themes" in comparison.sections
    assert "Requirements Unique to Document B" not in comparison.sections
    assert {item["document_id"] for item in comparison.cited_evidence} == {"doc-a", "doc-b"}

    documents = [
        {"filename": "A.pdf", "pages": [{"extraction_method": "Native text"}], "chunks": [raw[0]], "status": "Processed"},
        {"filename": "B.pdf", "pages": [{"extraction_method": "OCR"}], "chunks": [raw[2]], "status": "Processed"},
    ]
    sources = [
        {
            "source_filename": item["source_filename"],
            "page_number": item["page_number"],
            "chunk_id": item["chunk_id"],
        }
        for item in comparison.cited_evidence
    ]
    report = build_markdown_report(
        "Policy Comparison",
        documents,
        "### Similarities\n- Supported similarity. [A1, B1]",
        comparison.evidence_strength,
        sources,
        comparison.cited_evidence,
        question="Compare requirements",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )
    assert DISCLAIMER in report
    assert "GEMINI_API_KEY" not in report and str(PROJECT_ROOT) not in report
    assert "Native-text pages: 1" in report and "OCR pages: 1" in report

    # Summary evidence comes directly from the selected chunks; no Query
    # Assistant result or retrieval history is created for this test.
    original_generate = intelligence_service._generate
    try:
        def fake_generate(prompt, schema, *_args):
            evidence_ids = list(dict.fromkeys(re.findall(r'<evidence id="([A-Z0-9]+)">', prompt)))
            return schema(
                purpose_scope=[
                    GroundedPoint(
                        text="Grounded scope summary.", evidence_ids=evidence_ids[:8]
                    )
                ]
            )

        intelligence_service._generate = fake_generate
        single_document = generate_executive_summary_from_chunks([raw[0]])
        entire_corpus = generate_executive_summary_from_chunks([raw[0], raw[2]])
        large_corpus_chunks = [
            {
                **evidence(
                    "doc-a" if index < 20 else "doc-b",
                    "A.pdf" if index < 20 else "B.pdf",
                    index + 1,
                    f"chunk-{index}",
                    0.7,
                ),
                "text": "Grounded policy requirement. " * 60,
            }
            for index in range(40)
        ]
        hierarchical = generate_executive_summary_from_chunks(large_corpus_chunks)
        assert single_document.sections and entire_corpus.sections and hierarchical.sections
        assert {item["document_id"] for item in entire_corpus.cited_evidence} == {
            "doc-a",
            "doc-b",
        }
        assert {item["document_id"] for item in hierarchical.cited_evidence} == {
            "doc-a",
            "doc-b",
        }
    finally:
        intelligence_service._generate = original_generate
    print("Grounding validation, document attribution, and Markdown export checks passed.")


if __name__ == "__main__":
    main()
