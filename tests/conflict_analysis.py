"""Regression checks for grounded cross-document conflict analysis."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import src.conflict_service as service  # noqa: E402
from src.conflict_service import (  # noqa: E402
    ConflictAnalysisSchema,
    ConflictFindingSchema,
    generate_conflict_analysis,
    group_evidence_by_topic,
)


def evidence(evidence_id, document_id, filename, page, text):
    return {
        "evidence_id": evidence_id,
        "document_id": document_id,
        "source_filename": filename,
        "page_number": page,
        "chunk_id": f"page-{page}-chunk-1",
        "extraction_method": "Native text",
        "similarity": 0.75,
        "text": text,
    }


def main():
    evidence_a = [
        evidence(
            "A1", "doc-a", "Policy-A.pdf", 4,
            "Applicants must have 5 years of service. Applications close 30 June.",
        )
    ]
    evidence_b = [
        evidence(
            "B1", "doc-b", "Policy-B.pdf", 9,
            "Applicants must have 8 years of service. Applications close 31 August.",
        )
    ]
    grouped = group_evidence_by_topic(evidence_a + evidence_b)
    assert "Eligibility / Thresholds" in grouped and "Dates / Deadlines" in grouped

    original_generate = service.generate_structured_response
    try:
        service.generate_structured_response = lambda *_args, **_kwargs: ConflictAnalysisSchema(
            findings=[
                ConflictFindingSchema(
                    topic="Eligibility threshold",
                    document_a_statement="A requires five years of service.",
                    document_b_statement="B requires eight years of service.",
                    relationship="POTENTIAL CONFLICT",
                    explanation="The thresholds differ for the same eligibility subject.",
                    document_a_evidence_ids=["A1"],
                    document_b_evidence_ids=["B1"],
                ),
                ConflictFindingSchema(
                    topic="Unsupported authority difference",
                    document_a_statement="A assigns Agency Alpha.",
                    document_b_statement="B assigns Agency Beta.",
                    relationship="POTENTIAL CONFLICT",
                    explanation="Unsupported bilateral claim.",
                    document_a_evidence_ids=["A1"],
                    document_b_evidence_ids=["B99"],
                ),
            ]
        )
        result = generate_conflict_analysis(
            evidence_a, evidence_b, "Policy-A.pdf", "Policy-B.pdf"
        )
    finally:
        service.generate_structured_response = original_generate

    assert result.findings[0]["relationship"] == "POTENTIAL CONFLICT"
    assert result.findings[0]["document_a_evidence_ids"] == ["A1"]
    assert result.findings[0]["document_b_evidence_ids"] == ["B1"]
    assert result.findings[1]["relationship"] == "INSUFFICIENT EVIDENCE"
    assert result.findings[1]["document_b_statement"] == "Not stated in retrieved evidence."
    assert {item["document_id"] for item in result.cited_evidence} == {"doc-a", "doc-b"}

    print("Conflict grouping, bilateral citation, downgrade, and attribution checks passed.")


if __name__ == "__main__":
    main()
