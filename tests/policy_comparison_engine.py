"""Regression checks for grounded, document-isolated policy comparison."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import src.intelligence_service as service  # noqa: E402
from src.intelligence_service import (  # noqa: E402
    ComparisonPoint,
    PolicyComparisonSchema,
    RAGServiceError,
    generate_policy_comparison,
    prepare_evidence,
)


def chunk(document_id, filename, page, chunk_id, text):
    return {
        "document_id": document_id,
        "source_filename": filename,
        "page_number": page,
        "chunk_id": chunk_id,
        "extraction_method": "Native text",
        "similarity": 0.72,
        "text": text,
    }


def main():
    evidence_a = prepare_evidence(
        [chunk("doc-a", "Policy-A.pdf", 4, "a-1", "Applications close 30 June. Agency Alpha administers grants.")],
        "A",
    )
    evidence_b = prepare_evidence(
        [chunk("doc-b", "Policy-B.pdf", 9, "b-1", "Applications close 31 August. Agency Bravo administers loans.")],
        "B",
    )
    original_generate = service._generate
    try:
        service._generate = lambda *_args, **_kwargs: PolicyComparisonSchema(
            document_a=[ComparisonPoint(text="A uses grants.", evidence_ids=["A1"])],
            document_b=[ComparisonPoint(text="B uses loans.", evidence_ids=["B1"])],
            differences=[
                ComparisonPoint(
                    text="The deadlines and financing mechanisms differ.",
                    evidence_ids=["A1", "B1"],
                ),
                # Must be rejected: a claimed difference cannot cite only A.
                ComparisonPoint(text="Unsupported difference.", evidence_ids=["A1"]),
            ],
            financial_budget=[
                ComparisonPoint(text="A describes grants; B describes loans.", evidence_ids=["A1", "B1"])
            ],
            unique_a=[ComparisonPoint(text="A states a grant mechanism.", evidence_ids=["A1"])],
            conclusion=[
                ComparisonPoint(text="The policies use different mechanisms.", evidence_ids=["A1", "B1"])
            ],
        )
        result = generate_policy_comparison(
            evidence_a + evidence_b, "Policy-A.pdf", "Policy-B.pdf", ""
        )
    finally:
        service._generate = original_generate

    assert "Document A" in result.sections and "Document B" in result.sections
    assert len(result.sections["Key Differences"]) == 1
    assert "Not stated in retrieved evidence" in result.sections[
        "Information Stated Only in Document A"
    ][0]["text"]
    assert {item["document_id"] for item in result.cited_evidence} == {"doc-a", "doc-b"}
    assert result.sections["Document A"][0]["evidence_ids"] == ["A1"]
    assert result.sections["Document B"][0]["evidence_ids"] == ["B1"]

    try:
        generate_policy_comparison(evidence_a, "Policy-A.pdf", "Policy-B.pdf", "")
    except RAGServiceError as error:
        assert "Document B has insufficient retrieved evidence" in str(error)
    else:
        raise AssertionError("Comparison should not run without evidence from both documents")

    print("Policy comparison grounding, attribution, differences, and missing-evidence checks passed.")


if __name__ == "__main__":
    main()
