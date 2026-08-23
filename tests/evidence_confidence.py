"""Deterministic evidence-confidence regression checks."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evidence_confidence import calculate_evidence_confidence  # noqa: E402


def evidence(text, score, document_id, source="Both", keyword_score=2.0):
    return {
        "text": text,
        "document_id": document_id,
        "semantic_similarity": score,
        "similarity": score,
        "keyword_score": keyword_score,
        "retrieval_source": source,
    }


def main():
    strong = [
        evidence("Personnel must complete annual security training.", 0.78, "doc-a"),
        evidence("Annual security training is mandatory for personnel.", 0.71, "doc-a"),
        evidence("The agency requires annual security training.", 0.66, "doc-b"),
    ]
    strong_result = calculate_evidence_confidence(
        "What annual security training is required for personnel?",
        strong,
        strong,
        selected_chunk_count=4,
    )
    assert strong_result.level == "HIGH"
    assert strong_result.supporting_chunks == 3 and strong_result.distinct_documents == 2

    weak = [
        evidence(
            "General administrative information.",
            0.29,
            "doc-a",
            source="Semantic",
            keyword_score=None,
        )
    ]
    weak_result = calculate_evidence_confidence(
        "What is the procurement deadline?", weak, weak, selected_chunk_count=4
    )
    assert weak_result.level == "LOW"

    no_evidence = calculate_evidence_confidence(
        "What is required?", [], [], selected_chunk_count=0
    )
    assert no_evidence.level == "INSUFFICIENT"

    conflicting = [
        evidence("Remote access must be permitted for analysts.", 0.74, "doc-a"),
        evidence("Remote access is prohibited for analysts.", 0.72, "doc-b"),
    ]
    conflict_result = calculate_evidence_confidence(
        "Is remote access permitted for analysts?",
        conflicting,
        conflicting,
        selected_chunk_count=2,
    )
    assert conflict_result.level == "LOW" and conflict_result.conflict_detected
    assert "conflicting" in conflict_result.reason.lower()

    print("Strong, weak, no-evidence, and conflicting-evidence confidence checks passed.")


if __name__ == "__main__":
    main()
