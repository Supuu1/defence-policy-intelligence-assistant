"""Regression checks for evidence-grounded entity and fact extraction."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.entity_service import entity_counts, extract_entities  # noqa: E402


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
            2,
            "page-2-chunk-1",
            "The Ministry of Defence issued the National Defence Policy on 15 March 2026. "
            "The Defence Procurement Agency administers a budget of ₹5 crore with a 25 percent reserve. "
            "The Falcon Radar System is deployed and headquartered in New Delhi. "
            "The Department of Military Affairs coordinates Northern Command and the National Security Council. "
            "General Arun Kumar chairs the review.",
        ),
        chunk(
            "doc-b",
            "Policy-B.pdf",
            7,
            "page-7-chunk-1",
            "The Ministry of Defence reviews the National Security Act, 2025 and the Defence Acquisition Rules, 2024. "
            "The Joint Defence Committee oversees the Modernisation Mission within 30 days.",
            "OCR",
        ),
        # Repeated mention must merge while retaining the second source reference.
        chunk(
            "doc-b",
            "Policy-B.pdf",
            8,
            "page-8-chunk-1",
            "The Ministry of Defence approves implementation.",
        ),
    ]

    corpus = extract_entities(chunks)
    ministry = next(item for item in corpus if item["entity"] == "Ministry of Defence")
    assert ministry["category"] == "Ministries"
    assert {item["document_id"] for item in ministry["references"]} == {"doc-a", "doc-b"}
    assert len(ministry["references"]) == 3
    assert len({ref["evidence_id"] for item in corpus for ref in item["references"]}) == sum(
        len(item["references"]) for item in corpus
    )

    expected = {
        ("National Defence Policy", "Policies"),
        ("Defence Procurement Agency", "Agencies"),
        ("15 March 2026", "Dates"),
        ("₹5 crore", "Monetary amounts"),
        ("25 percent", "Percentages"),
        ("New Delhi", "Locations"),
        ("Falcon Radar System", "Equipment/platform names"),
        ("National Security Act, 2025", "Acts"),
        ("Defence Acquisition Rules, 2024", "Rules/regulations"),
        ("Joint Defence Committee", "Committees"),
        ("Modernisation Mission", "Programs/schemes"),
        ("within 30 days", "Dates"),
        ("Department of Military Affairs", "Government departments"),
        ("Northern Command", "Defence organizations"),
        ("National Security Council", "Organizations"),
        ("General Arun Kumar", "People/designations"),
    }
    actual = {(item["entity"], item["category"]) for item in corpus}
    assert expected <= actual

    counts = entity_counts(corpus)
    assert counts["Ministries"] == 1 and counts["Dates"] >= 2
    document_a = extract_entities(chunks, document_id="doc-a")
    assert document_a and all(
        reference["document_id"] == "doc-a"
        for item in document_a
        for reference in item["references"]
    )
    assert not extract_entities([], document_id="doc-a")

    print("Entity categories, facts, deduplication, attribution, filtering, and evidence IDs passed.")


if __name__ == "__main__":
    main()
