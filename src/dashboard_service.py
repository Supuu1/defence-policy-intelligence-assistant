"""Lightweight deterministic aggregation for the Intelligence Dashboard."""

from collections import Counter
import math

from src.entity_service import entity_counts


def build_dashboard_snapshot(
    corpus_documents: list[dict],
    indexed_document_ids: set[str],
    vector_count: int,
    entities: list[dict] | None = None,
    timeline_events: list[dict] | None = None,
    conflict_analyses: list[dict] | None = None,
) -> dict:
    """Build dashboard data only from records already held by the application."""
    entities_available = entities is not None
    timeline_available = timeline_events is not None
    conflict_available = bool(conflict_analyses)
    entities = entities or []
    timeline_events = timeline_events or []
    conflict_analyses = conflict_analyses or []

    pages = [page for document in corpus_documents for page in document.get("pages", [])]
    chunks = [
        chunk
        for document in corpus_documents
        if document.get("document_id") in indexed_document_ids
        for chunk in document.get("chunks", [])
    ]
    native_pages = sum(page.get("extraction_method") == "Native text" for page in pages)
    ocr_pages = sum(page.get("extraction_method") == "OCR" for page in pages)

    entity_documents = Counter()
    for record in entities:
        for document_id in {
            reference.get("document_id") for reference in record.get("references", [])
        }:
            if document_id:
                entity_documents[document_id] += 1
    timeline_documents = Counter(
        event.get("document_id") for event in timeline_events if event.get("document_id")
    )
    document_names = {
        document.get("document_id"): document.get("filename", "Unknown document")
        for document in corpus_documents
    }

    document_rows = []
    coverage_rows = []
    total_chunks = len(chunks)
    for document in corpus_documents:
        document_pages = document.get("pages", [])
        document_chunks = (
            document.get("chunks", [])
            if document.get("document_id") in indexed_document_ids
            else []
        )
        document_rows.append(
            {
                "Document": document.get("filename", "Unknown document"),
                "Pages": len(document_pages),
                "Chunks": len(document_chunks),
                "Native Pages": sum(
                    page.get("extraction_method") == "Native text"
                    for page in document_pages
                ),
                "OCR Pages": sum(
                    page.get("extraction_method") == "OCR" for page in document_pages
                ),
                "Entities": entity_documents.get(document.get("document_id"), 0)
                if entities_available
                else None,
                "Timeline Events": timeline_documents.get(document.get("document_id"), 0)
                if timeline_available
                else None,
            }
        )
        coverage_rows.append(
            {
                "Document": document.get("filename", "Unknown document"),
                "Chunk Count": len(document_chunks),
                "Percentage of Corpus": (
                    round(len(document_chunks) * 100 / total_chunks, 1)
                    if total_chunks
                    else 0.0
                ),
            }
        )

    sortable_events = [
        event for event in timeline_events if event.get("date_precision") != "relative"
    ]
    potential_conflicts = [
        finding
        for analysis in conflict_analyses
        for finding in analysis.get("findings", [])
        if finding.get("relationship") == "POTENTIAL CONFLICT"
    ]
    conflict_documents = sorted(
        {
            document_name
            for analysis in conflict_analyses
            for document_name in (analysis.get("document_a"), analysis.get("document_b"))
            if document_name
        }
    )
    conflict_topics = Counter(
        finding.get("topic", "Unspecified topic") for finding in potential_conflicts
    )

    successful_documents = sum(
        document.get("status") == "Processed" for document in corpus_documents
    )
    warning_documents = sum(
        bool(document.get("safeguard_warning"))
        or document.get("status") != "Processed"
        or any(page.get("ocr_error") for page in document.get("pages", []))
        for document in corpus_documents
    )
    total_pages = len(pages)
    native_percentage = round(native_pages * 100 / total_pages, 1) if total_pages else 0.0
    ocr_percentage = round(ocr_pages * 100 / total_pages, 1) if total_pages else 0.0
    index_online = vector_count > 0 and bool(indexed_document_ids)
    if (
        corpus_documents
        and index_online
        and successful_documents == len(corpus_documents)
        and warning_documents == 0
    ):
        health = "Good"
    elif (
        corpus_documents
        and index_online
        and successful_documents >= max(1, math.ceil(len(corpus_documents) * 0.75))
    ):
        health = "Moderate"
    else:
        health = "Needs Attention"

    return {
        "metrics": {
            "Total Documents": len(corpus_documents),
            "Total Pages": total_pages,
            "Total Text Chunks": total_chunks,
            "Native Text Pages": native_pages,
            "OCR Pages": ocr_pages,
            "Total Extracted Entities": len(entities) if entities_available else None,
            "Timeline Events": len(timeline_events) if timeline_available else None,
            "Detected Conflicts": len(potential_conflicts) if conflict_available else None,
        },
        "document_rows": document_rows,
        "entity_distribution": entity_counts(entities) if entities_available else None,
        "coverage_rows": coverage_rows,
        "timeline": {
            "available": timeline_available,
            "total": len(timeline_events),
            "earliest": sortable_events[0]["date"] if sortable_events else None,
            "latest": sortable_events[-1]["date"] if sortable_events else None,
            "by_document": {
                document_names.get(document_id, "Unknown document"): count
                for document_id, count in timeline_documents.items()
            },
        },
        "conflicts": {
            "available": conflict_available,
            "total": len(potential_conflicts),
            "documents": conflict_documents,
            "categories": dict(conflict_topics),
        },
        "health": {
            "classification": health,
            "native_page_percentage": native_percentage,
            "ocr_page_percentage": ocr_percentage,
            "successful_documents": successful_documents,
            "warning_documents": warning_documents,
            "index_status": "Online" if index_online else "Offline",
        },
    }
