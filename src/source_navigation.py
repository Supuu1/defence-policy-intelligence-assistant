"""Resolve deterministic citations against already processed corpus metadata."""

from dataclasses import dataclass


@dataclass
class SourceView:
    filename: str
    document_id: str
    page_number: int
    extraction_method: str
    page_text: str
    chunk_id: str
    evidence_id: str
    passage: str


def resolve_source_view(
    corpus_documents: list[dict],
    document_id: str,
    page_number: int,
    chunk_id: str = "",
    evidence_id: str = "",
    passage: str = "",
) -> SourceView | None:
    """Resolve a source without reopening or reprocessing its uploaded PDF."""
    document = next(
        (
            item
            for item in corpus_documents
            if item.get("document_id") == document_id
            and item.get("status") == "Processed"
        ),
        None,
    )
    if document is None:
        return None

    page = next(
        (
            item
            for item in document.get("pages", [])
            if item.get("page_number") == page_number
        ),
        None,
    )
    if page is None:
        return None

    page_chunks = [
        item
        for item in document.get("chunks", [])
        if item.get("page_number") == page_number
    ]
    selected_chunk = next(
        (item for item in page_chunks if item.get("chunk_id") == chunk_id),
        None,
    )
    if selected_chunk is None and not passage and len(page_chunks) == 1:
        selected_chunk = page_chunks[0]

    return SourceView(
        filename=document.get("filename", page.get("source_filename", "Unknown document")),
        document_id=document_id,
        page_number=page_number,
        extraction_method=page.get("extraction_method", "Unknown"),
        page_text=page.get("text", ""),
        chunk_id=(selected_chunk or {}).get("chunk_id", chunk_id),
        evidence_id=evidence_id,
        passage=passage or (selected_chunk or {}).get("text", ""),
    )


def chunks_for_page(document: dict, page_number: int) -> list[dict]:
    """Return existing page chunks in their original deterministic order."""
    return [
        item
        for item in document.get("chunks", [])
        if item.get("page_number") == page_number
    ]
