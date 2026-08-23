"""Build downloadable Markdown reports from results already held by the app."""

from datetime import datetime


DISCLAIMER = (
    "This report is generated from the uploaded document corpus and should be "
    "verified against the cited source documents before operational or policy use."
)


def build_markdown_report(
    analysis_type: str,
    corpus_documents: list[dict],
    content_markdown: str,
    evidence_strength: str,
    sources: list[dict],
    evidence: list[dict],
    question: str = "",
    embedding_model: str = "",
) -> str:
    """Serialize trusted application state without another model request."""
    pages = [page for document in corpus_documents for page in document.get("pages", [])]
    native_pages = sum(page.get("extraction_method") == "Native text" for page in pages)
    ocr_pages = sum(page.get("extraction_method") == "OCR" for page in pages)
    lines = [
        "# Defence Policy Intelligence Report",
        "",
        f"**Analysis Type:** {analysis_type}",
        f"**Generated Timestamp:** {datetime.now().astimezone().isoformat(timespec='minutes')}",
        f"**Evidence Strength:** {evidence_strength}",
    ]
    if question:
        lines.extend([f"**User Query / Comparison Question:** {question}"])
    lines.extend(["", "## Corpus Documents", ""])
    for document in corpus_documents:
        lines.append(
            f"- {document['filename']} — {len(document.get('pages', []))} pages, "
            f"{len(document.get('chunks', []))} chunks, status: {document.get('status', 'Unknown')}"
        )
    lines.extend(["", f"## {analysis_type}", "", content_markdown or "No generated analysis available."])
    lines.extend(["", "## Cited Sources", ""])
    for source in sources:
        lines.append(
            f"- {source['source_filename']} · Page {source['page_number']} · "
            f"{source.get('chunk_id', 'chunk unavailable')}"
        )
    if not sources:
        lines.append("- None")
    lines.extend(["", "## Supporting Evidence", ""])
    for item in evidence:
        lines.extend(
            [
                f"### {item.get('evidence_id', 'Evidence')} — {item['source_filename']} · Page {item['page_number']}",
                "",
                item["text"],
                "",
            ]
        )
    if not evidence:
        lines.append("No supporting evidence recorded.")
    lines.extend(
        [
            "",
            "## Processing Metadata",
            "",
            f"- Documents: {len(corpus_documents)}",
            f"- Pages: {len(pages)}",
            f"- Native-text pages: {native_pages}",
            f"- OCR pages: {ocr_pages}",
            f"- Embedding model: {embedding_model}",
            "",
            "## Disclaimer",
            "",
            DISCLAIMER,
        ]
    )
    return "\n".join(lines)
