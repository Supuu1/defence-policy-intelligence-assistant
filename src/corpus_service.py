"""Small corpus-state helpers shared by the Streamlit application and tests."""


def deduplicate_documents_by_content(
    documents: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Keep the first upload for each content hash and report later duplicates."""
    unique_documents = []
    duplicate_documents = []
    seen_document_ids = set()
    for document in documents:
        document_id = document.get("document_id")
        if document_id and document_id in seen_document_ids:
            duplicate_documents.append(document)
            continue
        if document_id:
            seen_document_ids.add(document_id)
        unique_documents.append(document)
    return unique_documents, duplicate_documents
