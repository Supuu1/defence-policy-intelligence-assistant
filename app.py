import hashlib
import html
import gc
import logging
from datetime import datetime
import time

import fitz  # PyMuPDF
import streamlit as st

from src.cache_service import (
    cache_statistics,
    clear_local_cache,
    chunk_fingerprint,
    load_document,
    load_embeddings,
    relabel_document,
    save_document,
    save_embeddings,
)
from src.document_processing import CHUNK_OVERLAP, CHUNK_SIZE, process_pdf
from src.dashboard_service import build_dashboard_snapshot
from src.evidence_confidence import calculate_evidence_confidence
from src.entity_service import ENTITY_CATEGORIES, entity_counts, extract_entities
from src.conflict_service import generate_conflict_analysis
from src.corpus_service import deduplicate_documents_by_content
from src.intelligence_service import (
    NoSummarizableContentError,
    summary_cache_key,
    cached_executive_summary,
    local_extractive_summary,
    generate_policy_comparison,
    prepare_evidence,
)
from src.ocr_service import get_tesseract_status
from src.hybrid_retrieval import build_bm25_index, hybrid_search
from src.rag_service import (
    GeminiRequestError,
    INSUFFICIENT_EVIDENCE_MESSAGE,
    MalformedStructuredResponseError,
    MissingAPIKeyError,
    RAGServiceError,
    TOP_K,
    assign_evidence_ids,
    build_verified_sources,
    calculate_evidence_strength,
    generate_grounded_answer,
    get_gemini_availability,
    get_chunks_by_evidence_id,
    select_evidence_for_generation,
)
from src.semantic_retrieval import (
    EMBEDDING_MODEL_NAME,
    build_vector_index_from_parts,
    embed_chunks_batched,
    load_embedding_model,
    recommended_embedding_batch_size,
    semantic_search,
)
from src.report_service import build_markdown_report
from src.source_navigation import chunks_for_page, resolve_source_view
from src.timeline_service import extract_timeline_events


st.set_page_config(
    page_title="Research Intelligence Assistant",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    """Load the embedding model once per Streamlit server process."""
    return load_embedding_model()


MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024
MAX_PAGES_PER_DOCUMENT = 500
LARGE_FILE_WARNING_BYTES = 25 * 1024 * 1024
LARGE_PAGE_WARNING = 75
MAX_PREVIEW_CHARACTERS = 150_000
logger = logging.getLogger("defence_policy_rag")


def failed_document_record(filename, document_id, file_size, failure_stage):
    """Create a safe UI record while detailed errors remain in terminal logs."""
    return {
        "filename": filename,
        "document_id": document_id,
        "content_hash": document_id,
        "file_size": file_size,
        "safeguard_warning": "",
        "pages": [],
        "chunks": [],
        "status": "Failed",
        "error": "Processing failed for this document.",
        "failure_stage": failure_stage,
    }


def retrieve_policy_evidence(
    vector_index, model, keyword_index, document_id, queries, limit=18
):
    """Retrieve broad, deduplicated evidence without sending a whole corpus."""
    candidates = []
    for query in queries:
        candidates.extend(
            hybrid_search(
                query,
                vector_index,
                model,
                keyword_index,
                top_k=4,
                document_id=document_id,
            )
        )
    candidates.sort(
        key=lambda item: item.get("hybrid_score", item.get("similarity", 0.0)),
        reverse=True,
    )
    deduplicated = []
    seen = set()
    for item in candidates:
        if (
            item.get("retrieval_source", "Semantic") == "Semantic"
            and item.get("similarity", 0.0) < 0.25
        ):
            continue
        key = (item.get("document_id"), item.get("chunk_id"))
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(item)
        if len(deduplicated) == limit:
            break
    return deduplicated


def intelligence_markdown(
    sections: dict[str, list[dict]], evidence: list[dict]
) -> str:
    """Render validated points with citations built from retrieved metadata."""
    evidence_map = {item["evidence_id"]: item for item in evidence}
    lines = []
    for heading, points in sections.items():
        lines.extend([f"### {heading}", ""])
        for point in points:
            citations = ", ".join(
                f"[{item}] {evidence_map[item]['source_filename']}, "
                f"Page {evidence_map[item]['page_number']}"
                for item in point["evidence_ids"]
                if item in evidence_map
            )
            lines.append(f"- {point['text']} {citations}")
        lines.append("")
    return "\n".join(lines).strip()


def evidence_sources(evidence: list[dict]) -> list[dict]:
    """Create deterministic source records from retrieved metadata only."""
    return [
        {
            "evidence_id": item["evidence_id"],
            "source_filename": item["source_filename"],
            "document_id": item.get("document_id", ""),
            "page_number": item["page_number"],
            "chunk_id": item["chunk_id"],
            "extraction_method": item["extraction_method"],
        }
        for item in evidence
    ]


def select_source_for_viewer(source: dict) -> None:
    """Store only validated source metadata for the shared Source Viewer."""
    st.session_state.source_viewer_document_id = source.get("document_id", "")
    st.session_state.source_viewer_page_number = int(source.get("page_number", 1))
    st.session_state.source_viewer_chunk_id = source.get("chunk_id", "")
    st.session_state.source_viewer_citation = dict(source)


def metric_card(label: str, value: str, value_class: str = "") -> None:
    """Render one compact dashboard metric."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{html.escape(label)}</div>
            <div class="metric-value {value_class}">{html.escape(value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Restrained dashboard styling for a professional policy workspace.
st.markdown(
    """
    <style>
    /* Native colors come from .streamlit/config.toml. These rules add hierarchy
       and targeted contrast without maintaining a second theme system. */
    .block-container { max-width: 1500px; padding-top: 1.4rem; padding-bottom: 2rem; }
    .main-header {
        padding: 1.5rem 1.7rem; margin-bottom: 1.2rem;
        border: 1px solid #334352; border-left: 5px solid #7f9b83;
        border-radius: 8px; background: #121c26;
    }
    .eyebrow { color: #a7b3bd; font-size: .72rem; letter-spacing: .14rem; }
    .main-title { margin: .35rem 0; color: #f2f5f7; font-size: 2rem; font-weight: 650; }
    .main-subtitle { color: #b1bdc7; font-size: .92rem; }
    .panel-heading {
        padding: 1rem 1.1rem; margin: .3rem 0 1rem;
        border: 1px solid #334352; border-radius: 7px; background: #121c26;
    }
    .panel-heading h3 { margin: 0 0 .3rem; color: #edf1f4; font-size: 1.05rem; }
    .panel-heading p { margin: 0; color: #a8b4be; font-size: .83rem; }
    .metric-card {
        min-height: 86px; padding: .9rem .35rem; text-align: center;
        border: 1px solid #334352; border-radius: 7px; background: #101923;
    }
    .metric-title { color: #a8b4be; font-size: .67rem; text-transform: uppercase; }
    .metric-value { margin-top: .35rem; color: #eef2f5; font-size: 1.15rem; font-weight: 650; }
    .status-ready { color: #9cc2a1; }
    .status-warning { color: #ddbd78; }
    .file-card {
        padding: 1rem; margin-top: 1rem; overflow-wrap: anywhere;
        border: 1px solid #334352; border-radius: 7px; background: #101923;
    }
    .file-card small { color: #a8b4be; }
    .evidence-card {
        padding: 1rem 1.1rem; margin: .75rem 0;
        border: 1px solid #3a4b5a; border-left: 4px solid #7f9b83;
        border-radius: 7px; background: #101923;
    }
    .evidence-meta { color: #b9c4cc; font-size: .78rem; }
    .response-card {
        padding: 1.25rem 1.35rem; margin: .6rem 0 .8rem;
        border: 1px solid #435564; border-left: 5px solid #88a68c;
        border-radius: 8px; background: #121c26; color: #edf1f5;
        font-size: .96rem; line-height: 1.65;
    }
    .response-badge { color: #a0c5a5; font-weight: 650; }
    .source-reference {
        display: inline-block; padding: .3rem .55rem; margin: .18rem .2rem .18rem 0;
        border: 1px solid #405261; border-radius: 5px;
        background: #101923; color: #d0d8de; font-size: .78rem;
    }

    /* Inputs and uploader surfaces. BaseWeb owns their visible inner shells. */
    div[data-testid="stFileUploader"] {
        padding: .45rem; border: 1px solid #334352; border-radius: 7px;
        background: #0f1821;
    }
    div[data-testid="stFileUploaderDropzone"] {
        background: #121c26; border-color: #425464;
    }
    div[data-testid="stTextArea"] [data-baseweb="textarea"],
    div[data-testid="stTextInput"] [data-baseweb="input"],
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background-color: #0f1821 !important;
        border: 1px solid #425464 !important;
        border-radius: 7px;
    }
    div[data-testid="stTextArea"] [data-baseweb="textarea"]:focus-within,
    div[data-testid="stTextInput"] [data-baseweb="input"]:focus-within,
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within {
        border-color: #7f9b83 !important;
        box-shadow: 0 0 0 1px #7f9b83;
    }
    div[data-testid="stTextArea"] textarea,
    div[data-testid="stTextInput"] input,
    div[data-testid="stSelectbox"] [data-baseweb="select"] * {
        color: #e5e7eb !important;
        caret-color: #e5e7eb;
    }
    div[data-testid="stTextArea"] textarea {
        background-color: #0f1821 !important;
        padding: 1rem !important;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
            "Segoe UI", sans-serif;
        font-size: .94rem;
        line-height: 1.65;
    }
    /* Browsers otherwise dim disabled text even when a color is supplied. */
    div[data-testid="stTextArea"] textarea:disabled,
    div[data-testid="stTextInput"] input:disabled {
        color: #e5e7eb !important;
        -webkit-text-fill-color: #e5e7eb !important;
        opacity: 1 !important;
    }
    div[data-testid="stTextArea"] textarea::-webkit-scrollbar { width: 11px; }
    div[data-testid="stTextArea"] textarea::-webkit-scrollbar-track { background: #0b121a; }
    div[data-testid="stTextArea"] textarea::-webkit-scrollbar-thumb {
        background: #526575; border: 3px solid #0b121a; border-radius: 8px;
    }
    div[data-testid="stTextArea"] textarea { scrollbar-color: #526575 #0b121a; }
    div[data-testid="stTextArea"] textarea::placeholder,
    div[data-testid="stTextInput"] input::placeholder {
        color: #b0bbc4 !important;
        -webkit-text-fill-color: #b0bbc4 !important;
        opacity: 1 !important;
    }
    div[data-testid="stTextArea"] label p,
    div[data-testid="stTextInput"] label p,
    div[data-testid="stFileUploader"] label p,
    div[data-testid="stSelectbox"] label p {
        color: #dce2e7 !important;
    }

    /* Buttons share one restrained green accent; download buttons match actions. */
    .stButton > button,
    div[data-testid="stDownloadButton"] > button {
        color: #e5e7eb !important;
        background: #17242d !important;
        border-color: #526675 !important;
        border-radius: 6px;
    }
    .stButton > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {
        color: #f3f6f4 !important; background: #20302a !important;
        border-color: #7f9b83 !important;
    }
    .stButton > button:disabled,
    div[data-testid="stDownloadButton"] > button:disabled {
        color: #9ba7b1 !important;
        -webkit-text-fill-color: #9ba7b1 !important;
        background-color: #141d25 !important;
        border-color: #303f4c !important;
        opacity: 1 !important;
    }

    /* Navigation, disclosures, alerts, progress, and native metric containers. */
    button[data-baseweb="tab"] { color: #b8c2ca !important; }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #f0f4f1 !important; border-bottom-color: #7f9b83 !important;
    }
    div[data-testid="stExpander"] details {
        background: #101923; border-color: #334352; border-radius: 7px;
    }
    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary p { color: #e1e6ea !important; }
    div[data-testid="stAlert"] {
        background: #121c26; border: 1px solid #3a4b5a; color: #e3e8ec;
    }
    div[data-testid="stProgress"] > div > div > div { background-color: #7f9b83 !important; }
    div[data-testid="stMetric"] {
        padding: .75rem; background: #101923; border: 1px solid #334352;
        border-radius: 7px;
    }
    div[data-testid="stMetricLabel"] p { color: #aeb9c2 !important; }
    div[data-testid="stMetricValue"] { color: #eef2f5 !important; }
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        border-color: #334352 !important; background: #101923;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="main-header">
        <div class="eyebrow">DOCUMENT INTELLIGENCE / POLICY ANALYSIS</div>
        <div class="main-title">🛡️ Research Intelligence Assistant</div>
        <div class="main-subtitle">
            Secure workspace for inspecting and preparing policy documents for analysis
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Defaults keep the dashboard stable before an upload or after an error.
uploaded_files = []
pages: list[dict] = []
extracted_text = ""
text_chunks: list[dict] = []
corpus_documents: list[dict] = []
processing_status = "Awaiting corpus"
index_error = ""
vector_index = None
keyword_index = None
corpus_key = ""
native_page_count = 0
ocr_page_count = 0
ocr_failures: list[dict] = []
indexed_document_ids: set[str] = set()
processing_diagnostics = {
    "document_cache": [],
    "processed_this_run": [],
    "reused_this_run": [],
    "embedding_batches": 0,
    "embedding_cache": [],
    "elapsed_seconds": 0.0,
    "index_reused": False,
    "duplicate_uploads": [],
}

left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    st.markdown(
        """
        <div class="panel-heading">
            <h3>Document Control</h3>
            <p>Upload and inspect a searchable policy document corpus.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Local Document Cache", expanded=False):
        local_cache = cache_statistics()
        st.caption(
            f"{local_cache['documents']} cached documents · "
            f"{local_cache['bytes'] / (1024 * 1024):.1f} MB"
        )
        confirm_cache_clear = st.checkbox(
            "Confirm removal of locally cached document text and embeddings",
            key="confirm-local-cache-clear",
        )
        if st.button(
            "Clear local document cache",
            disabled=not confirm_cache_clear,
            use_container_width=True,
            key="clear-local-document-cache",
        ):
            try:
                removed_documents = clear_local_cache()
                for state_key in (
                    "document_processing_cache",
                    "document_embedding_cache",
                    "upload_hash_cache",
                    "retry_document_hashes",
                    "corpus_index_state",
                    "last_rag_result",
                    "executive_summary_cache",
                    "executive_summary_results",
                    "executive_summary_local",
                    "policy_comparison_cache",
                    "conflict_analysis_cache",
                    "timeline_analysis_cache",
                    "entity_analysis_cache",
                    "source_viewer_citation",
                ):
                    st.session_state.pop(state_key, None)
                st.success(
                    f"Cleared local cache for {removed_documents} document"
                    f"{'s' if removed_documents != 1 else ''}."
                )
                st.caption(
                    "PDFs still present in the uploader will be processed and cached again."
                )
            except Exception:
                logger.exception("Local document cache clearing failed")
                st.error("The local document cache could not be cleared safely.")
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        help="PDF files only. Scanned documents may require OCR.",
        accept_multiple_files=True,
    )

    if uploaded_files:
        run_started_at = time.perf_counter()
        st.session_state.setdefault("document_processing_cache", {})
        st.session_state.setdefault("document_embedding_cache", {})
        st.session_state.setdefault("upload_hash_cache", {})
        st.session_state.setdefault("retry_document_hashes", set())
        document_cache = st.session_state.document_processing_cache
        embedding_cache = st.session_state.document_embedding_cache
        upload_hash_cache = st.session_state.upload_hash_cache
        retry_document_hashes = st.session_state.retry_document_hashes
        corpus_members = []
        progress_placeholder = st.empty()
        status_placeholder = st.empty()

        for document_position, uploaded_file in enumerate(uploaded_files, start=1):
            document_record = None
            upload_id = getattr(uploaded_file, "file_id", None)
            pdf_bytes = None
            try:
                if upload_id and upload_id in upload_hash_cache:
                    content_hash = upload_hash_cache[upload_id]
                else:
                    pdf_bytes = uploaded_file.getvalue()
                    content_hash = hashlib.sha256(pdf_bytes).hexdigest()
                    if upload_id:
                        upload_hash_cache[upload_id] = content_hash
            except MemoryError:
                logger.exception("MemoryError reading uploaded PDF %s", uploaded_file.name)
                content_hash = hashlib.sha256(
                    f"unreadable:{upload_id}:{uploaded_file.name}:{uploaded_file.size}".encode()
                ).hexdigest()
                document_record = failed_document_record(
                    uploaded_file.name, content_hash, uploaded_file.size, "Upload read"
                )
            except Exception:
                logger.exception("Failed to read uploaded PDF %s", uploaded_file.name)
                content_hash = hashlib.sha256(
                    f"unreadable:{upload_id}:{uploaded_file.name}:{uploaded_file.size}".encode()
                ).hexdigest()
                document_record = failed_document_record(
                    uploaded_file.name, content_hash, uploaded_file.size, "Upload read"
                )

            document_id = content_hash
            corpus_members.append((uploaded_file.name, content_hash))
            cache_source = "miss"
            if document_record is not None:
                document_cache[content_hash] = document_record
                if upload_id:
                    upload_hash_cache[upload_id] = content_hash

            retry_requested = content_hash in retry_document_hashes
            if retry_requested:
                document_cache.pop(content_hash, None)
                embedding_cache.pop(content_hash, None)
                st.session_state.pop("corpus_index_state", None)

            if document_record is None and not retry_requested and content_hash in document_cache:
                document_record = relabel_document(
                    document_cache[content_hash], uploaded_file.name, document_id
                )
                cache_source = "session"
            elif document_record is None and not retry_requested:
                document_record = load_document(content_hash, uploaded_file.name)
                if document_record is not None:
                    document_cache[content_hash] = document_record
                    cache_source = "disk"

            if document_record is None:
                processing_status = "Processing"
                processing_diagnostics["processed_this_run"].append(uploaded_file.name)
                progress_bar = progress_placeholder.progress(
                    (document_position - 1) / len(uploaded_files),
                    text=(
                        f"Processing document {document_position} of {len(uploaded_files)}: "
                        f"{uploaded_file.name}"
                    ),
                )
                if pdf_bytes is None:
                    pdf_bytes = uploaded_file.getvalue()

                def update_extraction_progress(
                    page_number: int,
                    total_pages: int,
                    stage: str,
                    current_document: int = document_position,
                    current_name: str = uploaded_file.name,
                ) -> None:
                    """Show bounded page extraction/OCR progress for the corpus."""
                    page_fraction = page_number / max(total_pages, 1)
                    overall = min(
                        ((current_document - 1) + page_fraction) / len(uploaded_files),
                        0.99,
                    )
                    stage_detail = (
                        f"{stage}: {current_name}"
                        if stage == "Creating chunks"
                        else f"{stage}: {current_name} — page {page_number} / {total_pages}"
                    )
                    status_placeholder.info(stage_detail)
                    progress_bar.progress(overall, text=f"{stage}: {current_name}")

                try:
                    safeguard_warning = ""
                    if uploaded_file.size > MAX_FILE_SIZE_BYTES:
                        raise ValueError(
                            f"File exceeds the {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB limit."
                        )
                    try:
                        with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf_check:
                            page_count = len(pdf_check)
                    except Exception:
                        logger.exception("PDF open/read validation failed for %s", uploaded_file.name)
                        raise
                    if page_count > MAX_PAGES_PER_DOCUMENT:
                        raise ValueError(
                            f"Document has {page_count} pages; the supported limit is "
                            f"{MAX_PAGES_PER_DOCUMENT}."
                        )
                    if (
                        uploaded_file.size >= LARGE_FILE_WARNING_BYTES
                        or page_count >= LARGE_PAGE_WARNING
                    ):
                        safeguard_warning = (
                            f"Large document: {uploaded_file.size / (1024 * 1024):.1f} MB · "
                            f"{page_count} pages"
                        )
                        status_placeholder.warning(
                            f"Large document detected: {uploaded_file.name} · "
                            f"{uploaded_file.size / (1024 * 1024):.1f} MB · {page_count} pages"
                        )

                    document_pages, _, document_chunks = process_pdf(
                        pdf_bytes,
                        uploaded_file.name,
                        document_id=document_id,
                        progress_callback=update_extraction_progress,
                        include_preview=False,
                    )
                    document_record = {
                        "filename": uploaded_file.name,
                        "document_id": document_id,
                        "content_hash": content_hash,
                        "file_size": uploaded_file.size,
                        "safeguard_warning": safeguard_warning,
                        "pages": document_pages,
                        "chunks": document_chunks,
                        "status": "Processed" if document_chunks else "Failed",
                        "error": "" if document_chunks else "No readable text was found.",
                    }
                    if document_chunks:
                        try:
                            save_document(content_hash, document_record)
                        except (OSError, ValueError):
                            # Streamlit Cloud storage is ephemeral and may be read-only.
                            # The processed document remains available in this session.
                            logger.warning(
                                "Document cache persistence unavailable for %s",
                                uploaded_file.name,
                            )
                except (fitz.FileDataError, fitz.EmptyFileError, ValueError):
                    logger.exception("Document validation/extraction failed for %s", uploaded_file.name)
                    document_record = failed_document_record(
                        uploaded_file.name, document_id, uploaded_file.size, "PDF extraction"
                    )
                except MemoryError:
                    logger.exception("MemoryError processing document %s", uploaded_file.name)
                    document_record = failed_document_record(
                        uploaded_file.name, document_id, uploaded_file.size, "Memory allocation"
                    )
                except Exception:
                    logger.exception("Unexpected document processing failure for %s", uploaded_file.name)
                    document_record = failed_document_record(
                        uploaded_file.name, document_id, uploaded_file.size, "Document processing"
                    )
                document_cache[content_hash] = document_record
                retry_document_hashes.discard(content_hash)
                pdf_bytes = None
                gc.collect()
            else:
                processing_diagnostics["reused_this_run"].append(uploaded_file.name)

            document_record["runtime_cache_status"] = (
                "Loaded from cache"
                if cache_source in {"session", "disk"}
                else "Processing complete"
            )

            processing_diagnostics["document_cache"].append(
                {
                    "filename": uploaded_file.name,
                    "hash": content_hash,
                    "cache": cache_source,
                }
            )
            corpus_documents.append(document_record)

        progress_placeholder.empty()
        status_placeholder.empty()
        corpus_documents, duplicate_documents = deduplicate_documents_by_content(
            corpus_documents
        )
        processing_diagnostics["duplicate_uploads"] = [
            document["filename"] for document in duplicate_documents
        ]
        if duplicate_documents:
            duplicate_names = ", ".join(
                sorted({document["filename"] for document in duplicate_documents})
            )
            st.info(
                "Duplicate PDF content was uploaded more than once. Only the first "
                f"copy was indexed: {duplicate_names}"
            )
        corpus_members = [
            (document["filename"], document["document_id"])
            for document in corpus_documents
        ]
        corpus_key = "corpus-v3:" + hashlib.sha256(
            repr(sorted(corpus_members)).encode("utf-8")
        ).hexdigest()

        successful_documents = [
            document
            for document in corpus_documents
            if document["chunks"] and document["status"] == "Processed"
        ]
        pages = [page for document in corpus_documents for page in document["pages"]]
        text_chunks = [
            chunk for document in successful_documents for chunk in document["chunks"]
        ]
        preview_parts = []
        preview_length = 0
        for document in successful_documents:
            for page in document["pages"]:
                if not page["text"] or preview_length >= MAX_PREVIEW_CHARACTERS:
                    continue
                section = (
                    f"[{document['filename']} · Page {page['page_number']}]\n{page['text']}"
                )
                remaining = MAX_PREVIEW_CHARACTERS - preview_length
                preview_parts.append(section[:remaining])
                preview_length += min(len(section), remaining)
        extracted_text = "\n\n".join(preview_parts)
        native_page_count = sum(
            page["extraction_method"] == "Native text" for page in pages
        )
        ocr_page_count = sum(page["extraction_method"] == "OCR" for page in pages)
        ocr_failures = [page for page in pages if page["ocr_error"]]

        indexed_document_ids = set()
        if text_chunks:
            existing_index_state = st.session_state.get("corpus_index_state")
            if existing_index_state and existing_index_state["key"] == corpus_key:
                vector_index = existing_index_state["index"]
                keyword_index = existing_index_state.get("keyword_index")
                if keyword_index is None:
                    try:
                        keyword_index = build_bm25_index(vector_index.chunks)
                        existing_index_state["keyword_index"] = keyword_index
                    except Exception:
                        logger.exception(
                            "BM25 index rebuild failed; FAISS retrieval remains available"
                        )
                indexed_document_ids = {
                    document["document_id"] for document in successful_documents
                }
                processing_diagnostics["index_reused"] = True
            else:
                document_parts = []
                for document in sorted(
                    successful_documents, key=lambda item: item["document_id"]
                ):
                    try:
                        fingerprint = chunk_fingerprint(document["chunks"])
                        cached_embedding = embedding_cache.get(document["document_id"])
                        if (
                            cached_embedding
                            and cached_embedding["fingerprint"] == fingerprint
                        ):
                            embeddings = cached_embedding["matrix"]
                            embedding_source = "session"
                        else:
                            embeddings = load_embeddings(
                                document["document_id"], document["chunks"]
                            )
                            embedding_source = "disk" if embeddings is not None else "miss"
                        if embeddings is None:
                            def update_embedding_progress(batch_number, total_batches):
                                processing_diagnostics["embedding_batches"] += 1
                                status_placeholder.info(
                                    f"{document['filename']} — Embedding batch "
                                    f"{batch_number} / {total_batches}"
                                )

                            embeddings = embed_chunks_batched(
                                document["chunks"],
                                get_embedding_model(),
                                batch_size=recommended_embedding_batch_size(
                                    len(document["chunks"])
                                ),
                                progress_callback=update_embedding_progress,
                            )
                            try:
                                save_embeddings(
                                    document["document_id"],
                                    document["chunks"],
                                    embeddings,
                                )
                            except (OSError, ValueError):
                                logger.warning(
                                    "Embedding cache persistence unavailable for %s",
                                    document["filename"],
                                )
                        embedding_cache[document["document_id"]] = {
                            "fingerprint": fingerprint,
                            "matrix": embeddings,
                        }
                        processing_diagnostics["embedding_cache"].append(
                            {
                                "filename": document["filename"],
                                "cache": embedding_source,
                            }
                        )
                        document_parts.append((document["chunks"], embeddings))
                        indexed_document_ids.add(document["document_id"])
                    except MemoryError:
                        logger.exception("MemoryError embedding %s", document["filename"])
                        document["status"] = "Failed"
                        document["error"] = "Processing failed for this document."
                        document["failure_stage"] = "Embedding generation"
                        document_cache[document["document_id"]] = document
                        embedding_cache.pop(document["document_id"], None)
                        gc.collect()
                    except Exception:
                        logger.exception("Embedding generation failed for %s", document["filename"])
                        document["status"] = "Failed"
                        document["error"] = "Processing failed for this document."
                        document["failure_stage"] = "Embedding generation"
                        document_cache[document["document_id"]] = document
                        embedding_cache.pop(document["document_id"], None)
                        gc.collect()

                if document_parts:
                  try:
                    status_placeholder.info("Updating FAISS corpus index")
                    vector_index = build_vector_index_from_parts(document_parts)
                    try:
                        keyword_index = build_bm25_index(vector_index.chunks)
                    except Exception:
                        keyword_index = None
                        logger.exception(
                            "BM25 index creation failed; FAISS retrieval remains available"
                        )
                    st.session_state.corpus_index_state = {
                        "key": corpus_key,
                        "index": vector_index,
                        "keyword_index": keyword_index,
                    }
                    status_placeholder.empty()
                  except MemoryError:
                    logger.exception("MemoryError updating the FAISS corpus index")
                    processing_status = "Failed"
                    index_error = "The corpus index exceeded available memory."
                  except Exception:
                    logger.exception("FAISS corpus index update failed")
                    processing_status = "Failed"
                    index_error = "The FAISS corpus index could not be updated."
            if vector_index is not None:
                processing_status = (
                    "Partial"
                    if len(indexed_document_ids) < len(corpus_documents)
                    else "Indexed"
                )
        else:
            processing_status = "Failed"

        # Embedding isolation may have marked a previously extracted document
        # failed. Report only chunks that actually reached the current index.
        text_chunks = [
            chunk
            for document in corpus_documents
            if document["document_id"] in indexed_document_ids
            for chunk in document["chunks"]
        ]

        processing_diagnostics["elapsed_seconds"] = time.perf_counter() - run_started_at

    metric_columns = st.columns(4)
    metric_values = [
        ("Documents", str(len(corpus_documents)), ""),
        ("Total pages", str(len(pages)), ""),
        ("Text chunks", str(len(text_chunks)), ""),
        ("Corpus status", processing_status,
         "status-ready" if processing_status in {"Indexed", "Partial"} else "status-warning"),
    ]
    for column, (label, value, value_class) in zip(metric_columns, metric_values):
        with column:
            metric_card(label, value, value_class)

    if index_error:
        st.error(
            "Corpus text was extracted, but the semantic index could not be created. "
            "Check the model installation or network connection, then try again. "
            f"Details: {index_error}"
        )
    elif uploaded_files and not extracted_text:
        tesseract_available, tesseract_message = get_tesseract_status()
        if not tesseract_available:
            st.error(
                f"OCR could not run: {tesseract_message} Install Tesseract as a "
                "system package and ensure it is available on PATH."
            )
        else:
            st.warning("No readable text was found after native extraction and OCR.")
    elif processing_status == "Partial":
        st.warning("Available documents were indexed; one or more documents failed.")
    elif vector_index is not None:
        st.success("Policy corpus processed and FAISS index created successfully.")

    if ocr_failures and extracted_text:
        failed_pages = ", ".join(
            f"{page['source_filename']} page {page['page_number']}"
            for page in ocr_failures
        )
        tesseract_available, tesseract_message = get_tesseract_status()
        if not tesseract_available:
            st.warning(
                f"OCR was needed on {failed_pages}, but {tesseract_message} "
                "Other readable pages were retained. Install Tesseract as a system "
                "package and ensure it is available on PATH."
            )
        else:
            st.warning(
                f"OCR could not complete on {failed_pages}. Other readable "
                "pages were retained and indexed."
            )

    extraction_columns = st.columns(2)
    with extraction_columns[0]:
        metric_card("Native text pages", str(native_page_count))
    with extraction_columns[1]:
        metric_card("OCR pages", str(ocr_page_count))

    st.markdown("##### Vector Index")
    index_columns = st.columns(3)
    with index_columns[0]:
        metric_card("Embedding model", "all-MiniLM-L6-v2")
    with index_columns[1]:
        metric_card("Vectors indexed", str(vector_index.index.ntotal if vector_index else 0))
    with index_columns[2]:
        metric_card("Index status", "Online" if vector_index else "Offline",
                    "status-ready" if vector_index else "status-warning")
    st.caption(f"Model: {EMBEDDING_MODEL_NAME}")

    if corpus_documents:
        st.markdown("##### Corpus Documents")
        for document in corpus_documents:
            document_pages = document["pages"]
            native_count = sum(
                page["extraction_method"] == "Native text" for page in document_pages
            )
            ocr_count = sum(
                page["extraction_method"] == "OCR" for page in document_pages
            )
            safe_name = html.escape(document["filename"])
            if document["status"] != "Processed":
                displayed_document_status = "Failed"
            elif document.get("runtime_cache_status") == "Loaded from cache":
                displayed_document_status = "Loaded from cache"
            elif document["document_id"] in indexed_document_ids:
                displayed_document_status = "Indexed"
            else:
                displayed_document_status = "Processing"
            status_class = (
                "status-ready" if document["status"] == "Processed" else "status-warning"
            )
            st.markdown(
                f"""
                <div class="file-card">
                    <strong>{safe_name}</strong><br>
                    <small>ID {document['document_id'][:10]} · {len(document_pages)} pages ·
                    Native {native_count} · OCR {ocr_count} · {len(document['chunks'])} chunks</small><br>
                    <span class="{status_class}">{displayed_document_status}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if document["error"]:
                st.error(document["error"])
                st.caption(
                    f"Failed stage: {document.get('failure_stage', 'Document processing')}"
                )
                if st.button(
                    "Retry failed document",
                    key=f"retry-{document['document_id']}",
                    use_container_width=True,
                ):
                    st.session_state.retry_document_hashes.add(document["document_id"])
                    st.session_state.pop("corpus_index_state", None)
                    st.rerun()
            if document.get("safeguard_warning"):
                st.caption(document["safeguard_warning"])

        with st.expander("Corpus Processing Diagnostics", expanded=False):
            st.write(f"Corpus hash: `{corpus_key}`")
            st.write(f"Documents processed this run: {len(processing_diagnostics['processed_this_run'])}")
            st.write(f"Documents reused from cache: {len(processing_diagnostics['reused_this_run'])}")
            st.write(
                f"Duplicate uploads excluded: {len(processing_diagnostics['duplicate_uploads'])}"
            )
            st.write(f"Chunks in corpus: {len(text_chunks)}")
            st.write(f"Embedding batches generated: {processing_diagnostics['embedding_batches']}")
            st.write(
                "FAISS index: "
                + ("reused from session" if processing_diagnostics["index_reused"] else "updated this run")
            )
            st.write(f"Processing time this run: {processing_diagnostics['elapsed_seconds']:.2f} seconds")
            st.markdown("**Document cache**")
            for cache_item in processing_diagnostics["document_cache"]:
                st.caption(
                    f"{cache_item['filename']} · {cache_item['hash'][:12]} · "
                    f"{cache_item['cache']}"
                )
            if processing_diagnostics["embedding_cache"]:
                st.markdown("**Embedding cache**")
                for cache_item in processing_diagnostics["embedding_cache"]:
                    st.caption(f"{cache_item['filename']} · {cache_item['cache']}")

with right_col:
    st.markdown(
        """
        <div class="panel-heading">
            <h3>Query Assistant</h3>
            <p>Search the indexed policy corpus for semantically relevant evidence.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    searchable_documents = [
        document
        for document in corpus_documents
        if document["document_id"] in indexed_document_ids
    ]
    filter_labels = {"Search All Documents": None}
    for document in searchable_documents:
        label = f"{document['filename']} · {document['document_id'][:8]}"
        filter_labels[label] = document["document_id"]
    selected_filter_label = st.selectbox(
        "Retrieval scope",
        options=list(filter_labels),
        disabled=vector_index is None,
    )
    selected_document_id = filter_labels[selected_filter_label]
    user_query = st.text_input(
        "Policy question",
        placeholder="Example: What are the stated eligibility requirements?",
        disabled=vector_index is None,
    )
    search_submitted = st.button(
        "Search document", use_container_width=True, disabled=vector_index is None
    )
    st.session_state.setdefault("query_history", [])

    if search_submitted:
        if not user_query.strip():
            st.warning("Enter a question before submitting.")
        else:
            st.session_state.pop("last_rag_result", None)
            query_status = st.status("Retrieving relevant evidence...", expanded=True)
            try:
                retrieved_results = hybrid_search(
                    user_query,
                    vector_index,
                    get_embedding_model(),
                    keyword_index,
                    top_k=TOP_K,
                    document_id=selected_document_id,
                )
                if not retrieved_results:
                    query_status.update(
                        label="No relevant evidence was retrieved.", state="error"
                    )
                else:
                    retrieved_results = assign_evidence_ids(retrieved_results)
                    selected_chunks = select_evidence_for_generation(retrieved_results)
                    answer = ""
                    generation_error = ""
                    generation_error_type = ""
                    evidence_ids = []
                    sufficient_evidence = False
                    preliminary_confidence = calculate_evidence_confidence(
                        user_query,
                        retrieved_results,
                        selected_chunks,
                        len(selected_chunks),
                    )
                    try:
                        if preliminary_confidence.level == "INSUFFICIENT":
                            answer = INSUFFICIENT_EVIDENCE_MESSAGE
                            query_status.update(
                                label="Insufficient supporting evidence.",
                                state="complete",
                                expanded=False,
                            )
                        else:
                            query_status.update(label="Generating grounded response...")
                            grounded_answer = generate_grounded_answer(
                                user_query, selected_chunks
                            )
                            answer = grounded_answer.answer
                            evidence_ids = grounded_answer.evidence_ids
                            sufficient_evidence = grounded_answer.sufficient_evidence
                            query_status.update(
                                label=(
                                    "Grounded response complete."
                                    if sufficient_evidence
                                    else "Insufficient supporting evidence."
                                ),
                                state="complete",
                                expanded=False,
                            )
                    except MissingAPIKeyError as error:
                        generation_error = str(error)
                        generation_error_type = "missing_api_key"
                        query_status.update(
                            label="Gemini API key is not configured.",
                            state="error",
                            expanded=False,
                        )
                    except MalformedStructuredResponseError as error:
                        generation_error = str(error)
                        generation_error_type = "malformed_response"
                        query_status.update(
                            label="Evidence retrieved; Gemini response could not be validated.",
                            state="error",
                            expanded=False,
                        )
                    except GeminiRequestError as error:
                        generation_error = str(error)
                        generation_error_type = "api_request_failed"
                        query_status.update(
                            label="Evidence retrieved; Gemini request failed.",
                            state="error",
                            expanded=False,
                        )
                    except RAGServiceError as error:
                        generation_error = str(error)
                        generation_error_type = "generation_failed"
                        query_status.update(
                            label="Evidence retrieved; generation failed.",
                            state="error",
                            expanded=False,
                        )

                    verified_sources = build_verified_sources(
                        retrieved_results, evidence_ids
                    )
                    supporting_chunks = get_chunks_by_evidence_id(
                        retrieved_results, evidence_ids
                    )
                    evidence_confidence = calculate_evidence_confidence(
                        user_query,
                        retrieved_results,
                        supporting_chunks,
                        len(selected_chunks),
                        sufficient_evidence=sufficient_evidence,
                    )
                    rag_result = {
                        "result_version": 5,
                        "corpus_key": corpus_key,
                        "question": user_query.strip(),
                        "retrieval_scope": selected_filter_label,
                        "answer": answer,
                        "sources": verified_sources,
                        "retrieved_results": retrieved_results,
                        "selected_evidence_ids": [
                            chunk["evidence_id"] for chunk in selected_chunks
                        ],
                        "cited_evidence_ids": evidence_ids,
                        "evidence_strength": calculate_evidence_strength(
                            supporting_chunks
                        ),
                        "evidence_confidence": evidence_confidence.level,
                        "evidence_confidence_reason": evidence_confidence.reason,
                        "sufficient_evidence": sufficient_evidence,
                        "generation_error": generation_error,
                        "generation_error_type": generation_error_type,
                        "timestamp": datetime.now().astimezone().strftime(
                            "%Y-%m-%d %H:%M"
                        ),
                    }
                    st.session_state.last_rag_result = rag_result
                    if answer:
                        st.session_state.query_history.insert(
                            0,
                            {
                                "question": rag_result["question"],
                                "answer": answer,
                                "sources": verified_sources,
                                "evidence_strength": rag_result["evidence_strength"],
                                "evidence_confidence": rag_result["evidence_confidence"],
                                "evidence_confidence_reason": rag_result[
                                    "evidence_confidence_reason"
                                ],
                                "timestamp": rag_result["timestamp"],
                                "corpus_key": corpus_key,
                            },
                        )
                        st.session_state.query_history = st.session_state.query_history[:10]
            except Exception:
                logger.exception("Hybrid retrieval failed")
                query_status.update(label="Retrieval failed.", state="error")
                st.error(
                    "Retrieval could not be completed. The indexed corpus remains available."
                )

    active_result = st.session_state.get("last_rag_result")
    if active_result and (
        active_result.get("result_version") != 5
        or active_result["corpus_key"] != corpus_key
    ):
        active_result = None

    if active_result and active_result["answer"]:
        st.markdown("### INTELLIGENCE RESPONSE")
        # Streamlit renders Markdown safely here; arbitrary model HTML is not enabled.
        with st.container(border=True):
            st.markdown(active_result["answer"], unsafe_allow_html=False)
        response_label = (
            "Grounded Response ✓"
            if active_result["sufficient_evidence"]
            else "Insufficient evidence"
        )
        st.markdown(
            f"**{response_label}** · Cited Sources: **{len(active_result['sources'])}**"
        )
        st.markdown(
            f"**Evidence Confidence: {active_result['evidence_confidence'].title()}**"
        )
        st.caption(
            active_result["evidence_confidence_reason"]
            + " This is a qualitative evidence assessment, not a probability or factual certainty."
        )
        if active_result["sources"]:
            st.markdown("**Cited Sources**")
            for source in active_result["sources"]:
                extraction_method = (
                    "Native"
                    if source["extraction_method"] == "Native text"
                    else source["extraction_method"]
                )
                st.markdown(
                    f"- `{source['source_filename']} · ID {source['document_id'][:8]} · "
                    f"Page {source['page_number']} · "
                    f"{source['chunk_id']} · {extraction_method}`"
                )
                source_chunk = next(
                    (
                        item
                        for item in active_result["retrieved_results"]
                        if item.get("document_id") == source.get("document_id")
                        and item.get("chunk_id") == source.get("chunk_id")
                    ),
                    {},
                )
                st.button(
                    "Inspect cited page",
                    key=f"view-query-{source.get('evidence_id', '')}-{source['document_id']}-{source['chunk_id']}",
                    on_click=select_source_for_viewer,
                    args=({**source, "text": source_chunk.get("text", "")},),
                )
        query_report = build_markdown_report(
            "Individual Intelligence Response",
            corpus_documents,
            active_result["answer"],
            active_result["evidence_strength"],
            active_result["sources"],
            active_result["retrieved_results"],
            question=active_result["question"],
            embedding_model=EMBEDDING_MODEL_NAME,
        )
        st.download_button(
            "Export Analysis Report",
            data=query_report,
            file_name="defence-policy-intelligence-response.md",
            mime="text/markdown",
            use_container_width=True,
            key="download-query-response",
        )
    elif active_result and active_result["generation_error"]:
        st.warning(active_result["generation_error"])
        if active_result.get("generation_error_type") == "missing_api_key":
            st.caption(
                "Configure GEMINI_API_KEY in local environment settings or Streamlit "
                "Secrets to enable grounded generation."
            )
        else:
            st.caption(
                "Retrieved semantic evidence remains available below for review."
            )

    retrieved_results = active_result["retrieved_results"] if active_result else []
    if retrieved_results:
        st.markdown("### RETRIEVED EVIDENCE")
        st.caption(
            "Top fused semantic and keyword matches, including evidence not cited "
            "in the answer."
        )
        selected_ids = set(active_result["selected_evidence_ids"])
        for result in retrieved_results:
            safe_source = html.escape(result["source_filename"])
            safe_chunk_id = html.escape(result["chunk_id"])
            retrieval_source = html.escape(result.get("retrieval_source", "Semantic"))
            semantic_score = result.get("semantic_similarity", result.get("similarity"))
            semantic_label = f"{semantic_score:.3f}" if semantic_score is not None else "n/a"
            keyword_score = result.get("keyword_score")
            keyword_label = f"{keyword_score:.3f}" if keyword_score is not None else "n/a"
            fusion_label = f"{result.get('hybrid_score', 0.0):.3f}"
            selection_note = (
                "Selected for Gemini context"
                if result["evidence_id"] in selected_ids
                else "Weak match · not sent to Gemini"
            )
            st.markdown(
                f"""
                <div class="evidence-card">
                    <strong>{result['evidence_id']}</strong><br>
                    <span class="evidence-meta">
                        {safe_source} · Document {result['document_id'][:8]} ·
                        Page {result['page_number']} · {safe_chunk_id} ·
                        {result['extraction_method']} · Retrieval {retrieval_source} ·
                        Semantic {semantic_label} · BM25 {keyword_label} · Fusion {fusion_label}<br>
                        {selection_note}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write(result["text"])
            st.divider()

        with st.expander("Retrieval Diagnostics", expanded=False):
            st.markdown(f"**Query:** {active_result['question']}")
            st.write(f"Retrieval scope: {active_result['retrieval_scope']}")
            st.write(f"Chunks retrieved: {len(retrieved_results)}")
            st.write(
                "Chunks selected as candidate support: "
                f"{len(active_result['selected_evidence_ids'])}"
            )
            st.write(
                f"Chunks cited in final answer: {len(active_result['cited_evidence_ids'])}"
            )
            st.write(f"Hybrid retrieval scores: " + ", ".join(
                f"{result['evidence_id']}={result.get('hybrid_score', 0.0):.3f} "
                f"({result.get('retrieval_source', 'Semantic')})"
                for result in retrieved_results
            ))
            st.write(f"Embedding model: `{EMBEDDING_MODEL_NAME}`")
            st.write(f"Top-k: {TOP_K}")

    current_query_history = [
        item
        for item in st.session_state.query_history
        if item.get("corpus_key") == corpus_key
    ]
    if current_query_history:
        with st.expander("Recent Queries"):
            for history_item in current_query_history:
                st.caption(history_item["timestamp"])
                st.markdown(f'**Question:** {history_item["question"]}')
                st.markdown(history_item["answer"], unsafe_allow_html=False)
                source_summary = ", ".join(
                    f'{source["source_filename"]} · Page {source["page_number"]} · '
                    f'{source.get("chunk_id", "chunk unavailable")}'
                    for source in history_item["sources"]
                )
                st.caption(
                    f"Evidence confidence: {history_item.get('evidence_confidence', 'Not recorded').title()} · "
                    f"Cited sources: {source_summary or 'None'}"
                )
                st.divider()

    st.subheader("Document Preview")
    if extracted_text:
        st.text_area(
            "Extracted page-aware text",
            value=extracted_text,
            height=390,
            disabled=True,
            help="Each section begins with its original PDF page number.",
        )
        with st.expander("Preview first text chunk"):
            st.code(text_chunks[0]["text"], language=None, wrap_lines=True)
    elif uploaded_files:
        st.info("No corpus text preview is available. Review document processing status.")
    else:
        st.info("Upload one or more PDFs in Document Control to begin.")

    if pages:
        text_page_count = sum(bool(page["text"]) for page in pages)
        st.caption(
            f"Text extracted from {text_page_count} of {len(pages)} pages · "
            f"Native: {native_page_count} · OCR: {ocr_page_count} · "
            "Chunk size: 1,000 characters · Overlap: 200 characters"
        )

    st.divider()
    (
        summary_tab,
        comparison_tab,
        conflict_tab,
        timeline_tab,
        entities_tab,
        dashboard_tab,
        overview_tab,
    ) = st.tabs(
        [
            "Executive Summary",
            "Policy Comparison",
            "Conflict Analysis",
            "Policy Timeline",
            "Entities & Facts",
            "Intelligence Dashboard",
            "Corpus Overview",
        ]
    )
    st.session_state.setdefault("executive_summary_cache", {})
    st.session_state.setdefault("policy_comparison_cache", {})
    st.session_state.setdefault("conflict_analysis_cache", {})
    st.session_state.setdefault("timeline_analysis_cache", {})
    st.session_state.setdefault("entity_analysis_cache", {})
    gemini_available, gemini_diagnostic = get_gemini_availability()

    with summary_tab:
        st.markdown("### Executive Summary")
        summary_labels = {"Entire Corpus": None}
        for document in searchable_documents:
            summary_labels[
                f"{document['filename']} · {document['document_id'][:8]}"
            ] = document["document_id"]
        summary_scope = st.selectbox(
            "Summary scope",
            options=list(summary_labels),
            disabled=vector_index is None,
            key="summary_scope",
        )
        summary_document_id = summary_labels[summary_scope]
        scoped_chunks = [chunk for chunk in text_chunks
                         if summary_document_id is None or chunk.get("document_id") == summary_document_id]
        try:
            summary_key = summary_cache_key(scoped_chunks)
        except RAGServiceError:
            summary_key = f"local-summary:{corpus_key}:{summary_document_id or 'all'}"
        cached_summary = st.session_state.executive_summary_cache.get(summary_key)
        summary_ready = bool(
            vector_index is not None
            and searchable_documents
            and text_chunks
        )
        if not gemini_available:
            st.warning(f"Gemini unavailable: {gemini_diagnostic}")
        summary_action = st.button(
            "Regenerate Summary" if cached_summary else "Generate Summary",
            disabled=not summary_ready,
            use_container_width=True,
        )
        st.session_state.setdefault("executive_summary_results", {})
        st.session_state.setdefault("executive_summary_local", {})
        if summary_action:
            st.session_state.executive_summary_cache.pop(summary_key, None)
            st.session_state.executive_summary_results.pop(summary_key, None)
            st.session_state.executive_summary_local.pop(summary_key, None)
            with st.status("Preparing grounded summary evidence...", expanded=True) as status:
                try:
                    scoped_chunks = [
                        chunk
                        for chunk in text_chunks
                        if summary_document_id is None
                        or chunk.get("document_id") == summary_document_id
                    ]

                    def update_summary_progress(group_number, total_groups, stage):
                        status.update(
                            label=f"{stage}: {group_number} / {total_groups}"
                        )

                    generated = cached_executive_summary(
                        scoped_chunks, st.session_state.executive_summary_results,
                        progress_callback=update_summary_progress,
                    )
                    cached_summary = {
                        "sections": generated.sections,
                        "evidence": generated.cited_evidence,
                        "cited_evidence": generated.cited_evidence,
                        "evidence_strength": generated.evidence_strength,
                        "scope": summary_scope,
                    }
                    st.session_state.executive_summary_cache[summary_key] = cached_summary
                    status.update(label="Executive summary complete.", state="complete", expanded=False)
                except NoSummarizableContentError:
                    status.update(
                        label="No summarizable document content is available.",
                        state="error",
                    )
                    st.warning("No summarizable document content is available.")
                except RAGServiceError as error:
                    status.update(label="Summary generation failed.", state="error")
                    st.warning(f"Summary generation failed: {error}")
                    try:
                        local = local_extractive_summary(scoped_chunks)
                        st.session_state.executive_summary_local[summary_key] = {
                            "sections": local.sections, "evidence": local.cited_evidence,
                            "cited_evidence": local.cited_evidence,
                            "evidence_strength": local.evidence_strength, "scope": summary_scope,
                            "local_error": str(error),
                        }
                    except NoSummarizableContentError as local_error:
                        st.warning(str(local_error))
                except Exception as error:
                    logger.error("Executive summary failed: exception_type=%s", type(error).__name__)
                    status.update(label="Summary generation failed.", state="error")
                    st.error("The executive summary could not be generated.")

        cached_summary = (st.session_state.executive_summary_cache.get(summary_key)
                          or st.session_state.executive_summary_local.get(summary_key))
        if cached_summary:
            if cached_summary.get("local_error"):
                st.info("Local extractive summary — actual source excerpts, not an AI-generated executive summary.")
                st.caption(cached_summary["local_error"])
            summary_markdown = intelligence_markdown(
                cached_summary["sections"], cached_summary["cited_evidence"]
            )
            if summary_markdown:
                st.markdown(summary_markdown, unsafe_allow_html=False)
            else:
                st.info("No summary sections had sufficient supporting evidence.")
            st.caption(f"Evidence strength: {cached_summary['evidence_strength']}")
            st.markdown("#### Deterministic Source References")
            for item in cached_summary["cited_evidence"]:
                st.markdown(
                    f"- **[{item['evidence_id']}]** `{item['source_filename']} · "
                    f"Page {item['page_number']} · {item['chunk_id']}`"
                )
                st.button(
                    "Inspect cited page",
                    key=f"view-summary-{summary_key}-{item['evidence_id']}",
                    on_click=select_source_for_viewer,
                    args=(item,),
                )
            summary_sources = evidence_sources(cached_summary["cited_evidence"])
            summary_report = build_markdown_report(
                "Executive Summary",
                corpus_documents,
                summary_markdown,
                cached_summary["evidence_strength"],
                summary_sources,
                cached_summary["evidence"],
                question=f"Scope: {cached_summary['scope']}",
                embedding_model=EMBEDDING_MODEL_NAME,
            )
            st.download_button(
                "Export Analysis Report",
                data=summary_report,
                file_name="defence-policy-executive-summary.md",
                mime="text/markdown",
                use_container_width=True,
                key="download-summary",
            )

    with comparison_tab:
        st.markdown("### Policy Comparison")
        comparison_enabled = len(searchable_documents) >= 2
        comparison_ready = comparison_enabled and gemini_available
        if not comparison_enabled:
            st.info("Upload and successfully index at least two documents to compare policies.")
        elif not gemini_available:
            st.warning(f"Gemini unavailable: {gemini_diagnostic}")
        comparison_options = {
            f"{document['filename']} · {document['document_id'][:8]}": document
            for document in searchable_documents
        }
        option_labels = list(comparison_options)
        document_a_label = st.selectbox(
            "Document A",
            option_labels,
            disabled=not comparison_ready,
            key="comparison_a",
        ) if option_labels else None
        remaining_labels = [label for label in option_labels if label != document_a_label]
        document_b_label = st.selectbox(
            "Document B",
            remaining_labels,
            disabled=not comparison_ready,
            key="comparison_b",
        ) if remaining_labels else None
        comparison_question = st.text_input(
            "Optional comparison question",
            placeholder="Example: Compare reporting obligations",
            disabled=not comparison_ready,
        )
        document_a = comparison_options.get(document_a_label)
        document_b = comparison_options.get(document_b_label)
        comparison_identity = (
            f"{corpus_key}:"
            f"{document_a['document_id'] if document_a else ''}:"
            f"{document_b['document_id'] if document_b else ''}:"
            f"{comparison_question.strip().lower()}"
        )
        comparison_key = "comparison-v3:" + hashlib.sha256(
            comparison_identity.encode("utf-8")
        ).hexdigest()
        cached_comparison = st.session_state.policy_comparison_cache.get(comparison_key)
        compare_action = st.button(
            "Regenerate Comparison" if cached_comparison else "Generate Comparison",
            disabled=not comparison_ready,
            use_container_width=True,
        )
        if compare_action and document_a and document_b:
            st.session_state.policy_comparison_cache.pop(comparison_key, None)
            comparison_queries = [
                "policy purpose scope common provisions themes",
                "eligibility requirements qualifications criteria",
                "important dates deadlines effective dates thresholds",
                "financial budget funding costs appropriations",
                "authorities agencies departments organizations",
                "obligations responsibilities duties reporting procedures",
                "exceptions exemptions waivers limitations restrictions",
            ]
            if comparison_question.strip():
                comparison_queries.insert(0, comparison_question.strip())
            with st.status("Retrieving evidence independently from both documents...", expanded=True) as status:
                try:
                    evidence_a = prepare_evidence(
                        retrieve_policy_evidence(
                            vector_index, get_embedding_model(), keyword_index,
                            document_a["document_id"], comparison_queries, limit=18
                        ),
                        "A",
                    )
                    evidence_b = prepare_evidence(
                        retrieve_policy_evidence(
                            vector_index, get_embedding_model(), keyword_index,
                            document_b["document_id"], comparison_queries, limit=18
                        ),
                        "B",
                    )
                    status.update(label="Generating grounded policy comparison...")
                    generated = generate_policy_comparison(
                        evidence_a + evidence_b,
                        document_a["filename"],
                        document_b["filename"],
                        comparison_question,
                    )
                    cached_comparison = {
                        "sections": generated.sections,
                        "evidence": evidence_a + evidence_b,
                        "cited_evidence": generated.cited_evidence,
                        "evidence_strength": generated.evidence_strength,
                        "document_a": document_a["filename"],
                        "document_b": document_b["filename"],
                        "question": comparison_question.strip() or "General comparison",
                    }
                    st.session_state.policy_comparison_cache[comparison_key] = cached_comparison
                    status.update(label="Policy comparison complete.", state="complete", expanded=False)
                except RAGServiceError as error:
                    status.update(label="Comparison generation failed.", state="error")
                    st.warning(f"Comparison generation failed: {error}")
                except Exception:
                    logger.exception("Policy comparison generation failed")
                    status.update(label="Comparison generation failed.", state="error")
                    st.error("The policy comparison could not be generated.")

        cached_comparison = st.session_state.policy_comparison_cache.get(comparison_key)
        if cached_comparison:
            comparison_markdown = intelligence_markdown(
                cached_comparison["sections"], cached_comparison["cited_evidence"]
            )
            if comparison_markdown:
                st.markdown(comparison_markdown, unsafe_allow_html=False)
            else:
                st.info("No comparison points had sufficient supporting evidence.")
            st.caption(f"Evidence strength: {cached_comparison['evidence_strength']}")
            evidence_columns = st.columns(2)
            for column, prefix, heading in (
                (evidence_columns[0], "A", "Evidence from Document A"),
                (evidence_columns[1], "B", "Evidence from Document B"),
            ):
                with column:
                    st.markdown(f"#### {heading}")
                    for item in cached_comparison["evidence"]:
                        if not item["evidence_id"].startswith(prefix):
                            continue
                        with st.container(border=True):
                            st.markdown(
                                f"**[{item['evidence_id']}]** `{item['source_filename']} · "
                                f"Page {item['page_number']} · {item['chunk_id']}`"
                            )
                            st.write(item["text"])
                            st.button(
                                "Inspect source page",
                                key=f"view-comparison-{comparison_key}-{item['evidence_id']}",
                                on_click=select_source_for_viewer,
                                args=(item,),
                            )
            comparison_sources = evidence_sources(cached_comparison["cited_evidence"])
            comparison_report = build_markdown_report(
                "Policy Comparison",
                corpus_documents,
                comparison_markdown,
                cached_comparison["evidence_strength"],
                comparison_sources,
                cached_comparison["evidence"],
                question=cached_comparison["question"],
                embedding_model=EMBEDDING_MODEL_NAME,
            )
            st.download_button(
                "Export Analysis Report",
                data=comparison_report,
                file_name="defence-policy-comparison.md",
                mime="text/markdown",
                use_container_width=True,
                key="download-comparison",
            )

    with conflict_tab:
        st.markdown("### Conflict Analysis")
        conflict_enabled = len(searchable_documents) >= 2
        conflict_ready = conflict_enabled and gemini_available
        if not conflict_enabled:
            st.info("Upload and successfully index at least two documents to analyze conflicts.")
        elif not gemini_available:
            st.warning(f"Gemini unavailable: {gemini_diagnostic}")

        conflict_options = {
            f"{document['filename']} · {document['document_id'][:8]}": document
            for document in searchable_documents
        }
        conflict_labels = list(conflict_options)
        conflict_a_label = st.selectbox(
            "Conflict analysis — Document A",
            conflict_labels,
            disabled=not conflict_ready,
            key="conflict_document_a",
        ) if conflict_labels else None
        conflict_remaining = [label for label in conflict_labels if label != conflict_a_label]
        conflict_b_label = st.selectbox(
            "Conflict analysis — Document B",
            conflict_remaining,
            disabled=not conflict_ready,
            key="conflict_document_b",
        ) if conflict_remaining else None
        conflict_document_a = conflict_options.get(conflict_a_label)
        conflict_document_b = conflict_options.get(conflict_b_label)
        conflict_identity = (
            f"{corpus_key}:"
            f"{conflict_document_a['document_id'] if conflict_document_a else ''}:"
            f"{conflict_document_b['document_id'] if conflict_document_b else ''}"
        )
        conflict_key = "conflict-v1:" + hashlib.sha256(
            conflict_identity.encode("utf-8")
        ).hexdigest()
        cached_conflict = st.session_state.conflict_analysis_cache.get(conflict_key)
        conflict_action = st.button(
            "Regenerate Conflict Analysis" if cached_conflict else "Analyze Conflicts",
            disabled=not conflict_ready,
            use_container_width=True,
            key="generate-conflict-analysis",
        )
        if conflict_action and conflict_document_a and conflict_document_b:
            st.session_state.conflict_analysis_cache.pop(conflict_key, None)
            conflict_queries = [
                "eligibility criteria minimum maximum thresholds qualifications",
                "dates deadlines effective expiration duration",
                "financial amounts funding budgets costs payments",
                "responsible authorities agencies departments administration",
                "definitions terminology means defined as",
                "requirements obligations restrictions prohibited permitted exemptions",
            ]
            with st.status(
                "Retrieving conflict evidence independently from both documents...",
                expanded=True,
            ) as status:
                try:
                    conflict_evidence_a = prepare_evidence(
                        retrieve_policy_evidence(
                            vector_index,
                            get_embedding_model(),
                            keyword_index,
                            conflict_document_a["document_id"],
                            conflict_queries,
                            limit=18,
                        ),
                        "A",
                    )
                    conflict_evidence_b = prepare_evidence(
                        retrieve_policy_evidence(
                            vector_index,
                            get_embedding_model(),
                            keyword_index,
                            conflict_document_b["document_id"],
                            conflict_queries,
                            limit=18,
                        ),
                        "B",
                    )
                    status.update(label="Classifying grounded topic relationships...")
                    conflict_result = generate_conflict_analysis(
                        conflict_evidence_a,
                        conflict_evidence_b,
                        conflict_document_a["filename"],
                        conflict_document_b["filename"],
                    )
                    cached_conflict = {
                        "corpus_key": corpus_key,
                        "findings": conflict_result.findings,
                        "cited_evidence": conflict_result.cited_evidence,
                        "evidence": conflict_evidence_a + conflict_evidence_b,
                        "document_a": conflict_document_a["filename"],
                        "document_b": conflict_document_b["filename"],
                    }
                    st.session_state.conflict_analysis_cache[conflict_key] = cached_conflict
                    status.update(
                        label="Conflict analysis complete.", state="complete", expanded=False
                    )
                except RAGServiceError as error:
                    status.update(label="Conflict analysis failed.", state="error")
                    st.warning(f"Conflict analysis failed: {error}")
                except Exception:
                    logger.exception("Conflict analysis generation failed")
                    status.update(label="Conflict analysis failed.", state="error")
                    st.error("The conflict analysis could not be generated.")

        cached_conflict = st.session_state.conflict_analysis_cache.get(conflict_key)
        if cached_conflict:
            conflict_evidence_map = {
                item["evidence_id"]: item for item in cached_conflict["evidence"]
            }
            conflict_report_lines = []
            if not cached_conflict["findings"]:
                st.info("No grounded topic relationships were identified.")
            for finding_number, finding in enumerate(cached_conflict["findings"], start=1):
                relationship = finding["relationship"]
                with st.container(border=True):
                    st.markdown(f"#### {finding['topic']}")
                    st.markdown(f"**Relationship:** `{relationship}`")
                    st.markdown(f"**Document A — {cached_conflict['document_a']}**")
                    st.write(finding["document_a_statement"])
                    for evidence_id in finding["document_a_evidence_ids"]:
                        item = conflict_evidence_map[evidence_id]
                        st.caption(
                            f"[{evidence_id}] {item['source_filename']} · Page "
                            f"{item['page_number']} · {item['chunk_id']}"
                        )
                    st.markdown(f"**Document B — {cached_conflict['document_b']}**")
                    st.write(finding["document_b_statement"])
                    for evidence_id in finding["document_b_evidence_ids"]:
                        item = conflict_evidence_map[evidence_id]
                        st.caption(
                            f"[{evidence_id}] {item['source_filename']} · Page "
                            f"{item['page_number']} · {item['chunk_id']}"
                        )
                    st.markdown("**Explanation**")
                    st.write(finding["explanation"])
                    st.markdown("**Sources**")
                    st.caption(
                        ", ".join(
                            f"[{evidence_id}] "
                            f"{conflict_evidence_map[evidence_id]['source_filename']} · "
                            f"Page {conflict_evidence_map[evidence_id]['page_number']}"
                            for evidence_id in (
                                finding["document_a_evidence_ids"]
                                + finding["document_b_evidence_ids"]
                            )
                        )
                    )

                all_finding_ids = (
                    finding["document_a_evidence_ids"]
                    + finding["document_b_evidence_ids"]
                )
                for evidence_id in all_finding_ids:
                    item = conflict_evidence_map[evidence_id]
                    st.button(
                        f"Inspect source [{evidence_id}]",
                        key=f"view-conflict-{conflict_key}-{finding_number}-{evidence_id}",
                        on_click=select_source_for_viewer,
                        args=(item,),
                    )
                conflict_report_lines.extend(
                    [
                        f"### {finding['topic']}",
                        f"**Relationship:** {relationship}",
                        f"**Document A:** {finding['document_a_statement']}",
                        f"**Document B:** {finding['document_b_statement']}",
                        f"**Explanation:** {finding['explanation']}",
                        "**Sources:** " + ", ".join(
                            f"[{evidence_id}] {conflict_evidence_map[evidence_id]['source_filename']} · "
                            f"Page {conflict_evidence_map[evidence_id]['page_number']}"
                            for evidence_id in all_finding_ids
                        ),
                        "",
                    ]
                )

            conflict_sources = evidence_sources(cached_conflict["cited_evidence"])
            conflict_report = build_markdown_report(
                "Conflict Analysis",
                corpus_documents,
                "\n\n".join(conflict_report_lines),
                "Grounded classifications",
                conflict_sources,
                cached_conflict["cited_evidence"],
                question=(
                    f"Potential conflicts between {cached_conflict['document_a']} and "
                    f"{cached_conflict['document_b']}"
                ),
                embedding_model=EMBEDDING_MODEL_NAME,
            )
            st.download_button(
                "Export Conflict Analysis",
                data=conflict_report,
                file_name="defence-policy-conflict-analysis.md",
                mime="text/markdown",
                use_container_width=True,
                key="download-conflict-analysis",
            )

    with timeline_tab:
        st.markdown("### Policy Timeline")
        timeline_scopes = {"Entire Corpus": None}
        for document in searchable_documents:
            timeline_scopes[
                f"{document['filename']} · {document['document_id'][:8]}"
            ] = document["document_id"]
        timeline_scope_label = st.selectbox(
            "Timeline scope",
            options=list(timeline_scopes),
            disabled=not text_chunks,
            key="timeline_scope",
        )
        timeline_document_id = timeline_scopes[timeline_scope_label]
        timeline_analysis_key = f"{corpus_key}:{timeline_document_id or 'all'}"
        if timeline_analysis_key not in st.session_state.timeline_analysis_cache:
            st.session_state.timeline_analysis_cache[timeline_analysis_key] = (
                extract_timeline_events(text_chunks, document_id=timeline_document_id)
            )
        timeline_events = st.session_state.timeline_analysis_cache[timeline_analysis_key]
        if not text_chunks:
            st.info("No indexed document text is available for timeline extraction.")
        elif not timeline_events:
            st.info("No explicitly dated events were found in the selected scope.")
        else:
            st.caption(
                f"{len(timeline_events)} explicitly dated events · chronological order. "
                "Month-only, fiscal-year, and relative expressions remain exactly as written."
            )
            timeline_report_lines = []
            for event in timeline_events:
                with st.container(border=True):
                    st.markdown(f"#### {event['date']} — {event['event_type']}")
                    st.write(event["event"])
                    st.caption(
                        f"[{event['evidence_id']}] {event['source_filename']} · "
                        f"Page {event['page_number']} · {event['chunk_id']} · "
                        f"{event['extraction_method']}"
                    )
                    st.button(
                        f"Inspect source [{event['evidence_id']}]",
                        key=(
                            f"view-timeline-{corpus_key}-{timeline_document_id or 'all'}-"
                            f"{event['evidence_id']}"
                        ),
                        on_click=select_source_for_viewer,
                        args=(event,),
                    )
                timeline_report_lines.extend(
                    [
                        f"### {event['date']} — {event['event_type']}",
                        event["event"],
                        (
                            f"**Source:** [{event['evidence_id']}] "
                            f"{event['source_filename']} · Page {event['page_number']} · "
                            f"{event['chunk_id']}"
                        ),
                        "",
                    ]
                )

            timeline_sources = evidence_sources(timeline_events)
            timeline_report = build_markdown_report(
                "Policy Timeline",
                corpus_documents,
                "\n\n".join(timeline_report_lines),
                "Explicitly dated source evidence",
                timeline_sources,
                timeline_events,
                question=f"Timeline scope: {timeline_scope_label}",
                embedding_model=EMBEDDING_MODEL_NAME,
            )
            st.download_button(
                "Export Policy Timeline",
                data=timeline_report,
                file_name="defence-policy-timeline.md",
                mime="text/markdown",
                use_container_width=True,
                key="download-policy-timeline",
            )

    with entities_tab:
        st.markdown("### Entities & Facts")
        entity_scopes = {"Entire Corpus": None}
        for document in searchable_documents:
            entity_scopes[
                f"{document['filename']} · {document['document_id'][:8]}"
            ] = document["document_id"]
        entity_scope_label = st.selectbox(
            "Entity analysis scope",
            options=list(entity_scopes),
            disabled=not text_chunks,
            key="entity-analysis-scope",
        )
        entity_document_id = entity_scopes[entity_scope_label]
        entity_analysis_key = f"{corpus_key}:{entity_document_id or 'all'}"
        if entity_analysis_key not in st.session_state.entity_analysis_cache:
            st.session_state.entity_analysis_cache[entity_analysis_key] = extract_entities(
                text_chunks, document_id=entity_document_id
            )
        scoped_entities = st.session_state.entity_analysis_cache[entity_analysis_key]
        counts_by_category = entity_counts(scoped_entities)
        category_filter = st.selectbox(
            "Entity category",
            options=["All Categories"] + list(ENTITY_CATEGORIES),
            disabled=not scoped_entities,
            key="entity-category-filter",
        )

        if not text_chunks:
            st.info("No indexed document text is available for entity extraction.")
        elif not scoped_entities:
            st.info("No explicitly supported entities or factual values were found.")
        else:
            st.caption(
                f"{len(scoped_entities)} unique grounded entities in {entity_scope_label}. "
                "Counts represent merged entities; every source mention remains available below."
            )
            category_items = list(counts_by_category.items())
            for start in range(0, len(category_items), 3):
                count_columns = st.columns(3)
                for column, (category, count) in zip(
                    count_columns, category_items[start : start + 3]
                ):
                    with column:
                        metric_card(category, str(count))

            visible_entities = [
                record
                for record in scoped_entities
                if category_filter == "All Categories"
                or record["category"] == category_filter
            ]
            if not visible_entities:
                st.info("No entities match the selected category.")
            for entity_number, record in enumerate(visible_entities, start=1):
                with st.container(border=True):
                    st.markdown(f"#### {record['entity']}")
                    st.caption(
                        f"Category: {record['category']} · "
                        f"Source references: {len(record['references'])}"
                    )
                    for reference in record["references"]:
                        st.markdown(f"**Context — [{reference['evidence_id']}]**")
                        st.write(reference["context"])
                        st.caption(
                            f"{reference['source_filename']} · Page "
                            f"{reference['page_number']} · {reference['chunk_id']} · "
                            f"{reference['extraction_method']}"
                        )
                        st.button(
                            f"Inspect source [{reference['evidence_id']}]",
                            key=(
                                f"view-entity-{corpus_key}-{entity_document_id or 'all'}-"
                                f"{entity_number}-{reference['evidence_id']}"
                            ),
                            on_click=select_source_for_viewer,
                            args=(reference,),
                        )

    with dashboard_tab:
        st.markdown("### Intelligence Dashboard")
        if not corpus_documents:
            st.info(
                "Upload and index one or more documents to populate corpus intelligence."
            )
        else:
            corpus_timeline = st.session_state.timeline_analysis_cache.get(
                f"{corpus_key}:all"
            )
            corpus_entities = st.session_state.entity_analysis_cache.get(
                f"{corpus_key}:all"
            )
            corpus_conflicts = [
                analysis
                for analysis in st.session_state.conflict_analysis_cache.values()
                if analysis.get("corpus_key") == corpus_key
            ]
            dashboard = build_dashboard_snapshot(
                corpus_documents,
                indexed_document_ids,
                vector_index.index.ntotal if vector_index else 0,
                entities=corpus_entities,
                timeline_events=corpus_timeline,
                conflict_analyses=corpus_conflicts,
            )

            st.markdown("#### Corpus Metrics")
            metric_items = list(dashboard["metrics"].items())
            for start in range(0, len(metric_items), 4):
                dashboard_columns = st.columns(4)
                for column, (label, value) in zip(
                    dashboard_columns, metric_items[start : start + 4]
                ):
                    with column:
                        metric_card(label, "Not generated" if value is None else str(value))

            st.markdown("#### Document Analysis")
            st.dataframe(
                dashboard["document_rows"],
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("#### Entity Distribution")
            if dashboard["entity_distribution"] is None:
                st.info("Run Entities & Facts to populate entity intelligence.")
            elif not dashboard["entity_distribution"]:
                st.info("No supported entities were found in the corpus.")
            else:
                entity_chart_rows = [
                    {"Category": category, "Count": count}
                    for category, count in dashboard["entity_distribution"].items()
                ]
                st.bar_chart(entity_chart_rows, x="Category", y="Count")

            st.markdown("#### Document Coverage")
            st.dataframe(
                dashboard["coverage_rows"],
                use_container_width=True,
                hide_index=True,
            )
            if any(row["Chunk Count"] for row in dashboard["coverage_rows"]):
                st.bar_chart(
                    dashboard["coverage_rows"],
                    x="Document",
                    y="Percentage of Corpus",
                )

            dashboard_columns = st.columns(2)
            with dashboard_columns[0]:
                st.markdown("#### Timeline Overview")
                timeline_overview = dashboard["timeline"]
                if not timeline_overview["available"]:
                    st.info("Run Policy Timeline to populate timeline intelligence.")
                else:
                    st.metric("Timeline events", timeline_overview["total"])
                    st.caption(
                        f"Earliest: {timeline_overview['earliest'] or 'No calendar date'} · "
                        f"Latest: {timeline_overview['latest'] or 'No calendar date'}"
                    )
                    if timeline_overview["by_document"]:
                        st.bar_chart(
                            [
                                {"Document": document, "Events": count}
                                for document, count in timeline_overview["by_document"].items()
                            ],
                            x="Document",
                            y="Events",
                        )
            with dashboard_columns[1]:
                st.markdown("#### Conflict Overview")
                conflict_overview = dashboard["conflicts"]
                if not conflict_overview["available"]:
                    st.info("Run Conflict Analysis to populate conflict intelligence.")
                else:
                    st.metric("Potential conflicts", conflict_overview["total"])
                    st.caption(
                        "Documents involved: "
                        + (", ".join(conflict_overview["documents"]) or "None")
                    )
                    if conflict_overview["categories"]:
                        st.bar_chart(
                            [
                                {"Topic": topic, "Conflicts": count}
                                for topic, count in conflict_overview["categories"].items()
                            ],
                            x="Topic",
                            y="Conflicts",
                        )

            st.markdown("#### Corpus Health")
            health = dashboard["health"]
            health_columns = st.columns(3)
            health_columns[0].metric("Health", health["classification"])
            health_columns[1].metric(
                "Native-text pages", f"{health['native_page_percentage']:.1f}%"
            )
            health_columns[2].metric(
                "OCR pages", f"{health['ocr_page_percentage']:.1f}%"
            )
            st.caption(
                f"Successfully processed documents: {health['successful_documents']} · "
                f"Documents with warnings: {health['warning_documents']} · "
                f"Vector index: {health['index_status']}"
            )

            with st.expander("Retrieval Diagnostics", expanded=False):
                st.write(f"Embedding model: `{EMBEDDING_MODEL_NAME}`")
                st.write(
                    f"Vectors indexed: {vector_index.index.ntotal if vector_index else 0}"
                )
                st.write(f"Chunk size: {CHUNK_SIZE}")
                st.write(f"Chunk overlap: {CHUNK_OVERLAP}")
                st.write(f"Indexed documents: {len(indexed_document_ids)}")

    with overview_tab:
        st.markdown("### Corpus Overview")
        overview_metrics = st.columns(3)
        overview_metrics[0].metric("Documents", len(corpus_documents))
        overview_metrics[1].metric("Pages", len(pages))
        overview_metrics[2].metric("Vectors", vector_index.index.ntotal if vector_index else 0)
        overview_metrics = st.columns(3)
        overview_metrics[0].metric("Native pages", native_page_count)
        overview_metrics[1].metric("OCR pages", ocr_page_count)
        overview_metrics[2].metric("Chunks", len(text_chunks))
        for document in corpus_documents:
            document_pages = document.get("pages", [])
            native_count = sum(
                page.get("extraction_method") == "Native text" for page in document_pages
            )
            ocr_count = sum(
                page.get("extraction_method") == "OCR" for page in document_pages
            )
            with st.container(border=True):
                st.markdown(f"**{document['filename']}**")
                st.caption(
                    f"{len(document_pages)} pages · {len(document.get('chunks', []))} chunks · "
                    f"Native {native_count} · OCR {ocr_count} · {document.get('status', 'Unknown')}"
                )

    st.divider()
    st.markdown("### Source Viewer")
    st.caption(
        "Inspect extracted source pages from the indexed corpus. This viewer reuses "
        "existing page and chunk metadata and does not reprocess PDFs."
    )
    viewer_documents = [
        document
        for document in searchable_documents
        if document.get("pages") and document.get("status") == "Processed"
    ]
    if not viewer_documents:
        st.info("No indexed source pages are currently available.")
    else:
        document_ids = [document["document_id"] for document in viewer_documents]
        if st.session_state.get("source_viewer_document_id") not in document_ids:
            st.session_state.source_viewer_document_id = document_ids[0]
        selected_source_document_id = st.selectbox(
            "Source document",
            options=document_ids,
            format_func=lambda item: next(
                document["filename"]
                for document in viewer_documents
                if document["document_id"] == item
            ),
            key="source_viewer_document_id",
        )
        selected_source_document = next(
            document
            for document in viewer_documents
            if document["document_id"] == selected_source_document_id
        )
        available_page_numbers = [
            page["page_number"] for page in selected_source_document["pages"]
        ]
        if st.session_state.get("source_viewer_page_number") not in available_page_numbers:
            st.session_state.source_viewer_page_number = available_page_numbers[0]
        selected_source_page = st.selectbox(
            "Source page",
            options=available_page_numbers,
            format_func=lambda page_number: f"Page {page_number}",
            key="source_viewer_page_number",
        )

        available_chunks = chunks_for_page(selected_source_document, selected_source_page)
        chunk_ids = [""] + [chunk["chunk_id"] for chunk in available_chunks]
        if st.session_state.get("source_viewer_chunk_id", "") not in chunk_ids:
            st.session_state.source_viewer_chunk_id = ""
        selected_source_chunk = st.selectbox(
            "Retrieved passage",
            options=chunk_ids,
            format_func=lambda chunk_id: chunk_id or "Full page only",
            key="source_viewer_chunk_id",
        )

        selected_citation = st.session_state.get("source_viewer_citation", {})
        citation_matches = (
            selected_citation.get("document_id") == selected_source_document_id
            and selected_citation.get("page_number") == selected_source_page
            and selected_citation.get("chunk_id", "") == selected_source_chunk
        )
        source_view = resolve_source_view(
            corpus_documents,
            selected_source_document_id,
            selected_source_page,
            chunk_id=selected_source_chunk,
            evidence_id=(selected_citation.get("evidence_id", "") if citation_matches else ""),
            passage=(selected_citation.get("text", "") if citation_matches else ""),
        )
        if source_view is None:
            st.warning("The selected source page is no longer available in the indexed corpus.")
        else:
            with st.container(border=True):
                st.markdown(f"**{source_view.filename} · Page {source_view.page_number}**")
                reference_parts = [f"Extraction: {source_view.extraction_method}"]
                if source_view.chunk_id:
                    reference_parts.append(f"Chunk: {source_view.chunk_id}")
                if source_view.evidence_id:
                    reference_parts.append(f"Evidence: {source_view.evidence_id}")
                st.caption(" · ".join(reference_parts))
                if source_view.passage:
                    st.markdown("#### Retrieved Passage")
                    st.info(source_view.passage)
                else:
                    st.info("No retrieved passage is associated with this page selection.")
                st.text_area(
                    "Extracted page text",
                    value=source_view.page_text or "No extracted text is available for this page.",
                    height=320,
                    disabled=True,
                    key=(
                        f"source-page-text-{source_view.document_id}-"
                        f"{source_view.page_number}"
                    ),
                )
