"""PDF text extraction and page-aware chunk creation."""

from collections.abc import Callable
import gc
import logging

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ocr_service import extract_text_with_ocr


MIN_NATIVE_TEXT_CHARACTERS = 40
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
logger = logging.getLogger(__name__)


def process_pdf(
    pdf_bytes: bytes,
    source_filename: str,
    document_id: str | None = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    progress_callback: Callable[[int, int, str], None] | None = None,
    include_preview: bool = True,
) -> tuple[list[dict], str, list[dict]]:
    """Extract each page natively, falling back to OCR for sparse pages."""
    pages = []
    chunks = []
    resolved_document_id = document_id or source_filename
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        logger.exception("PDF open/read failed for %s", source_filename)
        raise

    try:
        if pdf_document.needs_pass:
            raise ValueError("This PDF is password-protected and cannot be opened.")
        total_pages = len(pdf_document)
        for page_index in range(total_pages):
            page_number = page_index + 1
            if progress_callback:
                progress_callback(page_number, total_pages, "Extracting pages")

            page = None
            try:
                page = pdf_document.load_page(page_index)
                native_text = page.get_text("text").strip()
                page_text = native_text
                extraction_method = "Native text"
                ocr_error = ""

                # OCR only genuinely sparse pages; normal native-text pages never
                # allocate a raster image.
                useful_character_count = len("".join(native_text.split()))
                if useful_character_count < MIN_NATIVE_TEXT_CHARACTERS:
                    if progress_callback:
                        progress_callback(page_number, total_pages, "OCR processing")
                    try:
                        ocr_text = extract_text_with_ocr(page)
                        if ocr_text:
                            page_text = ocr_text
                            extraction_method = "OCR"
                        elif not native_text:
                            extraction_method = "Failed"
                            ocr_error = "OCR completed but found no readable text."
                    except Exception as error:
                        logger.exception(
                            "OCR failed for %s page %s/%s",
                            source_filename,
                            page_number,
                            total_pages,
                        )
                        # Retain sparse native text and continue other pages.
                        ocr_error = type(error).__name__
                        if not native_text:
                            extraction_method = "Failed"

                page_record = {
                    "source_filename": source_filename,
                    "document_id": resolved_document_id,
                    "page_number": page_number,
                    "text": page_text,
                    "extraction_method": extraction_method,
                    "ocr_error": ocr_error,
                }
                pages.append(page_record)

                # Chunk immediately while this page is current. This avoids a
                # second full-document traversal and isolates chunk failures.
                if page_text:
                    for page_chunk_number, chunk_text in enumerate(
                        text_splitter.split_text(page_text), start=1
                    ):
                        chunks.append(
                            {
                                "text": chunk_text,
                                "source_filename": source_filename,
                                "document_id": resolved_document_id,
                                "page_number": page_number,
                                "chunk_id": f"page-{page_number}-chunk-{page_chunk_number}",
                                "extraction_method": extraction_method,
                            }
                        )
            except MemoryError:
                logger.exception(
                    "MemoryError processing %s page %s/%s",
                    source_filename,
                    page_number,
                    total_pages,
                )
                raise
            except Exception:
                logger.exception(
                    "Page extraction/chunking failed for %s page %s/%s",
                    source_filename,
                    page_number,
                    total_pages,
                )
                raise
            finally:
                page = None
                # PyMuPDF page wrappers are released by ref-counting. A periodic
                # collection prevents buildup without pausing every native page.
                if page_number % 20 == 0:
                    gc.collect()
    finally:
        pdf_document.close()
        gc.collect()

    if progress_callback:
        progress_callback(total_pages, total_pages, "Creating chunks")

    if include_preview:
        preview_sections = [
            f"[Page {page['page_number']}]\n{page['text']}"
            for page in pages
            if page["text"]
        ]
        preview_text = "\n\n".join(preview_sections)
    else:
        preview_text = ""
    return pages, preview_text, chunks
