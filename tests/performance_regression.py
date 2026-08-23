"""Local regression harness for large-document caching and incremental indexing."""

from hashlib import sha256
from pathlib import Path
import sys
import tempfile
import time

import fitz
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import src.cache_service as cache_service  # noqa: E402
from src.document_processing import process_pdf  # noqa: E402
from src.semantic_retrieval import (  # noqa: E402
    build_vector_index_from_parts,
    embed_chunks_batched,
    load_embedding_model,
    recommended_embedding_batch_size,
    semantic_search,
)


class DeterministicEmbeddingModel:
    """Small test double that exposes batching without downloading a model."""

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts, **_kwargs):
        self.calls += 1
        rows = []
        for text in texts:
            digest = sha256(text.encode("utf-8")).digest()
            vector = np.frombuffer(digest, dtype=np.uint8).astype("float32")
            vector /= max(np.linalg.norm(vector), 1.0)
            rows.append(vector)
        return np.vstack(rows)


class CountingModel:
    """Count calls while exercising the real local MiniLM model."""

    def __init__(self, model) -> None:
        self.model = model
        self.calls = 0

    def encode(self, texts, **kwargs):
        self.calls += 1
        return self.model.encode(texts, **kwargs)


def make_pdf(page_count: int, label: str, padded_size: int = 0) -> bytes:
    document = fitz.open()
    for page_number in range(1, page_count + 1):
        page = document.new_page()
        policy_text = (
            f"{label} defence policy page {page_number}. "
            "Readiness, logistics, command resilience, and civil coordination "
            "are assessed through documented governance controls. " * 8
        )
        page.insert_textbox(fitz.Rect(50, 50, 545, 790), policy_text, fontsize=10)
    data = document.tobytes(deflate=True)
    document.close()
    if padded_size and len(data) < padded_size:
        data += b"\n% performance-test-padding\n" + b"0" * (padded_size - len(data) - 28)
    return data


def process_and_cache(name: str, data: bytes) -> tuple[dict, float]:
    started = time.perf_counter()
    document_hash = sha256(data).hexdigest()
    pages, _, chunks = process_pdf(data, name, document_hash, include_preview=False)
    record = {
        "filename": name,
        "document_id": document_hash,
        "content_hash": document_hash,
        "file_size": len(data),
        "pages": pages,
        "chunks": chunks,
        "status": "Processed",
        "error": "",
    }
    cache_service.save_document(document_hash, record)
    return record, time.perf_counter() - started


def embeddings_for(record: dict, model, created: list[str]):
    matrix = cache_service.load_embeddings(record["document_id"], record["chunks"])
    if matrix is None:
        matrix = embed_chunks_batched(
            record["chunks"],
            model,
            batch_size=recommended_embedding_batch_size(len(record["chunks"])),
        )
        cache_service.save_embeddings(record["document_id"], record["chunks"], matrix)
        created.append(record["filename"])
    return matrix


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="defence-rag-test-") as directory:
        cache_service.CACHE_ROOT = Path(directory)
        small_a = make_pdf(2, "Alpha")
        small_b = make_pdf(3, "Bravo")
        large = make_pdf(98, "Large corpus", padded_size=32 * 1024 * 1024)
        small_c = make_pdf(2, "Charlie")

        model = (
            CountingModel(load_embedding_model())
            if "--real-model" in sys.argv
            else DeterministicEmbeddingModel()
        )
        a, a_time = process_and_cache("A.pdf", small_a)
        b, b_time = process_and_cache("B.pdf", small_b)
        assert len(a["pages"]) == 2 and len(b["pages"]) == 3
        print(f"A two small PDFs: {a_time + b_time:.2f}s, {len(a['chunks']) + len(b['chunks'])} chunks")

        large_record, large_time = process_and_cache("large-98-pages.pdf", large)
        assert len(large_record["pages"]) == 98
        assert all(
            page["extraction_method"] == "Native text"
            for page in large_record["pages"]
        ), "Native pages unexpectedly entered OCR"
        print(
            f"B large PDF: {large_time:.2f}s, {len(large) / (1024 * 1024):.1f}MB, "
            f"{len(large_record['chunks'])} chunks"
        )

        created = []
        parts = [
            (record["chunks"], embeddings_for(record, model, created))
            for record in (large_record, a)
        ]
        index = build_vector_index_from_parts(parts)
        assert index.index.ntotal == len(large_record["chunks"]) + len(a["chunks"])
        print(f"C mixed corpus: {index.index.ntotal} vectors, embedded={created}")

        calls_before = model.calls
        reloaded = [
            cache_service.load_document(record["document_id"], record["filename"])
            for record in (large_record, a)
        ]
        recreated = []
        reloaded_parts = [
            (record["chunks"], embeddings_for(record, model, recreated))
            for record in reloaded
        ]
        stable_index = build_vector_index_from_parts(reloaded_parts)
        for query in ("logistics", "command resilience", "governance controls"):
            assert semantic_search(query, stable_index, model, top_k=4)
        assert recreated == [] and model.calls == calls_before + 3
        print("D rerun + three queries: 0 documents reprocessed, 0 document embeddings regenerated")

        c, _ = process_and_cache("C.pdf", small_c)
        created_after_add = []
        updated_parts = [
            (record["chunks"], embeddings_for(record, model, created_after_add))
            for record in (*reloaded, c)
        ]
        updated_index = build_vector_index_from_parts(updated_parts)
        assert created_after_add == ["C.pdf"]
        assert updated_index.index.ntotal == sum(len(record["chunks"]) for record in (*reloaded, c))
        print(f"E add C.pdf: embedded only {created_after_add}; {updated_index.index.ntotal} vectors")
        print(f"Embedding model calls (bounded batch + queries): {model.calls}")

        # A corrupt document must not prevent a healthy cached document from
        # remaining searchable.
        failure_type = ""
        try:
            process_pdf(b"not a PDF", "broken.pdf", "broken", include_preview=False)
        except Exception as error:
            failure_type = type(error).__name__
        assert failure_type
        healthy_matrix = embeddings_for(a, model, [])
        healthy_index = build_vector_index_from_parts([(a["chunks"], healthy_matrix)])
        assert semantic_search("readiness", healthy_index, model, top_k=4)
        print(
            f"Failure isolation: broken.pdf raised {failure_type}; "
            "A.pdf remained searchable"
        )


if __name__ == "__main__":
    main()
