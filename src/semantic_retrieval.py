"""Local embeddings, FAISS indexing, and semantic similarity search."""

from dataclasses import dataclass
import gc
import logging

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_BATCH_SIZE = 16
logger = logging.getLogger(__name__)


def recommended_embedding_batch_size(chunk_count: int) -> int:
    """Use smaller batches as a document grows to limit peak model memory."""
    if chunk_count > 2_000:
        return 4
    if chunk_count > 500:
        return 8
    return DEFAULT_EMBEDDING_BATCH_SIZE


@dataclass
class VectorIndex:
    """Keep the FAISS index beside metadata in matching row order."""

    index: faiss.Index
    chunks: list[dict]


def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """Load a local Sentence Transformers model."""
    return SentenceTransformer(model_name)


def build_vector_index(chunks: list[dict], model: SentenceTransformer) -> VectorIndex:
    """Embed chunks and build a cosine-similarity FAISS index."""
    if not chunks:
        raise ValueError("No text chunks are available for indexing.")

    embeddings = embed_chunks_batched(chunks, model)

    return build_vector_index_from_embeddings(chunks, embeddings)


def embed_chunks_batched(
    chunks: list[dict],
    model: SentenceTransformer,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    progress_callback=None,
) -> np.ndarray:
    """Embed chunks in bounded batches and optionally report batch progress."""
    if not chunks:
        raise ValueError("No text chunks are available for embedding.")
    if batch_size < 1:
        raise ValueError("Embedding batch size must be positive.")

    embedding_matrix = None
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    for batch_number, start in enumerate(range(0, len(chunks), batch_size), start=1):
        batch_chunks = chunks[start : start + batch_size]
        batch_texts = [chunk["text"] for chunk in batch_chunks]
        try:
            batch_embeddings = model.encode(
                batch_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32")
        except Exception:
            logger.exception("Embedding generation failed at batch %s/%s", batch_number, total_batches)
            raise
        if embedding_matrix is None:
            embedding_matrix = np.empty(
                (len(chunks), batch_embeddings.shape[1]), dtype="float32"
            )
        embedding_matrix[start : start + len(batch_chunks)] = batch_embeddings
        if progress_callback:
            progress_callback(batch_number, total_batches)
        del batch_texts, batch_embeddings
        gc.collect()

    return np.ascontiguousarray(embedding_matrix, dtype="float32")


def build_vector_index_from_embeddings(
    chunks: list[dict], embeddings: np.ndarray
) -> VectorIndex:
    """Build FAISS from existing embeddings without running the model again."""
    if not chunks:
        raise ValueError("No text chunks are available for indexing.")
    embeddings = np.asarray(embeddings, dtype="float32")

    if embeddings.ndim != 2 or len(embeddings) != len(chunks):
        raise ValueError("The embedding model returned an unexpected result.")

    # Inner product over normalized vectors is cosine similarity.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.ascontiguousarray(embeddings))
    return VectorIndex(index=index, chunks=chunks)


def build_vector_index_from_parts(
    document_parts: list[tuple[list[dict], np.ndarray]],
) -> VectorIndex:
    """Build one FAISS index incrementally from cached per-document matrices."""
    if not document_parts:
        raise ValueError("No document embeddings are available for indexing.")

    vector_index = None
    corpus_chunks = []
    for chunks, embeddings in document_parts:
        matrix = np.asarray(embeddings, dtype="float32")
        if matrix.ndim != 2 or len(matrix) != len(chunks):
            raise ValueError("Document embeddings do not match their chunks.")
        if vector_index is None:
            vector_index = faiss.IndexFlatIP(matrix.shape[1])
        elif vector_index.d != matrix.shape[1]:
            raise ValueError("Document embedding dimensions do not match.")
        try:
            vector_index.add(np.ascontiguousarray(matrix))
        except Exception:
            logger.exception("FAISS update failed while adding a document matrix")
            raise
        corpus_chunks.extend(chunks)

    return VectorIndex(index=vector_index, chunks=corpus_chunks)


def semantic_search(
    query: str,
    vector_index: VectorIndex,
    model: SentenceTransformer,
    top_k: int = 4,
    document_id: str | None = None,
) -> list[dict]:
    """Return top semantic matches across the corpus or within one document."""
    if not query.strip():
        return []

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    # Search the full index when filtering so relevant chunks from the chosen
    # document are not displaced by higher-scoring chunks from other documents.
    result_count = (
        vector_index.index.ntotal
        if document_id
        else min(top_k, vector_index.index.ntotal)
    )
    scores, positions = vector_index.index.search(query_embedding, result_count)

    results = []
    for score, position in zip(scores[0], positions[0]):
        if position < 0:
            continue
        result = dict(vector_index.chunks[position])
        if document_id and result.get("document_id") != document_id:
            continue
        result["similarity"] = float(score)
        results.append(result)
        if len(results) == top_k:
            break
    return results
