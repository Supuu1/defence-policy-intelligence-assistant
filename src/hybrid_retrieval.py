"""Deterministic BM25 keyword retrieval and fusion with existing FAISS results."""

from collections import Counter
from dataclasses import dataclass
import logging
import math
import re

from src.semantic_retrieval import VectorIndex, semantic_search


logger = logging.getLogger(__name__)
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[./-][A-Za-z0-9]+)*")
DEFAULT_RRF_K = 60


def tokenize(text: str) -> list[str]:
    """Tokenize text while retaining dates, acronyms, and policy identifiers."""
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


@dataclass
class BM25Index:
    """Store compact BM25 statistics beside chunks in corpus order."""

    chunks: list[dict]
    term_frequencies: list[Counter]
    document_frequencies: Counter
    document_lengths: list[int]
    average_document_length: float
    k1: float = 1.5
    b: float = 0.75


def build_bm25_index(chunks: list[dict]) -> BM25Index:
    """Build a lexical index over the same page-aware chunks used by FAISS."""
    if not chunks:
        raise ValueError("No text chunks are available for BM25 indexing.")

    term_frequencies = []
    document_frequencies = Counter()
    document_lengths = []
    for chunk in chunks:
        tokens = tokenize(chunk.get("text", ""))
        frequencies = Counter(tokens)
        term_frequencies.append(frequencies)
        document_frequencies.update(frequencies.keys())
        document_lengths.append(len(tokens))

    average_length = sum(document_lengths) / max(len(document_lengths), 1)
    return BM25Index(
        chunks=chunks,
        term_frequencies=term_frequencies,
        document_frequencies=document_frequencies,
        document_lengths=document_lengths,
        average_document_length=max(average_length, 1.0),
    )


def keyword_search(
    query: str,
    keyword_index: BM25Index,
    top_k: int = 12,
    document_id: str | None = None,
) -> list[dict]:
    """Return BM25 matches, optionally restricted to one source document."""
    query_terms = tokenize(query)
    if not query_terms or top_k < 1:
        return []

    corpus_size = len(keyword_index.chunks)
    scored_rows = []
    for row, (chunk, frequencies, length) in enumerate(
        zip(
            keyword_index.chunks,
            keyword_index.term_frequencies,
            keyword_index.document_lengths,
        )
    ):
        if document_id and chunk.get("document_id") != document_id:
            continue
        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            document_frequency = keyword_index.document_frequencies.get(term, 0)
            # Robertson/Sparck Jones IDF with a positive floor for common terms.
            inverse_frequency = math.log(
                1.0 + (corpus_size - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            length_normalizer = frequency + keyword_index.k1 * (
                1.0
                - keyword_index.b
                + keyword_index.b * length / keyword_index.average_document_length
            )
            score += inverse_frequency * frequency * (keyword_index.k1 + 1.0) / length_normalizer
        if score > 0:
            scored_rows.append((score, row))

    scored_rows.sort(key=lambda item: (-item[0], item[1]))
    results = []
    for score, row in scored_rows[:top_k]:
        result = dict(keyword_index.chunks[row])
        result["keyword_score"] = float(score)
        results.append(result)
    return results


def _chunk_key(chunk: dict) -> tuple[str, str]:
    """Use stable corpus metadata to identify a chunk across retrievers."""
    return str(chunk.get("document_id", "")), str(chunk.get("chunk_id", ""))


def _semantic_only_results(results: list[dict], rrf_k: int) -> list[dict]:
    """Annotate semantic results when BM25 is unavailable."""
    annotated = []
    maximum_rrf = 1.0 / (rrf_k + 1)
    for rank, result in enumerate(results, start=1):
        item = dict(result)
        item["semantic_similarity"] = float(item.get("similarity", 0.0))
        item["keyword_score"] = None
        item["retrieval_source"] = "Semantic"
        item["rrf_score"] = 1.0 / (rrf_k + rank)
        item["hybrid_score"] = item["rrf_score"] / maximum_rrf
        annotated.append(item)
    return annotated


def hybrid_search(
    query: str,
    vector_index: VectorIndex,
    model,
    keyword_index: BM25Index | None,
    top_k: int = 4,
    document_id: str | None = None,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[dict]:
    """Fuse FAISS and BM25 rankings with Reciprocal Rank Fusion (RRF)."""
    candidate_count = max(12, top_k * 4)
    semantic_results = semantic_search(
        query, vector_index, model, top_k=candidate_count, document_id=document_id
    )
    if keyword_index is None:
        return _semantic_only_results(semantic_results[:top_k], rrf_k)

    try:
        keyword_results = keyword_search(
            query, keyword_index, top_k=candidate_count, document_id=document_id
        )
    except Exception:
        logger.exception("BM25 retrieval failed; continuing with FAISS-only retrieval")
        return _semantic_only_results(semantic_results[:top_k], rrf_k)

    fused: dict[tuple[str, str], dict] = {}
    semantic_ranks = {}
    keyword_ranks = {}
    for rank, result in enumerate(semantic_results, start=1):
        key = _chunk_key(result)
        semantic_ranks[key] = rank
        fused[key] = dict(result)
        fused[key]["semantic_similarity"] = float(result["similarity"])
    for rank, result in enumerate(keyword_results, start=1):
        key = _chunk_key(result)
        keyword_ranks[key] = rank
        fused.setdefault(key, dict(result))
        fused[key]["keyword_score"] = float(result["keyword_score"])

    maximum_rrf = 2.0 / (rrf_k + 1)
    for key, result in fused.items():
        semantic_rank = semantic_ranks.get(key)
        keyword_rank = keyword_ranks.get(key)
        rrf_score = 0.0
        if semantic_rank:
            rrf_score += 1.0 / (rrf_k + semantic_rank)
        if keyword_rank:
            rrf_score += 1.0 / (rrf_k + keyword_rank)
        result["retrieval_source"] = (
            "Both" if semantic_rank and keyword_rank else "Semantic" if semantic_rank else "Keyword"
        )
        result.setdefault("semantic_similarity", None)
        result.setdefault("similarity", 0.0)
        result.setdefault("keyword_score", None)
        result["rrf_score"] = rrf_score
        result["hybrid_score"] = rrf_score / maximum_rrf

    return sorted(
        fused.values(),
        key=lambda item: (
            -item["rrf_score"],
            str(item.get("source_filename", "")),
            int(item.get("page_number", 0)),
            str(item.get("chunk_id", "")),
        ),
    )[:top_k]
