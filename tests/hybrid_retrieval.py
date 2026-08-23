"""Regression checks for exact-term and semantic hybrid retrieval."""

from unittest.mock import patch

from src.hybrid_retrieval import build_bm25_index, hybrid_search, keyword_search


CHUNKS = [
    {
        "text": "Directive 17-B takes effect on 2028-04-15. JADC2 reporting is mandatory.",
        "source_filename": "operations.pdf",
        "document_id": "doc-operations",
        "page_number": 7,
        "chunk_id": "chunk-3",
        "extraction_method": "Native text",
    },
    {
        "text": "Units must maintain continuity of essential operations during disruption.",
        "source_filename": "readiness.pdf",
        "document_id": "doc-readiness",
        "page_number": 2,
        "chunk_id": "chunk-1",
        "extraction_method": "OCR",
    },
    {
        "text": "General administrative procedures and annual review requirements.",
        "source_filename": "administration.pdf",
        "document_id": "doc-administration",
        "page_number": 1,
        "chunk_id": "chunk-1",
        "extraction_method": "Native text",
    },
]


def semantic_result(position: int, similarity: float) -> dict:
    result = dict(CHUNKS[position])
    result["similarity"] = similarity
    return result


def run_checks() -> None:
    keyword_index = build_bm25_index(CHUNKS)

    exact = keyword_search("Directive 17-B 2028-04-15 JADC2", keyword_index, top_k=2)
    assert exact[0]["document_id"] == "doc-operations"
    assert exact[0]["keyword_score"] > 0

    semantic_rank = [semantic_result(2, 0.68), semantic_result(0, 0.42)]
    with patch("src.hybrid_retrieval.semantic_search", return_value=semantic_rank):
        fused = hybrid_search("Directive 17-B", object(), object(), keyword_index, top_k=3)
    assert fused[0]["document_id"] == "doc-operations"
    assert fused[0]["retrieval_source"] == "Both"
    assert len({_key(item) for item in fused}) == len(fused)
    assert fused[0]["source_filename"] == "operations.pdf"
    assert fused[0]["page_number"] == 7
    assert fused[0]["chunk_id"] == "chunk-3"
    assert fused[0]["extraction_method"] == "Native text"

    paraphrase_rank = [semantic_result(1, 0.73), semantic_result(2, 0.31)]
    with patch("src.hybrid_retrieval.semantic_search", return_value=paraphrase_rank):
        paraphrase = hybrid_search(
            "How are critical missions kept running during an interruption?",
            object(),
            object(),
            keyword_index,
            top_k=2,
        )
    assert paraphrase[0]["document_id"] == "doc-readiness"
    assert paraphrase[0]["retrieval_source"] in {"Semantic", "Both"}

    filtered = keyword_search(
        "requirements", keyword_index, top_k=4, document_id="doc-administration"
    )
    assert filtered and all(item["document_id"] == "doc-administration" for item in filtered)

    with patch("src.hybrid_retrieval.semantic_search", return_value=paraphrase_rank):
        fallback = hybrid_search("continuity", object(), object(), None, top_k=2)
    assert fallback and all(item["retrieval_source"] == "Semantic" for item in fallback)

    with (
        patch("src.hybrid_retrieval.semantic_search", return_value=paraphrase_rank),
        patch("src.hybrid_retrieval.keyword_search", side_effect=RuntimeError("test failure")),
        patch("src.hybrid_retrieval.logger.exception"),
    ):
        failed_bm25 = hybrid_search(
            "continuity", object(), object(), keyword_index, top_k=2
        )
    assert failed_bm25 and all(
        item["retrieval_source"] == "Semantic" for item in failed_bm25
    )


def _key(item: dict) -> tuple[str, str]:
    return item["document_id"], item["chunk_id"]


if __name__ == "__main__":
    run_checks()
    print("Hybrid retrieval checks passed.")
