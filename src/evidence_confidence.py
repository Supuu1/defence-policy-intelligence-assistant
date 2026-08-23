"""Deterministic, application-side evidence confidence assessment."""

from dataclasses import dataclass
import re


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[./-][A-Za-z0-9]+)*")
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "the", "these", "this",
    "to", "was", "were", "what", "when", "where", "which", "who", "with",
}
POSITIVE_MODALITIES = {"must", "shall", "required", "mandatory", "permitted", "allowed"}
NEGATIVE_PHRASES = {"must not", "shall not", "not permitted", "not allowed", "prohibited", "forbidden"}


@dataclass(frozen=True)
class EvidenceConfidence:
    """A qualitative evidence assessment, never a probability."""

    level: str
    reason: str
    supporting_chunks: int
    distinct_documents: int
    citation_coverage: float
    query_term_coverage: float
    hybrid_agreement: float
    conflict_detected: bool


def _tokens(text: str) -> set[str]:
    return {
        match.group(0).lower()
        for match in TOKEN_PATTERN.finditer(text)
        if match.group(0).lower() not in STOP_WORDS
        and (len(match.group(0)) >= 3 or any(character.isdigit() for character in match.group(0)))
    }


def _explicit_conflict(chunks: list[dict], query_terms: set[str]) -> bool:
    """Conservatively flag directly opposed modalities about overlapping terms."""
    if any(chunk.get("conflict_detected") is True for chunk in chunks):
        return True
    classified = []
    for chunk in chunks:
        text = chunk.get("text", "").lower()
        terms = _tokens(text)
        negative = any(phrase in text for phrase in NEGATIVE_PHRASES)
        positive = any(term in terms for term in POSITIVE_MODALITIES) and not negative
        classified.append((terms, positive, negative))
    for position, (left_terms, left_positive, left_negative) in enumerate(classified):
        for right_terms, right_positive, right_negative in classified[position + 1 :]:
            subject_overlap = (left_terms & right_terms) - POSITIVE_MODALITIES
            if query_terms:
                subject_overlap &= query_terms
            if subject_overlap and (
                (left_positive and right_negative) or (left_negative and right_positive)
            ):
                return True
    return False


def calculate_evidence_confidence(
    query: str,
    retrieved_chunks: list[dict],
    supporting_chunks: list[dict],
    selected_chunk_count: int,
    sufficient_evidence: bool = True,
) -> EvidenceConfidence:
    """Assess evidence using retrieval metadata and validated citation coverage."""
    support_count = len(supporting_chunks)
    document_count = len(
        {chunk.get("document_id") for chunk in supporting_chunks if chunk.get("document_id")}
    )
    citation_coverage = support_count / max(selected_chunk_count, 1)
    query_terms = _tokens(query)
    supporting_terms = set().union(
        *(_tokens(chunk.get("text", "")) for chunk in supporting_chunks)
    ) if supporting_chunks else set()
    query_term_coverage = (
        len(query_terms & supporting_terms) / len(query_terms) if query_terms else 0.0
    )
    hybrid_agreement = (
        sum(chunk.get("retrieval_source") == "Both" for chunk in supporting_chunks)
        / support_count
        if support_count
        else 0.0
    )
    semantic_scores = [
        score
        for chunk in supporting_chunks
        for score in [chunk.get("semantic_similarity", chunk.get("similarity"))]
        if score is not None
    ]
    top_score = max(semantic_scores, default=0.0)
    lexical_support = any(
        chunk.get("retrieval_source") in {"Keyword", "Both"}
        and chunk.get("keyword_score") is not None
        for chunk in supporting_chunks
    )
    conflict_detected = _explicit_conflict(supporting_chunks, query_terms)

    insufficient = (
        not sufficient_evidence
        or support_count == 0
        or (
            top_score < 0.20
            and not lexical_support
            and query_term_coverage < 0.20
        )
    )
    if insufficient:
        return EvidenceConfidence(
            "INSUFFICIENT",
            "Insufficient — no adequately relevant, validated evidence supports an answer.",
            support_count,
            document_count,
            citation_coverage,
            query_term_coverage,
            hybrid_agreement,
            conflict_detected,
        )

    score = 0
    score += 2 if top_score >= 0.60 else 1 if top_score >= 0.40 else 0
    score += 2 if support_count >= 3 else 1 if support_count >= 2 else 0
    score += 1 if document_count >= 2 else 0
    score += 2 if citation_coverage >= 0.75 else 1 if citation_coverage >= 0.50 else 0
    score += 1 if query_term_coverage >= 0.60 else 0
    score += 1 if hybrid_agreement >= 0.50 else 0

    if conflict_detected:
        level = "LOW"
        reason = (
            f"Low — {support_count} cited passages include potentially conflicting "
            "directives; review the cited sources."
        )
    elif score >= 7:
        level = "HIGH"
        reason = (
            f"High — {support_count} strongly relevant passages across "
            f"{document_count} document{'s' if document_count != 1 else ''} support the answer."
        )
    elif score >= 4:
        level = "MEDIUM"
        reason = (
            f"Medium — {support_count} relevant passage{'s' if support_count != 1 else ''} "
            f"across {document_count} document{'s' if document_count != 1 else ''} provide "
            "moderate support."
        )
    else:
        level = "LOW"
        reason = (
            f"Low — only {support_count} limited or weakly matched passage"
            f"{'s' if support_count != 1 else ''} support the answer."
        )

    return EvidenceConfidence(
        level,
        reason,
        support_count,
        document_count,
        citation_coverage,
        query_term_coverage,
        hybrid_agreement,
        conflict_detected,
    )
