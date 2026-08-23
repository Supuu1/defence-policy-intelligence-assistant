"""Task-specific Gemini token budgets and MAX_TOKENS retry regression tests."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import src.rag_service as rag  # noqa: E402
from src.intelligence_service import (  # noqa: E402
    COMPARISON_OUTPUT_TOKENS,
    ExecutiveSummarySchema,
    GroundedPoint,
    PolicyComparisonSchema,
    SUMMARY_OUTPUT_TOKENS,
    generate_executive_summary_from_chunks,
    generate_policy_comparison,
    prepare_evidence,
)


class Candidate:
    def __init__(self, reason):
        self.finish_reason = reason


class Response:
    def __init__(self, parsed, reason="STOP", text=""):
        self.parsed = parsed
        self.text = text
        self.candidates = [Candidate(reason)]


def chunk(evidence_id, document_id, text):
    return {
        "evidence_id": evidence_id,
        "text": text,
        "source_filename": f"{document_id}.pdf",
        "document_id": document_id,
        "page_number": 1,
        "chunk_id": "page-1-chunk-1",
        "extraction_method": "Native text",
        "similarity": 0.85,
    }


def parsed_for(schema):
    if schema is rag.GeminiGroundedResponse:
        return schema(
            answer="The resumes share Python and differ in ML tooling [E1] [E2].",
            evidence_ids=["E1", "E2"],
            sufficient_evidence=True,
        )
    if schema is ExecutiveSummarySchema:
        return schema(
            key_themes=[GroundedPoint(text="Technical skills are documented.", evidence_ids=["S1"])]
        )
    return schema(
        differences=[GroundedPoint(text="The technical toolsets differ.", evidence_ids=["A1", "B1"])]
    )


def main():
    a = chunk("E1", "resume-a", "Python, PyTorch, FAISS, and SQL are listed.")
    b = chunk("E2", "resume-b", "Python, TensorFlow, Docker, and Kubernetes are listed.")
    calls = []
    original_generate = rag.generate_structured_content
    try:
        def successful(prompt, system_instruction, response_schema, max_output_tokens):
            calls.append((prompt, response_schema, max_output_tokens))
            return Response(parsed_for(response_schema))

        rag.generate_structured_content = successful
        long_query = rag.generate_grounded_answer(
            "Compare the technical skills mentioned in these two resumes.", [a, b]
        )
        short_query = rag.generate_grounded_answer(
            "Which resume mentions PyTorch?", [a]
        )
        summary = generate_executive_summary_from_chunks([a, b])
        comparison = generate_policy_comparison(
            prepare_evidence([a], "A") + prepare_evidence([b], "B"),
            "resume-a.pdf",
            "resume-b.pdf",
            "Compare technical skills",
        )
        assert long_query.sufficient_evidence and short_query.sufficient_evidence
        assert summary.sections and comparison.sections
        assert [call[2] for call in calls] == [
            rag.QUERY_OUTPUT_TOKENS,
            rag.QUERY_OUTPUT_TOKENS,
            SUMMARY_OUTPUT_TOKENS,
            COMPARISON_OUTPUT_TOKENS,
        ]

        retry_calls = []

        def max_tokens_then_success(prompt, system_instruction, response_schema, max_output_tokens):
            retry_calls.append((prompt, max_output_tokens))
            if len(retry_calls) == 1:
                return Response(None, reason="FinishReason.MAX_TOKENS", text='{"answer":"cut')
            return Response(parsed_for(response_schema), reason="FinishReason.STOP")

        rag.generate_structured_content = max_tokens_then_success
        repaired = rag.generate_structured_response(
            "Question with <evidence id=\"E1\">same evidence</evidence>",
            "Grounded only.",
            rag.GeminiGroundedResponse,
            rag.QUERY_OUTPUT_TOKENS,
            task_type="query_assistant",
            retry_output_tokens=rag.QUERY_RETRY_OUTPUT_TOKENS,
        )
        assert repaired.answer
        assert [call[1] for call in retry_calls] == [1000, 1600]
        assert "same evidence" in retry_calls[0][0] and "same evidence" in retry_calls[1][0]
    finally:
        rag.generate_structured_content = original_generate

    print("Long/short Q&A, summary, comparison, and MAX_TOKENS larger-budget retry passed.")


if __name__ == "__main__":
    main()
