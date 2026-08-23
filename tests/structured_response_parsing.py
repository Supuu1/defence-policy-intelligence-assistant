"""Structured Gemini parsing, repair retry, and controlled-failure checks."""

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import src.rag_service as rag  # noqa: E402
from src.intelligence_service import (  # noqa: E402
    ExecutiveSummarySchema,
    GroundedPoint,
    PolicyComparisonSchema,
)


class Candidate:
    finish_reason = "MAX_TOKENS"


class Response:
    def __init__(self, parsed=None, text=""):
        self.parsed = parsed
        self.text = text
        self.candidates = [Candidate()]


def main():
    typed_summary = ExecutiveSummarySchema(
        purpose_scope=[GroundedPoint(text="Purpose with a \"quote\".", evidence_ids=["S1"])],
        important_dates_thresholds=[
            GroundedPoint(text="Review occurs annually.\nNo date is invented.", evidence_ids=["S2"])
        ],
        supporting_evidence_ids=["S1", "S2"],
    )
    parsed = rag.parse_structured_response(Response(parsed=typed_summary), ExecutiveSummarySchema)
    assert parsed.purpose_scope[0].evidence_ids == ["S1"]

    fenced = "```json\n" + typed_summary.model_dump_json() + "\n```"
    parsed_fenced = rag.parse_structured_response(Response(text=fenced), ExecutiveSummarySchema)
    assert "quote" in parsed_fenced.purpose_scope[0].text

    comparison = PolicyComparisonSchema(
        differences=[
            GroundedPoint(text="The provisions differ.", evidence_ids=["A1", "B1"])
        ]
    )
    comparison_text = "Model preface\n" + json.dumps(comparison.model_dump())
    parsed_comparison = rag.parse_structured_response(
        Response(text=comparison_text), PolicyComparisonSchema
    )
    assert parsed_comparison.differences[0].evidence_ids == ["A1", "B1"]

    partial_qa = Response(
        text=(
            '{"answer":"Annual review is required [E1].",'
            '"evidence_ids":[" e1 "]'
        )
    )
    recovered_qa = rag.parse_structured_response(
        partial_qa, rag.GeminiGroundedResponse
    )
    assert recovered_qa.answer.startswith("Annual review")
    assert rag.validate_evidence_ids(
        recovered_qa.evidence_ids,
        [{"evidence_id": "E1"}],
    ) == ["E1"]

    original_structured_response = rag.generate_structured_response
    try:
        rag.generate_structured_response = lambda *_args, **_kwargs: recovered_qa
        grounded = rag.generate_grounded_answer(
            "What review is required?",
            [{"evidence_id": "E1", "text": "Annual review is required."}],
        )
        assert grounded.answer and grounded.evidence_ids == ["E1"]

        rag.generate_structured_response = lambda *_args, **_kwargs: rag.GeminiGroundedResponse(
            answer="Unverifiable answer.", evidence_ids=["E99"], sufficient_evidence=True
        )
        try:
            rag.generate_grounded_answer(
                "Question?", [{"evidence_id": "E1", "text": "Evidence."}]
            )
            raise AssertionError("Unknown evidence IDs must not be accepted")
        except rag.MalformedStructuredResponseError:
            pass
    finally:
        rag.generate_structured_response = original_structured_response

    original_generate = rag.generate_structured_content
    calls = []
    try:
        def malformed_then_valid(prompt, system_instruction, response_schema, max_output_tokens):
            calls.append((prompt, max_output_tokens))
            if len(calls) == 1:
                return Response(text='{"purpose_scope": [{"text": "unterminated')
            return Response(parsed=typed_summary)

        rag.generate_structured_content = malformed_then_valid
        repaired = rag.generate_structured_response(
            "Summarize evidence containing quotes, newlines, and {braces}.",
            "Use only evidence.",
            ExecutiveSummarySchema,
            1800,
            task_type="executive_summary",
            retry_output_tokens=3000,
        )
        assert repaired.purpose_scope and len(calls) == 2
        assert calls[1][1] == 3000 and "RETRY REQUIREMENTS" in calls[1][0]

        def always_malformed(*_args, **_kwargs):
            return Response(text='```json\n{"purpose_scope": [')

        rag.generate_structured_content = always_malformed
        try:
            rag.generate_structured_response(
                "Summarize.", "Use evidence.", ExecutiveSummarySchema, 1800
            )
            raise AssertionError("Malformed output should have raised a controlled error")
        except rag.RAGServiceError as error:
            assert "structured-response token budget" in str(error)
    finally:
        rag.generate_structured_content = original_generate

    print("Typed, fenced, partial-Q&A recovery, citation validation, repair-retry, controlled-failure, and comparison parsing passed.")


if __name__ == "__main__":
    main()
