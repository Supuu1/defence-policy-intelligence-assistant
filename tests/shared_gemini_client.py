"""Verify Q&A, summaries, and comparisons share one Gemini client path."""

import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import src.intelligence_service as intelligence  # noqa: E402
import src.rag_service as rag  # noqa: E402


class FakeResponse:
    def __init__(self, parsed):
        self.parsed = parsed
        self.text = ""


class FakeModels:
    def generate_content(self, **kwargs):
        schema = kwargs["config"].response_schema
        if schema is rag.GeminiGroundedResponse:
            return FakeResponse(
                schema(answer="Grounded answer.", evidence_ids=["E1"], sufficient_evidence=True)
            )
        if schema is intelligence.ExecutiveSummarySchema:
            return FakeResponse(
                schema(
                    purpose_scope=[
                        intelligence.GroundedPoint(text="Grounded summary.", evidence_ids=["S1"])
                    ]
                )
            )
        return FakeResponse(
            schema(
                differences=[
                    intelligence.GroundedPoint(
                        text="Grounded difference.", evidence_ids=["A1", "B1"]
                    )
                ]
            )
        )


class FakeClient:
    instances = 0

    def __init__(self, **_kwargs):
        FakeClient.instances += 1
        self.models = FakeModels()


class FakeGenAI:
    Client = FakeClient


def chunk(evidence_id, document_id):
    return {
        "evidence_id": evidence_id,
        "text": "Policy evidence.",
        "source_filename": f"{document_id}.pdf",
        "document_id": document_id,
        "page_number": 1,
        "chunk_id": "page-1-chunk-1",
        "extraction_method": "Native text",
        "similarity": 0.8,
    }


def main():
    previous_key = os.environ.get("GEMINI_API_KEY")
    previous_genai = rag.genai
    try:
        os.environ["GEMINI_API_KEY"] = "test-key-not-a-real-secret"
        rag.genai = FakeGenAI()
        rag.get_gemini_client.cache_clear()

        answer = rag.generate_grounded_answer("Question?", [chunk("E1", "doc-a")])
        summary = intelligence.generate_executive_summary([chunk("S1", "doc-a")])
        comparison = intelligence.generate_policy_comparison(
            [chunk("A1", "doc-a"), chunk("B1", "doc-b")],
            "doc-a.pdf",
            "doc-b.pdf",
            "Compare requirements",
        )
        assert answer.sufficient_evidence and summary.sections and comparison.sections
        assert FakeClient.instances == 1
        assert rag.get_gemini_availability() == (True, "Gemini configured")
        print("Q&A, Executive Summary, and Policy Comparison reused one Gemini client.")
    finally:
        rag.get_gemini_client.cache_clear()
        rag.genai = previous_genai
        if previous_key is None:
            os.environ.pop("GEMINI_API_KEY", None)
        else:
            os.environ["GEMINI_API_KEY"] = previous_key


if __name__ == "__main__":
    main()
