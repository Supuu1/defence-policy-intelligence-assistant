"""Gemini primary, quota failover, transient retry, and schema checks."""

from pathlib import Path
import sys
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import src.rag_service as rag  # noqa: E402


class ProviderError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class Response:
    def __init__(self, parsed):
        self.parsed = parsed
        self.text = ""
        self.candidates = []


class ScriptedModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs["model"])
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class Client:
    def __init__(self, outcomes):
        self.models = ScriptedModels(outcomes)


def valid_response():
    return Response(
        rag.GeminiGroundedResponse(
            answer="The evidence supports the answer [E1].",
            evidence_ids=["E1"],
            sufficient_evidence=True,
        )
    )


def generate(client):
    with patch("src.rag_service.get_gemini_client", return_value=client):
        return rag.generate_structured_content(
            "Grounded prompt",
            "Use evidence only",
            rag.GeminiGroundedResponse,
            1000,
        )


def main():
    primary = Client([valid_response()])
    assert rag.parse_structured_response(
        generate(primary), rag.GeminiGroundedResponse
    ).evidence_ids == ["E1"]
    assert primary.models.calls == [rag.PRIMARY_GEMINI_MODEL]

    # Minute rate limits retry the same configured model; never switch on quota.
    rate_limited = Client([ProviderError(429, "per minute rate limit"), valid_response()])
    with patch("src.rag_service.time.sleep"):
        assert generate(rate_limited).parsed.answer
    assert rate_limited.models.calls == [rag.PRIMARY_GEMINI_MODEL] * 2

    transient = Client(
        [
            ProviderError(503, "Service unavailable"),
            ProviderError(503, "Temporarily unavailable"),
            valid_response(),
        ]
    )
    with patch("src.rag_service.time.sleep") as sleep, patch("src.rag_service.random.uniform", return_value=0):
        assert generate(transient).parsed.answer
    assert transient.models.calls == [
        rag.PRIMARY_GEMINI_MODEL,
        rag.PRIMARY_GEMINI_MODEL,
        rag.PRIMARY_GEMINI_MODEL,
    ]
    assert [call.args[0] for call in sleep.call_args_list] == [1.0, 2.0]

    unavailable = Client([ProviderError(429, "daily quota exhausted")])
    with patch("src.rag_service.time.sleep") as sleep:
        try:
            generate(unavailable)
            raise AssertionError("Daily quota must fail without retry or model switching")
        except rag.GeminiRequestError as error:
            assert "daily quota" in str(error)
        sleep.assert_not_called()
    assert unavailable.models.calls == [rag.PRIMARY_GEMINI_MODEL]
    print("Gemini configured model, rate-limit retry, daily quota, and schema checks passed.")


if __name__ == "__main__":
    main()
