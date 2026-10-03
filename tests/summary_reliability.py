"""Offline summary regression tests. No real credentials or document content logged."""
from pathlib import Path
import sys
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import rag_service as rag, intelligence_service as intel


class ProviderError(Exception):
    def __init__(self, code, message="sanitized test failure", headers=None):
        super().__init__(message)
        self.code = code
        self.response = SimpleNamespace(headers=headers or {})


def chunk(text="The authority requires annual policy reports and approval.", document="selected", page=1):
    return dict(text=text, document_id=document, page_number=page, chunk_id=f"page-{page}",
                source_filename="source.pdf", extraction_method="Native text", similarity=.8)


def response(**kwargs):
    return SimpleNamespace(parsed=None, text="", candidates=[], **kwargs)


class SummaryTests(unittest.TestCase):
    def call(self, outcomes):
        client = SimpleNamespace(models=SimpleNamespace())
        from unittest.mock import Mock
        client.models.generate_content = Mock(side_effect=outcomes)
        with patch.object(rag, "get_gemini_client", return_value=client), patch.object(rag, "get_gemini_model", return_value=rag.PRIMARY_GEMINI_MODEL):
            try:
                result = rag.generate_structured_content("placeholder", "evidence only", intel.ExecutiveSummarySchema, 2400)
                return result, client.models.generate_content
            except Exception as error:
                return error, client.models.generate_content

    def test_original_traceback_and_message_are_safely_logged(self):
        from google.genai.errors import ServerError
        private_text = "PRIVATE_POLICY_PARAGRAPH"
        key = "AIza-private-placeholder"
        try:
            try:
                raise ValueError(f"Authorization: Bearer {key}; contents={private_text}")
            except ValueError as cause:
                raise ServerError(503, {"error": {
                    "status": "UNAVAILABLE", "message": "This model is overloaded. Please try again later.",
                    "headers": {"x-goog-api-key": key}, "contents": private_text,
                }}) from cause
        except ServerError as error:
            with self.assertLogs(rag.logger, level="WARNING") as capture:
                rag.log_gemini_exception(error, rag.PRIMARY_GEMINI_MODEL, 1)
        output = "\n".join(capture.output)
        for expected in ["ServerError", "status=503", "UNAVAILABLE", "model is overloaded",
                         "Traceback", "test_original_traceback_and_message_are_safely_logged", "ValueError"]:
            self.assertIn(expected, output)
        for private in [private_text, key, "Authorization", "x-goog-api-key", "Bearer"]:
            self.assertNotIn(private, output)

    def test_model_resolution_environment_secrets_and_default(self):
        import streamlit as st
        with patch.object(rag, "load_dotenv"), patch.dict(os.environ, {"GEMINI_MODEL": "env-model"}), patch.object(st, "secrets", {"GEMINI_MODEL": "cloud-model"}):
            self.assertEqual(rag.get_gemini_model(), "env-model")
        with patch.object(rag, "load_dotenv"), patch.dict(os.environ, {"GEMINI_MODEL": " "}), patch.object(st, "secrets", {"GEMINI_MODEL": " cloud-model "}):
            self.assertEqual(rag.get_gemini_model(), "cloud-model")
        with patch.object(rag, "load_dotenv"), patch.dict(os.environ, {"GEMINI_MODEL": ""}), patch.object(st, "secrets", {}):
            self.assertEqual(rag.get_gemini_model(), rag.PRIMARY_GEMINI_MODEL)
        with patch.object(rag, "load_dotenv"), patch.dict(os.environ, {"GEMINI_MODEL": "invalid\nmodel"}):
            with self.assertRaises(rag.RAGServiceError):
                rag.get_gemini_model()

    def test_text_unavailable_does_not_prove_transient_failure(self):
        self.assertFalse(rag._is_transient_error(Exception("Model unavailable for this key")))
        self.assertEqual(rag._error_category(ProviderError(400, "Model not found")), "model_configuration")
        self.assertFalse(rag._is_transient_error(ProviderError(400, "Model not found")))
        self.assertEqual(rag._error_category(ProviderError(400, "API key not valid")), "authentication")

    def test_success_and_transient_retry(self):
        good = response()
        with patch.object(rag.time, "sleep") as sleep, patch.object(rag.random, "uniform", return_value=.25):
            result, calls = self.call([ProviderError(503), good])
        self.assertIs(result, good)
        self.assertEqual(calls.call_count, 2)
        sleep.assert_called_once_with(1.25)

    def test_exhaustion_and_sanitized_log(self):
        with patch.object(rag.time, "sleep"), self.assertLogs(rag.logger, level="WARNING") as logs:
            result, calls = self.call([ProviderError(503, "secret-key source-content") for _ in range(3)])
        self.assertIsInstance(result, rag.GeminiRequestError)
        self.assertEqual(calls.call_count, 3)
        self.assertNotIn("secret-key", " ".join(logs.output))
        self.assertNotIn("source-content", " ".join(logs.output))
        self.assertIn("status=503", " ".join(logs.output))
        self.assertIn("ProviderError", " ".join(logs.output))

    def test_nonretryable_credentials_model_input_daily_quota(self):
        for code, text, expected in [(401, "invalid", "API key"), (403, "denied", "permissions"),
                                     (404, "missing", "model"), (400, "bad", "parameters"),
                                     (429, "GenerateRequestsPerDay quota exhausted", "daily quota")]:
            with self.subTest(code=code), patch.object(rag.time, "sleep") as sleep:
                result, calls = self.call([ProviderError(code, text)])
                self.assertIsInstance(result, rag.GeminiRequestError)
                self.assertIn(expected, str(result))
                self.assertEqual(calls.call_count, 1)
                sleep.assert_not_called()

    def test_rate_limit_retry_after_and_long_wait(self):
        with patch.object(rag.time, "sleep") as sleep:
            result, calls = self.call([ProviderError(429, "per minute", {"Retry-After": "7"}), response()])
        self.assertEqual(calls.call_count, 2)
        sleep.assert_called_once_with(7)
        with patch.object(rag.time, "sleep") as sleep:
            result, calls = self.call([ProviderError(429, "per minute", {"Retry-After": "120"})])
        self.assertEqual(calls.call_count, 1)
        sleep.assert_not_called()
        self.assertIsInstance(result, rag.GeminiRequestError)

    def test_network_timeout(self):
        for error in [TimeoutError(), ConnectionError()]:
            with patch.object(rag.time, "sleep"):
                result, calls = self.call([error, response()])
                self.assertEqual(calls.call_count, 2)

    def test_sdk_error_details_and_retry_info(self):
        from google.genai.errors import ClientError
        error = ClientError(429, {"error": {"message": "quota reached", "details": [
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
                {"quotaId": "GenerateRequestsPerDayPerProject", "quotaValue": "20"}]},
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "12s"}
        ]}})
        self.assertEqual(rag._error_category(error), "quota_exhausted")
        self.assertEqual(rag._retry_after(error), 12)
        minute = ClientError(429, {"error": {"message": "quota exhausted per minute"}})
        self.assertEqual(rag._error_category(minute), "rate_limit")
        zero = ClientError(429, {"error": {"details": [{"quotaValue": "0"}]}})
        self.assertEqual(rag._error_category(zero), "quota_exhausted")

    def test_selected_pdf_extraction_to_parsed_summary(self):
        import fitz
        from src.document_processing import process_pdf
        pdf = fitz.open()
        page = pdf.new_page()
        page.insert_text((72, 72), "The director requires annual policy review and approves all annual reports.")
        data = pdf.tobytes()
        pdf.close()
        pages, _, chunks = process_pdf(data, "selected.pdf", "selected", include_preview=False)
        parsed = intel.ExecutiveSummarySchema(purpose_scope=[
            intel.GroundedPoint(text="Annual policy review is required.", evidence_ids=["S1"])])
        with patch.object(rag, "generate_structured_content", return_value=SimpleNamespace(parsed=parsed, text="", candidates=[])):
            result = intel.generate_executive_summary_from_chunks(chunks)
        self.assertEqual(pages[0]["extraction_method"], "Native text")
        self.assertEqual(result.cited_evidence[0]["document_id"], "selected")
        self.assertEqual(result.cited_evidence[0]["page_number"], 1)

    def test_empty_text_no_call(self):
        with patch.object(intel, "generate_structured_response") as generate:
            for chunks in [[], [chunk(" \n")], [chunk("\x00\x00")], [chunk(None)]]:
                with self.assertRaises(intel.NoSummarizableContentError):
                    intel.generate_executive_summary_from_chunks(chunks)
            generate.assert_not_called()

    def test_empty_safety_and_malformed_responses(self):
        for value in [response(), response(prompt_feedback=SimpleNamespace(block_reason="SAFETY"))]:
            with patch.object(rag, "generate_structured_content", return_value=value) as generate:
                with self.assertRaises(rag.MalformedStructuredResponseError):
                    rag.generate_structured_response("p", "s", intel.ExecutiveSummarySchema, 2400)
                self.assertEqual(generate.call_count, 1)
        malformed = response()
        malformed.text = "not JSON"
        with patch.object(rag, "generate_structured_content", return_value=malformed) as generate:
            with self.assertRaises(rag.MalformedStructuredResponseError):
                rag.generate_structured_response("p", "s", intel.ExecutiveSummarySchema, 2400)
            self.assertEqual(generate.call_count, 2)
        with patch.object(intel, "generate_structured_response", return_value=intel.ExecutiveSummarySchema()):
            with self.assertRaises(rag.MalformedStructuredResponseError):
                intel.generate_executive_summary_from_chunks([chunk()])

    def test_large_multilingual_hierarchy_preserves_scope_and_pages(self):
        calls = []
        def generate(prompt, *args, **kwargs):
            import re
            self.assertLessEqual(intel.estimate_text_tokens(prompt), intel.SUMMARY_INPUT_TOKEN_BUDGET)
            self.assertNotIn("unselected", prompt)
            refs = re.findall(r'<evidence id="(S\d+)">', prompt)
            calls.append(refs)
            return intel.ExecutiveSummarySchema(purpose_scope=[intel.GroundedPoint(text="Policy requirements.", evidence_ids=[ref]) for ref in refs[:3]])
        chunks = [chunk("政策文本。" * 7000, page=1), chunk("Annual approvals are required. " * 2500, page=2)]
        with patch.object(intel, "generate_structured_response", side_effect=generate):
            result = intel.generate_executive_summary_from_chunks(chunks)
        self.assertGreater(len(calls), 3)
        self.assertEqual({i["document_id"] for i in result.cited_evidence}, {"selected"})
        self.assertEqual({i["page_number"] for i in result.cited_evidence}, {1, 2})
        valid_ids = {i["evidence_id"] for i in result.cited_evidence}
        self.assertTrue(all(set(p["evidence_ids"]) <= valid_ids for points in result.sections.values() for p in points))

    def test_rerun_cache_isolation_and_failed_requests(self):
        chunks = [chunk()]
        good = intel.IntelligenceResult({"Purpose": [{"text": "Policy", "evidence_ids": ["S1"]}]}, [], "Low")
        cache, other_user = {}, {}
        with patch.object(intel, "get_gemini_model", return_value=rag.PRIMARY_GEMINI_MODEL), patch.object(intel, "generate_executive_summary_from_chunks", return_value=good) as generate:
            self.assertIs(intel.cached_executive_summary(chunks, cache), good)
            self.assertIs(intel.cached_executive_summary(chunks, cache), good)
            self.assertEqual(generate.call_count, 1)
            intel.cached_executive_summary(chunks, other_user)
            self.assertEqual(generate.call_count, 2)
            intel.cached_executive_summary([chunk("Different actual document text.")], cache)
            self.assertEqual(generate.call_count, 3)
            with patch.object(intel, "get_gemini_model", return_value="other-model"):
                intel.cached_executive_summary(chunks, cache)
            self.assertEqual(generate.call_count, 4)
        failed = {}
        with patch.object(intel, "generate_executive_summary_from_chunks", side_effect=rag.GeminiRequestError("failed")):
            with self.assertRaises(rag.GeminiRequestError):
                intel.cached_executive_summary(chunks, failed)
        self.assertEqual(failed, {})

    def test_local_extract_is_actual_source(self):
        chunks = [chunk(), chunk("The director must approve exceptional requests.", page=2)]
        with patch.object(rag, "get_gemini_client") as api:
            result = intel.local_extractive_summary(chunks)
            api.assert_not_called()
        for points in result.sections.values():
            for point in points:
                self.assertTrue(any(point["text"] in c["text"] for c in chunks))
        self.assertEqual({i["page_number"] for i in result.cited_evidence}, {1, 2})

    def test_key_sources_and_sdk_retry_settings(self):
        import streamlit as st
        with patch.object(rag, "load_dotenv"), patch.dict(os.environ, {"GEMINI_API_KEY": " ", "GOOGLE_API_KEY": ""}), patch.object(st, "secrets", {"GEMINI_API_KEY": " cloud-placeholder "}):
            self.assertEqual(rag._resolve_gemini_api_key(), "cloud-placeholder")
        rag.get_gemini_client.cache_clear()
        with patch.object(rag, "_resolve_gemini_api_key", return_value="placeholder"), patch.object(rag.genai, "Client") as client:
            rag.get_gemini_client()
            self.assertEqual(client.call_args.kwargs["http_options"].retry_options.attempts, 1)
            self.assertEqual(client.call_args.kwargs["http_options"].timeout, 45000)
        rag.get_gemini_client.cache_clear()


if __name__ == "__main__":
    unittest.main()
