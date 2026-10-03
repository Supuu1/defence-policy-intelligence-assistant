"""Diagnostic ordering and confidentiality tests, with no live API calls."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
from contextlib import redirect_stdout
from io import StringIO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.verify_gemini import verify
from src import rag_service as rag


class DiagnosticTests(unittest.TestCase):
    def run_probe(self, client, pdf=False):
        output = StringIO()
        with patch.object(rag, 'get_gemini_client', return_value=client), patch.object(rag, 'get_gemini_model', return_value=rag.PRIMARY_GEMINI_MODEL), redirect_stdout(output):
            status = verify(pdf)
        return status, output.getvalue()

    def client(self):
        models = Mock()
        models.list.return_value = [SimpleNamespace(name=f'models/{rag.PRIMARY_GEMINI_MODEL}', supported_actions=['generateContent'])]
        models.generate_content.return_value = SimpleNamespace(text='OK', parsed=None, candidates=[])
        return SimpleNamespace(models=models)

    def test_minimal_request_same_client_and_model(self):
        client = self.client()
        status, output = self.run_probe(client)
        self.assertEqual(status, 0)
        client.models.generate_content.assert_called_once_with(model=rag.PRIMARY_GEMINI_MODEL, contents='Reply OK')
        self.assertEqual([call[0] for call in client.models.mock_calls], ['list', 'generate_content'])
        self.assertIn('Minimal Reply OK request: passed', output)

    def test_unsupported_model_stops_before_generation(self):
        client = self.client()
        client.models.list.return_value = []
        status, output = self.run_probe(client, pdf=True)
        self.assertEqual(status, 1)
        client.models.generate_content.assert_not_called()
        self.assertIn('Verification failed at models.list', output)

    def test_failed_minimal_request_does_not_summarize_or_retry(self):
        from google.genai.errors import ServerError
        client = self.client()
        client.models.generate_content.side_effect = ServerError(503, {'error': {'status': 'UNAVAILABLE', 'message': 'source-private-secret-key'}})
        with patch('src.intelligence_service.generate_executive_summary_from_chunks') as summary, self.assertLogs(rag.logger, level='WARNING') as logs:
            status, output = self.run_probe(client, pdf=True)
        self.assertEqual(status, 1)
        self.assertEqual(client.models.generate_content.call_count, 1)
        summary.assert_not_called()
        self.assertNotIn('source-private-secret-key', output + ' '.join(logs.output))
        self.assertIn('ServerError', output)
        self.assertIn('status=503', output)

    def test_pdf_test_runs_after_minimal_request(self):
        client = self.client()
        def summary(chunks):
            client.models.generate_content.assert_called_once_with(model=rag.PRIMARY_GEMINI_MODEL, contents='Reply OK')
            self.assertEqual(chunks[0]['document_id'], 'synthetic-diagnostic')
            return SimpleNamespace(sections={'Purpose': []}, cited_evidence=[chunks[0]])
        with patch('src.intelligence_service.generate_executive_summary_from_chunks', side_effect=summary) as generation:
            status, output = self.run_probe(client, pdf=True)
        self.assertEqual(status, 0)
        generation.assert_called_once()
        self.assertIn('Synthetic PDF summary: passed', output)


if __name__ == '__main__':
    unittest.main()
