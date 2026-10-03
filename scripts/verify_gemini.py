"""Verify the application Gemini configuration without printing secrets/content.

Run `python scripts/verify_gemini.py --pdf-smoke` locally or in the deployment
runtime. The optional PDF test runs only after model support and Reply OK pass.
It sends a small synthetic source, never an uploaded document.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import rag_service as rag


def verify(pdf_smoke=False):
    stage = "configuration"
    model = "unresolved"
    try:
        import google.genai
        model = rag.get_gemini_model()
        client = rag.get_gemini_client()
        print(f"SDK: google-genai {google.genai.__version__}; resolved model: {model}")
        stage = "models.list"
        available = client.models.list()
        supported = any(
            item.name == f"models/{model.removeprefix('models/')}"
            and "generateContent" in (item.supported_actions or [])
            for item in available
        )
        if not supported:
            raise rag.RAGServiceError("Configured model is not listed with generateContent support for this key. No generation attempted.")
        print("Configured model listed with generateContent support: yes")
        stage = "Reply OK"
        # One direct request, no application retry or fallback, same client/model.
        reply = client.models.generate_content(model=model, contents="Reply OK")
        rag._validate_response_status(reply)
        if (reply.text or "").strip().strip('"') != "OK":
            raise rag.RAGServiceError("Reply OK did not return the expected response; response content withheld.")
        print("Minimal Reply OK request: passed")
        if pdf_smoke:
            stage = "synthetic PDF summary"
            import fitz
            from src.document_processing import process_pdf
            from src.intelligence_service import generate_executive_summary_from_chunks
            pdf = fitz.open()
            try:
                page = pdf.new_page()
                page.insert_text((72, 72), "The policy requires an annual review. The director approves each annual review.")
                data = pdf.tobytes()
            finally:
                pdf.close()
            _, _, chunks = process_pdf(data, "synthetic-diagnostic.pdf", "synthetic-diagnostic", include_preview=False)
            result = generate_executive_summary_from_chunks(chunks)
            if not result.sections or not result.cited_evidence:
                raise rag.RAGServiceError("Synthetic PDF summary returned no validated sections or citations.")
            print(f"Synthetic PDF summary: passed; sections={len(result.sections)}; page_references={len(result.cited_evidence)}")
        return 0
    except Exception as error:
        cause = error.__cause__ or error
        rag.log_gemini_exception(cause, model)
        print(f"Verification failed at {stage}: class={type(cause).__name__}; status={rag._error_status_code(cause)}; category={rag._error_category(cause)}")
        # Only our fixed, safe application messages may be displayed verbatim.
        if isinstance(error, rag.RAGServiceError):
            print(str(error))
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-smoke", action="store_true")
    sys.exit(verify(parser.parse_args().pdf_smoke))
