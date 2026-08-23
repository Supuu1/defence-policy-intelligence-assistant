"""End-to-end local checks for text/OCR evidence used by intelligence features."""

from io import BytesIO
from pathlib import Path
import sys

import fitz
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import src.intelligence_service as intelligence_service  # noqa: E402
from src.document_processing import process_pdf  # noqa: E402
from src.report_service import build_markdown_report  # noqa: E402
from src.semantic_retrieval import (  # noqa: E402
    EMBEDDING_MODEL_NAME,
    build_vector_index,
    load_embedding_model,
    semantic_search,
)


def text_pdf(filename, text):
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_textbox(fitz.Rect(50, 50, 545, 790), text * 6, fontsize=11)
    data = pdf.tobytes()
    pdf.close()
    return process_pdf(data, filename, filename, include_preview=False)


def ocr_pdf(filename):
    image = Image.new("RGB", (1600, 1000), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 38)
    except OSError:
        font = ImageFont.load_default()
    draw.multiline_text(
        (80, 100),
        "SCANNED SECURITY POLICY\nReporting is required within 30 days.\n"
        "The Security Director approves exceptional access.",
        fill="black",
        font=font,
        spacing=20,
    )
    png = BytesIO()
    image.save(png, format="PNG")
    image.close()
    pdf = fitz.open()
    page = pdf.new_page(width=800, height=500)
    page.insert_image(page.rect, stream=png.getvalue())
    data = pdf.tobytes(deflate=True)
    pdf.close()
    return process_pdf(data, filename, filename, include_preview=False)


def main():
    a_pages, _, a_chunks = text_pdf(
        "eligibility.pdf",
        "The policy establishes eligibility requirements. Applicants must be 18 years old. ",
    )
    b_pages, _, b_chunks = text_pdf(
        "reporting.pdf",
        "The reporting policy requires quarterly reports and manager approval. ",
    )
    o_pages, _, o_chunks = ocr_pdf("scanned-security.pdf")
    assert a_pages[0]["extraction_method"] == "Native text"
    assert o_pages[0]["extraction_method"] == "OCR"

    chunks = a_chunks + b_chunks + o_chunks
    model = load_embedding_model()
    index = build_vector_index(chunks, model)

    summary_text = semantic_search(
        "purpose eligibility requirements", index, model, top_k=4, document_id="eligibility.pdf"
    )
    summary_ocr = semantic_search(
        "security reporting deadline", index, model, top_k=4, document_id="scanned-security.pdf"
    )
    summary_corpus = semantic_search("policy requirements", index, model, top_k=4)
    assert summary_text and summary_ocr and len({x["document_id"] for x in chunks}) == 3

    original_generate = intelligence_service._generate
    try:
        def fake_generate(_prompt, schema, *_args):
            if schema is intelligence_service.ExecutiveSummarySchema:
                return schema(
                    purpose_scope=[
                        intelligence_service.GroundedPoint(text="Supported purpose.", evidence_ids=["S1"])
                    ]
                )
            return schema(
                differences=[
                    intelligence_service.GroundedPoint(
                        text="The retrieved requirements differ.", evidence_ids=["A1", "B1"]
                    )
                ]
            )

        intelligence_service._generate = fake_generate
        for retrieved in (summary_text, summary_ocr, summary_corpus):
            prepared = intelligence_service.prepare_evidence(retrieved, "S")
            result = intelligence_service.generate_executive_summary(prepared)
            assert result.sections and result.cited_evidence

        evidence_a = intelligence_service.prepare_evidence(
            semantic_search("requirements", index, model, top_k=4, document_id="eligibility.pdf"), "A"
        )
        evidence_b = intelligence_service.prepare_evidence(
            semantic_search("requirements", index, model, top_k=4, document_id="reporting.pdf"), "B"
        )
        general = intelligence_service.generate_policy_comparison(
            evidence_a + evidence_b, "eligibility.pdf", "reporting.pdf", ""
        )
        focused = intelligence_service.generate_policy_comparison(
            evidence_a + evidence_b, "eligibility.pdf", "reporting.pdf", "Compare requirements"
        )
        mixed = intelligence_service.generate_policy_comparison(
            evidence_a
            + intelligence_service.prepare_evidence(summary_ocr, "B"),
            "eligibility.pdf",
            "scanned-security.pdf",
            "Compare reporting obligations",
        )
        assert general.sections and focused.sections and mixed.sections
        assert {item["document_id"] for item in mixed.cited_evidence} == {
            "eligibility.pdf",
            "scanned-security.pdf",
        }

        empty = intelligence_service._validated_result(
            intelligence_service.PolicyComparisonSchema(),
            evidence_a + evidence_b,
            intelligence_service.COMPARISON_SECTIONS,
        )
        assert empty.sections == {}

        documents = [
            {"filename": "eligibility.pdf", "pages": a_pages, "chunks": a_chunks, "status": "Processed"},
            {"filename": "scanned-security.pdf", "pages": o_pages, "chunks": o_chunks, "status": "Processed"},
        ]
        for analysis_type, result in (("Executive Summary", focused), ("Policy Comparison", mixed)):
            sources = [
                {
                    "source_filename": item["source_filename"],
                    "page_number": item["page_number"],
                    "chunk_id": item["chunk_id"],
                }
                for item in result.cited_evidence
            ]
            report = build_markdown_report(
                analysis_type,
                documents,
                "Grounded result",
                result.evidence_strength,
                sources,
                result.cited_evidence,
                embedding_model=EMBEDDING_MODEL_NAME,
            )
            assert "Defence Policy Intelligence Report" in report
            assert all(source["source_filename"] in report for source in sources)
    finally:
        intelligence_service._generate = original_generate

    print(
        "Text, OCR, corpus summary, text/text and text/OCR comparison, focused/general/empty "
        "comparison, citation, and two report-export checks passed."
    )


if __name__ == "__main__":
    main()
