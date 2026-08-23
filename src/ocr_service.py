"""Render PDF pages and extract text with the local Tesseract executable."""

from io import BytesIO
import gc
import logging
import shutil

import fitz
from PIL import Image

try:
    import pytesseract
except ImportError:  # Keep the app usable until Python dependencies are installed.
    pytesseract = None


logger = logging.getLogger(__name__)
MAX_OCR_PIXELS = 20_000_000


class OCRUnavailableError(RuntimeError):
    """Raised when the local OCR runtime is not ready."""


def get_tesseract_status() -> tuple[bool, str]:
    """Return whether both pytesseract and the Tesseract executable are available."""
    if shutil.which("tesseract") is None:
        return False, "Tesseract is not installed or is not available on PATH."
    if pytesseract is None:
        return False, "The pytesseract Python package is not installed."

    try:
        pytesseract.get_tesseract_version()
    except Exception as error:
        return False, f"Tesseract could not be started: {error}"
    return True, "Tesseract OCR is available."


def extract_text_with_ocr(page: fitz.Page, dpi: int = 200) -> str:
    """Render one PDF page to an in-memory image and run Tesseract OCR."""
    available, message = get_tesseract_status()
    if not available:
        raise OCRUnavailableError(message)

    # Reduce DPI only for unusually large page dimensions. This bounds the raw
    # RGB raster to roughly 60 MB while retaining normal pages at 200 DPI.
    width_inches = page.rect.width / 72
    height_inches = page.rect.height / 72
    estimated_pixels = width_inches * dpi * height_inches * dpi
    render_dpi = dpi
    if estimated_pixels > MAX_OCR_PIXELS:
        render_dpi = max(100, int(dpi * (MAX_OCR_PIXELS / estimated_pixels) ** 0.5))

    pixmap = None
    image_bytes = None
    try:
        pixmap = page.get_pixmap(dpi=render_dpi, alpha=False)
        image_bytes = pixmap.tobytes("png")
        with Image.open(BytesIO(image_bytes)) as image:
            grayscale = image.convert("L")
            try:
                return pytesseract.image_to_string(grayscale, timeout=90).strip()
            finally:
                grayscale.close()
    except Exception:
        logger.exception("OCR failed while rendering or recognizing a PDF page")
        raise
    finally:
        # Release every large page object before moving to the next page.
        image_bytes = None
        pixmap = None
        gc.collect()
