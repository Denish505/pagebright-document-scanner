"""PDF page rendering helpers for PageBright."""
from __future__ import annotations

import cv2
import numpy as np
import pypdfium2 as pdfium

MAX_PDF_PAGES = 20


def get_pdf_page_count(pdf_bytes: bytes) -> int:
    try:
        document = pdfium.PdfDocument(pdf_bytes)
        count = len(document)
        document.close()
    except Exception as exc:
        raise ValueError("Could not open this PDF. Password-protected PDFs are not supported.") from exc
    if count < 1:
        raise ValueError("The PDF has no pages.")
    if count > MAX_PDF_PAGES:
        raise ValueError(
            f"This PDF has {count} pages. PageBright currently processes up to "
            f"{MAX_PDF_PAGES} pages at a time; split it and upload each part."
        )
    return count


def render_pdf_page(pdf_bytes: bytes, page_index: int, dpi: int = 144) -> np.ndarray:
    """Render one page as an OpenCV BGR image."""
    if not 72 <= dpi <= 240:
        raise ValueError("PDF render resolution must be between 72 and 240 DPI.")
    try:
        document = pdfium.PdfDocument(pdf_bytes)
        if page_index < 0 or page_index >= len(document):
            raise ValueError("PDF page index is out of range.")
        page = document[page_index]
        bitmap = page.render(scale=dpi / 72.0, limit_image_cache=True)
        rgb = np.asarray(bitmap.to_pil().convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        page.close()
        document.close()
        return bgr
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not render PDF page {page_index + 1}.") from exc
