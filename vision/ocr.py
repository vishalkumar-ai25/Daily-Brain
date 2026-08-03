"""
vision/ocr.py
─────────────
Layer 4 — Step 1: Extract text from images using Tesseract OCR.

What this does:
  - Takes an image file path (screenshot, scan, photo of notes)
  - Uses Tesseract to extract any text in the image
  - Returns that text as a plain string, ready to flow through the
    existing Layer 1 pipeline (clean → chunk → embed → store)

Why Tesseract?
──────────────
  - Free, open source, well-established (Google-maintained)
  - Works offline — no API calls needed
  - Good accuracy on clean, typed text (screenshots, PDFs)
  - Lower accuracy on handwriting (for that, use a multimodal LLM instead)

Prerequisites
─────────────
  Install Tesseract on macOS:
    brew install tesseract

  Then install the Python wrapper:
    pip install pytesseract Pillow

Usage
─────
    from vision.ocr import extract_text_from_image

    text = extract_text_from_image("screenshot.png")
    print(text[:500])
"""

from pathlib import Path


def extract_text_from_image(image_path: str | Path) -> str:
    """
    Extract text from an image file using Tesseract OCR.

    Parameters
    ----------
    image_path : str or Path
        Path to the image file (PNG, JPG, PDF page, etc.)

    Returns
    -------
    str
        Extracted text. Empty string if no text found or OCR fails.

    Raises
    ------
    FileNotFoundError
        If the image file doesn't exist.
    ImportError
        If pytesseract or Pillow is not installed.
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        raise ImportError(
            "pytesseract and Pillow are required for OCR. "
            "Install them with: pip install pytesseract Pillow\n"
            "Also install Tesseract: brew install tesseract"
        )

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = Image.open(path)
    text = pytesseract.image_to_string(image)
    return text.strip()


# ── Manual test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 ocr.py <image_path>")
    else:
        result = extract_text_from_image(sys.argv[1])
        print(f"Extracted {len(result.split())} words:\n")
        print(result)
