"""
vision/describe_image.py
────────────────────────
Layer 4 — Step 2: Describe diagrams and figures using a multimodal LLM.

For images that are NOT just text (architecture diagrams, charts, figures),
OCR won't help — it only extracts text pixels. Instead, we send the image
to a vision-capable LLM and ask it to describe what's in the diagram.

The description is then stored as text and flows through the same
clean → chunk → embed → store pipeline as everything else.

Architectural decision (interview-ready)
─────────────────────────────────────────
We deliberately convert images → text descriptions, rather than building
a separate image-embedding pipeline. Why?
  ✓ One unified retrieval system serves both text and image-derived content
  ✓ No changes needed to Layer 2 (RAG) or Layer 3 (agents)
  ✓ Simpler: one Chroma collection, one query function
  ✗ Trade-off: we lose some visual nuance that an image embedding might capture
  ✗ Dependent on the quality of the LLM's visual description

For a personal knowledge assistant, this trade-off is clearly worth it.

Usage
─────
    from vision.describe_image import describe_image

    description = describe_image("architecture_diagram.png")
    print(description)
"""

import base64
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


def _encode_image_base64(image_path: Path) -> str:
    """Read an image file and return its base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def describe_image(image_path: str | Path, context_hint: str = "") -> str:
    """
    Generate a detailed text description of an image using a multimodal LLM.

    Parameters
    ----------
    image_path : str or Path
        Path to the image file.
    context_hint : str, optional
        A hint about what the image is about (e.g., "neural network diagram
        from a paper about attention mechanisms"). Helps the LLM focus.

    Returns
    -------
    str
        A detailed text description, ready to be treated as a note entry.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    prompt_text = (
        f"This image was saved from a reading/study session"
        + (f" related to: {context_hint}" if context_hint else "")
        + ".\n\n"
        "Please provide a detailed, technical description of this image that captures:\n"
        "1. What type of diagram/figure/chart this is\n"
        "2. The key components, labels, and relationships shown\n"
        "3. The main concept or idea the image is communicating\n"
        "4. Any specific values, metrics, or data shown\n\n"
        "Write the description as plain prose that would allow someone who "
        "cannot see the image to fully understand its content and meaning. "
        "Be specific and technical — this will be stored as a searchable note."
    )

    if LLM_PROVIDER == "gemini":
        return _describe_with_gemini(path, prompt_text)
    elif LLM_PROVIDER == "openai":
        return _describe_with_openai(path, prompt_text)
    else:
        return (
            f"[Vision description not available: LLM_PROVIDER={LLM_PROVIDER!r} "
            "doesn't support images. Set LLM_PROVIDER=gemini or openai in .env]"
        )


def _describe_with_gemini(image_path: Path, prompt: str) -> str:
    """Use Gemini's vision API to describe the image."""
    try:
        import google.generativeai as genai
        from PIL import Image

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        image = Image.open(image_path)
        response = model.generate_content([prompt, image])
        return response.text.strip()
    except Exception as e:
        return f"[Gemini vision error: {e}]"


def _describe_with_openai(image_path: Path, prompt: str) -> str:
    """Use OpenAI's vision API to describe the image."""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)
        b64 = _encode_image_base64(image_path)
        suffix = image_path.suffix.lower().lstrip(".")
        mime = f"image/{suffix}" if suffix in ("png", "jpg", "jpeg", "gif", "webp") else "image/png"

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[OpenAI vision error: {e}]"


# ── Manual test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 describe_image.py <image_path> [context_hint]")
    else:
        path = sys.argv[1]
        hint = sys.argv[2] if len(sys.argv) > 2 else ""
        print(describe_image(path, hint))
