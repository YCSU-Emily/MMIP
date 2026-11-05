"""OCR-based image prompt-injection inspection."""

from pathlib import Path

from PIL import Image, UnidentifiedImageError
import pytesseract

from security import inspect_text


OCR_LANGUAGES = "eng+chi_tra"
OCR_CONFIG = "--psm 6"


def inspect_image_text(image_path):
    """
    Extract visible text and inspect it for known prompt-injection patterns.

    OCR failure is fail-open: it does not automatically block an image.
    The caller should still treat all image content as untrusted.
    """
    result = {
        "scan_status": "not_scanned",
        "extracted_text": "",
        "is_suspicious": False,
        "matched_patterns": [],
        "error": None,
    }

    if image_path is None:
        result["scan_status"] = "no_image"
        return result

    path = Path(image_path)

    if not path.is_file():
        result["scan_status"] = "error"
        result["error"] = "Image file does not exist."
        return result

    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            extracted_text = pytesseract.image_to_string(
                image,
                lang=OCR_LANGUAGES,
                config=OCR_CONFIG,
            )

        result["extracted_text"] = extracted_text.strip()
        result["scan_status"] = (
            "completed" if result["extracted_text"] else "no_text"
        )

        # Check both forms:
        # 1. Original OCR text preserves English word boundaries.
        # 2. Compact text handles spaces inserted between Chinese characters.
        original_result = inspect_text(result["extracted_text"])

        normalized_text = "".join(result["extracted_text"].split())
        compact_result = inspect_text(normalized_text)

        matches = list(dict.fromkeys(
            original_result["matched_patterns"]
            + compact_result["matched_patterns"]
        ))

        result["is_suspicious"] = bool(matches)
        result["matched_patterns"] = matches

    except (
        pytesseract.TesseractError,
        pytesseract.TesseractNotFoundError,
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as exc:
        result["scan_status"] = "error"
        result["error"] = type(exc).__name__

    return result
