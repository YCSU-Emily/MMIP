from pathlib import Path
from PIL import Image, UnidentifiedImageError


SUPPORTED_FORMATS = {
    "JPEG",
    "PNG",
    "WEBP"
}


def analyze_image(image_path):
    """
    Validate an image and extract basic image information.
    This version does not perform AI-based image recognition.
    """

    path = Path(image_path)

    # 1. Check whether the file exists
    if not path.exists():
        return {
            "success": False,
            "error": "Image file does not exist."
        }

    if not path.is_file():
        return {
            "success": False,
            "error": "Image path is not a file."
        }

    # 2. Check file extension
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        return {
            "success": False,
            "error": "Unsupported image extension."
        }

    try:
        # 3. Open and verify image
        with Image.open(path) as image:
            image.verify()

        # 4. Reopen image to extract metadata
        with Image.open(path) as image:
            image_format = image.format
            width, height = image.size
            mode = image.mode

        if image_format not in SUPPORTED_FORMATS:
            return {
                "success": False,
                "error": "Unsupported image format."
            }

        # 5. Return image information
        return {
            "success": True,
            "image_path": str(path),
            "image_format": image_format,
            "width": width,
            "height": height,
            "color_mode": mode
        }

    except (UnidentifiedImageError, OSError, ValueError):
        return {
            "success": False,
            "error": "Invalid or corrupted image file."
        }
