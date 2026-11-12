from pathlib import Path
import json

import pytesseract
from PIL import Image, ImageEnhance, ImageOps
from pytesseract import Output


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "outputs" / "video_manifest.json"
OUTPUT_PATH = ROOT / "transcripts" / "wifi_troubleshooting_silent_ocr.json"

OCR_LANGUAGES = "chi_tra+eng"


def preprocess_image(image_path):
    """Enlarge and improve contrast before OCR."""
    image = Image.open(image_path).convert("RGB")

    # The silent video has relatively low resolution.
    image = image.resize(
        (image.width * 3, image.height * 3),
        Image.Resampling.LANCZOS,
    )

    image = ImageOps.grayscale(image)
    image = ImageEnhance.Contrast(image).enhance(1.8)

    return image


def extract_frame_text(frame_path):
    image = preprocess_image(frame_path)

    raw_text = pytesseract.image_to_string(
        image,
        lang=OCR_LANGUAGES,
        config="--psm 6",
    ).strip()

    data = pytesseract.image_to_data(
        image,
        lang=OCR_LANGUAGES,
        config="--psm 6",
        output_type=Output.DICT,
    )

    tokens = []

    for i, word in enumerate(data["text"]):
        word = word.strip()

        try:
            confidence = float(data["conf"][i])
        except (ValueError, TypeError):
            confidence = -1

        if word and confidence >= 0:
            tokens.append({
                "text": word,
                "confidence": round(confidence, 2),
                "box": {
                    "left": data["left"][i],
                    "top": data["top"][i],
                    "width": data["width"][i],
                    "height": data["height"][i],
                },
            })

    return {
        "raw_text": raw_text,
        "tokens": tokens,
    }


def main():
    manifest = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )

    silent_records = [
        item for item in manifest
        if item["video_type"] == "silent"
    ]

    if not silent_records:
        raise RuntimeError("No silent video found in manifest.")

    all_results = []

    for video in silent_records:
        print(f"\n[OCR VIDEO] {video['video_id']}")

        for index, frame in enumerate(video["frames"]):
            frame_path = ROOT / frame["frame_path"]

            result = extract_frame_text(frame_path)

            record = {
                "video_id": video["video_id"],
                "frame_index": index,
                "timestamp_seconds": frame["timestamp_seconds"],
                "frame_path": frame["frame_path"],
                **result,
            }

            all_results.append(record)

            preview = result["raw_text"].replace("\n", " ")
            print(
                f"[{frame['timestamp_seconds']:6.2f}s] "
                f"{preview[:120] if preview else '(no text detected)'}"
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    detected = sum(
        bool(item["raw_text"]) for item in all_results
    )

    print("\n========== OCR SUMMARY ==========")
    print(f"Frames processed: {len(all_results)}")
    print(f"Frames with detected text: {detected}")
    print(f"Frames without detected text: {len(all_results) - detected}")
    print(f"Output: {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
