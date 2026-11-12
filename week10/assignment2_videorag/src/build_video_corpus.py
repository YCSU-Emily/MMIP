from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]

SPEECH_PATH = ROOT / "transcripts" / "wifi_troubleshooting_speech.json"
OCR_PATH = ROOT / "transcripts" / "wifi_troubleshooting_silent_ocr.json"

JSON_OUTPUT = ROOT / "outputs" / "video_corpus.json"
JSONL_OUTPUT = ROOT / "outputs" / "video_corpus.jsonl"


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def normalize_text(text):
    """Normalize whitespace while preserving the original wording."""
    return " ".join(text.split()).strip()


def build_speech_records(data):
    records = []

    for index, segment in enumerate(data["segments"]):
        text = normalize_text(segment["text"])

        if not text:
            continue

        records.append({
            "record_id": f"speech_{index:04d}",
            "video_id": "wifi_troubleshooting_speech",
            "modality": "speech",
            "start_seconds": segment["start"],
            "end_seconds": segment["end"],
            "text": text,
            "source_file": data["audio_file"],
            "metadata": {
                "language": data["language"],
                "model": data["model"],
            },
        })

    return records


def build_visual_records(items):
    records = []

    for item in items:
        text = normalize_text(item["raw_text"])

        # Skip frames where OCR found no readable text.
        if not text:
            continue

        timestamp = item["timestamp_seconds"]

        records.append({
            "record_id": f"visual_{item['frame_index']:04d}",
            "video_id": item["video_id"],
            "modality": "visual_ocr",
            "start_seconds": timestamp,
            "end_seconds": timestamp,
            "text": text,
            "source_file": item["frame_path"],
            "metadata": {
                "frame_index": item["frame_index"],
                "ocr_token_count": len(item["tokens"]),
                "ocr_engine": "tesseract",
                "ocr_languages": "chi_tra+eng",
            },
        })

    return records


def main():
    speech_data = load_json(SPEECH_PATH)
    visual_data = load_json(OCR_PATH)

    speech_records = build_speech_records(speech_data)
    visual_records = build_visual_records(visual_data)

    all_records = speech_records + visual_records

    all_records.sort(
        key=lambda item: (
            item["video_id"],
            item["start_seconds"],
            item["modality"],
        )
    )

    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    JSON_OUTPUT.write_text(
        json.dumps(all_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with JSONL_OUTPUT.open("w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    speech_count = sum(
        item["modality"] == "speech" for item in all_records
    )
    visual_count = sum(
        item["modality"] == "visual_ocr" for item in all_records
    )

    print("\n========== VIDEO CORPUS SUMMARY ==========")
    print(f"Speech records: {speech_count}")
    print(f"Visual OCR records: {visual_count}")
    print(f"Total records: {len(all_records)}")
    print(f"JSON output: {JSON_OUTPUT.relative_to(ROOT)}")
    print(f"JSONL output: {JSONL_OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
