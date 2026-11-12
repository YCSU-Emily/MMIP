from pathlib import Path
import json

from faster_whisper import WhisperModel


ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = ROOT / "outputs" / "audio"
OUTPUT_DIR = ROOT / "transcripts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Use CPU to avoid competing with other GPU workloads.
MODEL_SIZE = "small"
model = WhisperModel(
    MODEL_SIZE,
    device="cpu",
    compute_type="int8",
)


def transcribe(audio_path):
    segments, info = model.transcribe(
        str(audio_path),
        language=None,  # Automatically detect language.
        beam_size=5,
        vad_filter=True,
    )

    results = []

    for segment in segments:
        results.append({
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "text": segment.text.strip(),
        })

    return {
        "audio_file": str(audio_path.relative_to(ROOT)),
        "model": MODEL_SIZE,
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
        "segments": results,
        "full_text": " ".join(
            item["text"] for item in results
        ),
    }


def main():
    audio_files = sorted(AUDIO_DIR.glob("*.wav"))

    if not audio_files:
        raise FileNotFoundError(
            f"No WAV files found in {AUDIO_DIR}"
        )

    for audio_path in audio_files:
        print(f"\n[TRANSCRIBING] {audio_path.name}")

        result = transcribe(audio_path)

        output_path = (
            OUTPUT_DIR / f"{audio_path.stem}.json"
        )

        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"Language: {result['language']}")
        print(f"Segments: {len(result['segments'])}")
        print(f"Transcript: {result['full_text']}")
        print(f"Saved: {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
