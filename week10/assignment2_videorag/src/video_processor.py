from pathlib import Path
from fractions import Fraction
import json
import subprocess

import cv2


ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = ROOT / "videos"
FRAME_DIR = ROOT / "frames"
AUDIO_DIR = ROOT / "outputs" / "audio"
MANIFEST_PATH = ROOT / "outputs" / "video_manifest.json"

SAMPLE_INTERVAL = 2.0


def probe_video(video_path):
    """Read video metadata using ffprobe."""
    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries",
        "stream=codec_type,width,height,r_frame_rate",
        "-show_entries",
        "format=duration",
        "-of", "json",
        str(video_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    info = json.loads(result.stdout)
    streams = info.get("streams", [])

    video_stream = next(
        (s for s in streams if s.get("codec_type") == "video"),
        None,
    )

    if video_stream is None:
        raise RuntimeError(f"No video stream found: {video_path}")

    audio_present = any(
        s.get("codec_type") == "audio" for s in streams
    )

    fps_text = video_stream.get("r_frame_rate", "0/1")
    fps = float(Fraction(fps_text))

    duration = float(info["format"]["duration"])

    return {
        "duration_seconds": duration,
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "fps": fps,
        "audio_present": audio_present,
    }


def extract_frames(video_path, output_dir):
    """Sample one frame every SAMPLE_INTERVAL seconds."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid FPS: {video_path}")

    frame_index = 0
    next_sample_time = 0.0
    frames = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        timestamp = frame_index / fps

        if timestamp + 1e-6 >= next_sample_time:
            filename = f"frame_{len(frames):04d}.jpg"
            frame_path = output_dir / filename

            saved = cv2.imwrite(
                str(frame_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )

            if not saved:
                cap.release()
                raise RuntimeError(f"Failed to save frame: {frame_path}")

            frames.append({
                "timestamp_seconds": round(timestamp, 3),
                "frame_path": str(frame_path.relative_to(ROOT)),
            })

            next_sample_time += SAMPLE_INTERVAL

        frame_index += 1

    cap.release()
    return frames


def extract_audio(video_path, output_path):
    """Extract mono 16 kHz WAV audio."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(output_path),
    ]

    subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )


def main():
    records = []

    video_groups = [
        ("speech", VIDEO_DIR / "speech"),
        ("silent", VIDEO_DIR / "silent"),
    ]

    for video_type, folder in video_groups:
        video_files = sorted(folder.glob("*.mp4"))

        if not video_files:
            print(f"[WARNING] No MP4 files in {folder}")
            continue

        for video_path in video_files:
            print(f"\n[PROCESSING] {video_path.name}")

            metadata = probe_video(video_path)

            frame_output = (
                FRAME_DIR / video_type / video_path.stem
            )

            frames = extract_frames(video_path, frame_output)

            audio_path = None

            if video_type == "speech" and metadata["audio_present"]:
                audio_file = AUDIO_DIR / f"{video_path.stem}.wav"
                extract_audio(video_path, audio_file)
                audio_path = str(audio_file.relative_to(ROOT))
                print(f"[AUDIO] {audio_path}")

            elif video_type == "speech":
                print("[WARNING] Speech video has no audio stream.")

            record = {
                "video_id": video_path.stem,
                "video_type": video_type,
                "video_path": str(video_path.relative_to(ROOT)),
                **metadata,
                "audio_path": audio_path,
                "sample_interval_seconds": SAMPLE_INTERVAL,
                "frames": frames,
            }

            records.append(record)

            print(
                f"[DONE] {len(frames)} frames | "
                f"{metadata['duration_seconds']:.2f} seconds | "
                f"{metadata['width']}x{metadata['height']}"
            )

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print("\n========== SUMMARY ==========")
    print(f"Videos processed: {len(records)}")
    print(f"Manifest: {MANIFEST_PATH.relative_to(ROOT)}")
    print(f"Total frames: {sum(len(r['frames']) for r in records)}")


if __name__ == "__main__":
    main()
