import os
import json
import time
from pathlib import Path

from google import genai
from PIL import Image


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "gemini-3.8-flash"

CAPTION_FILE = Path("results/generated_captions.json")
IMAGE_DIR = Path("data/images/test")
OUTPUT_FILE = Path("results/gemini_evaluation.json")

MAX_IMAGES = 20


# ============================================================
# Gemini Client
# ============================================================

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. "
        "Please run: export GEMINI_API_KEY='YOUR_API_KEY'"
    )

client = genai.Client(api_key=api_key)


# ============================================================
# Evaluation Prompt
# ============================================================

def build_prompt(caption):
    return f"""
You are evaluating an image captioning model.

Look carefully at the provided image and compare it with the generated caption.

Generated caption:
"{caption}"

Classify the caption into exactly ONE of these categories:

MATCH:
The caption correctly describes the main content of the image.
Minor wording differences are acceptable.

PARTIAL:
The caption describes some important content correctly, but contains
a notable error, omission, or incorrect detail.

MISMATCH:
The caption does not correctly describe the main content of the image.

Return ONLY valid JSON in exactly this format:

{{
  "label": "MATCH",
  "reason": "short explanation"
}}

The label must be exactly one of:
MATCH
PARTIAL
MISMATCH

Keep the reason under 30 words.
"""


# ============================================================
# Parse Gemini Response
# ============================================================

def parse_response(text):
    text = text.strip()

    # Remove Markdown code fences if Gemini adds them
    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)

        label = str(data.get("label", "")).upper()
        reason = str(data.get("reason", "")).strip()

        if label not in {"MATCH", "PARTIAL", "MISMATCH"}:
            raise ValueError("Invalid label")

        return label, reason

    except Exception:
        # Fallback if Gemini returns unexpected formatting
        upper = text.upper()

        if "MISMATCH" in upper:
            label = "MISMATCH"
        elif "PARTIAL" in upper:
            label = "PARTIAL"
        elif "MATCH" in upper:
            label = "MATCH"
        else:
            label = "ERROR"

        return label, text[:300]


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("Week 4 Quiz 4 - Gemini Image Caption Evaluation")
    print("=" * 70)

    print(f"Gemini model: {MODEL_NAME}")
    print(f"Caption file: {CAPTION_FILE}")
    print(f"Image directory: {IMAGE_DIR}")
    print(f"Output file: {OUTPUT_FILE}")
    print()

    # --------------------------------------------------------
    # Load generated captions
    # --------------------------------------------------------

    if not CAPTION_FILE.exists():
        raise FileNotFoundError(
            f"Caption file not found: {CAPTION_FILE}"
        )

    with open(CAPTION_FILE, "r", encoding="utf-8") as f:
        caption_data = json.load(f)

    results = caption_data.get("results", [])

    if not results:
        raise RuntimeError(
            "No generated captions found in generated_captions.json"
        )

    results = results[:MAX_IMAGES]

    print(f"Images to evaluate: {len(results)}")
    print()

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    evaluations = []

    total_start = time.perf_counter()

    for i, item in enumerate(results, start=1):

        image_name = item["image"]
        generated_caption = item["generated_caption"]

        image_path = IMAGE_DIR / image_name

        print(f"[{i:03d}/{len(results):03d}] {image_name}")
        print(f"    Caption: {generated_caption}")

        if not image_path.exists():
            print("    ERROR: Image not found")
            evaluations.append({
                "image": image_name,
                "generated_caption": generated_caption,
                "label": "ERROR",
                "reason": "Image file not found",
                "evaluation_time_sec": 0.0
            })
            continue

        start_time = time.perf_counter()

        try:
            image = Image.open(image_path).convert("RGB")

            prompt = build_prompt(generated_caption)

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[prompt, image]
            )

            response_text = response.text.strip()

            label, reason = parse_response(response_text)

            elapsed = time.perf_counter() - start_time

            print(f"    Gemini: {label}")
            print(f"    Reason: {reason}")
            print(f"    Time: {elapsed * 1000:.2f} ms")

            evaluations.append({
                "image": image_name,
                "generated_caption": generated_caption,
                "label": label,
                "reason": reason,
                "evaluation_time_sec": elapsed
            })

        except Exception as e:

            elapsed = time.perf_counter() - start_time
            error_text = str(e)

            print(f"    ERROR: {type(e).__name__}: {error_text}")

            # Stop immediately when Gemini quota is exhausted.
            if (
                "RESOURCE_EXHAUSTED" in error_text
                or "429" in error_text
                or "quota" in error_text.lower()
            ):
                print()
                print("    Gemini API quota exhausted.")
                print("    Stopping further evaluations.")
                print("    Previous valid evaluations will be preserved.")

                break

            evaluations.append({
                "image": image_name,
                "generated_caption": generated_caption,
                "label": "ERROR",
                "reason": error_text,
                "evaluation_time_sec": elapsed
            })

        print()

        # Small delay to reduce the chance of hitting rate limits
        if i < len(results):
            time.sleep(0.5)

    total_time = time.perf_counter() - total_start

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    match_count = sum(
        x["label"] == "MATCH"
        for x in evaluations
    )

    partial_count = sum(
        x["label"] == "PARTIAL"
        for x in evaluations
    )

    mismatch_count = sum(
        x["label"] == "MISMATCH"
        for x in evaluations
    )

    error_count = sum(
        x["label"] == "ERROR"
        for x in evaluations
    )

    valid_count = match_count + partial_count + mismatch_count

    match_rate = (
        match_count / valid_count * 100
        if valid_count > 0 else 0
    )

    acceptable_rate = (
        (match_count + partial_count) / valid_count * 100
        if valid_count > 0 else 0
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output = {
        "model": MODEL_NAME,
        "evaluation_type": "Gemini visual caption matching",
        "num_images": len(evaluations),
        "valid_evaluations": valid_count,

        "statistics": {
            "MATCH": match_count,
            "PARTIAL": partial_count,
            "MISMATCH": mismatch_count,
            "ERROR": error_count,

            "match_rate_percent": match_rate,
            "acceptable_rate_percent": acceptable_rate
        },

        "total_evaluation_time_sec": total_time,

        "average_evaluation_time_sec": (
            total_time / len(evaluations)
            if evaluations else 0
        ),

        "evaluations": evaluations
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("=" * 70)
    print("Gemini Evaluation Complete")
    print("=" * 70)

    print(f"Images:              {len(evaluations)}")
    print(f"Valid evaluations:   {valid_count}")
    print(f"MATCH:               {match_count}")
    print(f"PARTIAL:             {partial_count}")
    print(f"MISMATCH:            {mismatch_count}")
    print(f"ERROR:               {error_count}")
    print()
    print(f"Match rate:          {match_rate:.2f}%")
    print(f"Match + Partial:     {acceptable_rate:.2f}%")
    print()
    print(f"Total Gemini time:   {total_time:.2f} sec")
    print(f"Average per image:   {total_time / len(evaluations):.2f} sec")
    print()
    print(f"Results: {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
