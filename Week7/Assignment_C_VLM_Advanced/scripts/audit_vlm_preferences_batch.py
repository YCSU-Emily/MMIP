import json
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

PROJECT = Path.home() / "MMIP/Week7/Assignment_C_VLM_Advanced"
INPUT = PROJECT / "data/vlm_preferences_candidates.jsonl"
OUTPUT = PROJECT / "results/preference_audit_batch120.jsonl"
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

VALID_VERDICTS = {"visible", "not_visible", "uncertain"}


def extract_added_claim(row):
    chosen = row["chosen"].strip()
    rejected = row["rejected"].strip()

    if rejected.startswith(chosen):
        claim = rejected[len(chosen):].strip()
    else:
        claim = row.get("added_claim", "").strip()

    return claim.lstrip(" .,:;-")


def parse_json_answer(answer):
    text = answer.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {
                "object": "",
                "verdict": "uncertain",
                "reason": "Could not parse model response.",
            }
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {
                "object": "",
                "verdict": "uncertain",
                "reason": "Model response was not valid JSON.",
            }

    verdict = str(data.get("verdict", "uncertain")).lower().strip()
    if verdict not in VALID_VERDICTS:
        verdict = "uncertain"

    return {
        "object": str(data.get("object", "")),
        "verdict": verdict,
        "reason": str(data.get("reason", "")),
    }


if not INPUT.exists():
    raise FileNotFoundError(f"Input not found: {INPUT}")

rows = [
    json.loads(line)
    for line in INPUT.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# Resume: preserve existing results and skip image IDs already recorded.
completed = {}
if OUTPUT.exists():
    with OUTPUT.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                result = json.loads(line)
                if "image_id" in result:
                    completed[str(result["image_id"])] = result
            except json.JSONDecodeError:
                print("Warning: ignored an incomplete output line.")

pending = [
    row for row in rows
    if str(row["image_id"]) not in completed
]

print("Input rows:", len(rows))
print("Already audited:", len(completed))
print("Remaining:", len(pending))
print("Output:", OUTPUT)

if not pending:
    print("All input rows already have results.")
    raise SystemExit(0)

print("Loading local Qwen2.5-VL model...")
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    local_files_only=True,
)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
)
model.eval()

with OUTPUT.open("a", encoding="utf-8") as out:
    for index, row in enumerate(pending, 1):
        image_id = str(row["image_id"])
        claim = extract_added_claim(row)
        image_path = Path(row["image"]).expanduser()

        print(f"\n[{index}/{len(pending)}] image_id={image_id}")
        print("Claim:", claim)

        result = {
            "image_id": image_id,
            "image": str(image_path),
            "chosen": row.get("chosen", ""),
            "added_claim": claim,
            "model_audit": {
                "object": "",
                "verdict": "uncertain",
                "reason": "",
            },
            "raw_answer": "",
            "human_review_required": True,
        }

        try:
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            image = Image.open(image_path).convert("RGB")

            prompt_text = (
                "You are auditing a visual claim against an image. "
                "Inspect the image itself, not any caption or prior label. "
                "Decide whether the specific object in the claim is clearly "
                "visible anywhere in the image. Do not treat an object as "
                "present merely because it is plausible in the scene. "
                "If the image is ambiguous, small, obscured, or uncertain, "
                "choose uncertain. Return one JSON object only with keys "
                "object, verdict, reason. Verdict must be exactly one of "
                "visible, not_visible, uncertain. Keep the reason brief.\n"
                f"Claim to verify: {claim}"
            )

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }]

            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = processor(
                text=[prompt],
                images=[image],
                return_tensors="pt",
            ).to(model.device)

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=120,
                    do_sample=False,
                )

            new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]
            answer = processor.batch_decode(
                new_tokens,
                skip_special_tokens=True,
            )[0]

            result["raw_answer"] = answer
            result["model_audit"] = parse_json_answer(answer)

        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["model_audit"]["reason"] = "Audit failed; manual review needed."

        # Write each result immediately so an interruption is recoverable.
        out.write(json.dumps(result, ensure_ascii=False) + "\n")
        out.flush()

        print("Verdict:", result["model_audit"]["verdict"])
        if result.get("error"):
            print("ERROR:", result["error"])

print("\nBatch audit finished.")
print("Results saved to:", OUTPUT)
print("All model judgments still require quality checks.")
