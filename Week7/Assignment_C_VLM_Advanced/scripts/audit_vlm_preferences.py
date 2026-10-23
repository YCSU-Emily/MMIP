import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

PROJECT = Path.home() / "MMIP/Week7/Assignment_C_VLM_Advanced"
INPUT = PROJECT / "data/vlm_preferences_candidates.jsonl"
OUTPUT = PROJECT / "results/preference_audit_first10.jsonl"
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

rows = [json.loads(line) for line in INPUT.read_text().splitlines()[:10]]

processor = AutoProcessor.from_pretrained(
    MODEL_ID, local_files_only=True
)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
)
model.eval()

results = []

for i, row in enumerate(rows, 1):
    image = Image.open(row["image"]).convert("RGB")
    added_claim = row["rejected"].removeprefix(
        row["chosen"]
    ).strip()

    prompt_text = (
        "You are auditing an image-caption preference dataset. "
        "Inspect the image and judge only the added claim below. "
        "Is the named object clearly visible? "
        "Return JSON only with keys object, verdict, reason. "
        "verdict must be one of: visible, not_visible, uncertain. "
        "Do not infer from the reference caption alone.\n"
        f"Added claim: {added_claim}"
    )

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ],
    }]

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[prompt], images=[image], return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=100, do_sample=False
        )

    new_tokens = output[:, inputs["input_ids"].shape[1]:]
    answer = processor.batch_decode(
        new_tokens, skip_special_tokens=True
    )[0]

    result = {
        "image_id": row["image_id"],
        "image": row["image"],
        "chosen": row["chosen"],
        "added_claim": added_claim,
        "model_audit": answer,
        "human_review_required": True,
    }
    results.append(result)
    print(f"\n[{i}/10] image_id={row['image_id']}")
    print("Claim:", added_claim)
    print("Audit:", answer)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT.open("w") as f:
    for result in results:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

print("\nSaved audit:", OUTPUT)
print("Audited:", len(results))
