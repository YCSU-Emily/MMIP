import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]

BASE_MODEL = (
    "/home/emily/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/"
    "66285546d2b821cf421d4f5eb2576359d3770cd3"
)
ADAPTER_DIR = ROOT / "checkpoints/dpo_lora_v1/final_adapter"
BASELINE_FILE = ROOT / "results/eval_v1/baseline_predictions.jsonl"
OUTPUT_DIR = ROOT / "results/eval_dpo_lora_v1"
PREDICTIONS_FILE = OUTPUT_DIR / "dpo_predictions.jsonl"
SUMMARY_FILE = OUTPUT_DIR / "dpo_summary.json"

EXPECTED_PROMPT = (
    "Describe this image in one concise sentence. "
    "Mention only objects and actions supported by the image. "
    "Do not guess details that cannot be determined from the image."
)

def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def main():
    print("=" * 65)
    print("DPO + LoRA evaluation")
    print("=" * 65)

    # Safety checks before loading the model.
    for path in [Path(BASE_MODEL), ADAPTER_DIR, BASELINE_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output directory already exists: {OUTPUT_DIR}\n"
            "Refusing to overwrite existing evaluation results."
        )

    rows = load_jsonl(BASELINE_FILE)
    if len(rows) != 100:
        raise ValueError(f"Expected 100 baseline rows, found {len(rows)}")

    for i, row in enumerate(rows):
        if row.get("prompt") != EXPECTED_PROMPT:
            raise ValueError(f"Unexpected prompt in row {i}")
        if not Path(row["image"]).is_file():
            raise FileNotFoundError(
                f"Image missing for image_id={row.get('image_id')}: "
                f"{row['image']}"
            )

    image_ids = [r["image_id"] for r in rows]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Duplicate image_id found in baseline predictions")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    print(f"Evaluation images: {len(rows)}")
    print(f"Adapter: {ADAPTER_DIR}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Loading processor...")

    processor = AutoProcessor.from_pretrained(
        BASE_MODEL,
        local_files_only=True,
    )

    print("Loading original model in BF16...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        local_files_only=True,
    )

    print("Attaching trained LoRA adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        str(ADAPTER_DIR),
        is_trainable=False,
    )
    model.eval()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    elapsed_times = []
    total_start = time.perf_counter()

    with PREDICTIONS_FILE.open("w", encoding="utf-8") as fout:
        for i, row in enumerate(rows, start=1):
            image_path = Path(row["image"])

            with Image.open(image_path) as im:
                image = im.convert("RGB")

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": EXPECTED_PROMPT},
                ],
            }]

            # Use the model's own chat template and multimodal processor.
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = processor(
                text=[text],
                images=[image],
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=80,
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency = time.perf_counter() - start
            elapsed_times.append(latency)

            # Decode only newly generated tokens, not the prompt.
            input_length = inputs["input_ids"].shape[1]
            new_tokens = generated[:, input_length:]
            prediction = processor.batch_decode(
                new_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            result = {
                **row,
                "prediction": prediction,
                "latency_seconds": round(latency, 6),
                "model": "Qwen2.5-VL-3B-Instruct + DPO LoRA v1",
                "condition": "dpo_lora_v1",
            }
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()

            print(
                f"[{i:03d}/{len(rows)}] "
                f"id={row['image_id']} "
                f"latency={latency:.3f}s "
                f"prediction={prediction[:100]}"
            )

    total_elapsed = time.perf_counter() - total_start
    sorted_times = sorted(elapsed_times)
    n = len(sorted_times)
    median = (
        sorted_times[n // 2]
        if n % 2
        else (sorted_times[n // 2 - 1] + sorted_times[n // 2]) / 2
    )

    summary = {
        "model": "Qwen2.5-VL-3B-Instruct + DPO LoRA v1",
        "base_model": BASE_MODEL,
        "adapter": str(ADAPTER_DIR),
        "condition": "dpo_lora_v1",
        "num_images": len(rows),
        "total_elapsed_seconds": round(total_elapsed, 4),
        "total_generation_seconds": round(sum(elapsed_times), 4),
        "mean_generation_latency_seconds": round(
            sum(elapsed_times) / n, 4
        ),
        "median_generation_latency_seconds": round(median, 4),
        "prompt": EXPECTED_PROMPT,
        "generation_config": {
            "do_sample": False,
            "max_new_tokens": 80,
            "dtype": "bfloat16",
            "attn_implementation": "sdpa",
        },
        "predictions_file": str(PREDICTIONS_FILE.relative_to(ROOT)),
    }

    SUMMARY_FILE.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("Evaluation complete")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 65)

if __name__ == "__main__":
    main()
