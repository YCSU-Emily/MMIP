import gc
import csv
import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageEnhance
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = (
    "/home/emily/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/"
    "66285546d2b821cf421d4f5eb2576359d3770cd3"
)
ADAPTER_DIR = ROOT / "checkpoints/dpo_lora_v1/final_adapter"
MANIFEST = ROOT / "results/eval_v1/manifest_100.jsonl"
OUT_DIR = ROOT / "results/brightness_v1"

PROMPT = (
    "Describe this image in one concise sentence. "
    "Mention only objects and actions supported by the image. "
    "Do not guess details that cannot be determined from the image."
)

IMAGE_IDS = [
    270386, 47010, 229997, 64868, 132408,
    52017, 8211, 2149, 4795, 7108,
    528705, 159684, 76261, 257478, 556873,
    30494, 180188, 434996, 55299, 463174,
]
CONDITIONS = {"original": 1.0, "dark_0.5": 0.5, "bright_1.5": 1.5}


def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_model(kind):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        local_files_only=True,
    )
    if kind == "dpo":
        model = PeftModel.from_pretrained(
            model, str(ADAPTER_DIR), is_trainable=False
        )
    model.eval()
    return model


def run_prediction(model, processor, image):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT},
        ],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=[image], return_tensors="pt"
    ).to(model.device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs, do_sample=False, max_new_tokens=80
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latency = time.perf_counter() - start
    n_input = inputs["input_ids"].shape[1]
    prediction = processor.batch_decode(
        generated[:, n_input:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return prediction, latency


def main():
    if OUT_DIR.exists():
        raise FileExistsError(
            f"{OUT_DIR} already exists; refusing to overwrite results."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    manifest_rows = load_jsonl(MANIFEST)
    by_id = {int(r["image_id"]): r for r in manifest_rows}
    missing = [i for i in IMAGE_IDS if i not in by_id]
    if missing:
        raise ValueError(f"IDs missing from manifest: {missing}")

    selected = [by_id[i] for i in IMAGE_IDS]
    for row in selected:
        if not Path(row["image"]).is_file():
            raise FileNotFoundError(row["image"])

    OUT_DIR.mkdir(parents=True)
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL, local_files_only=True
    )
    results = []

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print("20 images x 3 brightness conditions x 2 models = 120", flush=True)

    for kind in ("baseline", "dpo"):
        print(f"\nLoading model: {kind}", flush=True)
        model = load_model(kind)
        for idx, row in enumerate(selected, 1):
            for condition, factor in CONDITIONS.items():
                with Image.open(row["image"]) as im:
                    image = im.convert("RGB")
                if factor != 1.0:
                    image = ImageEnhance.Brightness(image).enhance(factor)

                prediction, latency = run_prediction(model, processor, image)
                results.append({
                    "image_id": int(row["image_id"]),
                    "file_name": row.get("file_name", ""),
                    "image": row["image"],
                    "model": kind,
                    "condition": condition,
                    "brightness_factor": factor,
                    "prediction": prediction,
                    "latency_seconds": round(latency, 6),
                    "reference_captions": row.get("reference_captions", []),
                    "coco_objects": row.get("coco_objects", []),
                    "prompt": PROMPT,
                })
                print(
                    f"[{kind} {idx:02d}/20 {condition}] "
                    f"id={row['image_id']} {latency:.3f}s "
                    f"{prediction[:100]}",
                    flush=True,
                )
                image.close()

        del model
        gc.collect()
        torch.cuda.empty_cache()

    with (OUT_DIR / "brightness_predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    lookup = {
        (r["image_id"], r["model"], r["condition"]): r
        for r in results
    }
    comparison = []
    for row in selected:
        iid = int(row["image_id"])
        for condition, factor in CONDITIONS.items():
            b = lookup[(iid, "baseline", condition)]
            d = lookup[(iid, "dpo", condition)]
            comparison.append({
                "image_id": iid,
                "condition": condition,
                "brightness_factor": factor,
                "baseline_prediction": b["prediction"],
                "dpo_prediction": d["prediction"],
                "baseline_latency_seconds": b["latency_seconds"],
                "dpo_latency_seconds": d["latency_seconds"],
                "text_identical": b["prediction"] == d["prediction"],
            })

    with (OUT_DIR / "brightness_comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)

    summary = {
        "experiment": "VLM brightness robustness exploratory evaluation",
        "base_model": BASE_MODEL,
        "adapter": str(ADAPTER_DIR),
        "image_ids": IMAGE_IDS,
        "conditions": CONDITIONS,
        "num_images": len(selected),
        "num_generations": len(results),
        "generation_settings": {
            "do_sample": False,
            "max_new_tokens": 80,
            "prompt": PROMPT,
        },
        "caveat": (
            "Exploratory evaluation only. Different captions do not by "
            "themselves prove hallucination or quality improvement."
        ),
    }
    (OUT_DIR / "brightness_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\nCompleted 120 generations.")
    print("Results:", OUT_DIR.resolve(), flush=True)


if __name__ == "__main__":
    main()
