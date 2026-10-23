import csv
import gc
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info

BASE = Path("Week7/Assignment_B_VLM")
BENCHMARK_PATH = BASE / "benchmarks" / "visual_benchmark.json"
RESULT_DIR = BASE / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MAX_NEW_TOKENS = 128

MODELS = [
    {
        "name": "Qwen2.5-VL-3B",
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "type": "qwen",
        "output_prefix": "english_qwen25_vl_3b",
    },
    {
        "name": "SmolVLM2-2.2B",
        "model_id": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "type": "smol",
        "output_prefix": "english_smolvlm2_22b",
    },
]


def save_results(results, prefix):
    json_path = RESULT_DIR / f"{prefix}.json"
    csv_path = RESULT_DIR / f"{prefix}.csv"

    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if results:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    return json_path, csv_path


def run_one_model(config, benchmark):
    print("\n" + "=" * 70, flush=True)
    print("MODEL:", config["name"], flush=True)
    print("MODEL ID:", config["model_id"], flush=True)
    print("=" * 70, flush=True)

    load_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(config["model_id"])

    if config["type"] == "qwen":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config["model_id"],
            torch_dtype="auto",
            device_map="auto",
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            config["model_id"],
            torch_dtype="auto",
            device_map="auto",
        )

    load_seconds = time.perf_counter() - load_start
    print(f"Model load time: {load_seconds:.3f} seconds", flush=True)

    results = []

    for index, item in enumerate(benchmark, start=1):
        image_path = BASE / item["image"]
        result = {
            "id": item["id"],
            "category": item["category"],
            "image": item["image"],
            "prompt": item["prompt"],
            "reference_answer": item["reference_answer"],
            "model": config["name"],
            "model_id": config["model_id"],
            "answer": "",
            "inference_seconds": None,
            "generated_tokens": None,
            "error": "",
        }

        try:
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            image = Image.open(image_path).convert("RGB")
            prompt = item["prompt"]

            if config["type"] == "qwen":
                messages = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": str(image_path.resolve()),
                        },
                        {"type": "text", "text": prompt},
                    ],
                }]

                chat_text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                image_inputs, video_inputs = process_vision_info(messages)

                inputs = processor(
                    text=[chat_text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
            else:
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }]

                chat_text = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                )
                inputs = processor(
                    text=chat_text,
                    images=[image],
                    return_tensors="pt",
                )

            device = next(model.parameters()).device
            inputs = inputs.to(device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            start = time.perf_counter()
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start

            input_length = inputs["input_ids"].shape[1]
            generated_ids = output_ids[:, input_length:]
            answer = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            result["answer"] = answer
            result["inference_seconds"] = round(elapsed, 4)
            result["generated_tokens"] = int(generated_ids.shape[-1])

            print(
                f'[{index:02d}/{len(benchmark)}] '
                f'ID={item["id"]:02d} '
                f'time={elapsed:.3f}s '
                f'tokens={result["generated_tokens"]}',
                flush=True,
            )
            print(f"  Answer: {answer}", flush=True)

            del inputs, output_ids, generated_ids

        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            print(
                f'[{index:02d}/{len(benchmark)}] '
                f'ID={item["id"]:02d} ERROR: {result["error"]}',
                flush=True,
            )

        results.append(result)

        # Save after each question so completed work is retained.
        save_results(results, config["output_prefix"])

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    successful = [r for r in results if not r["error"]]
    times = [r["inference_seconds"] for r in successful
             if r["inference_seconds"] is not None]
    tokens = [r["generated_tokens"] for r in successful
              if r["generated_tokens"] is not None]

    summary = {
        "model": config["name"],
        "model_id": config["model_id"],
        "questions_total": len(benchmark),
        "questions_successful": len(successful),
        "questions_with_errors": len(results) - len(successful),
        "model_load_seconds": round(load_seconds, 3),
        "mean_inference_seconds": round(sum(times) / len(times), 4)
            if times else None,
        "mean_generated_tokens": round(sum(tokens) / len(tokens), 2)
            if tokens else None,
        "evaluation_status": (
            "Not scored yet. Requires answer-level manual review or an "
            "independent judge. No keyword accuracy is reported."
        ),
    }

    summary_path = RESULT_DIR / (
        f'{config["output_prefix"]}_summary.json'
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nSUMMARY")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary, results


def main():
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))

    if len(benchmark) != 20:
        raise ValueError(f"Expected 20 questions, got {len(benchmark)}")

    if not all(
        "Answer in English only." in item["prompt"]
        for item in benchmark
    ):
        raise ValueError(
            "Benchmark prompts are not all marked for English-only answers."
        )

    print("English benchmark questions:", len(benchmark))
    print("GPU:", torch.cuda.get_device_name(0)
          if torch.cuda.is_available() else "CUDA unavailable")
    print("Decoding: greedy; max_new_tokens =", MAX_NEW_TOKENS)
    print("Existing Chinese result files will not be overwritten.")
    print("No keyword accuracy will be reported for this English run.")

    summaries = []
    all_results = []

    for config in MODELS:
        summary, results = run_one_model(config, benchmark)
        summaries.append(summary)
        all_results.extend(results)

    summary_json = RESULT_DIR / "english_benchmark_comparison_summary.json"
    summary_csv = RESULT_DIR / "english_benchmark_comparison_summary.csv"
    all_json = RESULT_DIR / "english_benchmark_all_answers.json"
    all_csv = RESULT_DIR / "english_benchmark_all_answers.csv"

    summary_json.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with summary_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    all_json.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if all_results:
        with all_csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(all_results[0].keys())
            )
            writer.writeheader()
            writer.writerows(all_results)

    print("\n" + "=" * 70)
    print("ENGLISH BENCHMARK COMPLETE")
    print("Comparison:", summary_json)
    print("All answers:", all_json)
    print("All answers CSV:", all_csv)
    print("=" * 70)


if __name__ == "__main__":
    main()
