import csv
import gc
import json
import re
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
    },
    {
        "name": "SmolVLM2-2.2B",
        "model_id": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "type": "smol",
    },
]

# 每題的初步關鍵答案檢查。
# 每個內層 list 是一組可接受的替代詞；每組至少命中一個才算該組通過。
# 這是初步自動檢查，不等同於完整語意評估。
CHECKS = {
    1: [["紅色", "red"], ["圓形", "圓"], ["藍色", "blue"], ["正方形", "方形"]],
    2: [["5 個", "5個", "五個", "共 5", "共五"],
        ["3 個", "3個", "三個", "3 個紅", "三個紅"],
        ["2 個", "2個", "兩個", "二個", "2 個藍", "兩個藍"]],
    3: [["紅色", "紅"], ["綠色", "綠"], ["藍色", "藍"]],
    4: [["藍色", "藍"], ["最大", "最大的是"]],
    5: [["紅色", "紅"], ["正方形", "方形"], ["黃色", "黃"],
        ["圓形", "圓"], ["藍色", "藍"], ["三角形", "三角"]],
    6: [["4", "四"], ["綠色", "綠"], ["三角形", "三角"]],
    7: [["紅色", "紅"], ["圓形", "圓"], ["上方", "上面"],
        ["藍色", "藍"], ["正方形", "方形"], ["下方", "下面"]],
    8: [["最右", "右邊", "圓形", "圓"], ["不同", "不一樣"],
        ["正方形", "方形"]],
    9: [["3", "三"], ["紅色", "紅"], ["圓形", "圓"]],
    10: [["水平", "橫向", "由左至右", "從左到右"],
         ["橘色", "橙色"], ["正方形", "方形"]],
    11: [["hello"]],
    12: [["2026"]],
    13: [["cat"], ["dog"]],
    14: [["黃色", "黃"], ["紫色", "紫"], ["綠色", "綠"],
         ["圓形", "圓"], ["正方形", "方形"], ["三角形", "三角"]],
    15: [["綠色", "綠"], ["最大"], ["紅色", "紅"], ["最小"]],
    16: [["4 個", "4個", "四個", "共 4", "共四"]],
    17: [["右側", "右邊", "右方"]],
    18: [["5", "五"], ["圓形", "圓"]],
    19: [["綠色", "綠"], ["正方形", "方形"],
         ["上方", "上面"], ["橘色", "橙色"], ["圓形", "圓"]],
    20: [["3 個", "3個", "三個", "共 3", "共三"],
         ["紅色", "紅"], ["圓形", "圓"], ["三角形", "三角"],
         ["正方形", "方形"]],
}

def normalize(text):
    text = text.lower()
    text = re.sub(r"\s+", "", text)
    return text

def keyword_check(question_id, answer):
    groups = CHECKS.get(question_id, [])
    normalized = normalize(answer)
    hits = []
    for group in groups:
        hit = any(normalize(term) in normalized for term in group)
        hits.append(hit)
    passed = bool(hits) and all(hits)
    group_score = sum(hits) / len(hits) if hits else 0.0
    return passed, group_score, sum(hits), len(hits)

def save_results(results, model_name):
    safe_name = model_name.lower().replace(".", "").replace("-", "_")
    json_path = RESULT_DIR / f"benchmark_{safe_name}.json"
    csv_path = RESULT_DIR / f"benchmark_{safe_name}.csv"

    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if results:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    return json_path, csv_path

def run_one_model(config, benchmark):
    print("\n" + "=" * 70)
    print("MODEL:", config["name"])
    print("ID:", config["model_id"])
    print("=" * 70)

    start_load = time.perf_counter()

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

    load_seconds = time.perf_counter() - start_load
    print(f"Model load time: {load_seconds:.2f} seconds")

    results = []
    model_peak_memory = 0

    for index, item in enumerate(benchmark, start=1):
        image_path = BASE / item["image"]
        image = Image.open(image_path).convert("RGB")
        prompt = item["prompt"]

        try:
            if config["type"] == "qwen":
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path.resolve())},
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

            inputs = inputs.to(model.device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            start_infer = time.perf_counter()

            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            inference_seconds = time.perf_counter() - start_infer
            input_length = inputs["input_ids"].shape[1]
            new_tokens = generated[0][input_length:]

            answer = processor.decode(
                new_tokens,
                skip_special_tokens=True,
            ).strip()

            if torch.cuda.is_available():
                peak_gib = torch.cuda.max_memory_allocated() / 1024**3
                model_peak_memory = max(model_peak_memory, peak_gib)
            else:
                peak_gib = 0.0

            passed, group_score, hits, total_groups = keyword_check(
                item["id"], answer
            )

            row = {
                "model": config["name"],
                "model_id": config["model_id"],
                "id": item["id"],
                "category": item["category"],
                "image": item["image"],
                "prompt": prompt,
                "reference_answer": item["reference_answer"],
                "answer": answer,
                "inference_seconds": round(inference_seconds, 4),
                "generated_tokens": len(new_tokens),
                "peak_gpu_memory_gib": round(peak_gib, 3),
                "keyword_pass": passed,
                "keyword_group_score": round(group_score, 4),
                "keyword_groups_hit": hits,
                "keyword_groups_total": total_groups,
                "error": "",
            }

            print(
                f'[{index:02d}/{len(benchmark)}] '
                f'ID={item["id"]:02d} '
                f'time={inference_seconds:.3f}s '
                f'keyword={hits}/{total_groups} '
                f'pass={passed}'
            )
            print("  Answer:", answer.replace("\n", " ")[:240])

        except Exception as exc:
            row = {
                "model": config["name"],
                "model_id": config["model_id"],
                "id": item["id"],
                "category": item["category"],
                "image": item["image"],
                "prompt": prompt,
                "reference_answer": item["reference_answer"],
                "answer": "",
                "inference_seconds": "",
                "generated_tokens": "",
                "peak_gpu_memory_gib": "",
                "keyword_pass": False,
                "keyword_group_score": 0.0,
                "keyword_groups_hit": 0,
                "keyword_groups_total": len(CHECKS.get(item["id"], [])),
                "error": repr(exc),
            }
            print(f'[{index:02d}/{len(benchmark)}] ERROR: {exc}')

        results.append(row)
        save_results(results, config["name"])

    successful = [
        r for r in results
        if not r["error"] and r["inference_seconds"] != ""
    ]

    if successful:
        mean_latency = sum(r["inference_seconds"] for r in successful) / len(successful)
        mean_tokens = sum(r["generated_tokens"] for r in successful) / len(successful)
        pass_rate = sum(bool(r["keyword_pass"]) for r in successful) / len(successful)
        mean_group_score = sum(r["keyword_group_score"] for r in successful) / len(successful)
    else:
        mean_latency = mean_tokens = pass_rate = mean_group_score = 0.0

    summary = {
        "model": config["name"],
        "model_id": config["model_id"],
        "questions_total": len(benchmark),
        "questions_successful": len(successful),
        "questions_with_errors": len(benchmark) - len(successful),
        "model_load_seconds": round(load_seconds, 3),
        "mean_inference_seconds": round(mean_latency, 4),
        "mean_generated_tokens": round(mean_tokens, 2),
        "keyword_question_pass_rate": round(pass_rate, 4),
        "mean_keyword_group_score": round(mean_group_score, 4),
        "peak_gpu_memory_gib_max": round(model_peak_memory, 3),
        "note": (
            "Keyword metrics are heuristic preliminary checks, not semantic "
            "accuracy or a substitute for commercial LLM judging."
        ),
    }

    summary_path = RESULT_DIR / (
        f"summary_{config['name'].lower().replace('.', '').replace('-', '_')}.json"
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nSUMMARY")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results, summary

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    if len(benchmark) != 20:
        raise RuntimeError(f"Expected 20 questions, got {len(benchmark)}")

    for item in benchmark:
        image_path = BASE / item["image"]
        if not image_path.exists():
            raise FileNotFoundError(image_path)

    print("Benchmark questions:", len(benchmark))
    print("GPU:", torch.cuda.get_device_name(0))
    print("Generation: greedy decoding, max_new_tokens =", MAX_NEW_TOKENS)
    print("Models will run sequentially to reduce VRAM pressure.")

    all_summaries = []
    for config in MODELS:
        _, summary = run_one_model(config, benchmark)
        all_summaries.append(summary)

    summary_path = RESULT_DIR / "benchmark_comparison_summary.json"
    summary_path.write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = RESULT_DIR / "benchmark_comparison_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
        writer.writeheader()
        writer.writerows(all_summaries)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("Summary JSON:", summary_path)
    print("Summary CSV :", csv_path)
    print("=" * 70)

if __name__ == "__main__":
    main()
