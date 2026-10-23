import csv
import json
import time
from pathlib import Path
from datetime import datetime

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

BASE = Path.home() / "MMIP" / "Week7" / "Assignment_A_LLM"
BENCHMARK = BASE / "benchmarks" / "instruction_benchmark.json"
OUT_JSON = BASE / "results" / "4bit_optimized_prompt_results.json"
OUT_CSV = BASE / "results" / "4bit_optimized_prompt_results.csv"

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS = 160

SYSTEM_PROMPT = """
你是一位重視準確性與指令遵循的助理。
回答前請確認以下要求：
1. 辨識使用者的主要任務與所有明確限制。
2. 遵守指定的語言、字數、項目數量及輸出格式。
3. 只根據題目提供的資訊作答；資訊不足時，清楚說明限制。
4. 數學或程式題要檢查結果是否符合要求。
5. 最終只輸出使用者需要的答案，不輸出檢核清單或內部分析。
除非題目另有要求，請使用繁體中文。
""".strip()

questions = json.loads(BENCHMARK.read_text(encoding="utf-8"))
assert len(questions) == 20, f"預期 20 題，實際為 {len(questions)}"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU 不可用。")

print("=" * 68)
print("4-bit NF4 + Optimized Prompt Benchmark")
print(f"Model: {MODEL_NAME}")
print(f"Questions: {len(questions)}")
print(f"Max new tokens: {MAX_NEW_TOKENS}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print("=" * 68)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

print("\n[1/3] Loading tokenizer and 4-bit model...")
load_start = time.perf_counter()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    dtype=torch.float16,
    device_map="auto",
)
model.eval()

torch.cuda.synchronize()
load_seconds = time.perf_counter() - load_start
device = model.get_input_embeddings().weight.device

if not getattr(model, "is_loaded_in_4bit", False):
    raise RuntimeError("模型未確認以 4-bit 載入，停止測試。")

memory_after_load_gib = torch.cuda.memory_allocated() / (1024 ** 3)
reserved_after_load_gib = torch.cuda.memory_reserved() / (1024 ** 3)

print(f"Loaded in 4-bit: {model.is_loaded_in_4bit}")
print(f"Model device: {device}")
print(f"Model load time: {load_seconds:.2f} seconds")
print(f"GPU allocated after load: {memory_after_load_gib:.3f} GiB")
print(f"GPU reserved after load: {reserved_after_load_gib:.3f} GiB")

results = []
print("\n[2/3] Running optimized prompt benchmark...")

for i, item in enumerate(questions, 1):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": item["prompt"]},
    ]

    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(formatted, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_tokens = inputs["input_ids"].shape[1]

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    generated = output[0][input_tokens:]
    answer = tokenizer.decode(generated, skip_special_tokens=True)
    generated_tokens = len(generated)

    keywords = item.get("evaluation_keywords", [])
    matched = [word for word in keywords if word in answer]
    keyword_coverage = len(matched) / len(keywords) if keywords else None

    peak_allocated_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    peak_reserved_gib = torch.cuda.max_memory_reserved() / (1024 ** 3)

    row = {
        "id": item["id"],
        "category": item["category"],
        "prompt": item["prompt"],
        "reference_answer": item["reference_answer"],
        "evaluation_keywords": keywords,
        "matched_keywords": matched,
        "keyword_coverage": round(keyword_coverage, 4)
            if keyword_coverage is not None else None,
        "model_answer": answer,
        "input_tokens": input_tokens,
        "generated_tokens": generated_tokens,
        "latency_seconds": round(elapsed, 4),
        "tokens_per_second": round(generated_tokens / elapsed, 4)
            if elapsed else 0,
        "hit_max_new_tokens": generated_tokens >= MAX_NEW_TOKENS,
        "precision": "4-bit NF4",
        "prompt_technique": "structured_instruction_constraint_check",
        "model_name": MODEL_NAME,
        "peak_allocated_gib": round(peak_allocated_gib, 4),
        "peak_reserved_gib": round(peak_reserved_gib, 4),
    }
    results.append(row)

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_name": MODEL_NAME,
        "precision": "4-bit NF4",
        "quantization": {
            "type": "NF4",
            "double_quant": True,
            "compute_dtype": "float16",
        },
        "prompt_technique": "structured_instruction_constraint_check",
        "max_new_tokens": MAX_NEW_TOKENS,
        "model_load_seconds": round(load_seconds, 4),
        "gpu_name": torch.cuda.get_device_name(0),
        "memory_after_load_gib": round(memory_after_load_gib, 4),
        "reserved_after_load_gib": round(reserved_after_load_gib, 4),
        "num_questions": len(results),
        "results": results,
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    coverage_text = f"{keyword_coverage:.0%}" if keyword_coverage is not None else "N/A"
    print(
        f"[{i:02d}/20] ID={item['id']:02d} {item['category']} | "
        f"{elapsed:.2f}s | {generated_tokens} tokens | "
        f"coverage={coverage_text} | peak={peak_allocated_gib:.2f} GiB"
    )

print("\n[3/3] Saving CSV and summary...")

with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
    columns = list(results[0].keys())
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    for row in results:
        copy_row = row.copy()
        copy_row["evaluation_keywords"] = json.dumps(
            copy_row["evaluation_keywords"], ensure_ascii=False
        )
        copy_row["matched_keywords"] = json.dumps(
            copy_row["matched_keywords"], ensure_ascii=False
        )
        writer.writerow(copy_row)

avg_latency = sum(r["latency_seconds"] for r in results) / len(results)
total_tokens = sum(r["generated_tokens"] for r in results)
total_time = sum(r["latency_seconds"] for r in results)
avg_coverage = sum(
    r["keyword_coverage"] for r in results
    if r["keyword_coverage"] is not None
) / sum(r["keyword_coverage"] is not None for r in results)
avg_peak = sum(r["peak_allocated_gib"] for r in results) / len(results)
max_peak = max(r["peak_allocated_gib"] for r in results)
truncated = sum(r["hit_max_new_tokens"] for r in results)

print("=" * 68)
print("4-bit + OPTIMIZED PROMPT SUMMARY")
print(f"Completed: {len(results)}/20")
print(f"Model load time: {load_seconds:.2f} seconds")
print(f"Average latency: {avg_latency:.4f} seconds/question")
print(f"Overall generation speed: {total_tokens / total_time:.2f} tokens/sec")
print(f"Average keyword coverage: {avg_coverage:.2%}")
print(f"Responses reaching token limit: {truncated}")
print(f"GPU allocated after load: {memory_after_load_gib:.3f} GiB")
print(f"Average per-question peak allocated: {avg_peak:.3f} GiB")
print(f"Maximum per-question peak allocated: {max_peak:.3f} GiB")
print(f"JSON: {OUT_JSON}")
print(f"CSV: {OUT_CSV}")
print("=" * 68)
