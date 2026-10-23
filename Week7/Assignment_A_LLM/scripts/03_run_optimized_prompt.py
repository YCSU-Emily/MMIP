import csv
import json
import time
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE = Path.home() / "MMIP" / "Week7" / "Assignment_A_LLM"
BENCHMARK = BASE / "benchmarks" / "instruction_benchmark.json"
OUT_JSON = BASE / "results" / "optimized_prompt_results.json"
OUT_CSV = BASE / "results" / "optimized_prompt_results.csv"

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
assert len(questions) == 20

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU 不可用。")

print("=" * 60)
print("Optimized Prompt Benchmark")
print("Technique: Structured instructions + constraint checking")
print(f"Model: {MODEL_NAME}")
print(f"Questions: {len(questions)}")
print("=" * 60)

print("\nLoading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto",
)
model.eval()
device = model.get_input_embeddings().weight.device

results = []

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
        "precision": "FP16",
        "prompt_technique": "structured_instruction_constraint_check",
        "model_name": MODEL_NAME,
    }
    results.append(row)

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_name": MODEL_NAME,
        "precision": "FP16",
        "prompt_technique": "structured_instruction_constraint_check",
        "max_new_tokens": MAX_NEW_TOKENS,
        "num_questions": len(results),
        "results": results,
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    coverage_text = (
        f"{keyword_coverage:.0%}" if keyword_coverage is not None else "N/A"
    )
    print(
        f"[{i:02d}/20] ID={item['id']:02d} {item['category']} | "
        f"{elapsed:.2f}s | {generated_tokens} tokens | "
        f"keyword coverage={coverage_text}"
    )

columns = list(results[0].keys())
with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    for row in results:
        row_copy = row.copy()
        row_copy["evaluation_keywords"] = json.dumps(
            row_copy["evaluation_keywords"], ensure_ascii=False
        )
        row_copy["matched_keywords"] = json.dumps(
            row_copy["matched_keywords"], ensure_ascii=False
        )
        writer.writerow(row_copy)

avg_latency = sum(r["latency_seconds"] for r in results) / len(results)
total_tokens = sum(r["generated_tokens"] for r in results)
total_time = sum(r["latency_seconds"] for r in results)
overall_speed = total_tokens / total_time if total_time else 0
avg_keyword_coverage = sum(
    r["keyword_coverage"] for r in results
    if r["keyword_coverage"] is not None
) / sum(r["keyword_coverage"] is not None for r in results)
truncated = sum(r["hit_max_new_tokens"] for r in results)

print("\n" + "=" * 60)
print("OPTIMIZED PROMPT SUMMARY")
print(f"Completed: {len(results)}/20")
print(f"Average latency: {avg_latency:.4f} seconds/question")
print(f"Overall generation speed: {overall_speed:.2f} tokens/sec")
print(f"Average keyword coverage: {avg_keyword_coverage:.2%}")
print(f"Responses reaching token limit: {truncated}")
print(f"JSON: {OUT_JSON}")
print(f"CSV: {OUT_CSV}")
print("=" * 60)
