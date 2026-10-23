import csv
import json
import time
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = Path.home() / "MMIP" / "Week7" / "Assignment_A_LLM"
BENCHMARK = BASE / "benchmarks" / "instruction_benchmark.json"
RESULT_JSON = BASE / "results" / "fp16_instruction_results.json"
RESULT_CSV = BASE / "results" / "fp16_instruction_results.csv"

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS = 160

questions = json.loads(BENCHMARK.read_text(encoding="utf-8"))
assert len(questions) == 20, f"預期 20 題，實際為 {len(questions)}"

print("=" * 60)
print("FP16 Instruction Benchmark")
print(f"Model: {MODEL_NAME}")
print(f"Questions: {len(questions)}")
print("=" * 60)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU 不可用，停止測試。")

# 載入模型與 tokenizer
print("\n[1/3] Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto",
)
model.eval()

input_device = model.get_input_embeddings().weight.device
print("GPU:", torch.cuda.get_device_name(0))
print("Model device:", input_device)

results = []

# 如果之前有完整結果，這次仍重新執行全部 20 題，
# 確保同一輪結果的測試條件一致。
print("\n[2/3] Running questions...")

for index, item in enumerate(questions, start=1):
    messages = [
        {
            "role": "system",
            "content": "你是一位準確、清楚的助理。請依照使用者的指令回答，使用繁體中文，除非題目要求其他語言或格式。"
        },
        {
            "role": "user",
            "content": item["prompt"]
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(input_device) for k, v in inputs.items()}
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

    generated_ids = output[0][input_tokens:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    output_tokens = len(generated_ids)

    row = {
        "id": item["id"],
        "category": item["category"],
        "prompt": item["prompt"],
        "reference_answer": item["reference_answer"],
        "evaluation_keywords": item.get("evaluation_keywords", []),
        "model_answer": answer,
        "input_tokens": input_tokens,
        "generated_tokens": output_tokens,
        "latency_seconds": round(elapsed, 4),
        "tokens_per_second": round(output_tokens / elapsed, 4) if elapsed else 0,
        "hit_max_new_tokens": output_tokens >= MAX_NEW_TOKENS,
        "precision": "FP16",
        "model_name": MODEL_NAME,
    }
    results.append(row)

    # 每完成一題就保存進度
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_name": MODEL_NAME,
        "precision": "FP16",
        "max_new_tokens": MAX_NEW_TOKENS,
        "num_questions": len(results),
        "results": results,
    }
    RESULT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"[{index:02d}/20] ID={item['id']:02d} "
        f"{item['category']} | {elapsed:.2f}s | "
        f"{output_tokens} tokens"
    )

print("\n[3/3] Saving CSV and summary...")

columns = [
    "id", "category", "prompt", "reference_answer",
    "evaluation_keywords", "model_answer", "input_tokens",
    "generated_tokens", "latency_seconds", "tokens_per_second",
    "hit_max_new_tokens", "precision", "model_name",
]

with RESULT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    for row in results:
        csv_row = row.copy()
        csv_row["evaluation_keywords"] = json.dumps(
            row["evaluation_keywords"], ensure_ascii=False
        )
        writer.writerow(csv_row)

avg_latency = sum(r["latency_seconds"] for r in results) / len(results)
total_tokens = sum(r["generated_tokens"] for r in results)
total_time = sum(r["latency_seconds"] for r in results)
avg_speed = total_tokens / total_time if total_time else 0
truncated = sum(r["hit_max_new_tokens"] for r in results)

print("=" * 60)
print("BENCHMARK SUMMARY")
print(f"Questions completed: {len(results)}/20")
print(f"Average latency: {avg_latency:.4f} seconds/question")
print(f"Total generated tokens: {total_tokens}")
print(f"Overall generation speed: {avg_speed:.2f} tokens/sec")
print(f"Responses reaching token limit: {truncated}")
print(f"JSON: {RESULT_JSON}")
print(f"CSV:  {RESULT_CSV}")
print("=" * 60)
