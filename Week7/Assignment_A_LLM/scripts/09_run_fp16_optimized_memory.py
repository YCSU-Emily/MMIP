import csv
import json
import time
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE = Path.home() / "MMIP/Week7/Assignment_A_LLM"
BENCHMARK = BASE / "benchmarks/instruction_benchmark.json"
OUT_JSON = BASE / "results/fp16_optimized_memory_results.json"
OUT_CSV = BASE / "results/fp16_optimized_memory_results.csv"

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
assert len(questions) == 20, f"預期 20 題，實際 {len(questions)} 題"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU 不可用")

print("=" * 65)
print("FP16 + Optimized Prompt + Memory Benchmark")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Questions: {len(questions)}")
print("=" * 65)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
load_start = time.perf_counter()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto",
)
model.eval()

torch.cuda.synchronize()
load_seconds = time.perf_counter() - load_start
device = model.get_input_embeddings().weight.device

memory_after_load = torch.cuda.memory_allocated() / 1024**3
reserved_after_load = torch.cuda.memory_reserved() / 1024**3

print(f"Model load time: {load_seconds:.2f}s")
print(f"GPU allocated after load: {memory_after_load:.3f} GiB")
print(f"GPU reserved after load: {reserved_after_load:.3f} GiB")

results = []

for i, item in enumerate(questions, 1):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": item["prompt"]},
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_tokens = inputs["input_ids"].shape[1]

    # 重設每題的峰值統計；模型本身的記憶體仍計入 peak
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
    keywords = item.get("evaluation_keywords", [])
    matched = [kw for kw in keywords if kw in answer]

    row = {
        "id": item["id"],
        "category": item["category"],
        "generated_tokens": len(generated),
        "latency_seconds": round(elapsed, 4),
        "tokens_per_second": round(len(generated) / elapsed, 4)
            if elapsed else 0,
        "keyword_coverage": round(len(matched) / len(keywords), 4)
            if keywords else None,
        "hit_max_new_tokens": len(generated) >= MAX_NEW_TOKENS,
        "peak_allocated_gib": round(
            torch.cuda.max_memory_allocated() / 1024**3, 4
        ),
        "peak_reserved_gib": round(
            torch.cuda.max_memory_reserved() / 1024**3, 4
        ),
    }
    results.append(row)

    OUT_JSON.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_name": MODEL_NAME,
        "precision": "FP16",
        "prompt_technique": "structured_instruction_constraint_check",
        "max_new_tokens": MAX_NEW_TOKENS,
        "model_load_seconds": round(load_seconds, 4),
        "gpu_name": torch.cuda.get_device_name(0),
        "memory_after_load_gib": round(memory_after_load, 4),
        "reserved_after_load_gib": round(reserved_after_load, 4),
        "num_questions": len(results),
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[{i:02d}/20] ID={item['id']:02d} "
        f"{elapsed:.2f}s | {len(generated)} tokens | "
        f"peak={row['peak_allocated_gib']:.2f} GiB"
    )

with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

avg_latency = sum(r["latency_seconds"] for r in results) / len(results)
total_tokens = sum(r["generated_tokens"] for r in results)
total_time = sum(r["latency_seconds"] for r in results)
avg_peak = sum(r["peak_allocated_gib"] for r in results) / len(results)
max_peak = max(r["peak_allocated_gib"] for r in results)
valid_cov = [r["keyword_coverage"] for r in results
             if r["keyword_coverage"] is not None]
avg_cov = sum(valid_cov) / len(valid_cov)

print("\n" + "=" * 65)
print("FP16 + OPTIMIZED PROMPT SUMMARY")
print(f"Completed: {len(results)}/20")
print(f"Average latency: {avg_latency:.4f}s/question")
print(f"Overall generation speed: {total_tokens / total_time:.2f} tokens/s")
print(f"Average keyword coverage: {avg_cov:.2%}")
print(f"GPU allocated after load: {memory_after_load:.3f} GiB")
print(f"Average peak allocated: {avg_peak:.3f} GiB")
print(f"Maximum peak allocated: {max_peak:.3f} GiB")
print(f"JSON: {OUT_JSON}")
print(f"CSV: {OUT_CSV}")
print("=" * 65)
