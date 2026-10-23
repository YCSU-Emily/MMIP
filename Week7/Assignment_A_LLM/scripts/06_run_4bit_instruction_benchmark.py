import csv
import json
import time
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE = Path.home() / "MMIP" / "Week7" / "Assignment_A_LLM"
BENCHMARK = BASE / "benchmarks" / "instruction_benchmark.json"
RESULT_JSON = BASE / "results" / "4bit_instruction_results.json"
RESULT_CSV = BASE / "results" / "4bit_instruction_results.csv"

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS = 160

SYSTEM_PROMPT = (
    "你是一位準確、清楚的助理。請依照使用者的指令回答，"
    "使用繁體中文，除非題目要求其他語言或格式。"
)

questions = json.loads(BENCHMARK.read_text(encoding="utf-8"))
assert len(questions) == 20, f"預期 20 題，實際為 {len(questions)}"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU 不可用，停止測試。")

print("=" * 65)
print("4-bit Quantized Instruction Benchmark")
print(f"Model: {MODEL_NAME}")
print(f"Questions: {len(questions)}")
print(f"Max new tokens: {MAX_NEW_TOKENS}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print("=" * 65)

# 使用 bitsandbytes NF4 4-bit 量化。
# 不升級或修改其他套件，也不載入 FP16 完整權重。
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
model_load_seconds = time.perf_counter() - load_start
input_device = model.get_input_embeddings().weight.device

memory_after_load_gib = torch.cuda.memory_allocated() / (1024 ** 3)
reserved_after_load_gib = torch.cuda.memory_reserved() / (1024 ** 3)

print(f"Model device: {input_device}")
print(f"Loaded in 4-bit: {getattr(model, 'is_loaded_in_4bit', False)}")
print(f"Model load time: {model_load_seconds:.2f} seconds")
print(f"GPU allocated after load: {memory_after_load_gib:.3f} GiB")
print(f"GPU reserved after load: {reserved_after_load_gib:.3f} GiB")

if not getattr(model, "is_loaded_in_4bit", False):
    raise RuntimeError("模型未確認以 4-bit 載入，停止測試。")

results = []

print("\n[2/3] Running the same 20 questions...")

for index, item in enumerate(questions, start=1):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": item["prompt"]},
    ]

    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(formatted_text, return_tensors="pt")
    inputs = {key: value.to(input_device) for key, value in inputs.items()}
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

    peak_allocated_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    peak_reserved_gib = torch.cuda.max_memory_reserved() / (1024 ** 3)

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
        "precision": "4-bit NF4",
        "model_name": MODEL_NAME,
        "peak_allocated_gib": round(peak_allocated_gib, 4),
        "peak_reserved_gib": round(peak_reserved_gib, 4),
    }
    results.append(row)

    # 每題完成後立即存檔，避免中途停止而失去已完成的結果。
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_name": MODEL_NAME,
        "precision": "4-bit NF4",
        "quantization": {
            "type": "NF4",
            "double_quant": True,
            "compute_dtype": "float16",
        },
        "max_new_tokens": MAX_NEW_TOKENS,
        "model_load_seconds": round(model_load_seconds, 4),
        "gpu_name": torch.cuda.get_device_name(0),
        "memory_after_load_gib": round(memory_after_load_gib, 4),
        "reserved_after_load_gib": round(reserved_after_load_gib, 4),
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
        f"{output_tokens} tokens | "
        f"peak allocated {peak_allocated_gib:.2f} GiB"
    )

print("\n[3/3] Saving CSV and summary...")

columns = [
    "id", "category", "prompt", "reference_answer",
    "evaluation_keywords", "model_answer", "input_tokens",
    "generated_tokens", "latency_seconds", "tokens_per_second",
    "hit_max_new_tokens", "precision", "model_name",
    "peak_allocated_gib", "peak_reserved_gib",
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
overall_speed = total_tokens / total_time if total_time else 0
truncated = sum(r["hit_max_new_tokens"] for r in results)
avg_peak_allocated = sum(
    r["peak_allocated_gib"] for r in results
) / len(results)
max_peak_allocated = max(r["peak_allocated_gib"] for r in results)

print("=" * 65)
print("4-bit BENCHMARK SUMMARY")
print(f"Questions completed: {len(results)}/20")
print(f"Model load time: {model_load_seconds:.2f} seconds")
print(f"Average latency: {avg_latency:.4f} seconds/question")
print(f"Total generated tokens: {total_tokens}")
print(f"Overall generation speed: {overall_speed:.2f} tokens/sec")
print(f"Responses reaching token limit: {truncated}")
print(f"GPU allocated after load: {memory_after_load_gib:.3f} GiB")
print(f"Average per-question peak allocated: {avg_peak_allocated:.3f} GiB")
print(f"Maximum per-question peak allocated: {max_peak_allocated:.3f} GiB")
print(f"JSON: {RESULT_JSON}")
print(f"CSV:  {RESULT_CSV}")
print("=" * 65)
