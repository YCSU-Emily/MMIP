import json
import time
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ===== 基本設定 =====
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
BASE_DIR = Path.home() / "MMIP" / "Week7" / "Assignment_A_LLM"
RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = RESULT_DIR / "baseline_results.json"

PROMPT = (
    "請用繁體中文解釋什麼是人工智慧，"
    "並列出三個日常生活中的應用。"
)

print("=" * 60)
print("Assignment A - LLM Baseline Test")
print("=" * 60)
print(f"Model: {MODEL_NAME}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise RuntimeError("目前偵測不到 CUDA GPU，先停止測試。")

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB")

# ===== 1. 載入 Tokenizer =====
print("\n[1/4] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ===== 2. 載入 FP16 模型 =====
print("\n[2/4] Loading FP16 model...")
print("第一次執行可能需要下載數 GB 的模型，請耐心等待。")

load_start = time.perf_counter()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()

load_seconds = time.perf_counter() - load_start
print(f"Model loading time: {load_seconds:.2f} seconds")
print(f"Model device: {model.device}")

# ===== 3. 準備輸入 =====
messages = [
    {
        "role": "system",
        "content": "你是一位擅長解釋人工智慧概念的助理，請清楚且準確地回答。",
    },
    {
        "role": "user",
        "content": PROMPT,
    },
]

input_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(input_text, return_tensors="pt")
input_device = model.get_input_embeddings().weight.device
inputs = {key: value.to(input_device) for key, value in inputs.items()}

input_tokens = inputs["input_ids"].shape[1]

# ===== 4. 執行推論與測量 =====
print("\n[3/4] Running inference...")

# 先做一次短暫暖機，降低首次 CUDA 初始化對測量的影響
with torch.inference_mode():
    _ = model.generate(
        **inputs,
        max_new_tokens=8,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

if torch.cuda.is_available():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

start_time = time.perf_counter()

with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

if torch.cuda.is_available():
    torch.cuda.synchronize()

inference_seconds = time.perf_counter() - start_time

generated_ids = output_ids[0][input_tokens:]
generated_tokens = len(generated_ids)
response = tokenizer.decode(
    generated_ids,
    skip_special_tokens=True,
)

tokens_per_second = (
    generated_tokens / inference_seconds
    if inference_seconds > 0
    else 0
)

# PyTorch 記憶體統計：反映 PyTorch 管理的 GPU 記憶體
gpu_allocated_gib = torch.cuda.memory_allocated() / 1024**3
gpu_reserved_gib = torch.cuda.memory_reserved() / 1024**3
gpu_peak_allocated_gib = torch.cuda.max_memory_allocated() / 1024**3
gpu_peak_reserved_gib = torch.cuda.max_memory_reserved() / 1024**3

# ===== 儲存結果 =====
result = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "model_name": MODEL_NAME,
    "precision": "FP16",
    "prompt": PROMPT,
    "response": response,
    "input_tokens": input_tokens,
    "generated_tokens": generated_tokens,
    "model_load_seconds": round(load_seconds, 4),
    "inference_seconds": round(inference_seconds, 4),
    "tokens_per_second": round(tokens_per_second, 4),
    "gpu_name": torch.cuda.get_device_name(0),
    "gpu_total_memory_gib": round(
        torch.cuda.get_device_properties(0).total_memory / 1024**3, 3
    ),
    "gpu_allocated_gib_after_inference": round(gpu_allocated_gib, 4),
    "gpu_reserved_gib_after_inference": round(gpu_reserved_gib, 4),
    "gpu_peak_allocated_gib": round(gpu_peak_allocated_gib, 4),
    "gpu_peak_reserved_gib": round(gpu_peak_reserved_gib, 4),
    "measurement_note": (
        "GPU memory figures are PyTorch CUDA allocator statistics, "
        "not total system GPU usage. Peak statistics were reset after "
        "model loading and before measured inference."
    ),
}

OUTPUT_JSON.write_text(
    json.dumps(result, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\n[4/4] Baseline results")
print("=" * 60)
print("Response:")
print(response)
print("-" * 60)
print(f"Input tokens: {input_tokens}")
print(f"Generated tokens: {generated_tokens}")
print(f"Inference time: {inference_seconds:.4f} seconds")
print(f"Generation speed: {tokens_per_second:.2f} tokens/sec")
print(f"GPU allocated after inference: {gpu_allocated_gib:.3f} GiB")
print(f"GPU peak allocated: {gpu_peak_allocated_gib:.3f} GiB")
print(f"GPU peak reserved: {gpu_peak_reserved_gib:.3f} GiB")
print(f"Model loading time: {load_seconds:.2f} seconds")
print(f"\nResults saved to: {OUTPUT_JSON}")
print("=" * 60)
