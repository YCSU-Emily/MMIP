import json
import time
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUT_PATH = Path("results/4bit_smoke_test.json")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA 不可用，停止測試。")

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

print("=" * 60)
print("Model:", MODEL_NAME)
print("GPU:", torch.cuda.get_device_name(0))
print("Quantization: 4-bit NF4")
print("=" * 60)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

print("\n[1/4] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("\n[2/4] Loading 4-bit model...")
load_start = time.perf_counter()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    torch_dtype=torch.float16,
    device_map="auto",
)

load_seconds = time.perf_counter() - load_start
print(f"Model load time: {load_seconds:.2f} sec")
print("Model device:", model.device)
print("Model quantization:", getattr(model, "is_loaded_in_4bit", "check below"))

print("\n[3/4] Running inference...")
messages = [
    {
        "role": "system",
        "content": "你是一位精確、清楚的繁體中文助理。請直接回答問題。"
    },
    {
        "role": "user",
        "content": "請列出三個節省家庭用電的方法，每點用一句話說明。"
    },
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

torch.cuda.synchronize()
inference_seconds = time.perf_counter() - start

input_tokens = inputs["input_ids"].shape[1]
new_tokens = output.shape[1] - input_tokens
answer = tokenizer.decode(
    output[0][input_tokens:],
    skip_special_tokens=True,
)

allocated_gib = torch.cuda.memory_allocated() / 1024**3
peak_allocated_gib = torch.cuda.max_memory_allocated() / 1024**3
peak_reserved_gib = torch.cuda.max_memory_reserved() / 1024**3
tokens_per_second = new_tokens / inference_seconds if inference_seconds else 0

print("\n[4/4] Results")
print("-" * 60)
print("Answer:")
print(answer)
print("-" * 60)
print(f"Load time: {load_seconds:.3f} sec")
print(f"Inference time: {inference_seconds:.3f} sec")
print(f"Input tokens: {input_tokens}")
print(f"Generated tokens: {new_tokens}")
print(f"Generation speed: {tokens_per_second:.2f} tokens/sec")
print(f"GPU allocated after inference: {allocated_gib:.3f} GiB")
print(f"GPU peak allocated: {peak_allocated_gib:.3f} GiB")
print(f"GPU peak reserved: {peak_reserved_gib:.3f} GiB")

result = {
    "model": MODEL_NAME,
    "quantization": "4-bit NF4 with double quantization",
    "compute_dtype": "float16",
    "load_seconds": load_seconds,
    "inference_seconds": inference_seconds,
    "input_tokens": input_tokens,
    "generated_tokens": new_tokens,
    "tokens_per_second": tokens_per_second,
    "gpu_allocated_after_inference_gib": allocated_gib,
    "gpu_peak_allocated_gib": peak_allocated_gib,
    "gpu_peak_reserved_gib": peak_reserved_gib,
    "answer": answer,
}

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(
    json.dumps(result, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(f"\nResults saved to: {OUT_PATH.resolve()}")
