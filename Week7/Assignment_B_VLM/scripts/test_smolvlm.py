import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
IMAGE_PATH = "Week7/Assignment_B_VLM/images/vlm_smoke_test.png"
RESULT_PATH = Path("Week7/Assignment_B_VLM/results/smolvlm_smoke_test.txt")

PROMPT = (
    "請仔細觀察圖片，說明圖片中有哪些幾何形狀、"
    "各自的顏色，以及它們大致位於圖片的哪個位置。"
)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Please check the GPU environment.")

image = Image.open(IMAGE_PATH).convert("RGB")

print("Loading model:", MODEL_ID)
print("GPU:", torch.cuda.get_device_name(0))

start_load = time.perf_counter()

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
)

load_seconds = time.perf_counter() - start_load
print(f"Model loaded in {load_seconds:.2f} seconds")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": PROMPT},
        ],
    }
]

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

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start_infer = time.perf_counter()

with torch.inference_mode():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
    )

torch.cuda.synchronize()
inference_seconds = time.perf_counter() - start_infer

input_length = inputs["input_ids"].shape[1]
new_tokens = generated_ids[0][input_length:]

answer = processor.decode(
    new_tokens,
    skip_special_tokens=True,
)

peak_memory_gib = torch.cuda.max_memory_allocated() / 1024**3

print("\n========== MODEL ANSWER ==========")
print(answer)
print("\n========== PERFORMANCE ==========")
print(f"Model load time: {load_seconds:.2f} seconds")
print(f"Inference time: {inference_seconds:.3f} seconds")
print(f"Generated tokens: {len(new_tokens)}")
print(f"Peak GPU memory: {peak_memory_gib:.2f} GiB")

RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
RESULT_PATH.write_text(
    f"Model: {MODEL_ID}\n"
    f"Image: {IMAGE_PATH}\n"
    f"Prompt: {PROMPT}\n"
    f"Answer:\n{answer}\n\n"
    f"Model load time: {load_seconds:.2f} seconds\n"
    f"Inference time: {inference_seconds:.3f} seconds\n"
    f"Generated tokens: {len(new_tokens)}\n"
    f"Peak GPU memory: {peak_memory_gib:.2f} GiB\n",
    encoding="utf-8",
)

print("Saved result:", RESULT_PATH)
