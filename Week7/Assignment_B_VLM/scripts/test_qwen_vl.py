import time
import torch
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
IMAGE_PATH = "Week7/Assignment_B_VLM/images/vlm_smoke_test.png"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Please check the GPU environment.")

print("Loading model:", MODEL_ID)
print("GPU:", torch.cuda.get_device_name(0))

start_load = time.perf_counter()

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

load_seconds = time.perf_counter() - start_load
print(f"Model loaded in {load_seconds:.2f} seconds")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": str(Path(IMAGE_PATH).resolve())},
            {
                "type": "text",
                "text": (
                    "請仔細觀察圖片，說明圖片中有哪些幾何形狀、"
                    "各自的顏色，以及它們大致位於圖片的哪個位置。"
                ),
            },
        ],
    }
]

prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[prompt],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

if torch.cuda.is_available():
    torch.cuda.synchronize()

start_infer = time.perf_counter()

with torch.inference_mode():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
    )

if torch.cuda.is_available():
    torch.cuda.synchronize()

inference_seconds = time.perf_counter() - start_infer

generated_ids_trimmed = [
    output_ids[len(input_ids):]
    for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
]

answer = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]

print("\n========== MODEL ANSWER ==========")
print(answer)
print("\n========== PERFORMANCE ==========")
print(f"Model load time: {load_seconds:.2f} seconds")
print(f"Inference time: {inference_seconds:.3f} seconds")
print(f"Generated tokens: {len(generated_ids_trimmed[0])}")
print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")

result_path = Path("Week7/Assignment_B_VLM/results/qwen_smoke_test.txt")
result_path.write_text(
    f"Model: {MODEL_ID}\n"
    f"Image: {IMAGE_PATH}\n"
    f"Answer:\n{answer}\n\n"
    f"Model load time: {load_seconds:.2f} seconds\n"
    f"Inference time: {inference_seconds:.3f} seconds\n",
    encoding="utf-8",
)
print("Saved result:", result_path)
