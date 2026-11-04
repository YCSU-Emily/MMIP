from pathlib import Path
import torch
import time
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_PATH = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/"
      "snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
)

IMAGE_PATH = Path("images/real_product.jpg")

def main():
    print("===== QWEN3.5-4B VISION TEST =====")
    print("Model:", MODEL_PATH)
    print("Image:", IMAGE_PATH)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(IMAGE_PATH)

    print("\n===== LOAD IMAGE =====")
    image = Image.open(IMAGE_PATH).convert("RGB")
    print("Image size:", image.size)

    print("\n===== LOAD PROCESSOR =====")
    processor = AutoProcessor.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        min_pixels=256 * 256,
        max_pixels=1024 * 1024,
    )

    print("\n===== LOAD MODEL =====")
    model = AutoModelForImageTextToText.from_pretrained(
        str(MODEL_PATH),
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True,
    )
    model.eval()

    print("Model loaded successfully.")
    print("Device:", model.device)
    print(
        "GPU memory allocated:",
        round(torch.cuda.memory_allocated() / 1024**3, 2),
        "GB",
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": (
                        "請仔細描述這張圖片中的內容。"
                        "如果有產品，請指出可辨識的產品類型、"
                        "外觀特徵，以及是否能觀察到明顯損壞。"
                        "請只描述圖片中實際可見的資訊，"
                        "不確定的部分請明確說明。"
                    ),
                },
            ],
        }
    ]

    print("\n===== IMAGE INFERENCE =====")

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    torch.cuda.synchronize()
    inference_start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=320,
            do_sample=False,
        )

    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_start
    print(f"Inference time: {inference_seconds:.3f} seconds")
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    answer = processor.decode(
        new_tokens,
        skip_special_tokens=True,
    )

    print("\n===== MODEL ANSWER =====")
    print(answer)

    print("\n===== GPU MEMORY AFTER INFERENCE =====")
    print(
        "Allocated:",
        round(torch.cuda.memory_allocated() / 1024**3, 2),
        "GB",
    )
    print("\n===== TEST FINISHED =====")


if __name__ == "__main__":
    main()
