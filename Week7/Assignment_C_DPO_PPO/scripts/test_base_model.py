import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

print(f"Loading tokenizer: {MODEL_NAME}", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

print(f"Loading model with dtype: {dtype}", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=dtype,
    device_map="auto",
)
model.eval()

messages = [
    {
        "role": "system",
        "content": (
            "你是一個謹慎、誠實的繁體中文助理。"
            "如果資訊不足以確定答案，請明確說明，不要捏造事實。"
        ),
    },
    {
        "role": "user",
        "content": "請問台灣第一座設在月球上的捷運站叫什麼名字？",
    },
]

# 明確取得 Tensor，不將 dict 傳入 generate()
encoded = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)

input_ids = encoded["input_ids"].to(model.device)

print("Input type:", type(input_ids).__name__, flush=True)
print("Input shape:", tuple(input_ids.shape), flush=True)
print("\n===== BASE MODEL RESPONSE =====", flush=True)

with torch.inference_mode():
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=120,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

new_tokens = outputs[0, input_ids.shape[1]:]
answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

print(answer)
print("\n===== GPU MEMORY =====")
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
print(f"Reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GiB")
