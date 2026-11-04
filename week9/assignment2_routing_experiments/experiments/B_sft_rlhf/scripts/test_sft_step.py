import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/"
      "snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
)

DATA_PATH = ROOT / "experiments/B_sft_rlhf/data/train.jsonl"

print("===== 1. ENVIRONMENT =====")
assert torch.cuda.is_available(), "CUDA is not available"
assert torch.cuda.is_bf16_supported(), "BF16 is not supported"

device = torch.device("cuda:0")

print("GPU:", torch.cuda.get_device_name(0))
print("BF16:", torch.cuda.is_bf16_supported())

print("\n===== 2. LOAD TOKENIZER =====")

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True,
    trust_remote_code=True
)

print("Tokenizer:", type(tokenizer).__name__)

print("\n===== 3. PREPARE ONE TRAINING SAMPLE =====")

with DATA_PATH.open(encoding="utf-8") as f:
    sample = json.loads(next(f))

messages = sample["messages"]
assistant_text = messages[1]["content"]

encoded = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    return_dict=True
)

input_ids = encoded["input_ids"]
attention_mask = encoded["attention_mask"]

if hasattr(input_ids, "tolist"):
    input_ids = input_ids.tolist()
if hasattr(attention_mask, "tolist"):
    attention_mask = attention_mask.tolist()

if input_ids and isinstance(input_ids[0], list):
    input_ids = input_ids[0]
if attention_mask and isinstance(attention_mask[0], list):
    attention_mask = attention_mask[0]

target_ids = tokenizer(
    assistant_text,
    add_special_tokens=False
)["input_ids"]

def find_all(sequence, pattern):
    positions = []
    for i in range(len(sequence) - len(pattern) + 1):
        if sequence[i:i + len(pattern)] == pattern:
            positions.append(i)
    return positions

positions = find_all(input_ids, target_ids)

assert len(positions) == 1, (
    f"Expected one JSON boundary, found {len(positions)}"
)

start = positions[0]
end = start + len(target_ids)

im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

assert input_ids[end] == im_end_id, (
    "JSON is not immediately followed by <|im_end|>"
)

# Only assistant JSON and its ending token contribute to loss.
labels = (
    [-100] * start
    + input_ids[start:end + 1]
    + [-100] * (len(input_ids) - end - 1)
)

assert len(labels) == len(input_ids)
assert sum(x != -100 for x in labels) == len(target_ids) + 1

input_tensor = torch.tensor(
    [input_ids], dtype=torch.long, device=device
)
attention_tensor = torch.tensor(
    [attention_mask], dtype=torch.long, device=device
)
labels_tensor = torch.tensor(
    [labels], dtype=torch.long, device=device
)

print("Sample:", sample["metadata"]["sample_id"])
print("Sequence length:", len(input_ids))
print("JSON start:", start)
print("JSON target tokens:", len(target_ids))
print("Loss tokens (JSON + im_end):", sum(x != -100 for x in labels))

print("\n===== 4. LOAD QWEN3.5-4B =====")

model = AutoModelForImageTextToText.from_pretrained(
    str(MODEL_PATH),
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    local_files_only=True,
    trust_remote_code=True
)

model.config.use_cache = False

print("Model:", type(model).__name__)
print("Model dtype:", next(model.parameters()).dtype)
print("Model device:", next(model.parameters()).device)

print("\n===== 5. ENABLE GRADIENT CHECKPOINTING =====")

model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
model.enable_input_require_grads()

print("Gradient checkpointing enabled")

print("\n===== 6. ATTACH LORA =====")

target_modules = [
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=target_modules
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

trainable = [
    (name, param)
    for name, param in model.named_parameters()
    if param.requires_grad
]

vision_trainable = [
    name for name, _ in trainable
    if "visual" in name.lower()
]

print("Trainable tensors:", len(trainable))
print("Trainable vision tensors:", len(vision_trainable))

assert len(trainable) > 0, "No trainable LoRA parameters"
assert len(vision_trainable) == 0, (
    "Vision parameters unexpectedly trainable"
)

print("\n===== 7. FORWARD PASS =====")

model.train()
model.zero_grad(set_to_none=True)
torch.cuda.reset_peak_memory_stats()

outputs = model(
    input_ids=input_tensor,
    attention_mask=attention_tensor,
    labels=labels_tensor,
    use_cache=False
)

loss = outputs.loss

print("Loss:", float(loss.detach().cpu()))
print("Loss finite:", bool(torch.isfinite(loss).item()))

assert torch.isfinite(loss).item(), "Loss is NaN or Inf"

print("\n===== 8. BACKWARD PASS =====")

loss.backward()

grad_tensors = [
    (name, param.grad)
    for name, param in trainable
    if param.grad is not None
]

nonzero_grad_tensors = [
    (name, grad)
    for name, grad in grad_tensors
    if torch.count_nonzero(grad).item() > 0
]

print("Trainable tensors:", len(trainable))
print("Tensors with gradients:", len(grad_tensors))
print("Tensors with nonzero gradients:", len(nonzero_grad_tensors))

assert len(grad_tensors) > 0, "No LoRA gradients found"
assert len(nonzero_grad_tensors) > 0, "All LoRA gradients are zero"

grad_norm = torch.sqrt(sum(
    torch.sum(grad.float() ** 2)
    for _, grad in grad_tensors
))

print("LoRA gradient norm:", float(grad_norm.detach().cpu()))

print("\n===== 9. MEMORY =====")

print(
    "Peak allocated VRAM (GB):",
    round(torch.cuda.max_memory_allocated() / 1024**3, 2)
)
print(
    "Peak reserved VRAM (GB):",
    round(torch.cuda.max_memory_reserved() / 1024**3, 2)
)

print("\n===== RESULT =====")
print("PASS: one-sample BF16 LoRA forward/backward completed.")
print("No checkpoint was saved.")
