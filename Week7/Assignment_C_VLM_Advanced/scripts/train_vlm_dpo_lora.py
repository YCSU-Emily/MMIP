import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/vlm_preferences_train.jsonl"
MODEL_PATH = Path(
    "/home/emily/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-VL-3B-Instruct/"
    "snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
)
OUTPUT_DIR = ROOT / "checkpoints/dpo_lora_v1"

SEED = 20261024
EPOCHS = 1
LEARNING_RATE = 5e-6
BETA = 0.1

if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"{OUTPUT_DIR} already exists. "
        "Refusing to overwrite an earlier run."
    )

if not DATA_PATH.is_file():
    raise FileNotFoundError(DATA_PATH)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable.")

print("=" * 65)
print("1. Load and validate preference data")
print("=" * 65)

with DATA_PATH.open("r", encoding="utf-8") as f:
    raw_rows = [
        json.loads(line)
        for line in f
        if line.strip()
    ]

required = {"image", "prompt", "chosen", "rejected"}
seen_ids = set()
rows = []

for i, row in enumerate(raw_rows):
    missing = required - row.keys()
    if missing:
        raise ValueError(f"Row {i} missing keys: {sorted(missing)}")

    image_path = Path(row["image"])
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image missing in row {i}: {image_path}"
        )

    if not row["prompt"].strip():
        raise ValueError(f"Empty prompt in row {i}")
    if not row["chosen"].strip() or not row["rejected"].strip():
        raise ValueError(f"Empty preference answer in row {i}")
    if row["chosen"].strip() == row["rejected"].strip():
        raise ValueError(f"Identical chosen/rejected in row {i}")

    image_id = row.get("image_id")
    if image_id is not None:
        if image_id in seen_ids:
            raise ValueError(f"Duplicate image_id: {image_id}")
        seen_ids.add(image_id)

    # TRL 的 vision preference collator 需要 conversational format。
    rows.append({
        "image": str(image_path),
        "prompt": [{
            "role": "user",
            "content": row["prompt"],
        }],
        "chosen": [{
            "role": "assistant",
            "content": row["chosen"],
        }],
        "rejected": [{
            "role": "assistant",
            "content": row["rejected"],
        }],
    })

if not rows:
    raise ValueError("Preference dataset is empty.")

dataset = Dataset.from_list(rows)
print("Training examples:", len(dataset))
print("All image paths and preference fields validated.")

print("\n" + "=" * 65)
print("2. Load model and processor")
print("=" * 65)

processor = AutoProcessor.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True,
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    str(MODEL_PATH),
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
    local_files_only=True,
)

model.config.use_cache = False

print("Model:", type(model).__name__)
print("GPU:", torch.cuda.get_device_name(0))

print("\n" + "=" * 65)
print("3. Configure language-model-only LoRA")
print("=" * 65)

suffixes = {
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
}

target_modules = [
    name
    for name, module in model.named_modules()
    if name.startswith("model.language_model.")
    and name.rsplit(".", 1)[-1] in suffixes
    and isinstance(module, torch.nn.Linear)
]

if not target_modules:
    raise RuntimeError("No LoRA target modules found.")

print("LoRA target modules:", len(target_modules))

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=target_modules,
)

print("\n" + "=" * 65)
print("4. Configure DPO training")
print("=" * 65)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

args = DPOConfig(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=LEARNING_RATE,
    beta=BETA,
    bf16=True,
    gradient_checkpointing=False,
    max_length=None,
    precompute_ref_log_probs=False,
    remove_unused_columns=False,
    save_strategy="epoch",
    save_total_limit=2,
    logging_strategy="steps",
    logging_steps=5,
    report_to="none",
    dataloader_num_workers=0,
    optim="adamw_torch",
    seed=SEED,
    data_seed=SEED,
)

trainer = DPOTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    processing_class=processor,
    peft_config=peft_config,
)

print("Trainer initialized.")
print("Epochs:", EPOCHS)
print("Examples:", len(dataset))
print("Effective batch size:", args.per_device_train_batch_size)
print("Output:", OUTPUT_DIR)

print("\n" + "=" * 65)
print("5. Start training")
print("=" * 65)

train_result = trainer.train()

print("\n" + "=" * 65)
print("6. Save final adapter")
print("=" * 65)

trainer.save_model(str(OUTPUT_DIR / "final_adapter"))
processor.save_pretrained(str(OUTPUT_DIR / "final_adapter"))
trainer.save_state()

metrics_path = OUTPUT_DIR / "train_metrics.json"
metrics = {
    key: (
        float(value)
        if isinstance(value, (int, float))
        else str(value)
    )
    for key, value in train_result.metrics.items()
}
metrics["training_examples"] = len(dataset)
metrics["epochs_configured"] = EPOCHS
metrics["learning_rate"] = LEARNING_RATE
metrics["beta"] = BETA
metrics["seed"] = SEED

with metrics_path.open("w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print("Metrics:", metrics_path)
print("Adapter:", OUTPUT_DIR / "final_adapter")
print("Training finished.")
