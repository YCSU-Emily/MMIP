import argparse
import csv
import json
import math
import random
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

DATA_DIR = ROOT / "experiments/B_sft_rlhf/data"
CHECKPOINT_DIR = ROOT / "experiments/B_sft_rlhf/checkpoints/qwen35_lora"
RESULT_DIR = ROOT / "experiments/B_sft_rlhf/results"

TARGET_MODULES = [
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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--run_name", type=str, default="sft_v1")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def load_jsonl(path):
    records = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def find_all(sequence, pattern):
    positions = []

    for i in range(len(sequence) - len(pattern) + 1):
        if sequence[i:i + len(pattern)] == pattern:
            positions.append(i)

    return positions


def encode_record(record, tokenizer, max_length):
    messages = record["messages"]
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

    positions = find_all(input_ids, target_ids)

    if len(positions) != 1:
        raise ValueError(
            f"Invalid JSON target boundary: "
            f"{record.get('metadata', {}).get('sample_id')}, "
            f"matches={len(positions)}"
        )

    start = positions[0]
    end = start + len(target_ids)

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    if input_ids[end] != im_end_id:
        raise ValueError("Assistant JSON is not followed by <|im_end|>")

    labels = (
        [-100] * start
        + input_ids[start:end + 1]
        + [-100] * (len(input_ids) - end - 1)
    )

    if len(input_ids) > max_length:
        raise ValueError(
            f"Sequence exceeds max_length: {len(input_ids)}"
        )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "sample_id": record.get("metadata", {}).get("sample_id", "unknown")
    }


def evaluate(model, records, tokenizer, device, max_length):
    model.eval()

    losses = []

    with torch.no_grad():
        for record in records:
            item = encode_record(record, tokenizer, max_length)

            input_ids = torch.tensor(
                [item["input_ids"]],
                dtype=torch.long,
                device=device
            )

            attention_mask = torch.tensor(
                [item["attention_mask"]],
                dtype=torch.long,
                device=device
            )

            labels = torch.tensor(
                [item["labels"]],
                dtype=torch.long,
                device=device
            )

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False
            )

            loss = float(outputs.loss.detach().float().cpu())

            if not math.isfinite(loss):
                raise RuntimeError("Validation loss is not finite")

            losses.append(loss)

    model.train()

    return sum(losses) / len(losses)


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not support BF16")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda:0")

    train_path = DATA_DIR / "train.jsonl"
    validation_path = DATA_DIR / "validation.jsonl"

    train_records = load_jsonl(train_path)
    validation_records = load_jsonl(validation_path)

    if not train_records or not validation_records:
        raise RuntimeError("Training or validation dataset is empty")

    print("===== 1. EXPERIMENT CONFIG =====")
    print("Run name:", args.run_name)
    print("Training samples:", len(train_records))
    print("Validation samples:", len(validation_records))
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)
    print("Gradient accumulation:", args.gradient_accumulation)
    print("Learning rate:", args.learning_rate)
    print("Max length:", args.max_length)
    print("GPU:", torch.cuda.get_device_name(0))

    if args.batch_size != 1:
        raise ValueError("This initial implementation supports batch_size=1")

    print("\n===== 2. LOAD TOKENIZER =====")

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        trust_remote_code=True
    )

    print("Tokenizer:", type(tokenizer).__name__)

    print("\n===== 3. CHECK DATASET TOKEN BOUNDARIES =====")

    for record in train_records + validation_records:
        encode_record(record, tokenizer, args.max_length)

    print("All samples passed token-boundary checks.")

    print("\n===== 4. LOAD BASE MODEL =====")

    model = AutoModelForImageTextToText.from_pretrained(
        str(MODEL_PATH),
        dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
        trust_remote_code=True
    )

    model.config.use_cache = False

    print("Model:", type(model).__name__)
    print("Dtype:", next(model.parameters()).dtype)

    print("\n===== 5. ENABLE GRADIENT CHECKPOINTING =====")

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()

    print("Gradient checkpointing enabled.")

    print("\n===== 6. ATTACH LORA =====")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES
    )

    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    trainable_params = [
        p for p in model.parameters() if p.requires_grad
    ]

    vision_trainable = [
        name for name, p in model.named_parameters()
        if p.requires_grad and "visual" in name.lower()
    ]

    if vision_trainable:
        raise RuntimeError("Vision parameters unexpectedly trainable")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    updates_per_epoch = math.ceil(
        len(train_records) / args.gradient_accumulation
    )

    total_updates = updates_per_epoch * args.epochs

    if args.max_steps > 0:
        total_updates = min(total_updates, args.max_steps)

    warmup_steps = max(1, int(total_updates * 0.05))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        remaining = max(1, total_updates - warmup_steps)
        progress = min(
            1.0,
            (step - warmup_steps) / remaining
        )

        return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda
    )

    print("\n===== 7. TRAINING PLAN =====")
    print("Updates per epoch:", updates_per_epoch)
    print("Planned optimizer updates:", total_updates)
    print("Warmup steps:", warmup_steps)

    run_checkpoint_dir = CHECKPOINT_DIR / f"{args.run_name}_best"
    run_result_dir = RESULT_DIR / args.run_name

    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_result_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_validation_loss = float("inf")
    optimizer_steps = 0
    stop_training = False

    print("\n===== 8. START TRAINING =====")

    for epoch in range(args.epochs):
        model.train()

        indices = list(range(len(train_records)))
        random.Random(args.seed + epoch).shuffle(indices)

        epoch_loss_sum = 0.0
        epoch_sample_count = 0

        for group_start in range(
            0, len(indices), args.gradient_accumulation
        ):
            group_indices = indices[
                group_start:
                group_start + args.gradient_accumulation
            ]

            optimizer.zero_grad(set_to_none=True)

            group_loss_sum = 0.0

            for index in group_indices:
                record = train_records[index]
                item = encode_record(
                    record, tokenizer, args.max_length
                )

                input_ids = torch.tensor(
                    [item["input_ids"]],
                    dtype=torch.long,
                    device=device
                )

                attention_mask = torch.tensor(
                    [item["attention_mask"]],
                    dtype=torch.long,
                    device=device
                )

                labels = torch.tensor(
                    [item["labels"]],
                    dtype=torch.long,
                    device=device
                )

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False
                )

                loss = outputs.loss

                if not torch.isfinite(loss).item():
                    raise RuntimeError("Training loss is not finite")

                (loss / len(group_indices)).backward()

                loss_value = float(loss.detach().float().cpu())
                group_loss_sum += loss_value
                epoch_loss_sum += loss_value
                epoch_sample_count += 1

                del outputs, loss, input_ids, attention_mask, labels

            torch.nn.utils.clip_grad_norm_(
                trainable_params,
                max_norm=1.0
            )

            optimizer.step()
            scheduler.step()

            optimizer_steps += 1

            optimizer.zero_grad(set_to_none=True)

            if optimizer_steps % 5 == 0 or optimizer_steps == 1:
                print(
                    f"Epoch {epoch + 1}/{args.epochs} | "
                    f"Step {optimizer_steps}/{total_updates} | "
                    f"Group loss {group_loss_sum / len(group_indices):.4f} | "
                    f"LR {scheduler.get_last_lr()[0]:.2e}"
                )

            if args.max_steps > 0 and optimizer_steps >= args.max_steps:
                stop_training = True
                break

        train_loss = epoch_loss_sum / max(1, epoch_sample_count)

        print(f"\nEpoch {epoch + 1} training loss: {train_loss:.6f}")
        print("Running validation...")

        validation_loss = evaluate(
            model,
            validation_records,
            tokenizer,
            device,
            args.max_length
        )

        print(f"Epoch {epoch + 1} validation loss: {validation_loss:.6f}")

        history.append({
            "epoch": epoch + 1,
            "optimizer_steps": optimizer_steps,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": scheduler.get_last_lr()[0]
        })

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss

            model.save_pretrained(run_checkpoint_dir)

            print("Saved new best LoRA adapter:")
            print(run_checkpoint_dir)

        with (run_result_dir / "training_history.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        with (run_result_dir / "training_history.csv").open(
            "w", encoding="utf-8", newline=""
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "optimizer_steps",
                    "train_loss",
                    "validation_loss",
                    "learning_rate"
                ]
            )
            writer.writeheader()
            writer.writerows(history)

        if stop_training:
            break

    summary = {
        "experiment": "B - Qwen3.5-4B LoRA SFT",
        "run_name": args.run_name,
        "model": "Qwen3.5-4B",
        "precision": "BF16",
        "training_samples": len(train_records),
        "validation_samples": len(validation_records),
        "image_training_samples": 0,
        "epochs_completed": len(history),
        "optimizer_steps": optimizer_steps,
        "best_validation_loss": best_validation_loss,
        "lora": {
            "r": 8,
            "alpha": 16,
            "dropout": 0.05,
            "target_modules": TARGET_MODULES
        },
        "training_config": vars(args),
        "vision_encoder_finetuned": False,
        "history": history
    }

    with (run_result_dir / "summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== TRAINING FINISHED =====")
    print("Optimizer steps:", optimizer_steps)
    print("Best validation loss:", best_validation_loss)
    print("Checkpoint:", run_checkpoint_dir)
    print("Results:", run_result_dir)


if __name__ == "__main__":
    main()
