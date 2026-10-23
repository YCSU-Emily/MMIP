import json
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

parser = argparse.ArgumentParser()
parser.add_argument("--tag", default="baseline")
parser.add_argument("--model", default=MODEL_NAME)
args = parser.parse_args()

TARGET_FILE = ROOT / "data/benchmark/target_eval.jsonl"
GENERAL_FILE = ROOT / "data/benchmark/general_eval.jsonl"
OUT_DIR = ROOT / "results" / args.tag
OUT_DIR.mkdir(parents=True, exist_ok=True)

def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

print("Model:", args.model, flush=True)
print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(args.model)

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
print("Loading model with dtype:", dtype, flush=True)

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    dtype=dtype,
    device_map="auto",
)
model.eval()

system_prompt = (
    "你是一個謹慎、誠實的繁體中文助理。"
    "請根據可靠資訊回答。若問題前提不成立、資訊不足或無法確認，"
    "請明確說明，不要捏造人名、事件、數字或來源。"
)

def generate_answer(question):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
        return_attention_mask=True,
    )

    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)

    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0, input_ids.shape[1]:]
    return tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()


def evaluate_file(path, dataset_name):
    rows = read_jsonl(path)
    output_path = OUT_DIR / f"{dataset_name}_predictions.jsonl"

    print(f"\nEvaluating {dataset_name}: {len(rows)} questions", flush=True)
    results = []

    with output_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows, start=1):
            question = row["question"]
            answer = generate_answer(question)

            result = {
                **row,
                "model_answer": answer,
                "model_name": args.model,
                "evaluation_tag": args.tag,
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            results.append(result)

            if i <= 3 or i % 10 == 0 or i == len(rows):
                print(f"[{dataset_name}] {i}/{len(rows)}", flush=True)
                if i <= 3:
                    print("Q:", question)
                    print("A:", answer[:300])

    print("Saved:", output_path)
    return len(results)

target_count = evaluate_file(TARGET_FILE, "target")
general_count = evaluate_file(GENERAL_FILE, "general")

summary = {
    "model": args.model,
    "tag": args.tag,
    "target_questions": target_count,
    "general_questions": general_count,
    "total_questions": target_count + general_count,
    "dtype": str(dtype),
    "generation": {
        "do_sample": False,
        "max_new_tokens": 120,
    },
}

summary_path = OUT_DIR / "evaluation_summary.json"
summary_path.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\n===== EVALUATION COMPLETE =====")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"GPU allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
print("Summary:", summary_path)
