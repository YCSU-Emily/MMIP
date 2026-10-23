import ast, json, argparse
from pathlib import Path
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="models/dpo_lora_v1")
ap.add_argument("--max_steps", type=int, default=-1)
ap.add_argument("--lr", type=float, default=2e-5)
args = ap.parse_args()

MODEL = "Qwen/Qwen2.5-3B-Instruct"
tree = ast.parse(open("scripts/evaluate_model.py", encoding="utf-8").read())
SP = None
for node in ast.walk(tree):
    if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "system_prompt":
        SP = ast.literal_eval(node.value)
assert SP, "讀不到 system_prompt"
print("system prompt:", SP)

rows = [json.loads(l) for l in open("data/preference/dpo_train_final.jsonl", encoding="utf-8") if l.strip()]
data = [{
    "prompt": [{"role": "system", "content": SP}, {"role": "user", "content": r["prompt"]}],
    "chosen": [{"role": "assistant", "content": r["chosen"]}],
    "rejected": [{"role": "assistant", "content": r["rejected"]}],
} for r in rows]
ds = Dataset.from_list(data)
print("訓練筆數:", len(ds))

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto")
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
cfg = DPOConfig(output_dir=args.out, per_device_train_batch_size=8, gradient_accumulation_steps=1,
                learning_rate=args.lr, num_train_epochs=3, max_steps=args.max_steps,
                max_length=256, beta=0.1, warmup_steps=5, logging_steps=1,
                save_strategy="no", report_to="none", bf16=True,
                gradient_checkpointing=False, seed=42)
trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lora)
trainer.model.print_trainable_parameters()
trainer.train()
trainer.save_model(args.out)
print("已儲存 adapter:", args.out)
print("DONE")
