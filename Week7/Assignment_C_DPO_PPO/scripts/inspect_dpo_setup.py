import json, inspect
from transformers import AutoTokenizer

rows = [json.loads(l) for l in open("data/preference/dpo_train.jsonl", encoding="utf-8") if l.strip()]
print("筆數:", len(rows))
print("欄位:", list(rows[0].keys()))
print("第 1 筆:")
print(json.dumps(rows[0], ensure_ascii=False, indent=2)[:900])

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
def n(x):
    return len(tok(str(x))["input_ids"])
for k in rows[0]:
    if isinstance(rows[0][k], str):
        lens = [n(r[k]) for r in rows]
        print(f"{k}: max_tokens={max(lens)} mean={sum(lens)/len(lens):.0f}")

import trl
print("\nTRL 版本:", trl.__version__)
from trl import DPOConfig, DPOTrainer
print("DPOTrainer.__init__ 參數:", list(inspect.signature(DPOTrainer.__init__).parameters))
keys = ["beta", "max_length", "max_prompt_length", "max_completion_length",
        "loss_type", "precompute_ref_log_probs", "learning_rate",
        "per_device_train_batch_size", "gradient_accumulation_steps",
        "num_train_epochs", "bf16", "gradient_checkpointing", "padding_free"]
params = inspect.signature(DPOConfig.__init__).parameters
for k in keys:
    p = params.get(k)
    print(f"DPOConfig.{k}:", "不存在" if p is None else f"預設={p.default}")
