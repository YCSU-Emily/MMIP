import json, sys
from pathlib import Path
dst = Path("data/preference/dpo_train_clean.jsonl")
if dst.exists():
    sys.exit("已存在，中止")
core = lambda s: s.split("：")[-1].strip()
tg = [json.loads(l) for l in open("data/benchmark/target_eval.jsonl", encoding="utf-8") if l.strip()]
tg_cores = {core(r["question"]) for r in tg}
tr = [json.loads(l) for l in open("data/preference/dpo_train.jsonl", encoding="utf-8") if l.strip()]
keep = [r for r in tr if core(r["prompt"]) not in tg_cores]
with dst.open("w", encoding="utf-8") as f:
    for r in keep:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("原本:", len(tr), "| 移除:", len(tr) - len(keep), "| 剩餘:", len(keep))
print("剩餘核心題數:", len({core(r["prompt"]) for r in keep}))
