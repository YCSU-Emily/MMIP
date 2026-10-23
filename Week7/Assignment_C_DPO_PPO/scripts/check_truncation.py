import json
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

def load(tag, name):
    path = f"results/{tag}/{name}_predictions.jsonl"
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

for tag, cap in [("baseline_max120", 120), ("baseline_256", 256)]:
    for name in ["general", "target"]:
        rows = load(tag, name)
        lens = [len(tok(str(r.get("model_answer", "")))["input_ids"]) for r in rows]
        hit = sum(n >= cap - 1 for n in lens)
        print(f"{tag:16s} {name:8s} n={len(rows)} max_tokens={max(lens)} near_cap={hit}")

print()
for name in ["general", "target"]:
    a = {r["id"]: str(r["model_answer"]).strip() for r in load("baseline_max120", name)}
    b = {r["id"]: str(r["model_answer"]).strip() for r in load("baseline_256", name)}
    same = sum(a[i] == b[i] for i in a)
    prefix = sum(a[i] != b[i] and b[i].startswith(a[i]) for i in a)
    other = [i for i in a if a[i] != b[i] and not b[i].startswith(a[i])]
    print(name, "same =", same, "| prefix_extended =", prefix, "| other_diff =", len(other), other[:10])
