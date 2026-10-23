import json
tr = [json.loads(l) for l in open("data/preference/dpo_train.jsonl", encoding="utf-8") if l.strip()]
tg = [json.loads(l) for l in open("data/benchmark/target_eval.jsonl", encoding="utf-8") if l.strip()]
print("Target 欄位:", list(tg[0].keys()))
qkey = "question" if "question" in tg[0] else list(tg[0].keys())[1]
core = lambda s: s.split("：")[-1].strip()
train_cores = {core(r["prompt"]) for r in tr}
print("訓練不重複核心題數:", len(train_cores))
exact = [r[qkey] for r in tg if core(r[qkey]) in train_cores]
print("Target 與訓練核心題完全相同:", len(exact))
for q in exact[:5]:
    print(" -", q)
