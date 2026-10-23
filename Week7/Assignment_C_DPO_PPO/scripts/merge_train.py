import json, sys
from pathlib import Path

dst = Path("data/preference/dpo_train_final.jsonl")
if dst.exists():
    sys.exit("已存在，中止")

srcs = ["dpo_train_clean", "dpo_extra_sample", "dpo_extra_batch2"]
core = lambda s: s.split("：")[-1].strip()

bench = set()
for p in ["data/benchmark/target_eval.jsonl", "data/benchmark/general_eval.jsonl"]:
    for l in open(p, encoding="utf-8"):
        if l.strip():
            bench.add(core(json.loads(l)["question"]))

rows, seen = [], set()
for s in srcs:
    n = 0
    for l in open(f"data/preference/{s}.jsonl", encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        assert set(r) == {"prompt", "chosen", "rejected"}, f"{s}: 欄位不符 {list(r)}"
        assert all(str(r[k]).strip() for k in r), f"{s}: 有空欄位"
        assert r["prompt"] not in seen, f"重複 prompt: {r['prompt']}"
        assert core(r["prompt"]) not in bench, f"與 Benchmark 相同: {r['prompt']}"
        seen.add(r["prompt"])
        rows.append(r)
        n += 1
    print(f"{s}: {n} 筆")

assert len(rows) >= 100, f"筆數不足: {len(rows)}"
with dst.open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("已寫入", dst, "| 合計:", len(rows))
print("不重複 chosen:", len({r["chosen"] for r in rows}))
print("不重複核心題:", len({core(r["prompt"]) for r in rows}))
