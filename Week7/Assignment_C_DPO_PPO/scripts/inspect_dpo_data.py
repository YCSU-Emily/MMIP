import json
from collections import Counter
rows = [json.loads(l) for l in open("data/preference/dpo_train.jsonl", encoding="utf-8") if l.strip()]
c = Counter(r["chosen"] for r in rows)
print("chosen 不重複數:", len(c), "/", len(rows))
for s, k in c.most_common(5):
    print(k, "次:", s[:50])
markers = ["沒有", "無法", "不存在", "尚未", "並非", "不確定", "前提"]
refuse = sum(any(m in r["chosen"] for m in markers) for r in rows)
print("chosen 含拒答／不確定詞:", refuse, "/", len(rows))
print("---- 隨機看 5 筆 ----")
import random
random.seed(0)
for r in random.sample(rows, 5):
    print("P:", r["prompt"]); print("C:", r["chosen"]); print("R:", r["rejected"]); print()
