import json
from collections import defaultdict
def load(tag):
    p = f"results/{tag}/scoring_v4/general_scored.jsonl"
    return {r["id"]: r for r in map(json.loads, open(p, encoding="utf-8"))}
a, b = load("baseline_256"), load("dpo_256")
L = defaultdict(lambda: [[], []])
for i in a:
    L[a[i]["category"]][0].append(len(a[i]["model_answer"]))
    L[a[i]["category"]][1].append(len(b[i]["model_answer"]))
print("平均回答字數 (baseline -> dpo)")
for c, (x, y) in L.items():
    print(" ", c, round(sum(x) / len(x)), "->", round(sum(y) / len(y)))
print()
for cat in ["english_vocabulary", "basic_science"]:
    print("====", cat, "狀態改變的題目 ====")
    for i in sorted(a):
        if a[i]["category"] == cat and a[i]["scoring_status"] != b[i]["scoring_status"]:
            print(i, a[i]["scoring_status"], "->", b[i]["scoring_status"], "| ref:", a[i]["reference"])
            print("   base:", a[i]["model_answer"][:60].replace("\n", " "))
            print("   dpo :", b[i]["model_answer"][:60].replace("\n", " "))
