import json
a = json.load(open("results/baseline_256/evaluation_summary.json", encoding="utf-8"))
b = json.load(open("results/dpo_256/evaluation_summary.json", encoding="utf-8"))
for k in ["model", "dtype", "generation", "target_questions", "general_questions"]:
    print(k, "| 相同" if a[k] == b[k] else "| 不同!", a[k], b[k])
print("baseline adapter:", a.get("adapter"), "| dpo adapter:", b.get("adapter"))
