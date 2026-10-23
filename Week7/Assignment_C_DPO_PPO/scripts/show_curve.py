import re, ast
rows = []
for l in open("logs/train_dpo_v1.log", encoding="utf-8"):
    if l.startswith("{") and "'loss'" in l:
        rows.append(ast.literal_eval(l.strip()))
print("記錄步數:", len(rows))
for r in rows[::6] + [rows[-1]]:
    print("epoch", r["epoch"], "| loss", r["loss"], "| margins", r["rewards/margins"], "| acc", r["rewards/accuracies"])
