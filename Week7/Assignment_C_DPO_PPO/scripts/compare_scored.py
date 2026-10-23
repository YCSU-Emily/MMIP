import json, sys
a_tag, b_tag = sys.argv[1], sys.argv[2]
def load(tag):
    p = f"results/{tag}/scoring_v4/general_scored.jsonl"
    return {r["id"]: r for r in map(json.loads, open(p, encoding="utf-8"))}
a, b = load(a_tag), load(b_tag)
keys = ["content_correct", "format_compliant", "fully_correct", "scoring_status", "model_answer"]
n = 0
for i in sorted(a):
    d = [k for k in keys if a[i].get(k) != b[i].get(k)]
    if d:
        n += 1
        print(i, a[i]["category"], "變動欄位:", d)
print("逐題有差異的題數:", n)
