import csv, json
from collections import defaultdict

SYN = {"general_033", "general_035", "general_038", "general_040",
       "general_048", "general_075", "general_085"}
BASE_OVERRIDE = {"general_063": 0}

def run(tag, mode):
    rows = [json.loads(l) for l in open(f"results/{tag}/scoring_v4/general_scored.jsonl", encoding="utf-8")]
    man = {r["id"]: int(r["manual_content_correct"]) for r in
           csv.DictReader(open(f"annotation/general_manual_{tag}.csv", encoding="utf-8-sig"))}
    out = defaultdict(lambda: {"n": 0, "content": 0, "fmt": 0, "full": 0})
    for r in rows:
        i, cat = r["id"], r["category"]
        if tag == "baseline_256" and i in BASE_OVERRIDE:
            v = BASE_OVERRIDE[i]
        elif i in man:
            v = man[i]
        elif cat == "instruction_following":
            cc = r.get("content_correct")
            v = None if cc is None else int(cc is True)
        elif r.get("answer_correct") is True:
            v = 1
        elif r["scoring_status"] == "automatic_incorrect":
            v = 0
        else:
            v = None
        assert v is not None, f"{tag} {i} 沒有最終判定"
        if mode == "strict" and tag == "baseline_256" and i == "general_038":
            v = 0
        if mode == "lenient" and i in SYN:
            v = 1
        o = out[cat]
        o["n"] += 1
        o["content"] += v
        if cat == "instruction_following":
            f = r["format_compliant"] is True
            o["fmt"] += int(f)
            o["full"] += int(f and v == 1)
    return out

result = {}
for mode in ["strict", "lenient"]:
    print("=====", mode, "=====")
    for tag in ["baseline_256", "dpo_256"]:
        o = run(tag, mode)
        result[f"{mode}_{tag}"] = o
        tot = sum(x["content"] for x in o.values())
        print(tag, "| 內容正確合計:", tot, "/ 100")
        for c, x in o.items():
            extra = f" | 格式 {x['fmt']}/{x['n']} | 完整遵循 {x['full']}/{x['n']}" if c == "instruction_following" else ""
            print("   ", c, "內容", x["content"], "/", x["n"], extra)
json.dump(result, open("results/final_general_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("DONE")
