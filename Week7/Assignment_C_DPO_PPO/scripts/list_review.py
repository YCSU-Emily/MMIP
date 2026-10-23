import json, sys
tag = sys.argv[1]
rows = [json.loads(l) for l in open(f"results/{tag}/scoring_v4/general_scored.jsonl", encoding="utf-8")]
todo = [r for r in rows if r["scoring_status"] == "manual_review"
        or (r["category"] == "instruction_following" and r.get("content_correct") is None)]
print("==== 待人工複查:", len(todo), "題 ====")
for r in todo:
    print(r["id"], r["category"], "| ref:", r["reference"], "| format:", r.get("format_compliant"))
    print("   ", r["model_answer"].replace("\n", " "))
print()
print("==== 科學題自動判對的完整回答 ====")
for r in rows:
    if r["category"] == "basic_science" and r["scoring_status"] == "automatic_conservative_match":
        print(r["id"], "| ref:", r["reference"])
        print("   ", r["model_answer"].replace("\n", " "))
