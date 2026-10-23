import json, sys
tag = sys.argv[1]
ids = set(sys.argv[2:])
for l in open(f"results/{tag}/general_predictions.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r["id"] in ids:
        print(r["id"], "|", r["question"])
        print(r["model_answer"])
        print("-" * 40)
