import json
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parents[1]

files = {
    "DPO train": root / "data/preference/dpo_train.jsonl",
    "Target eval": root / "data/benchmark/target_eval.jsonl",
    "General eval": root / "data/benchmark/general_eval.jsonl",
}

def load(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

for name, path in files.items():
    rows = load(path)
    print(f"\n===== {name} =====")
    print("File:", path)
    print("Total rows:", len(rows))

    if name == "DPO train":
        required = {"prompt", "chosen", "rejected"}
        print("Required fields:", all(required <= row.keys() for row in rows))
        prompts = [row["prompt"].strip() for row in rows]
        print("Unique prompts:", len(set(prompts)))
        print("Duplicate prompt rows:", len(prompts) - len(set(prompts)))
        print("Chosen equals rejected:", sum(
            row["chosen"].strip() == row["rejected"].strip()
            for row in rows
        ))
        print("\nSample:")
        for row in rows[:2]:
            print(json.dumps(row, ensure_ascii=False, indent=2))
    else:
        field = "question"
        questions = [row[field].strip() for row in rows]
        print("Unique questions:", len(set(questions)))
        print("Duplicate question rows:", len(questions) - len(set(questions)))
        print("Category counts:", dict(Counter(
            row.get("category", "unknown") for row in rows
        )))
        print("\nSample:")
        for row in rows[:2]:
            print(json.dumps(row, ensure_ascii=False, indent=2))

print("\nAudit complete.")
