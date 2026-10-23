import csv
import json
import random
import sys
from pathlib import Path

BASE = Path("results/baseline_256/scoring_v4/target_scored.jsonl")
DPO = Path("results/dpo_256/scoring_v4/target_scored.jsonl")
OUT = Path("annotation/target_blind_annotation.csv")
KEY = Path("annotation/target_blind_key.json")

# 避免不小心覆寫既有標註或對照資料
if OUT.exists() or KEY.exists():
    sys.exit(
        "中止：盲標表或對照檔已存在。"
        "請先檢查檔案，不要直接覆寫。"
    )

def load_rows(path):
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    result = {r["id"]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f"{path} 有重複 ID")
    return result

baseline = load_rows(BASE)
dpo = load_rows(DPO)

if len(baseline) != 100 or len(dpo) != 100:
    raise ValueError(
        f"預期兩邊各 100 題，實際 Baseline={len(baseline)}, DPO={len(dpo)}"
    )

if set(baseline) != set(dpo):
    raise ValueError("Baseline 與 DPO 的題目 ID 不一致")

rng = random.Random(20261023)
output_rows = []
key = {}

for qid in sorted(baseline):
    b = baseline[qid]
    d = dpo[qid]

    if b["question"] != d["question"]:
        raise ValueError(f"{qid} 的問題文字不一致")
    if b["reference"] != d["reference"]:
        raise ValueError(f"{qid} 的參考答案不一致")

    # 每題獨立隨機決定 A/B 對應，方便盲標
    if rng.random() < 0.5:
        a, answer_a = "baseline_256", b["model_answer"]
        bb, answer_b = "dpo_256", d["model_answer"]
    else:
        a, answer_a = "dpo_256", d["model_answer"]
        bb, answer_b = "baseline_256", b["model_answer"]

    key[qid] = {"A": a, "B": bb}

    output_rows.append({
        "id": qid,
        "category": b["category"],
        "question": b["question"],
        "reference": b["reference"],
        "answer_A": answer_a,
        "A_factual_correct": "",
        "A_fabricated_or_false_detail": "",
        "A_uncertainty_appropriate": "",
        "A_note": "",
        "answer_B": answer_b,
        "B_factual_correct": "",
        "B_fabricated_or_false_detail": "",
        "B_uncertainty_appropriate": "",
        "B_note": "",
    })

OUT.parent.mkdir(parents=True, exist_ok=True)

with OUT.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
    writer.writeheader()
    writer.writerows(output_rows)

KEY.write_text(
    json.dumps(key, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("盲標表已建立：", OUT)
print("題目數：", len(output_rows))
print("每題有 A、B 兩份匿名回答，共", len(output_rows) * 2, "份回答")
print("對照檔已保存，請先不要開啟：", KEY)
print("原始預測與 v4 評分檔未修改。")
