import csv
import json
import math
from pathlib import Path

ROOT = Path.cwd()
ANN = ROOT / "annotation"

files = [
    ANN / "target_pilot_10.csv",
    ANN / "target_remaining_90.csv",
]
key_path = ANN / "target_blind_key.json"

# 讀取並合併 10 題 pilot + 90 題正式標註
rows = []
for path in files:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows.extend(csv.DictReader(f))

with key_path.open("r", encoding="utf-8") as f:
    key = json.load(f)

# 如果 key 外層有包一層字典，請先確認實際結構
if "assignments" in key and isinstance(key["assignments"], dict):
    key = key["assignments"]

assert len(rows) == 100, f"預期 100 題，實際 {len(rows)} 題"
assert len({r["id"] for r in rows}) == 100, "發現重複 ID"

metrics = {
    "factual_correct": (
        "A_factual_correct", "B_factual_correct", True
    ),
    "fabricated_or_false_detail": (
        "A_fabricated_or_false_detail",
        "B_fabricated_or_false_detail",
        False
    ),
    "uncertainty_appropriate": (
        "A_uncertainty_appropriate",
        "B_uncertainty_appropriate",
        True
    ),
}

def exact_mcnemar_p(b, c):
    """雙尾 exact McNemar test，以 Binomial(n, 0.5) 計算。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)

def model_for(id_, side):
    item = key[id_]
    # 支援 {"A":"baseline_256","B":"dpo_256"}
    # 以及 {"A":{"model":"baseline_256"}, ...} 這類格式
    value = item[side]
    if isinstance(value, dict):
        value = value.get("model", value.get("model_name"))
    return value

def to_int(value):
    value = int(value)
    if value not in (0, 1):
        raise ValueError(f"標籤不是 0/1：{value}")
    return value

print("=" * 72)
print("Exact McNemar Test：Baseline vs DPO")
print("=" * 72)
print(f"樣本數：{len(rows)} 題")
print("顯著水準：alpha = 0.05")
print("以下均採雙尾 exact McNemar 檢定。\n")

for metric, (acol, bcol, higher_is_better) in metrics.items():
    # b：Baseline 較好；c：DPO 較好
    baseline_wins = 0
    dpo_wins = 0
    ties = 0

    for row in rows:
        id_ = row["id"]
        a = to_int(row[acol])
        b = to_int(row[bcol])

        model_a = model_for(id_, "A")
        model_b = model_for(id_, "B")

        scores = {
            model_a: a,
            model_b: b,
        }
        baseline = scores["baseline_256"]
        dpo = scores["dpo_256"]

        # 捏造細節越少越好，所以該項反轉 0/1 分數
        if not higher_is_better:
            baseline = 1 - baseline
            dpo = 1 - dpo

        if baseline > dpo:
            baseline_wins += 1
        elif dpo > baseline:
            dpo_wins += 1
        else:
            ties += 1

    p = exact_mcnemar_p(baseline_wins, dpo_wins)

    print(f"[{metric}]")
    print(f"Baseline 較佳：{baseline_wins}")
    print(f"DPO 較佳：{dpo_wins}")
    print(f"平手：{ties}")
    print(f"雙尾 exact p-value：{p:.6f}")

    if baseline_wins + dpo_wins == 0:
        print("判讀：沒有不一致配對，無法觀察到模型間的配對差異。")
    elif p < 0.05:
        winner = (
            "DPO" if dpo_wins > baseline_wins
            else "Baseline"
        )
        print(f"判讀：在 0.05 水準下有顯著差異，方向偏向 {winner}。")
    else:
        print("判讀：在 0.05 水準下，未達統計顯著。")
    print()

print("注意：未校正三個指標的多重檢定；結果應視為探索性分析。")
