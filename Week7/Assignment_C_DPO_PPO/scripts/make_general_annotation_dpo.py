import csv, json, sys
from pathlib import Path

dst = Path("annotation/general_manual_dpo_256.csv")
if dst.exists():
    sys.exit(f"{dst} 已存在，中止")

L = {
 "general_031": (0, "近義"),
 "general_033": (0, "近義"),
 "general_035": (0, "近義"),
 "general_038": (0, "近義"),
 "general_040": (0, "近義"),
 "general_048": (0, "近義"),
 "general_055": (1, "簡體字氧气，非內容錯誤"),
 "general_060": (1, "32在參考範圍；不含智齒沿用題目措辭，邊界案例"),
 "general_061": (0, "未說遺傳；轉錄寫成轉譯"),
 "general_064": (0, "蒸發沸騰汽化混淆"),
 "general_075": (0, "近義"),
 "general_085": (1, "含真；格式另計"),
 "general_091": (0, "星號數量錯誤"),
 "general_097": (1, "含12；格式另計"),
 "general_067": (0, "覆寫自動判對：科學家名字疑似捏造"),
 "general_072": (0, "覆寫自動判對：列出濾光片，答案不明確，邊界案例"),
}
OVERRIDE = {"general_067", "general_072"}

rows = [json.loads(l) for l in open("results/dpo_256/scoring_v4/general_scored.jsonl", encoding="utf-8") if l.strip()]
todo = [r for r in rows if r["scoring_status"] == "manual_review"
        or (r["category"] == "instruction_following" and r.get("content_correct") is None)]
ids = {r["id"] for r in todo} | OVERRIDE
assert ids == set(L), f"題號不一致: 多 {ids - set(L)} 少 {set(L) - ids}"

pick = [r for r in rows if r["id"] in set(L)]
with dst.open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["id", "category", "question", "reference", "model_answer", "manual_content_correct", "note"])
    for r in sorted(pick, key=lambda r: r["id"]):
        v, note = L[r["id"]]
        w.writerow([r["id"], r["category"], r["question"], r["reference"], r["model_answer"], v, note])
ones = sum(v for v, _ in L.values())
print("已寫入", dst, "| 題數:", len(pick), "| 1:", ones, "| 0:", len(L) - ones)
