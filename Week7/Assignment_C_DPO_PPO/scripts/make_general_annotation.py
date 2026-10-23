import csv, json, sys
from pathlib import Path

tag = sys.argv[1]
dst = Path(f"annotation/general_manual_{tag}.csv")
if dst.exists():
    sys.exit(f"{dst} 已存在，中止")

L = {
 "general_085": (0, "近義"),
 "general_031": (0, ""),
 "general_033": (0, "近義"),
 "general_053": (1, "簡體字氧气，非內容錯誤"),
 "general_057": (1, "用詞不精確"),
 "general_059": (0, "聲速數值錯誤"),
 "general_060": (0, "牙齒種類與數量錯誤"),
 "general_063": (1, "出處疑似捏造"),
 "general_064": (0, "沸騰描述錯誤"),
 "general_067": (1, "歸因不精確"),
 "general_068": (0, "自相矛盾"),
 "general_069": (1, ""),
 "general_071": (1, ""),
 "general_072": (1, "三棱鏡即稜鏡"),
 "general_073": (1, ""),
 "general_074": (1, ""),
 "general_075": (0, "近義"),
 "general_087": (1, ""),
 "general_091": (0, "星號數量錯誤"),
 "general_092": (1, ""),
 "general_097": (1, ""),
}

src = Path(f"results/{tag}/scoring_v4/general_scored.jsonl")
rows = [json.loads(l) for l in src.open(encoding="utf-8") if l.strip()]
todo = [r for r in rows if r["scoring_status"] == "manual_review"
        or (r["category"] == "instruction_following" and r.get("content_correct") is None)]
assert {r["id"] for r in todo} == set(L), "待複查題號與標註表不一致"

with dst.open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["id", "category", "question", "reference", "model_answer",
                "manual_content_correct", "note"])
    for r in sorted(todo, key=lambda r: r["id"]):
        v, note = L[r["id"]]
        w.writerow([r["id"], r["category"], r["question"], r["reference"],
                    r["model_answer"], v, note])
ones = sum(v for v, _ in L.values())
print("已寫入", dst, "| 題數:", len(todo), "| 1:", ones, "| 0:", len(L) - ones)
