import csv
import json
import hashlib
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
CANDIDATES = BASE / "data/vlm_preferences_candidates.jsonl"
AUDIT = BASE / "data/vlm_preferences_manual_audit.csv"
OUTPUT = BASE / "data/vlm_preferences_train.jsonl"

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

# 保護來源檔案
for path in (CANDIDATES, AUDIT):
    if not path.is_file():
        raise SystemExit(f"錯誤：找不到來源檔案：{path}")

# 不覆寫任何既有輸出
if OUTPUT.exists():
    raise SystemExit(
        f"停止：輸出檔已存在：{OUTPUT}\n"
        "為避免覆寫，請先檢查該檔案，不要直接刪除。"
    )

candidate_hash_before = sha256(CANDIDATES)
audit_hash_before = sha256(AUDIT)

# 讀取原始候選資料
candidates = []
with CANDIDATES.open("r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, start=1):
        if not line.strip():
            continue
        try:
            candidates.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SystemExit(f"JSONL 第 {line_no} 行格式錯誤：{e}")

# 讀取人工審核 CSV
with AUDIT.open("r", encoding="utf-8-sig", newline="") as f:
    audit_rows = list(csv.DictReader(f))

candidate_ids = [str(x.get("image_id", "")).strip() for x in candidates]
audit_ids = [str(x.get("image_id", "")).strip() for x in audit_rows]

if len(candidates) != 120:
    raise SystemExit(f"預期 120 筆候選資料，實際 {len(candidates)} 筆。")

if len(audit_rows) != 120:
    raise SystemExit(f"預期 120 筆人工審核資料，實際 {len(audit_rows)} 筆。")

if len(set(candidate_ids)) != len(candidate_ids):
    raise SystemExit("候選 JSONL 有重複 image_id，停止建置。")

if len(set(audit_ids)) != len(audit_ids):
    raise SystemExit("人工審核 CSV 有重複 image_id，停止建置。")

if set(candidate_ids) != set(audit_ids):
    missing_in_audit = sorted(set(candidate_ids) - set(audit_ids))
    missing_in_candidates = sorted(set(audit_ids) - set(candidate_ids))
    raise SystemExit(
        "候選資料與人工審核資料的 ID 不一致。\n"
        f"候選有、審核沒有：{missing_in_audit}\n"
        f"審核有、候選沒有：{missing_in_candidates}"
    )

audit_by_id = {
    str(row["image_id"]).strip(): row
    for row in audit_rows
}

training_rows = []
excluded = {"positive": [], "uncertain": [], "other": []}

for item in candidates:
    image_id = str(item["image_id"]).strip()
    audit = audit_by_id[image_id]

    status = audit.get("review_status", "").strip().lower()
    false_claim = audit.get("is_false_claim", "").strip().lower()
    added_correct = audit.get("human_added_claim_correct", "").strip().lower()

    # 只納入人工確認為錯誤新增物件的配對
    if status == "reviewed" and false_claim == "yes" and added_correct == "no":
        for field in ("prompt", "chosen", "rejected", "image"):
            if not str(item.get(field, "")).strip():
                raise SystemExit(
                    f"ID {image_id} 缺少必要欄位 {field}，停止建置。"
                )

        image_path = Path(item["image"])
        if not image_path.is_file():
            raise SystemExit(
                f"ID {image_id} 的圖片不存在：{image_path}\n"
                "請先確認圖片路徑，沒有完成確認前不會輸出資料集。"
            )

        if item["chosen"].strip() == item["rejected"].strip():
            raise SystemExit(f"ID {image_id} 的 chosen 與 rejected 相同。")

        training_rows.append({
            "image_id": item["image_id"],
            "image": item["image"],
            "prompt": item["prompt"],
            "chosen": item["chosen"],
            "rejected": item["rejected"],
        })

    elif status == "uncertain":
        excluded["uncertain"].append(image_id)

    elif status == "reviewed" and false_claim == "no" and added_correct == "yes":
        excluded["positive"].append(image_id)

    else:
        excluded["other"].append({
            "image_id": image_id,
            "review_status": status,
            "is_false_claim": false_claim,
            "human_added_claim_correct": added_correct,
        })

# 預期標註分布不符時停止
if len(training_rows) != 115:
    raise SystemExit(
        f"預期 115 筆訓練配對，實際符合條件 {len(training_rows)} 筆。\n"
        "停止輸出，請先檢查標註。"
    )

if len(excluded["positive"]) != 1:
    raise SystemExit(f"預期排除 1 筆正例，實際 {len(excluded['positive'])} 筆。")

if len(excluded["uncertain"]) != 4:
    raise SystemExit(f"預期排除 4 筆不確定樣本，實際 {len(excluded['uncertain'])} 筆。")

if excluded["other"]:
    raise SystemExit(f"有資料不符合已知分類，停止建置：{excluded['other']}")

# 再次確認來源檔案沒有被其他程序修改
if sha256(CANDIDATES) != candidate_hash_before:
    raise SystemExit("候選 JSONL 在執行期間發生變化，停止輸出。")

if sha256(AUDIT) != audit_hash_before:
    raise SystemExit("人工審核 CSV 在執行期間發生變化，停止輸出。")

# 先寫入同目錄的暫存檔，再以不覆寫模式建立新檔
temp = OUTPUT.with_suffix(".jsonl.tmp")
try:
    with temp.open("x", encoding="utf-8") as f:
        for row in training_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # x 模式確保不覆寫已存在的目標檔
    with OUTPUT.open("x", encoding="utf-8") as f:
        with temp.open("r", encoding="utf-8") as src:
            f.write(src.read())
finally:
    if temp.exists():
        temp.unlink()

print("========== Preference dataset 建置成功 ==========")
print(f"輸出檔案：{OUTPUT}")
print(f"訓練配對：{len(training_rows)}")
print(f"排除正例：{len(excluded['positive'])} 筆，ID={excluded['positive']}")
print(f"排除不確定：{len(excluded['uncertain'])} 筆，ID={excluded['uncertain']}")
print(f"其他未分類：{len(excluded['other'])} 筆")
print(f"候選 JSONL SHA-256 未變：{sha256(CANDIDATES) == candidate_hash_before}")
print(f"人工審核 CSV SHA-256 未變：{sha256(AUDIT) == audit_hash_before}")
