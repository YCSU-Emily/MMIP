import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path("week5/quiz3")
RESULTS = ROOT / "results"
OUT_CSV = RESULTS / "per_image_bleu_comparison.csv"
OUT_JSON = RESULTS / "per_image_bleu_comparison.json"

MODEL_FILES = {
    "week4_baseline": RESULTS / "week4_baseline_shared100.json",
    "week4_continuation": RESULTS / "week4_continuation_shared100.json",
    "week5_clip_lstm": RESULTS / "clip_lstm_test_captions.json",
}

def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)

def ngram_counts(tokens, n):
    return Counter(
        tuple(tokens[i:i+n])
        for i in range(len(tokens) - n + 1)
    )

def closest_reference_length(candidate_len, references):
    lengths = [len(tokenize(ref)) for ref in references]
    return min(lengths, key=lambda length: (abs(length - candidate_len), length))


def sentence_bleu(candidate, references, max_n=4):
    cand_tokens = tokenize(candidate)
    ref_tokens = [tokenize(ref) for ref in references]

    if not cand_tokens:
        return [0.0] * max_n

    ref_len = closest_reference_length(len(cand_tokens), references)

    if len(cand_tokens) > ref_len:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(
            1.0 - ref_len / len(cand_tokens)
        )

    precisions = []

    for n in range(1, max_n + 1):
        cand_counts = ngram_counts(cand_tokens, n)
        denominator = sum(cand_counts.values())

        if denominator == 0:
            precisions.append(0.0)
            continue

        max_ref_counts = Counter()
        for tokens in ref_tokens:
            counts = ngram_counts(tokens, n)
            for gram, count in counts.items():
                max_ref_counts[gram] = max(
                    max_ref_counts[gram], count
                )

        clipped = sum(
            min(count, max_ref_counts[gram])
            for gram, count in cand_counts.items()
        )

        # Match evaluate_comparison.py exactly.
        precision = (clipped + 0.1) / denominator
        precisions.append(precision)

    scores = []
    for n in range(1, max_n + 1):
        selected = precisions[:n]
        if any(p <= 0 for p in selected):
            scores.append(0.0)
        else:
            scores.append(
                brevity_penalty
                * math.exp(
                    sum(math.log(p) for p in selected) / n
                )
            )

    return scores


def load_predictions(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["results"] if isinstance(data, dict) else data

    output = {}
    for row in rows:
        filename = row.get("file_name") or row.get("image")
        if not filename:
            raise ValueError(f"找不到圖片檔名欄位：{path}")
        filename = Path(filename).name

        caption = row.get("generated_caption")
        if caption is None:
            raise ValueError(
                f"找不到 generated_caption：{path}, {filename}"
            )
        if filename in output:
            raise ValueError(f"重複圖片檔名：{path}, {filename}")
        output[filename] = caption

    return output

def main():
    predictions = {
        name: load_predictions(path)
        for name, path in MODEL_FILES.items()
    }

    common = set.intersection(
        *(set(mapping) for mapping in predictions.values())
    )
    if not common:
        raise RuntimeError("三個模型沒有共同圖片，請檢查檔案。")

    coco_path = Path(
        "week4/quiz4/image_captioning/data/captions/test.json"
    )
    coco = json.loads(coco_path.read_text(encoding="utf-8"))

    id_to_filename = {
        image["id"]: Path(image["file_name"]).name
        for image in coco["images"]
    }
    references = {}
    for ann in coco["annotations"]:
        filename = id_to_filename.get(ann["image_id"])
        if filename:
            references.setdefault(filename, []).append(ann["caption"])

    rows = []
    for filename in sorted(common):
        refs = references.get(filename, [])
        if not refs:
            print(f"警告：沒有找到參考描述：{filename}")
            continue

        row = {
            "filename": filename,
            "reference_count": len(refs),
        }

        for model_name, mapping in predictions.items():
            scores = sentence_bleu(mapping[filename], refs)
            for n, score in enumerate(scores, start=1):
                row[f"{model_name}_BLEU{n}"] = score

        for n in range(1, 5):
            row[f"delta_week5_vs_continuation_BLEU{n}"] = (
                row[f"week5_clip_lstm_BLEU{n}"]
                - row[f"week4_continuation_BLEU{n}"]
            )
            row[f"delta_week5_vs_baseline_BLEU{n}"] = (
                row[f"week5_clip_lstm_BLEU{n}"]
                - row[f"week4_baseline_BLEU{n}"]
            )

        rows.append(row)

    if not rows:
        raise RuntimeError("沒有可供分析的圖片資料。")

    fieldnames = list(rows[0].keys())
    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    OUT_JSON.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"\n共同圖片數：{len(common)}")
    print(f"成功評估圖片數：{len(rows)}")
    print(f"CSV：{OUT_CSV}")
    print(f"JSON：{OUT_JSON}")

    for metric in ("BLEU1", "BLEU2", "BLEU3", "BLEU4"):
        delta_key = f"delta_week5_vs_continuation_{metric}"
        ordered = sorted(rows, key=lambda row: row[delta_key])

        print(f"\n===== Week 5 vs Week 4 Continuation：{metric} =====")
        print("退步最多：")
        for row in ordered[:5]:
            print(
                f"{row['filename']}  "
                f"Week4={row[f'week4_continuation_{metric}']:.4f}  "
                f"Week5={row[f'week5_clip_lstm_{metric}']:.4f}  "
                f"差值={row[delta_key]:+.4f}"
            )

        print("進步最多：")
        for row in reversed(ordered[-5:]):
            print(
                f"{row['filename']}  "
                f"Week4={row[f'week4_continuation_{metric}']:.4f}  "
                f"Week5={row[f'week5_clip_lstm_{metric}']:.4f}  "
                f"差值={row[delta_key]:+.4f}"
            )

if __name__ == "__main__":
    main()
