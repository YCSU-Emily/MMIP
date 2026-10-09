import json
import re
import math
import csv
from collections import Counter
from pathlib import Path

ROOT = Path.home() / "MMIP"
W4 = ROOT / "week4/quiz4/image_captioning"
W5 = ROOT / "week5/quiz3"
RESULTS = W5 / "results"

COCO_JSON = W4 / "data/captions/test.json"

MODEL_FILES = {
    "Week4_baseline": RESULTS / "week4_baseline_shared100.json",
    "Week4_continuation": RESULTS / "week4_continuation_shared100.json",
    "Week5_CLIP_LSTM": RESULTS / "clip_lstm_test_captions.json",
}

OUTPUT_JSON = RESULTS / "comparison_week4_week5_metrics.json"
OUTPUT_CSV = RESULTS / "comparison_week4_week5_metrics.csv"


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

    ref_len = closest_reference_length(
        len(cand_tokens), references
    )

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

        # Method-1-like smoothing, consistent for every model.
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
                * math.exp(sum(math.log(p) for p in selected) / n)
            )

    return scores


def load_predictions(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["results"] if isinstance(data, dict) else data

    predictions = {}
    for row in rows:
        filename = row.get("file_name") or row.get("image")
        caption = row.get("generated_caption")

        if not filename or caption is None:
            raise ValueError(
                f"無法解析預測紀錄：{path}，紀錄：{row}"
            )

        filename = Path(filename).name
        if filename in predictions:
            raise ValueError(f"{path} 中有重複圖片：{filename}")

        predictions[filename] = caption.strip()

    return predictions


def load_references(path):
    data = json.loads(path.read_text(encoding="utf-8"))

    id_to_name = {
        image["id"]: Path(image["file_name"]).name
        for image in data["images"]
    }

    references = {}
    for ann in data["annotations"]:
        filename = id_to_name.get(ann["image_id"])
        if filename:
            references.setdefault(filename, []).append(
                ann["caption"].strip()
            )

    return references


def main():
    references = load_references(COCO_JSON)
    predictions = {
        name: load_predictions(path)
        for name, path in MODEL_FILES.items()
    }

    # Require exactly the same images in all model prediction sets.
    image_sets = [set(p) for p in predictions.values()]
    if not all(s == image_sets[0] for s in image_sets):
        raise ValueError("三個模型的圖片集合不一致，停止評估。")

    filenames = sorted(image_sets[0])
    if len(filenames) != 100:
        raise ValueError(
            f"預期 100 張共同圖片，實際為 {len(filenames)} 張。"
        )

    missing_refs = [f for f in filenames if f not in references]
    if missing_refs:
        raise ValueError(
            f"有圖片缺少 COCO 參考 caption：{missing_refs[:10]}"
        )

    print(f"共同圖片數：{len(filenames)}")
    print(
        "每張圖片的參考 caption 數量：",
        sorted(set(len(references[f]) for f in filenames))
    )

    report = {
        "dataset": "COCO test captions",
        "num_images": len(filenames),
        "bleu_method": (
            "Sentence-level BLEU averaged across images; "
            "lowercase regex tokenization, clipped n-gram counts, "
            "0.1 smoothing, brevity penalty; all available references"
        ),
        "bert_score_method": (
            "If installed: roberta-base, English; score each generated "
            "caption against every available reference, then average "
            "across references and images."
        ),
        "models": {},
    }

    bertscore_available = True
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        bertscore_available = False
        bert_score_fn = None
        print(
            "\n注意：未安裝 bert-score，這次先計算 BLEU。"
        )
        print(
            "若要計算 BERTScore，請執行："
            " python -m pip install bert-score"
        )

    for model_name, pred_map in predictions.items():
        print(f"\n評估：{model_name}")

        per_image_bleu = []
        for filename in filenames:
            per_image_bleu.append(
                sentence_bleu(
                    pred_map[filename],
                    references[filename],
                )
            )

        mean_bleu = [
            sum(row[i] for row in per_image_bleu) / len(per_image_bleu)
            for i in range(4)
        ]

        model_result = {
            "num_images": len(filenames),
            "BLEU1": mean_bleu[0],
            "BLEU2": mean_bleu[1],
            "BLEU3": mean_bleu[2],
            "BLEU4": mean_bleu[3],
        }

        if bertscore_available:
            candidates = []
            ref_texts = []

            for filename in filenames:
                caption = pred_map[filename]
                for ref in references[filename]:
                    candidates.append(caption)
                    ref_texts.append(ref)

            precision, recall, f1 = bert_score_fn(
                candidates,
                ref_texts,
                model_type="roberta-base",
                lang="en",
                verbose=True,
            )

            # Each image contributes equally, regardless of reference count.
            offset = 0
            image_p, image_r, image_f = [], [], []

            for filename in filenames:
                count = len(references[filename])
                image_p.append(
                    precision[offset:offset+count].mean().item()
                )
                image_r.append(
                    recall[offset:offset+count].mean().item()
                )
                image_f.append(
                    f1[offset:offset+count].mean().item()
                )
                offset += count

            model_result.update({
                "BERTScore_P": sum(image_p) / len(image_p),
                "BERTScore_R": sum(image_r) / len(image_r),
                "BERTScore_F1": sum(image_f) / len(image_f),
            })

        report["models"][model_name] = model_result

        for metric, value in model_result.items():
            if metric != "num_images":
                print(f"  {metric}: {value:.6f}")

    OUTPUT_JSON.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    metric_names = [
        "BLEU1", "BLEU2", "BLEU3", "BLEU4",
        "BERTScore_P", "BERTScore_R", "BERTScore_F1",
    ]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "NumImages"] + metric_names)
        for name, result in report["models"].items():
            writer.writerow([
                name,
                result["num_images"],
                *[result.get(metric, "") for metric in metric_names],
            ])

    print("\n===== 比較摘要 =====")
    print(f"JSON：{OUTPUT_JSON}")
    print(f"CSV： {OUTPUT_CSV}")
    print("評估完成。")


if __name__ == "__main__":
    main()
