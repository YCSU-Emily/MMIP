import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median, pstdev

REFERENCE_JSON = Path("data/captions/test.json")

MODELS = {
    "epoch5": Path("results/experiments/baseline_epoch5_captions.json"),
    "epoch8": Path("results/experiments/continuation_epoch8_captions.json"),
}

OUTPUT_JSON = Path(
    "results/experiments/epoch5_vs_epoch8_metrics_v1.json"
)


def tokenize(text):
    # Lowercase words and punctuation consistently for both models.
    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)


def ngram_counts(tokens, n):
    return Counter(
        tuple(tokens[i:i+n])
        for i in range(len(tokens) - n + 1)
    )


def closest_reference_length(candidate_length, reference_tokens):
    lengths = [len(ref) for ref in reference_tokens]
    return min(lengths, key=lambda length: (abs(length - candidate_length), length))


def sentence_bleu(candidate, references, max_n):
    if not candidate:
        return 0.0

    precisions = []

    for n in range(1, max_n + 1):
        candidate_counts = ngram_counts(candidate, n)
        denominator = sum(candidate_counts.values())

        if denominator == 0:
            precisions.append(0.1)
            continue

        max_reference_counts = Counter()
        for ref in references:
            ref_counts = ngram_counts(ref, n)
            for gram, count in ref_counts.items():
                max_reference_counts[gram] = max(
                    max_reference_counts[gram], count
                )

        clipped = sum(
            min(count, max_reference_counts[gram])
            for gram, count in candidate_counts.items()
        )

        # NLTK-style method1 smoothing for zero clipped matches.
        precision = (clipped + 0.1) / denominator
        precisions.append(precision)

    if max_n == 1:
        weights = [1.0]
    else:
        weights = [1.0 / max_n] * max_n

    score = math.exp(sum(
        weight * math.log(max(p, 1e-16))
        for weight, p in zip(weights, precisions)
    ))

    ref_length = closest_reference_length(len(candidate), references)
    brevity_penalty = (
        1.0 if len(candidate) > ref_length
        else math.exp(1.0 - ref_length / len(candidate))
    )

    return brevity_penalty * score


def load_references():
    data = json.loads(REFERENCE_JSON.read_text(encoding="utf-8"))

    image_id_to_filename = {
        image["id"]: image["file_name"]
        for image in data["images"]
    }

    refs_by_filename = {}
    for ann in data["annotations"]:
        filename = image_id_to_filename.get(ann["image_id"])
        if filename is not None:
            refs_by_filename.setdefault(filename, []).append(
                ann["caption"].strip()
            )

    return refs_by_filename


def load_predictions(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        item["image"]: item["generated_caption"].strip()
        for item in data["results"]
    }


def summarize(values):
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "std_population": pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def evaluate(predictions, references):
    samples = []

    for filename in sorted(predictions):
        if filename not in references:
            continue

        candidate_text = predictions[filename]
        reference_texts = references[filename]

        candidate = tokenize(candidate_text)
        refs = [tokenize(text) for text in reference_texts]

        row = {
            "image": filename,
            "generated_caption": candidate_text,
            "num_references": len(reference_texts),
        }

        for n in range(1, 5):
            row[f"BLEU-{n}"] = sentence_bleu(candidate, refs, n)

        samples.append(row)

    if not samples:
        raise RuntimeError("沒有任何生成字幕能與參考字幕配對。")

    metrics = {
        key: summarize([row[key] for row in samples])
        for key in ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]
    }

    return metrics, samples


def main():
    references = load_references()
    predictions = {
        name: load_predictions(path)
        for name, path in MODELS.items()
    }

    common_images = set.intersection(
        *(set(items) for items in predictions.values())
    )

    if set(predictions["epoch5"]) != set(predictions["epoch8"]):
        raise RuntimeError("兩個模型的圖片集合不同，停止比較。")

    missing_refs = sorted(
        filename for filename in common_images
        if filename not in references
    )
    if missing_refs:
        raise RuntimeError(
            f"有 {len(missing_refs)} 張圖片缺少參考字幕，例如：{missing_refs[:3]}"
        )

    model_results = {}
    for name in ["epoch5", "epoch8"]:
        subset = {
            filename: predictions[name][filename]
            for filename in sorted(common_images)
        }
        metrics, samples = evaluate(subset, references)
        model_results[name] = {
            "num_images": len(samples),
            "metrics": metrics,
            "samples": samples,
        }

    deltas = {}
    for metric in ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]:
        old = model_results["epoch5"]["metrics"][metric]["mean"]
        new = model_results["epoch8"]["metrics"][metric]["mean"]
        deltas[metric] = {
            "epoch5_mean": old,
            "epoch8_mean": new,
            "absolute_change": new - old,
            "relative_change_percent": (
                (new - old) / old * 100 if old else None
            ),
        }

    report = {
        "evaluation": "Paired sentence-level BLEU comparison",
        "tokenization": "lowercase regex tokens; punctuation retained",
        "smoothing": "method1-style epsilon 0.1",
        "reference_captions": "all available COCO references per image",
        "num_common_images": len(common_images),
        "models": model_results,
        "deltas_epoch8_minus_epoch5": deltas,
        "warning": (
            "BLEU measures text overlap, not image-grounded correctness. "
            "Do not compare these scores directly with reports that use "
            "different tokenization or aggregation settings."
        ),
    }

    OUTPUT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 65)
    print("Epoch 5 vs Epoch 8：BLEU 比較")
    print("共同圖片數：", len(common_images))
    print()

    for name in ["epoch5", "epoch8"]:
        print(f"[{name}]")
        for metric in ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]:
            print(
                f"  {metric}: "
                f"{model_results[name]['metrics'][metric]['mean']:.4f}"
            )

    print("\n[Epoch 8 相對 Epoch 5]")
    for metric, values in deltas.items():
        change = values["absolute_change"]
        relative = values["relative_change_percent"]
        print(
            f"  {metric}: {change:+.4f} "
            f"({relative:+.2f}%)"
        )

    print("\n新報告已儲存：", OUTPUT_JSON)


if __name__ == "__main__":
    main()
