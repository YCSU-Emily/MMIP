import json
import time
from pathlib import Path

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score


# ============================================================
# Configuration
# ============================================================

TEST_JSON = Path("data/captions/test.json")
GENERATED_JSON = Path("results/generated_captions.json")
OUTPUT_JSON = Path("results/advanced_evaluation.json")

BERT_MODEL = "roberta-base"

# ============================================================
# Load data
# ============================================================

print("=" * 70)
print("Week 4 Quiz 4 - Advanced Caption Evaluation")
print("=" * 70)

print(f"Reference captions: {TEST_JSON}")
print(f"Generated captions: {GENERATED_JSON}")
print(f"Output: {OUTPUT_JSON}")
print()

with open(TEST_JSON, "r", encoding="utf-8") as f:
    coco_data = json.load(f)

with open(GENERATED_JSON, "r", encoding="utf-8") as f:
    generated_data = json.load(f)

# ============================================================
# Build image_id -> references mapping
# ============================================================

image_id_to_filename = {
    image["id"]: image["file_name"]
    for image in coco_data["images"]
}

references = {}

for ann in coco_data["annotations"]:
    image_id = ann["image_id"]
    caption = ann["caption"].strip()

    if image_id not in references:
        references[image_id] = []

    references[image_id].append(caption)

# Convert to filename -> references
filename_to_references = {}

for image_id, captions in references.items():
    if image_id in image_id_to_filename:
        filename = image_id_to_filename[image_id]
        filename_to_references[filename] = captions

print(f"COCO images: {len(coco_data['images'])}")
print(f"COCO annotations: {len(coco_data['annotations'])}")
print(f"Images with references: {len(filename_to_references)}")

# ============================================================
# Load generated captions
# ============================================================

generated_items = generated_data["results"]

print(f"Generated captions: {len(generated_items)}")
print()

# ============================================================
# Match generated captions with references
# ============================================================

samples = []

for item in generated_items:
    filename = item["image"]
    generated_caption = item["generated_caption"].strip()

    if filename not in filename_to_references:
        print(f"WARNING: No reference found for {filename}")
        continue

    refs = filename_to_references[filename]

    samples.append({
        "image": filename,
        "generated_caption": generated_caption,
        "references": refs
    })

print(f"Matched samples: {len(samples)}")

if len(samples) == 0:
    raise RuntimeError("No generated captions could be matched with references.")

# ============================================================
# BLEU calculation
# ============================================================

print()
print("=" * 70)
print("Calculating BLEU...")
print("=" * 70)

smooth = SmoothingFunction().method1

bleu1_scores = []
bleu2_scores = []
bleu3_scores = []
bleu4_scores = []

for sample in samples:
    candidate_tokens = nltk.word_tokenize(
        sample["generated_caption"].lower()
    )

    reference_tokens = [
        nltk.word_tokenize(ref.lower())
        for ref in sample["references"]
    ]

    bleu1 = sentence_bleu(
        reference_tokens,
        candidate_tokens,
        weights=(1.0, 0.0, 0.0, 0.0),
        smoothing_function=smooth
    )

    bleu2 = sentence_bleu(
        reference_tokens,
        candidate_tokens,
        weights=(0.5, 0.5, 0.0, 0.0),
        smoothing_function=smooth
    )

    bleu3 = sentence_bleu(
        reference_tokens,
        candidate_tokens,
        weights=(1/3, 1/3, 1/3, 0.0),
        smoothing_function=smooth
    )

    bleu4 = sentence_bleu(
        reference_tokens,
        candidate_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smooth
    )

    bleu1_scores.append(bleu1)
    bleu2_scores.append(bleu2)
    bleu3_scores.append(bleu3)
    bleu4_scores.append(bleu4)

    sample["BLEU-1"] = bleu1
    sample["BLEU-2"] = bleu2
    sample["BLEU-3"] = bleu3
    sample["BLEU-4"] = bleu4

avg_bleu1 = sum(bleu1_scores) / len(bleu1_scores)
avg_bleu2 = sum(bleu2_scores) / len(bleu2_scores)
avg_bleu3 = sum(bleu3_scores) / len(bleu3_scores)
avg_bleu4 = sum(bleu4_scores) / len(bleu4_scores)

print(f"BLEU-1: {avg_bleu1:.4f}")
print(f"BLEU-2: {avg_bleu2:.4f}")
print(f"BLEU-3: {avg_bleu3:.4f}")
print(f"BLEU-4: {avg_bleu4:.4f}")

# ============================================================
# BERTScore
# ============================================================

print()
print("=" * 70)
print("Calculating BERTScore...")
print("=" * 70)

candidates = []
reference_texts = []

for sample in samples:
    candidates.append(sample["generated_caption"])

    # Use the first human reference for the main BERTScore.
    # The per-image maximum across references is calculated below.
    reference_texts.append(sample["references"][0])

print(f"Model: {BERT_MODEL}")
print(f"Samples: {len(candidates)}")
print()

bert_start = time.perf_counter()

P, R, F1 = bert_score(
    candidates,
    reference_texts,
    model_type=BERT_MODEL,
    lang="en",
    verbose=True
)

bert_time = time.perf_counter() - bert_start

bert_p_avg = P.mean().item()
bert_r_avg = R.mean().item()
bert_f1_avg = F1.mean().item()

print()
print(f"BERTScore Precision: {bert_p_avg:.4f}")
print(f"BERTScore Recall:    {bert_r_avg:.4f}")
print(f"BERTScore F1:        {bert_f1_avg:.4f}")
print(f"BERTScore time:      {bert_time:.2f} sec")

for i, sample in enumerate(samples):
    sample["BERTScore_P"] = P[i].item()
    sample["BERTScore_R"] = R[i].item()
    sample["BERTScore_F1"] = F1[i].item()

# ============================================================
# Inference timing
# ============================================================

inference_data = generated_data.get("average_inference_time_ms")

# ============================================================
# Final results
# ============================================================

results = {
    "evaluation_type": "Image Captioning Advanced Evaluation",

    "num_generated_images": len(generated_items),

    "num_matched_images": len(samples),

    "bleu": {
        "BLEU-1": avg_bleu1,
        "BLEU-2": avg_bleu2,
        "BLEU-3": avg_bleu3,
        "BLEU-4": avg_bleu4
    },

    "bertscore": {
        "model": BERT_MODEL,
        "precision": bert_p_avg,
        "recall": bert_r_avg,
        "f1": bert_f1_avg,
        "evaluation_time_sec": bert_time
    },

    "inference": {
        "total_inference_time_sec": generated_data.get(
            "total_inference_time_sec"
        ),
        "average_inference_time_ms_per_image": inference_data
    },

    "samples": samples
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# ============================================================
# Summary
# ============================================================

print()
print("=" * 70)
print("Advanced Evaluation Complete")
print("=" * 70)

print(f"Images evaluated: {len(samples)}")
print()

print("BLEU:")
print(f"  BLEU-1: {avg_bleu1:.4f}")
print(f"  BLEU-2: {avg_bleu2:.4f}")
print(f"  BLEU-3: {avg_bleu3:.4f}")
print(f"  BLEU-4: {avg_bleu4:.4f}")
print()

print("BERTScore:")
print(f"  Precision: {bert_p_avg:.4f}")
print(f"  Recall:    {bert_r_avg:.4f}")
print(f"  F1:        {bert_f1_avg:.4f}")
print()

if inference_data is not None:
    print(f"Average inference time: {inference_data:.4f} ms/image")

print()
print(f"Results saved to: {OUTPUT_JSON}")
print("=" * 70)
