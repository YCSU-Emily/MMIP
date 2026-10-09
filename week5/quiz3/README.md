# Week 5 Quiz 3 — Image Captioning Model Comparison

## 1. Objective

Compare three image captioning systems on the same 100 COCO test images:

1. Week 4 baseline
2. Week 4 continuation
3. Week 5 CLIP + LSTM

The goal is to investigate whether using pretrained CLIP image features with a trainable LSTM caption decoder improves generated captions relative to the Week 4 models.

## 2. Week 5 Model

The Week 5 model uses:

- **Image encoder:** pretrained CLIP ViT-B/32
- **Image feature processing:** trainable image projection layer
- **Text decoder:** LSTM
- **Training epochs:** 5
- **Learning rate:** 0.001
- **Training objective:** caption token prediction loss

The CLIP encoder provides visual features. The trainable projection maps image features into the decoder's representation space, and the LSTM generates a caption token by token.

## 3. Training Results

| Epoch | Train Loss | Validation Loss |
|---:|---:|---:|
| 1 | 4.2557 | 3.5112 |
| 2 | 3.3855 | 3.0467 |
| 3 | 3.0089 | 2.8375 |
| 4 | 2.8273 | 2.7204 |
| 5 | 2.6585 | 2.6415 |

Both training and validation loss decreased throughout the five epochs. The lowest recorded validation loss was **2.6415 at epoch 5**.

The best and last checkpoints are stored locally under `models/`. Model checkpoint files are excluded from Git by the repository's `*.pth` ignore rule.

## 4. Evaluation Setup

All three models were compared on the same 100 COCO test images, with five reference captions per image.

Metrics:

- **BLEU-1 to BLEU-4:** measure n-gram overlap between generated captions and reference captions.
- **BERTScore Precision, Recall, and F1:** estimate semantic similarity using contextual text embeddings.

The reported BLEU results use sentence-level scores averaged across images. These scores should be interpreted as measurements on this 100-image subset, not as results on the complete COCO test set.

## 5. Aggregate Results

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | BERTScore P | BERTScore R | BERTScore F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Week 4 baseline | 0.577859 | 0.387809 | 0.237471 | 0.162732 | 0.900748 | 0.899160 | 0.899887 |
| Week 4 continuation | 0.612501 | 0.416234 | 0.264923 | 0.176010 | 0.900261 | 0.898865 | 0.899494 |
| Week 5 CLIP + LSTM | **0.657494** | **0.441106** | **0.271387** | 0.170365 | 0.893477 | **0.899920** | 0.896624 |

### Main Findings

- Week 5 achieved the highest BLEU-1, BLEU-2, and BLEU-3 among the three models.
- Week 4 continuation achieved the highest BLEU-4.
- Week 4 baseline achieved the highest BERTScore Precision and F1.
- Week 5 achieved the highest BERTScore Recall, but its BERTScore F1 was lower than both Week 4 models.
- Therefore, Week 5 improved some lexical-overlap metrics but did not outperform the Week 4 models on every metric.

These results are descriptive comparisons on a 100-image subset; no statistical significance claim is made.

## 6. Per-Image Analysis

The script `analyze_per_image.py` calculates BLEU scores for each image and compares Week 5 with Week 4 continuation.

Selected examples:

| Image ID | Week 4 continuation BLEU-1 | Week 5 BLEU-1 | Difference |
|---|---:|---:|---:|
| `000000128699.jpg` | 0.6065 | 0.9048 | +0.2983 |
| `000000199055.jpg` | 0.3643 | 0.7889 | +0.4246 |
| `000000372317.jpg` | 0.9100 | 0.4250 | -0.4850 |

The skateboard example `000000128699.jpg` illustrates a substantial improvement: Week 5 describes a skateboard trick, while the Week 4 models describe flying a kite.

The bus example `000000372317.jpg` illustrates a failure case: Week 5 generates a repetitive bus-stop description, while the Week 4 captions are more consistent with the references.

These examples show that aggregate scores can hide image-level improvements and regressions.

## 7. Files

### Source code

- `train_clip_caption.py` — trains the CLIP + LSTM captioning model.
- `inference_clip_caption.py` — generates captions for validation or test images.
- `evaluate_comparison.py` — compares aggregate captioning metrics.
- `analyze_per_image.py` — evaluates per-image BLEU scores.
- `vocab.json` — vocabulary used by the captioning model.

### Results

- `results/training_history.json` — per-epoch training and validation loss.
- `results/clip_lstm_test_captions.json` — Week 5 generated captions.
- `results/week4_baseline_shared100.json` — Week 4 baseline captions.
- `results/week4_continuation_shared100.json` — Week 4 continuation captions.
- `results/comparison_week4_week5_metrics.csv` — aggregate metrics in CSV format.
- `results/comparison_week4_week5_metrics.json` — aggregate metrics in JSON format.
- `results/per_image_bleu_comparison.csv` — per-image BLEU comparison.
- `results/per_image_bleu_comparison.json` — per-image BLEU comparison in JSON format.

Model checkpoints and log files are kept locally and are not required for reading the summarized results.

## 8. Running the Scripts

Run commands from the repository root, `~/MMIP`, using an environment with the required dependencies installed.

### Train

```bash
python week5/quiz3/train_clip_caption.py --epochs 5 --lr 0.001
```

Other available options include `--batch-size`, `--workers`, `--min-freq`, `--max-length`, and `--limit-train`. The last option is intended for debugging on a limited number of training images.

### Generate captions

```bash
python week5/quiz3/inference_clip_caption.py --checkpoint week5/quiz3/models/best.pth --split test --limit 100
```

### Evaluate aggregate metrics

```bash
python week5/quiz3/evaluate_comparison.py
```

### Analyze individual images

```bash
python week5/quiz3/analyze_per_image.py
```

The evaluation and per-image scripts execute their analysis when run; they do not provide a conventional `--help` interface.

The exact data paths and available dependencies may need adjustment in another environment. The shared COCO data is referenced through the paths configured by the project.

## 9. Limitations and Future Work

- The comparison uses 100 images rather than the complete test set.
- BLEU scores are sensitive to wording and n-gram overlap; a lower score does not necessarily mean that a caption is semantically incorrect.
- Some Week 5 captions contain repeated phrases or incorrect scene details.
- BERTScore results should be interpreted together with qualitative examples rather than used as the sole measure of caption quality.
- Future work can investigate repeated-token control, decoder training, image-scene relationships, and broader evaluation on a larger test subset.

## 10. Conclusion

The Week 5 CLIP + LSTM model achieved the highest BLEU-1, BLEU-2, and BLEU-3 scores in the 100-image comparison. However, Week 4 continuation remained stronger on BLEU-4, and Week 4 baseline had the highest BERTScore F1. The results suggest that the CLIP-based model can improve some aspects of caption generation, but further work is needed to improve phrase consistency and scene-level accuracy.

## 11. Week 4 Advanced Artifacts

Selected Week 4 advanced image-captioning code and evaluation artifacts are included in [`week4_advanced/`](week4_advanced/README.md).

This folder preserves the earlier training, inference, automatic evaluation, Gemini evaluation, and selected human-review results. The original Week 4 files remain in their original directory.
