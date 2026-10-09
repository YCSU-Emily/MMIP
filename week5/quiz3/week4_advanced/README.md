# Week 4 Advanced — Image Captioning

## Purpose

This folder preserves selected source code and experiment artifacts from the Week 4 advanced image-captioning work. It is included with Week 5 Quiz 3 to provide the earlier model, evaluation results, and comparison history.

The original files remain under `week4/quiz4/image_captioning/`.

## Source Code

- `train_caption.py` — trains the original captioning model.
- `continue_train_caption.py` — continues training from an existing checkpoint.
- `inference.py` — generates image captions.
- `inference_compare.py` — compares caption outputs.
- `evaluate_caption.py` — evaluates generated captions.
- `evaluate_comparison_v1.py` — compares model evaluation results.
- `gemini_judge.py` — performs Gemini-based caption evaluation.

## Main Results

- `results/advanced_evaluation.json` — advanced evaluation output.
- `results/generated_captions.json` — generated captions.
- `results/history.json` — training history.
- `results/config.json` — experiment configuration.
- `results/gemini_evaluation.json` — Gemini evaluation output.
- `results/automatic_metric_summary_v1.json` — automatic metric summary.
- `results/clip_consistency_summary_v1.json` — CLIP consistency summary.
- `results/caption_audit_100.csv` — caption audit records.
- `results/manual_caption_audit.csv` — manual caption audit.
- `results/human_review_v1.csv` — human-review records.

## Epoch 5 vs. Epoch 8

The `results/experiments/` folder contains selected captions, metrics, configuration, training history, and a visual audit for the baseline and continuation experiments.

## Relationship to Week 5 Quiz 3

Week 4 Advanced provides the earlier experiment and evaluation artifacts. The parent `week5/quiz3/` folder contains the CLIP + LSTM implementation and the shared-100-image comparison against the Week 4 baseline and continuation models.

The results should be interpreted using the evaluation setup and dataset subset documented in the parent README.

## Reproducibility Notes

The scripts and result files are copied for submission and reference. Model checkpoints and dataset files are not duplicated here. Running the scripts may require the original project data paths, dependencies, and local model checkpoints from the repository.
