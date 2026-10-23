## Results and Discussion

### Overall comparison

| Model | Manually scored | Score rate | Mean latency (s) | Successful inference calls |
|---|---:|---:|---:|---:|
| Gemini Flash Lite | 40/40 | 100.00% | 1.9607 | 20 |
| Qwen2.5-VL-3B-Instruct | 34/40 | 85.00% | 0.2049 | 20 |
| SmolVLM2-2.2B-Instruct | 30/40 | 75.00% | 0.3434 | 20 |

### Evaluation method

Each benchmark response was manually evaluated on a 0–2 scale using the reference answer. The score rate is the total awarded score divided by the maximum possible score for the questions scored for that model. Category-level results are provided in `results/category_score_summary.csv`, and the overall summary is provided in `results/model_comparison_summary.csv`.

### Discussion

The results should be interpreted within the scope of this 20-question visual benchmark. A higher score indicates better agreement with the reference answers on this test set, not a guarantee of general visual-question-answering performance. Differences among counting, spatial-relation, OCR, shape/color, comparison, and visual-reasoning categories can help identify task-specific strengths and weaknesses.

Latency is not a perfectly controlled cross-model comparison. Gemini latency measures successful cloud API request time, while Qwen and SmolVLM2 latency values come from their local inference result CSVs. Hardware, preprocessing, image transfer, network conditions, warm-up, and measurement boundaries may differ. Therefore, latency should be interpreted as observed runtime in these experiments rather than a hardware-independent speed ranking.

Gemini's latency summary uses successful initial benchmark requests and the successful retests for questions 17 and 18. Failed initial requests are excluded from the latency mean; the successful retests are used for those question IDs.

### Reproducibility and files

- `results/model_comparison_summary.csv`: overall score and latency comparison.
- `results/category_score_summary.csv`: score breakdown by category and model.
- `results/gemini_manual_review_progress.csv`: completed Gemini manual review.
- The original inference result CSVs and the original Gemini pending-review CSV are retained.

### Limitations

The benchmark contains only 20 questions, so each question has a substantial effect on the overall score. The reported scores reflect manual judgments against the supplied reference answers and should not be generalized to all images, prompts, or real-world use cases.
