# Assignment B: Vision-Language Model Benchmark

## 1. Overview

This project evaluates and compares three vision-language models (VLMs) on a small image-based visual question-answering benchmark.

The objective is to examine their performance on visual understanding tasks, including object counting, spatial relations, optical character recognition (OCR), shape and color recognition, comparison, and visual reasoning.

The evaluated models are:

- **Gemini Flash Lite** — accessed through the Gemini API.
- **Qwen2.5-VL-3B-Instruct** — executed locally.
- **SmolVLM2-2.2B-Instruct** — executed locally.

The evaluation considers both answer quality and observed inference latency. Since the models run in different environments, latency results are reported with their measurement limitations.

## 2. Benchmark Dataset

The benchmark contains **20 image-based questions**. Each question includes an image, a prompt, and a reference answer.

The questions cover six categories:

- Counting
- Spatial relations
- OCR
- Shape and color recognition
- Comparison
- Visual reasoning

The benchmark definition is stored in `benchmarks/visual_benchmark.json`, and the corresponding images are stored in `images/benchmark/`.

## 3. Models and Execution Environments

| Model | Execution method | Inference source |
|---|---|---|
| Gemini Flash Lite | Cloud API | Gemini API |
| Qwen2.5-VL-3B-Instruct | Local inference | Local result CSV |
| SmolVLM2-2.2B-Instruct | Local inference | Local result CSV |

Gemini responses were collected through API requests. Questions 17 and 18 were successfully retested after the initial requests encountered API quota errors. The successful retests were used in the final 20-question comparison.

The local models were evaluated using their saved inference results. The original inference result files were retained.

## 4. Evaluation Method

Each model's response was manually compared with the corresponding reference answer using a **0–2 scoring scale**:

| Score | Interpretation |
|---:|---|
| 2 | Correct answer |
| 1 | Partially correct answer |
| 0 | Incorrect answer |

For 20 questions, the maximum possible score is 40 points per model.

The score rate is calculated as:

`Score rate (%) = Total score / Maximum possible score × 100`

Inference latency is reported as the mean measured time for successful inference calls. For Gemini, the calculation includes successful initial requests and the successful retests for questions 17 and 18. Failed initial requests are excluded from the latency mean.

## 5. Overall Results

| Model | Score | Score Rate | Mean Latency (s) | Successful Calls |
|---|---:|---:|---:|---:|
| Gemini Flash Lite | 40/40 | 100.00% | 1.9607 | 20 |
| Qwen2.5-VL-3B-Instruct | 34/40 | 85.00% | 0.2049 | 20 |
| SmolVLM2-2.2B-Instruct | 30/40 | 75.00% | 0.3434 | 20 |

### Overall Observations

**Gemini Flash Lite** achieved the highest score, receiving 40 out of 40 points in this benchmark.

**Qwen2.5-VL-3B-Instruct** achieved 34 out of 40 points and had the lowest measured mean latency among the three models in these experiments.

**SmolVLM2-2.2B-Instruct** achieved 30 out of 40 points. Its observed mean latency was higher than Qwen's in the recorded local inference results.

These findings indicate that Gemini performed best in answer accuracy on this test set, while Qwen offered a favorable combination of answer quality and observed latency in the local experiments.

## 6. Category-Level Results

The score rate for each category is summarized below.

| Category | Gemini Flash Lite | Qwen2.5-VL-3B-Instruct | SmolVLM2-2.2B-Instruct |
|---|---:|---:|---:|
| Comparison | 100.00% | 100.00% | 100.00% |
| Counting | 100.00% | 87.50% | 62.50% |
| OCR | 100.00% | 100.00% | 100.00% |
| Shape / Color | 100.00% | 66.67% | 83.33% |
| Spatial Relation | 100.00% | 90.00% | 80.00% |
| Visual Reasoning | 100.00% | 66.67% | 33.33% |

### Category-Level Observations

- **OCR:** All three models achieved full scores, suggesting that the text-recognition questions in this benchmark were manageable for each model.
- **Comparison:** All three models also achieved full scores.
- **Counting:** Gemini achieved a full score, while Qwen outperformed SmolVLM2.
- **Shape and Color:** Qwen had the lowest score rate of the three models in this category; SmolVLM2 performed better than Qwen.
- **Spatial Relations:** Gemini achieved a full score, followed by Qwen and SmolVLM2.
- **Visual Reasoning:** The largest difference between the models appeared in this category. Qwen scored 66.67%, while SmolVLM2 scored 33.33%.

These results suggest that model performance depends on the type of visual task. However, each category contains only two to five questions, so these observations should not be treated as definitive measures of general model capabilities.

## 7. Latency and Fairness Considerations

The recorded mean latency values were:

- Gemini Flash Lite: **1.9607 seconds**
- Qwen2.5-VL-3B-Instruct: **0.2049 seconds**
- SmolVLM2-2.2B-Instruct: **0.3434 seconds**

These values are observations from the current experiments, not a controlled hardware-independent speed ranking.

Gemini latency measures cloud API request time, whereas the local model latency values come from their saved local inference results. The measurement boundaries may differ in terms of preprocessing, image handling, network transfer, warm-up, and other overheads.

Consequently, latency comparisons should be interpreted cautiously. A rigorous speed comparison would require a standardized measurement procedure and clearly defined timing boundaries.

## 8. Results and Project Files

Important files in this project include:

```text
Assignment_B_VLM/
├── benchmarks/
│   └── visual_benchmark.json
├── images/
│   └── benchmark/
│       ├── 01.png
│       ├── ...
│       └── 20.png
├── results/
│   ├── model_comparison_summary.csv
│   ├── category_score_summary.csv
│   ├── english_qwen25_vl_3b.csv
│   ├── english_smolvlm2_22b.csv
│   ├── english_manual_review_scored_v2.csv
│   ├── english_question_by_question_review_v2.csv
│   ├── english_category_score_summary_v2.csv
│   ├── gemini_manual_review_progress.csv
│   ├── gemini_benchmark_results_20261023_153220.csv
│   ├── gemini_retest_17_18_20261023_153553.csv
│   └── assignment_b_results_report_v2.md
└── README.md
```

The directory may contain additional intermediate or diagnostic files.

### Main Result Files

- `results/model_comparison_summary.csv` — overall score and latency summary.
- `results/category_score_summary.csv` — score breakdown by category and model.
- `results/english_manual_review_scored_v2.csv` — manual evaluation results for the two local models.
- `results/english_question_by_question_review_v2.csv` — question-by-question comparison of the local models.
- `results/gemini_manual_review_progress.csv` — completed manual evaluation of Gemini responses.
- `results/assignment_b_results_report_v2.md` — supplementary results report.

The original inference results and the original Gemini pending-review file were retained during the report-generation process.

## 9. Reproducibility

The saved CSV files provide the inference outputs and evaluation results used to produce the comparison tables.

To inspect the generated summary files from the project directory:

```bash
cat results/model_comparison_summary.csv
cat results/category_score_summary.csv
```

To reproduce the complete inference experiments, install the required dependencies, configure the relevant model weights and API credentials, and run the corresponding inference scripts. Exact commands and package versions should be documented alongside the scripts used for the experiments.

**API security:** Gemini API credentials should be provided through an environment variable such as `GEMINI_API_KEY`. Do not commit API keys or other secrets to version control.

## 10. Limitations

This benchmark has several limitations:

1. **Small sample size:** Only 20 questions were evaluated, and each question can noticeably affect the overall score.
2. **Manual evaluation:** Scores depend on comparisons with the supplied reference answers and may involve some subjective judgment.
3. **Limited task coverage:** The results do not represent all visual question-answering scenarios or real-world images.
4. **Different inference environments:** Cloud API and local inference latency measurements are not perfectly equivalent.
5. **API quota constraints:** Some initial Gemini requests failed because of quota limits and required successful retests.

The results should therefore be interpreted as a comparative study of these three models on the specified benchmark, rather than a universal ranking of vision-language models.

## 11. Conclusion

In this 20-question benchmark, Gemini Flash Lite achieved the highest answer score, followed by Qwen2.5-VL-3B-Instruct and SmolVLM2-2.2B-Instruct. The category-level results showed differences in counting, shape and color recognition, spatial relations, and visual reasoning, while all three models achieved full scores in OCR and comparison.

Qwen had the lowest observed mean latency among the evaluated models in these experiments. Nevertheless, differences in execution environments and timing procedures prevent a definitive cross-platform speed conclusion.

Overall, this experiment demonstrates the value of evaluating VLMs across multiple task categories and considering both answer quality and inference cost when selecting a model for a particular application.
