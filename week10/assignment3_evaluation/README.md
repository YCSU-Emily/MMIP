# Week 10 Assignment 3 — RAG System Evaluation

## 1. Overview

This assignment evaluates the retrieval performance of two existing systems:

1. **Smart3C RAG** from Assignment 1, evaluated using RAGAS.
2. **VideoRAG** from Assignment 2, evaluated using evidence-level retrieval metrics and additional RRF experiments.

The goal is to measure retrieval quality, identify weaknesses, and investigate whether query-aware multimodal fusion can improve VideoRAG retrieval performance.

---

## 2. Part A — RAGAS Evaluation

### 2.1 Experimental Setup

| Item | Configuration |
|---|---|
| RAG System | Smart3C Canonical V4 |
| Evaluation Framework | RAGAS 0.4.3 |
| LLM | Qwen/Qwen3.5-4B |
| Embedding Model | BAAI/bge-m3 |
| Retrieval Top-K | 3 |
| Test Samples | 16 |

The evaluation uses the existing Canonical V4 test dataset and the corresponding ChromaDB collection.

### 2.2 Evaluation Metrics

Four metrics are used:

- **Faithfulness**: Measures whether the generated answer is supported by the retrieved context.
- **Answer Relevancy**: Measures how relevant the generated answer is to the user question.
- **Context Precision**: Measures whether relevant retrieved contexts are ranked above irrelevant contexts.
- **Context Recall**: Measures whether the retrieved context contains the information needed to answer the reference answer.

### 2.3 Results

| Metric | Score |
|---|---:|
| Faithfulness | 0.7310 |
| Answer Relevancy | 0.1717 |
| Context Precision | 0.1250 |
| Context Recall | 0.1771 |

### 2.4 Discussion

Faithfulness achieved 0.7310 among valid evaluation results, indicating that many generated answers were supported by the retrieved context.

However, Answer Relevancy, Context Precision, and Context Recall were relatively low. This suggests that the system still has difficulties retrieving sufficiently relevant evidence and covering all information required by the reference answers.

Three Faithfulness evaluations returned parser errors and were recorded as NaN rather than zero. Therefore, the Faithfulness score is based on 13 valid samples out of 16.

The reference answers were taken from the existing Canonical V4 test dataset. They are not claimed to have been independently human-validated.

---

## 3. Part B — VideoRAG Retrieval Evaluation

### 3.1 Dataset

The VideoRAG corpus contains 22 evidence records:

| Modality | Number of Records |
|---|---:|
| Speech | 8 |
| Visual OCR | 14 |
| Total | 22 |

Speech records contain transcribed spoken instructions, while OCR records contain text extracted from video frames.

**Important:** Speech and OCR originate from two separate source videos. Their timestamps do not represent a shared synchronized timeline.

### 3.2 Ground Truth

The original evaluation uses six manually annotated questions (VQ01–VQ06).

Each question is associated with relevant Speech evidence IDs, OCR evidence IDs, or both.

### 3.3 Evaluation Metrics

- **Recall@K**: The proportion of annotated gold evidence retrieved within the top K results.
- **HitRate@K**: The proportion of eligible questions with at least one gold evidence result in the top K.
- **MRR**: Mean Reciprocal Rank of the first relevant evidence.

### 3.4 Baseline Results

| Method | Eligible Questions | Recall@1 | Recall@3 | Recall@5 | HitRate@3 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| Speech-only | 6 | 0.5000 | 1.0000 | 1.0000 | 1.0000 | 0.7222 |
| OCR-only | 3 | 0.3333 | 0.6667 | 0.7778 | 1.0000 | 1.0000 |
| RRF Fusion | 6 | 0.2000 | 0.5333 | 0.7333 | 0.8333 | 0.6722 |

Speech-only and OCR-only use different eligible question sets. Their Recall values should not be interpreted as a direct comparison under an identical denominator.

---

## 4. Part C — RRF Weighting Experiments

### 4.1 Motivation

The two modalities provide different types of evidence:

- Speech is useful for retrieving spoken instructions and procedural explanations.
- OCR is useful for retrieving visible interface elements, screen text, and network names.

A fixed fusion weight may not be optimal for every query.

Therefore, additional experiments were conducted to investigate the effect of changing modality weights.

### 4.2 Weighted RRF

The experiments compare:

| Method | Speech Weight | OCR Weight |
|---|---:|---:|
| Baseline RRF | 1.0 | 1.0 |
| Fixed Speech-Weighted RRF | 1.5 | 1.0 |
| Fixed OCR-Weighted RRF | 1.0 | 1.5 |

The RRF constant is 60. Each modality retrieves up to 8 candidates, and the final ranking contains up to 8 results.

The weighted experiments reuse the original candidate lists, changing ranking weights rather than the underlying retrieval candidates.

### 4.3 Query-Adaptive RRF

Instead of applying one fixed weighting strategy to every query, Query-Adaptive RRF adjusts the modality weights according to manually annotated query intent.

| Query Intent | Speech Weight | OCR Weight |
|---|---:|---:|
| Speech-oriented | 1.5 | 1.0 |
| Visual-oriented | 1.0 | 1.5 |

This experiment uses six additional questions (VQ07–VQ12) and the same 22-record corpus.

### 4.4 Four-Way Comparison

| Method | Recall@1 | Recall@3 | Recall@5 | HitRate@3 | MRR |
|---|---:|---:|---:|---:|---:|
| Baseline RRF | 0.3333 | 0.6667 | 1.0000 | 0.6667 | 0.5194 |
| Fixed Speech-Weighted | 0.1667 | 0.5000 | 0.5000 | 0.5000 | 0.3333 |
| Fixed OCR-Weighted | 0.1667 | 0.5000 | 0.5000 | 0.5000 | 0.3056 |
| Query-Adaptive RRF | 0.3333 | 1.0000 | 1.0000 | 1.0000 | 0.6389 |

### 4.5 Discussion

Query-Adaptive RRF achieved the best Recall@3, Recall@5, HitRate@3, and MRR among the four methods.

Compared with Baseline RRF:

- Recall@3 increased from 0.6667 to 1.0000.
- MRR increased from 0.5194 to 0.6389.
- Recall@1 remained unchanged at 0.3333.
- Recall@5 remained unchanged at 1.0000.

The per-question results also show that visual-oriented questions benefit from OCR weighting, while speech-oriented questions benefit from Speech weighting.

These findings suggest that different queries require different modality preferences, and a query-aware fusion strategy may be more suitable than a single global weighting configuration.

However, the intent labels were manually annotated. This experiment evaluates intent-aware fusion with known labels, not an automatic query-intent classifier.

---

## 5. Limitations

1. The RAGAS evaluation contains only 16 test samples.
2. Three Faithfulness evaluations returned parser errors.
3. The RAGAS reference answers come from the existing test dataset and are not independently human-validated.
4. VideoRAG uses a small corpus of 22 evidence records.
5. The original VideoRAG evaluation uses six manually annotated questions.
6. The adaptive experiment uses six additional questions on the same corpus, not an independent external video dataset.
7. Query intent labels are manually annotated; automatic intent classification has not been evaluated.
8. The experiments evaluate retrieval ranking, not the quality of generated answers.
9. Speech and OCR come from separate source videos, so their timestamps must not be treated as synchronized.

---

## 6. Reproducibility

### Environment

```bash
conda activate mmipWeek10
cd ~/MMIP
```

### RAGAS Evaluation

```bash
python week10/assignment3_evaluation/evaluation/evaluate_ragas.py --reuse-raw
```

### VideoRAG Baseline Evaluation

```bash
python week10/assignment3_evaluation/videorag/evaluate_videorag.py
```

### Weighted RRF Evaluation

```bash
python week10/assignment3_evaluation/videorag/evaluate_weighted_rrf.py
```

### Query-Adaptive RRF Evaluation

```bash
python week10/assignment3_evaluation/videorag/evaluate_adaptive_rrf.py
```

### Four-Way RRF Evaluation

```bash
python week10/assignment3_evaluation/videorag/evaluate_four_way_rrf.py
```

---

## 7. Output Files

### RAGAS

- `reports/ragas_raw_results.json`
- `reports/ragas_scores.json`
- `reports/ragas_summary.json`

### VideoRAG

- `videorag/videorag_evaluation_results.json`
- `videorag/videorag_evaluation_summary.csv`
- `videorag/videorag_evaluation_report.md`

### Weighted RRF

- `videorag/weighted_rrf_results.json`
- `videorag/weighted_rrf_summary.csv`
- `videorag/weighted_rrf_report.md`

### Adaptive and Four-Way RRF

- `videorag/adaptive_rrf_results.json`
- `videorag/adaptive_rrf_summary.csv`
- `videorag/adaptive_rrf_report.md`
- `videorag/four_way_rrf_results.json`
- `videorag/four_way_rrf_summary.csv`
- `videorag/four_way_rrf_report.md`

---

## 8. Conclusion

This assignment evaluates both text-based RAG and multimodal VideoRAG systems.

RAGAS evaluation reveals that the Smart3C RAG system still has room for improvement in answer relevancy and retrieval quality.

For VideoRAG, the experiments demonstrate that a fixed modality weighting strategy may not perform well across different query types. Query-Adaptive RRF achieved better Top-3 retrieval performance and MRR in the six-question adaptive evaluation.

Overall, the results support the potential of query-aware multimodal retrieval fusion, while larger datasets and automatic intent classification are needed for further validation.
