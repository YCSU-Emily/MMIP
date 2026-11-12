# VideoRAG Retrieval Evaluation

- Corpus records: 22
- Ground Truth questions: 6
- Embedding model: BAAI/bge-m3
- Fusion: Reciprocal Rank Fusion (RRF), constant=60
- Candidate depth: 8 per modality

## Evaluation Metrics

- Evidence Recall@K: retrieved gold evidence / total gold evidence.
- HitRate@K: proportion of eligible questions with at least one gold result.
- MRR: mean reciprocal rank of the first relevant evidence.

## Results

| Method | Eligible Q | Recall@1 | Recall@3 | Recall@5 | HitRate@3 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| speech_only | 6 | 0.5000 | 1.0000 | 1.0000 | 1.0000 | 0.7222 |
| ocr_only | 3 | 0.3333 | 0.6667 | 0.7778 | 1.0000 | 1.0000 |
| rrf_fusion | 6 | 0.2000 | 0.5333 | 0.7333 | 0.8333 | 0.6722 |

## Annotation Notes

Ground truth evidence IDs are manually annotated from original speech transcript and OCR transcript. Speech and OCR are separate video sources, so their timestamps must not be treated as one shared timeline.

Speech-only and OCR-only metrics use only questions with gold evidence in that modality.
RRF Fusion uses the union of Speech and OCR gold evidence.
The speech and OCR recordings are separate video sources; their timestamps are not treated as one shared timeline.
