# VideoRAG Four-Way RRF Evaluation

## Experimental Setup

- Questions: 6
- Corpus records: 22
- RRF constant: 60
- Candidate depth per modality: 8
- Final Top-K: 8
- Speech and OCR candidates are retrieved independently.
- All methods use identical candidate lists for each query.
- Intent labels are manually annotated, not automatically predicted.
- Speech and OCR originate from separate source videos.

## Results

| Method | Recall@1 | Recall@3 | Recall@5 | HitRate@3 | MRR |
|---|---:|---:|---:|---:|---:|
| Baseline RRF (1.0 / 1.0) | 0.3333 | 0.6667 | 1.0000 | 0.6667 | 0.5194 |
| Fixed Speech-Weighted RRF (1.5 / 1.0) | 0.1667 | 0.5000 | 0.5000 | 0.5000 | 0.3333 |
| Fixed OCR-Weighted RRF (1.0 / 1.5) | 0.1667 | 0.5000 | 0.5000 | 0.5000 | 0.3056 |
| Query-Adaptive RRF | 0.3333 | 1.0000 | 1.0000 | 1.0000 | 0.6389 |

## Baseline Verification

- Reconstructed baseline matches the original VideoRetriever for 6/6 questions.

## Per-Question Adaptive Retrieval

| Question | Intent | Speech W | OCR W | First Gold Rank |
|---|---|---:|---:|---:|
| VQ07 | visual | 1.0 | 1.5 | 1 |
| VQ08 | speech | 1.5 | 1.0 | 2 |
| VQ09 | visual | 1.0 | 1.5 | 3 |
| VQ10 | speech | 1.5 | 1.0 | 1 |
| VQ11 | visual | 1.0 | 1.5 | 2 |
| VQ12 | speech | 1.5 | 1.0 | 2 |

## Limitations

- Only six additional questions are evaluated.
- Questions are new, but they use the existing video corpus.
- Ground Truth evidence IDs are manually annotated.
- Intent labels are provided manually; automatic intent classification has not been evaluated.
- The experiment evaluates retrieval ranking, not answer-generation quality.
- Speech and OCR timestamps belong to separate source videos and must not be interpreted as one shared timeline.
