# Weighted RRF Evaluation Report

## Experimental Setup

- Questions: 6
- RRF constant: 60
- Candidate depth per modality: 8
- Final Top-K: 8
- Candidate lists are reused from the original evaluation JSON.
- No embedding inference or ChromaDB query is performed.

## Results: All Six Questions

| Method | Speech W | OCR W | Recall@3 | Recall@5 | HitRate@3 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| baseline_1.0_1.0 | 1.0 | 1.0 | 0.5333 | 0.7333 | 0.8333 | 0.6722 |
| weighted_A_1.5_1.0 | 1.5 | 1.0 | 0.4000 | 0.4000 | 1.0000 | 0.7222 |
| weighted_B_2.0_1.0 | 2.0 | 1.0 | 0.4000 | 0.4000 | 1.0000 | 0.7222 |
| weighted_C_1.0_1.5 | 1.0 | 1.5 | 0.4000 | 0.4667 | 0.5000 | 0.5000 |

## Results: Common Questions (VQ01, VQ03, VQ04)

| Method | Recall@3 | Recall@5 | HitRate@3 | MRR |
|---|---:|---:|---:|---:|
| baseline_1.0_1.0 | 0.5000 | 0.6667 | 1.0000 | 0.8333 |
| weighted_A_1.5_1.0 | 0.2500 | 0.2500 | 1.0000 | 0.8333 |
| weighted_B_2.0_1.0 | 0.2500 | 0.2500 | 1.0000 | 0.8333 |
| weighted_C_1.0_1.5 | 0.5000 | 0.5833 | 1.0000 | 1.0000 |

## Baseline Verification

- Reconstructed baseline ranking matches saved original RRF for 6/6 questions.

## Limitations

- Only six manually annotated questions are included.
- Results depend on the current candidate lists and Ground Truth.
- Weighted RRF changes ranking weights, not the underlying retrieval candidates.
- The two modalities originate from separate video sources; their timestamps are not a shared timeline.
