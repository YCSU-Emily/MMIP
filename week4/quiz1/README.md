# Week 4 Quiz 1 — Dataset Preparation

## 1. Task Description

This quiz focuses on preparing datasets for three different AI tasks:

1. Sentiment Analysis
2. Image Classification
3. Image Captioning

The prepared datasets will be used in the following quizzes:

- Quiz 2: Sentiment Analysis
- Quiz 3: Vision Transformer
- Quiz 4: Image Captioning

---

## 2. Dataset Overview

| Task | Dataset | Training | Validation | Test |
|---|---|---:|---:|---:|
| Sentiment Analysis | IMDb | 25,000 | - | 25,000 |
| Image Classification | Wafer Defect | 3,924 | 837 | 846 |
| Image Captioning | COCO | 10,000 | 1,000 | 1,000 |

---

## 3. Sentiment Analysis Dataset

### Dataset

IMDb Movie Review Dataset

### Task

Binary sentiment classification.

Labels:

- `0`: Negative
- `1`: Positive

### Dataset Statistics

#### Training Set

- Samples: 25,000
- Negative: 12,500
- Positive: 12,500

#### Test Set

- Samples: 25,000
- Negative: 12,500
- Positive: 12,500

### Data Format

Each CSV file contains two columns:

```text
text,label
