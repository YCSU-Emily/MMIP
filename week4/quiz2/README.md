# Week 4 Quiz 2 - Sentiment Analysis

## 1. Task Description

本作業使用 IMDb Movie Review Dataset 進行情感分類（Sentiment Analysis）。

本 Quiz 分為兩個部分：

* **Basic**：使用 RNN（Recurrent Neural Network）進行情感分類。
* **Advanced**：使用 LSTM（Long Short-Term Memory）進行情感分類，並與 RNN 進行比較。

分類任務為二元分類：

* `0`：Negative
* `1`：Positive

---

## 2. Dataset

本實驗使用 IMDb Movie Review Dataset。

| Dataset      | Samples |
| ------------ | ------: |
| Training set |  25,000 |
| Test set     |  25,000 |
| Total        |  50,000 |

原始 Training set 再進一步切分：

| Split      | Samples |
| ---------- | ------: |
| Train      |  22,500 |
| Validation |   2,500 |
| Test       |  25,000 |

Training / Validation 使用 stratified split，並設定：

```text
random_state = 3407
validation ratio = 0.1
```

### Label Distribution

IMDb dataset 為平衡二元分類資料集：

```text
0 = Negative
1 = Positive
```

Training 與 Test 中兩個類別皆各有 12,500 筆資料。

---

## 3. Data Preprocessing

文字資料使用簡單的 lowercase whitespace tokenizer。

例如：

```text
Original:
This movie was really good!

Tokenized:
["this", "movie", "was", "really", "good!"]
```

建立 vocabulary 時：

```text
Vocabulary size = 20,000
```

另外使用兩個特殊 token：

```text
<PAD> = 0
<UNK> = 1
```

每筆文字固定處理為：

```text
MAX_LEN = 200
```

如果文字長度超過 200 tokens，截斷至前 200 tokens。

如果不足 200 tokens，使用 `<PAD>` 補至長度 200。

---

# 4. Experimental Environment

Hardware:

```text
GPU: NVIDIA GeForce RTX 5090
```

Software:

```text
PyTorch: 2.14.0+cu130
Device: CUDA
```

Random seed:

```text
3407
```

---

# 5. Model Configuration

RNN 與 LSTM 使用相同的主要設定，以進行公平比較。

| Parameter               |            Value |
| ----------------------- | ---------------: |
| Vocabulary Size         |           20,000 |
| Maximum Sequence Length |              200 |
| Embedding Dimension     |              128 |
| Hidden Dimension        |              128 |
| Number of Layers        |                1 |
| Batch Size              |              128 |
| Epochs                  |               10 |
| Learning Rate           |             1e-3 |
| Optimizer               |             Adam |
| Loss Function           | CrossEntropyLoss |

---

# 6. RNN Model

RNN architecture:

```text
Input Text
    │
    ▼
Tokenization
    │
    ▼
Vocabulary
    │
    ▼
Embedding
20,000 → 128
    │
    ▼
RNN
128 → 128
    │
    ▼
Last Hidden State
    │
    ▼
Fully Connected
128 → 2
    │
    ▼
Negative / Positive
```

The RNN model contains:

```text
Embedding(20000, 128)
RNN(128, 128)
Linear(128, 2)
```

Total parameters:

```text
2,593,282
```

Training time:

```text
7.72 sec
```

Inference time:

```text
0.62 sec
```

---

# 7. LSTM Model

LSTM architecture:

```text
Input Text
    │
    ▼
Tokenization
    │
    ▼
Vocabulary
    │
    ▼
Embedding
20,000 → 128
    │
    ▼
LSTM
128 → 128
    │
    ▼
Last Hidden State
    │
    ▼
Fully Connected
128 → 2
    │
    ▼
Negative / Positive
```

The LSTM model contains:

```text
Embedding(20000, 128)
LSTM(128, 128)
Linear(128, 2)
```

Total parameters:

```text
2,692,354
```

Training time:

```text
7.46 sec
```

Inference time:

```text
0.70 sec
```

---

# 8. RNN Results

The best RNN checkpoint was selected according to validation F1-score.

Best validation F1:

```text
0.6282
```

Test results:

| Metric         |       RNN |
| -------------- | --------: |
| Test Loss      |    0.6973 |
| Accuracy       |    51.06% |
| Precision      |    50.64% |
| Recall         |    84.84% |
| F1-score       |    63.42% |
| Parameters     | 2,593,282 |
| Training Time  |  7.72 sec |
| Inference Time |  0.62 sec |

Model checkpoint:

```text
models/best_rnn.pth
```

Results:

```text
results/rnn/
├── config.json
├── history.json
├── metrics.json
└── test_predictions.csv
```

---

# 9. LSTM Results

The best LSTM checkpoint was selected according to validation F1-score.

Best validation F1:

```text
0.6533
```

Test results:

| Metric         |      LSTM |
| -------------- | --------: |
| Test Loss      |    0.7176 |
| Accuracy       |    53.28% |
| Precision      |    51.94% |
| Recall         |    87.95% |
| F1-score       |    65.31% |
| Parameters     | 2,692,354 |
| Training Time  |  7.46 sec |
| Inference Time |  0.70 sec |

Model checkpoint:

```text
models/best_lstm.pth
```

Results:

```text
results/lstm/
├── config.json
├── history.json
├── metrics.json
└── test_predictions.csv
```

---

# 10. RNN vs LSTM Comparison

## Overall Comparison

| Metric             |        RNN |       LSTM |
| ------------------ | ---------: | ---------: |
| Accuracy           |     51.06% | **53.28%** |
| Precision          |     50.64% | **51.94%** |
| Recall             |     84.84% | **87.95%** |
| F1-score           |     63.42% | **65.31%** |
| Parameters         |  2,593,282 |  2,692,354 |
| Training Time      |     7.72 s | **7.46 s** |
| Inference Time     | **0.62 s** |     0.70 s |
| Best Validation F1 |     62.82% | **65.33%** |

---

## Metric Differences

Compared with RNN, LSTM produced:

```text
Accuracy:
53.28% - 51.06%
= +2.22 percentage points

Precision:
51.94% - 50.64%
= +1.30 percentage points

Recall:
87.95% - 84.84%
= +3.11 percentage points

F1-score:
65.31% - 63.42%
= +1.89 percentage points
```

The LSTM model has:

```text
2,692,354 - 2,593,282
= 99,072
```

more parameters than the RNN.

The measured inference time of LSTM was also slightly higher:

```text
0.70 sec vs 0.62 sec
```

---

# 11. Training Behavior

## RNN

The RNN training accuracy increased from:

```text
49.81%
```

at Epoch 1 to:

```text
67.84%
```

at Epoch 10.

However, validation accuracy remained close to 50% throughout training.

The validation loss also increased from:

```text
0.6956
```

to:

```text
0.9415
```

This indicates that the current RNN configuration does not generalize well to the validation/test data.

---

## LSTM

The LSTM training accuracy increased from:

```text
51.16%
```

at Epoch 1 to:

```text
75.27%
```

at Epoch 10.

The highest validation F1-score was:

```text
65.33%
```

at Epoch 4.

After Epoch 4, the validation metrics fluctuated considerably.

Therefore, the experiment uses the checkpoint with the best validation F1-score instead of simply using the final epoch.

---

# 12. Analysis

The experiment demonstrates the basic difference between RNN and LSTM architectures for text classification.

The RNN uses a standard recurrent structure to process the sequence, while LSTM introduces memory cells and gating mechanisms to maintain information across the sequence.

In this experiment, the LSTM obtained higher values for:

```text
Accuracy
Precision
Recall
F1-score
Best Validation F1
```

while the RNN had slightly lower measured inference time and fewer parameters.

However, both models achieved test accuracy close to 50%, indicating that the current simple preprocessing and model configuration has limited classification performance.

Possible factors include:

1. Simple whitespace tokenization.
2. Vocabulary limited to 20,000 words.
3. Fixed sequence length of 200 tokens.
4. Padding is directly passed through the recurrent layer.
5. Only one recurrent layer is used.
6. No pretrained word embeddings are used.
7. No bidirectional recurrent layer is used.
8. No attention mechanism is used.

Therefore, this experiment mainly focuses on implementing and comparing RNN and LSTM architectures rather than maximizing IMDb classification performance.

---

# 13. Output Directory

Final Quiz 2 structure:

```text
week4/quiz2/
├── README.md
└── sentiment/
    ├── data/
    │   ├── train.csv
    │   └── test.csv
    │
    ├── models/
    │   ├── best_rnn.pth
    │   └── best_lstm.pth
    │
    ├── results/
    │   ├── rnn/
    │   │   ├── config.json
    │   │   ├── history.json
    │   │   ├── metrics.json
    │   │   └── test_predictions.csv
    │   │
    │   └── lstm/
    │       ├── config.json
    │       ├── history.json
    │       ├── metrics.json
    │       └── test_predictions.csv
    │
    ├── train_rnn.py
    └── train_lstm.py
```

---

# 14. How to Run

## RNN

```bash
cd ~/MMIP/week4/quiz2/sentiment

python train_rnn.py
```

## LSTM

```bash
cd ~/MMIP/week4/quiz2/sentiment

python train_lstm.py
```

---

# 15. Conclusion

本 Quiz 完成了 IMDb 情感分析的 RNN 與 LSTM 實作。

實驗結果顯示，在相同資料集、資料切分與主要超參數設定下，LSTM 的測試 Accuracy、Precision、Recall 與 F1-score 均高於本次 RNN 實驗結果；另一方面，RNN 使用較少的參數，且本次測得的 inference time 略低。

本實驗也觀察到兩種模型的 validation/test performance 皆受到目前簡單文字前處理與模型架構限制，因此後續若要提升效能，可以考慮使用更完善的 tokenizer、pretrained embeddings、Bidirectional RNN/LSTM、attention mechanism 或 Transformer-based model。

---

# 16. Reproducibility

All experiments use:

```text
Random Seed = 3407
GPU = NVIDIA GeForce RTX 5090
PyTorch = 2.14.0+cu130
```

The same train/validation split and major hyperparameters were used for both RNN and LSTM to make the comparison consistent.
