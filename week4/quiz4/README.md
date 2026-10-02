# Week 4 Quiz 4 — Image Captioning

## 1. Task Description

本作業實作一個 Image Captioning 模型，輸入影像後自動產生自然語言描述。

本作業分為 Basic 與 Advanced 兩個部分。

### Basic

1. 訓練 Image Captioning 模型。
2. 對測試影像產生 Caption。
3. 使用 Gemini 判斷生成 Caption 是否與影像內容一致。

### Advanced

1. 使用 BLEU 評估生成 Caption 與人工參考 Caption 的文字重疊程度。
2. 使用 BERTScore 評估生成 Caption 與人工參考 Caption 的語意相似度。
3. 比較模型的 Inference Time 與整體效能。

---

## 2. Dataset

本實驗使用 COCO-style Image Captioning Dataset。

資料結構：

```text
data/
├── images/
└── captions/
    ├── train.json
    ├── val.json
    └── test.json
```

Dataset statistics:

| Split      | Images | Captions |
| ---------- | -----: | -------: |
| Train      | 10,000 |   50,031 |
| Validation |  1,000 |    5,005 |
| Test       |  1,000 |    5,005 |

每張影像具有多個人工撰寫的 Reference Captions，可用來評估模型產生的 Caption。

---

## 3. Model Architecture

本實驗採用 CNN Encoder + LSTM Decoder 的 Image Captioning 架構。

```text
                 Input Image
                      │
                      ▼
              ┌─────────────┐
              │   ResNet18  │
              │   Encoder   │
              └─────────────┘
                      │
                      ▼
                512-d Feature
                      │
                      ▼
              Linear Projection
                  512 → 256
                      │
                      ▼
               Image Feature
                      │
                      ▼
              ┌─────────────┐
              │     LSTM    │
              │   Decoder   │
              └─────────────┘
                      │
                      ▼
             Generated Caption
```

### Encoder

使用 pretrained ResNet18 作為 Image Encoder。

ResNet18 backbone 的主要參數固定，只訓練最後的 Linear Projection Layer，將 512 維影像特徵轉換為 256 維 embedding。

### Decoder

使用 LSTM 作為 Caption Decoder。

模型設定：

* Embedding Dimension: 256
* Hidden Dimension: 512
* Maximum Caption Length: 30
* Vocabulary Size: 3,360

影像特徵用於初始化 LSTM 的 hidden state 與 cell state，接著逐步產生 Caption。

---

## 4. Training Configuration

| Parameter              |                   Value |
| ---------------------- | ----------------------: |
| Image Size             |               224 × 224 |
| Batch Size             |                      64 |
| Embedding Dimension    |                     256 |
| Hidden Dimension       |                     512 |
| Epochs                 |                       5 |
| Learning Rate          |                    1e-3 |
| Optimizer              |                   AdamW |
| Loss Function          |      Cross Entropy Loss |
| Minimum Word Frequency |                       5 |
| Maximum Caption Length |                      30 |
| Gradient Clipping      |                     5.0 |
| Random Seed            |                    3407 |
| Device                 | NVIDIA GeForce RTX 5090 |
| PyTorch                |            2.14.0+cu130 |

### Trainable Parameters

| Component | Parameters |
| --------- | ---------: |
| Encoder   |    131,328 |
| Decoder   |  4,423,968 |
| Total     |  4,555,296 |

---

## 5. Training Results

模型訓練 5 個 Epoch。

| Epoch | Train Loss | Validation Loss |    Time |
| ----: | ---------: | --------------: | ------: |
|     1 |     3.6419 |          3.0500 | 51.21 s |
|     2 |     2.8460 |          2.8020 | 48.96 s |
|     3 |     2.5860 |          2.7145 | 49.19 s |
|     4 |     2.4143 |          2.6784 | 49.23 s |
|     5 |     2.2728 |          2.6657 | 49.16 s |

Best Validation Loss:

```text
2.6657
```

Total Training Time:

```text
248.15 seconds
```

Best Model Checkpoint:

```text
models/best_caption_model.pth
```

---

## 6. Caption Generation

訓練完成後，使用最佳模型對測試資料產生 Caption。

本實驗實際評估 100 張測試影像。

| Metric                 |          Result |
| ---------------------- | --------------: |
| Test Images            |             100 |
| Total Inference Time   |      0.7001 sec |
| Average Inference Time | 7.0014 ms/image |

Example:

```text
Image:
000000000776.jpg

Generated Caption:
a teddy bear is sitting on a table
```

Generated captions are saved to:

```text
results/generated_captions.json
```

---

## 7. Gemini Visual Evaluation

為了判斷生成 Caption 是否真正描述了影像內容，本實驗使用 Gemini Vision 進行影像與文字的一致性判斷。

Gemini 同時接收：

1. Original Image
2. Generated Caption

並將結果分類為：

* MATCH
* PARTIAL
* MISMATCH

### Gemini Evaluation Results

| Category          | Count |
| ----------------- | ----: |
| Total Images      |    20 |
| Valid Evaluations |    12 |
| MATCH             |     2 |
| PARTIAL           |     5 |
| MISMATCH          |     5 |
| HTTP 503 ERROR    |     8 |

在 12 筆有效評估中：

```text
MATCH = 2 / 12 = 16.67%

MATCH + PARTIAL = 7 / 12 = 58.33%
```

Gemini evaluation 中有 8 筆請求因 HTTP 503 暫時性服務錯誤而無法完成，因此上述比例只針對 12 筆有效結果計算，不能視為完整 20 張圖片的 Accuracy。

Gemini evaluation results:

```text
results/gemini_evaluation.json
```

---

## 8. Advanced Evaluation

### 8.1 BLEU

BLEU 用於衡量生成 Caption 與 Reference Caption 之間的 N-gram overlap。

本實驗使用 100 張測試影像進行評估。

| Metric |  Score |
| ------ | -----: |
| BLEU-1 | 0.5902 |
| BLEU-2 | 0.3740 |
| BLEU-3 | 0.2293 |
| BLEU-4 | 0.1529 |

結果顯示 BLEU-1 分數最高，隨著 N-gram 長度增加，分數逐漸下降。

這表示模型能夠產生與 Reference Caption 部分重疊的詞彙，但在較長的連續詞組與完整句子結構上，與人工描述仍存在差異。

---

### 8.2 BERTScore

BERTScore 用於衡量生成 Caption 與 Reference Caption 的語意相似度。

使用模型：

```text
roberta-base
```

Samples:

```text
100
```

結果：

| Metric    |  Score |
| --------- | -----: |
| Precision | 0.9021 |
| Recall    | 0.8996 |
| F1        | 0.9008 |

BERTScore F1：

```text
0.9008
```

BERTScore 與 BLEU 的主要差異在於，BLEU 比較重視文字與 N-gram 的直接重疊，而 BERTScore 使用 contextual embeddings 來衡量語意相似度。因此，即使兩個 Caption 使用不同的文字表達方式，只要語意接近，BERTScore 仍可能得到較高的分數。

BERTScore computation time:

```text
14.66 seconds
```

---

## 9. Inference Performance

模型平均每張影像的 Caption Generation Inference Time：

```text
7.0014 ms/image
```

根據平均單張影像處理時間估算：

```text
1000 / 7.0014 ≈ 142.83 images/second
```

因此理論上的單張影像處理速率約為：

```text
142.83 images/second
```

此數值是根據平均 latency 所計算的理論值，並非使用 batch inference 所測得的 sustained throughput。

---

## 10. Overall Results

| Category           | Metric               |          Result |
| ------------------ | -------------------- | --------------: |
| Training           | Best Validation Loss |          2.6657 |
| Training           | Trainable Parameters |       4,555,296 |
| Training           | Training Time        |        248.15 s |
| Caption Generation | Images Evaluated     |             100 |
| Caption Generation | Inference Time       | 7.0014 ms/image |
| BLEU               | BLEU-1               |          0.5902 |
| BLEU               | BLEU-2               |          0.3740 |
| BLEU               | BLEU-3               |          0.2293 |
| BLEU               | BLEU-4               |          0.1529 |
| BERTScore          | Precision            |          0.9021 |
| BERTScore          | Recall               |          0.8996 |
| BERTScore          | F1                   |          0.9008 |
| Gemini             | Valid Evaluations    |         12 / 20 |
| Gemini             | MATCH                |          2 / 12 |
| Gemini             | MATCH + PARTIAL      |          7 / 12 |

---

## 11. Results Discussion

從訓練結果可以觀察到，Train Loss 與 Validation Loss 隨著 Epoch 增加而下降。

Validation Loss 從第一個 Epoch 的 3.0500 降低至第五個 Epoch 的 2.6657，表示模型逐漸學習影像特徵與 Caption 之間的關係。

BLEU 評估結果中，BLEU-1 為 0.5902，而 BLEU-4 為 0.1529。隨著 N-gram 長度增加，模型與 Reference Caption 的精確文字重疊程度下降，顯示模型雖然可以學習主要物件與部分描述詞彙，但完整句子的文字結構仍與人工 Caption 有差異。

另一方面，BERTScore F1 達到 0.9008，表示從語意角度來看，生成 Caption 與 Reference Caption 具有較高的相似程度。這也說明 BLEU 與 BERTScore 衡量的是不同面向的 Caption Quality。

Gemini Evaluation 則從影像本身判斷生成 Caption 是否符合實際內容。在 12 筆有效結果中，共有 7 筆屬於 MATCH 或 PARTIAL，顯示部分生成 Caption 能夠正確描述影像中的主要內容。不過，由於其中 8 筆 Gemini request 發生 HTTP 503，因此 Gemini 結果只能作為部分樣本的輔助分析。

最後，模型平均 Inference Time 為 7.0014 ms/image，表示在 RTX 5090 上可以快速完成 Caption Generation。

---

## 12. Project Structure

```text
image_captioning/
├── train_caption.py
├── inference.py
├── gemini_judge.py
├── evaluate_caption.py
├── models/
│   └── best_caption_model.pth
├── results/
│   ├── config.json
│   ├── history.json
│   ├── generated_captions.json
│   ├── gemini_evaluation.json
│   └── advanced_evaluation.json
└── README.md
```

---

## 13. Main Files

### train_caption.py

負責：

* Dataset loading
* Vocabulary construction
* ResNet18 Encoder
* LSTM Decoder
* Model training
* Validation
* Checkpoint saving

### inference.py

負責：

* Loading trained checkpoint
* Loading test images
* Generating captions
* Measuring inference time
* Saving generated captions

### gemini_judge.py

負責：

* Loading generated captions
* Loading test images
* Sending image + caption to Gemini
* Classifying MATCH / PARTIAL / MISMATCH
* Recording evaluation results

### evaluate_caption.py

負責：

* BLEU-1
* BLEU-2
* BLEU-3
* BLEU-4
* BERTScore
* Inference time summary
* Saving advanced evaluation results

---

## 14. Output Files

Training results:

```text
results/history.json
results/config.json
```

Generated captions:

```text
results/generated_captions.json
```

Gemini evaluation:

```text
results/gemini_evaluation.json
```

Advanced evaluation:

```text
results/advanced_evaluation.json
```

---

## 15. Conclusion

本實驗完成了一個完整的 Image Captioning Pipeline：

```text
                 Input Image
                      │
                      ▼
                ResNet18
                 Encoder
                      │
                      ▼
              Image Features
                      │
                      ▼
                 LSTM
                 Decoder
                      │
                      ▼
             Generated Caption
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
       Gemini              Reference Captions
          │                       │
          │                ┌──────┴──────┐
          │                ▼             ▼
          │              BLEU       BERTScore
          │
          ▼
 Image-Caption Consistency
```

本作業成功完成：

* Image Captioning model training
* Image caption generation
* Gemini image-caption evaluation
* BLEU evaluation
* BERTScore evaluation
* Inference time measurement

最終模型在測試資料上得到 BLEU-1 0.5902、BLEU-4 0.1529，以及 BERTScore F1 0.9008，平均 Caption Generation Inference Time 為 7.0014 ms/image。

整體實驗結果顯示，ResNet18 + LSTM 架構可以學習影像與自然語言之間的基本對應關係，並能夠在 GPU 上快速產生影像描述。
