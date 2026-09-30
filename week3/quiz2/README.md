# Quiz 2：訓練 CNN 影像分類模型

MMIP（Multi-Modality Image Processing）Week 3

## 1. 題目說明

本題目使用 Wafer Defect Dataset 進行多類別影像分類，目標為建立 CNN 影像分類模型，並比較：

1. 自行設計的 Plain CNN
2. 經典 CNN Backbone：ResNet18
3. 不同 Learning Rate 對模型效能的影響
4. Test Dataset 的分類結果
5. Confusion Matrix
6. ROC Curve 與 Macro-AUC
7. 模型參數量與分類效能之比較

---

## 2. Dataset

本實驗使用 Wafer Defect Dataset，共包含 9 個類別：

```text
Center
Donut
Edge-Loc
Edge-Ring
Local
Near-Full
Normal
Random
Scratch
```

### Dataset 數量

| Dataset    | Samples |
| ---------- | ------: |
| Train      |   3,924 |
| Validation |     837 |
| Test       |     846 |
| Total      |   5,607 |

每個類別共有 623 張影像。

資料切分方式：

| Split      | 每類別數量 |
| ---------- | ----: |
| Train      |   436 |
| Validation |    93 |
| Test       |    94 |

因此：

```text
9 classes × 436 = 3,924 Train
9 classes × 93  = 837 Validation
9 classes × 94  = 846 Test
```

---

# 3. 實驗環境

* OS：Ubuntu
* Python：Python 3
* Framework：PyTorch
* GPU：NVIDIA GPU
* CUDA：CUDA-enabled training
* Dataset：Wafer Defect Dataset

主要訓練參數：

```text
Epochs       = 20
Batch Size   = 64
Weight Decay = 0.0001
Optimizer    = AdamW
Scheduler    = Cosine Annealing
```

超參數實驗中僅調整 Learning Rate，其餘設定保持固定。

---

# 4. Model 1：Plain CNN

## 4.1 模型設計

Plain CNN 為自行設計的基礎卷積神經網路，使用多層 Convolution、Batch Normalization、ReLU 與 Pooling 進行影像特徵擷取，最後透過 Fully Connected Layer 完成 9 類別分類。

模型參數量：

```text
391,689 parameters
```

模型架構概念：

```text
Input Image
     │
     ▼
Conv + BN + ReLU
     │
     ▼
Pooling
     │
     ▼
Conv + BN + ReLU
     │
     ▼
Pooling
     │
     ▼
Conv + BN + ReLU
     │
     ▼
Pooling
     │
     ▼
Fully Connected
     │
     ▼
9-Class Output
```

---

# 5. Model 2：ResNet18

## 5.1 模型設計

第二個模型使用經典 CNN Backbone：**ResNet18**。

ResNet18 使用 Residual Connection 解決深層網路訓練時的梯度傳遞問題，並利用多個 Residual Block 進行特徵擷取。

本實驗將最後分類層調整為 9 個類別。

模型參數量：

```text
11,181,129 parameters
```

模型架構概念：

```text
Input Image
     │
     ▼
Initial Convolution
     │
     ▼
Residual Blocks
     │
     ├── ResBlock
     ├── ResBlock
     ├── ResBlock
     └── ResBlock
     │
     ▼
Global Average Pooling
     │
     ▼
Fully Connected
     │
     ▼
9-Class Output
```

---

# 6. Evaluation Metrics

本實驗使用以下指標：

### Top-1 Accuracy

模型預測的第一名類別是否與 Ground Truth 相同。

$$
Top\text{-}1 =
\frac{\text{Correct Predictions}}
{\text{Total Samples}}
$$

### Top-5 Accuracy

Ground Truth 是否出現在模型預測機率最高的五個類別中。

### Macro-AUC

分別計算每一個類別的 ROC-AUC，再取所有類別的平均值：

$$
Macro\text{-}AUC =
\frac{1}{N}
\sum_{i=1}^{N} AUC_i
$$

其中本 Dataset：

$$
N=9
$$

---

# 7. Baseline Results

首先使用固定的 baseline Learning Rate：

* Plain CNN：LR = 0.001
* ResNet18：LR = 0.0001

## 7.1 Baseline Comparison

| Model     | Learning Rate | Parameters | Best Val Top-1 | Test Top-1 | Test Top-5 | Macro-AUC |
| --------- | ------------: | ---------: | -------------: | ---------: | ---------: | --------: |
| Plain CNN |         0.001 |    391,689 |         96.89% |     98.23% |    100.00% |  0.999632 |
| ResNet18  |        0.0001 | 11,181,129 |         98.33% |     98.70% |    100.00% |  0.999728 |

可以觀察到兩個模型在本 Dataset 上皆具有很高的分類準確率，且兩者 Test Top-5 Accuracy 均達到 100%。

ResNet18 的參數量為 11,181,129，而 Plain CNN 為 391,689：

$$
\frac{11,181,129}{391,689}
\approx 28.55
$$

因此 ResNet18 的參數量約為 Plain CNN 的 28.55 倍。

---

# 8. Plain CNN Hyperparameter Experiment

為了分析 Learning Rate 對 Plain CNN 的影響，本實驗固定：

```text
Epochs       = 20
Batch Size   = 64
Weight Decay = 0.0001
Model        = Plain CNN
```

僅改變 Learning Rate：

```text
0.0001
0.001
0.01
```

## 8.1 Experimental Results

| Learning Rate | Best Val Top-1 | Test Top-1 | Test Top-5 | Macro-AUC | Parameters |
| ------------: | -------------: | ---------: | ---------: | --------: | ---------: |
|        0.0001 |         90.68% |     92.55% |    100.00% |  0.994131 |    391,689 |
|         0.001 |         96.89% |     98.23% |    100.00% |  0.999632 |    391,689 |
|          0.01 |         97.73% |     99.41% |    100.00% |  0.999719 |    391,689 |

## 8.2 Analysis

當 Learning Rate 設定為 0.0001 時，Test Top-1 為 92.55%，相較於其他兩組較低。

Learning Rate 提升至 0.001 後，Test Top-1 提升至 98.23%。

當 Learning Rate 進一步提升至 0.01 時，在本實驗設定下，模型最後達到：

```text
Best Validation Top-1 = 97.73%
Test Top-1             = 99.41%
Test Top-5             = 100.00%
Macro-AUC              = 0.999719
```

LR=0.01 在訓練前期的 Validation Accuracy 有較明顯波動，例如：

```text
Epoch 4 : 86.14%
Epoch 5 : 72.16%
Epoch 7 : 94.27%
Epoch 15: 97.13%
Epoch 17: 97.73%
```

隨著 Cosine Learning Rate Scheduler 逐漸降低 Learning Rate，模型後期逐步收斂。

因此本實驗顯示，**在目前 Dataset、20 epochs 與 Cosine Annealing 設定下，較大的初始 Learning Rate 可以得到較高的最終 Test Accuracy，但訓練前期的 Validation Accuracy 波動也較明顯。**

---

# 9. ResNet18 Hyperparameter Experiment

ResNet18 同樣進行 Learning Rate 實驗。

固定：

```text
Epochs       = 20
Batch Size   = 64
Weight Decay = 0.0001
Model        = ResNet18
```

Learning Rate：

```text
0.00001
0.0001
0.001
```

其中：

```text
0.0001 = Baseline
```

## 9.1 Experimental Results

| Learning Rate | Best Val Top-1 | Test Top-1 | Test Top-5 | Macro-AUC | Parameters |
| ------------: | -------------: | ---------: | ---------: | --------: | ---------: |
|       0.00001 |         97.61% |     98.35% |    100.00% |  0.999197 | 11,181,129 |
|        0.0001 |         98.33% |     98.70% |    100.00% |  0.999728 | 11,181,129 |
|         0.001 |         98.33% |     99.17% |    100.00% |  0.999580 | 11,181,129 |

## 9.2 Analysis

當 Learning Rate = 0.00001 時：

```text
Test Top-1 = 98.35%
Macro-AUC  = 0.999197
```

提高 Learning Rate 至 0.0001：

```text
Test Top-1 = 98.70%
Macro-AUC  = 0.999728
```

當 Learning Rate = 0.001：

```text
Test Top-1 = 99.17%
Macro-AUC  = 0.999580
```

因此在本次實驗設定下，三組 Learning Rate 都能達到非常高的分類準確率，而且 Test Top-5 均為 100%。

值得注意的是，Test Top-1 與 Macro-AUC 的變化並非完全同步，因此模型效能比較不應只使用單一指標判斷。

---

# 10. Model Comparison

## 10.1 Baseline Model Comparison

| Model     | Parameters | Test Top-1 | Test Top-5 | Macro-AUC |
| --------- | ---------: | ---------: | ---------: | --------: |
| Plain CNN |    391,689 |     98.23% |    100.00% |  0.999632 |
| ResNet18  | 11,181,129 |     98.70% |    100.00% |  0.999728 |

ResNet18 的參數量約為 Plain CNN 的 28.55 倍。

在 baseline 設定下：

```text
Plain CNN Test Top-1 = 98.23%
ResNet18 Test Top-1   = 98.70%
```

兩者相差：

$$
98.70-98.23=0.47
$$

即約 **0.47 percentage points**。

這顯示在本次 Wafer Dataset 上，即使 Plain CNN 的模型規模明顯較小，也能取得接近 ResNet18 的分類結果。

---

# 11. Test Dataset Prediction

兩個模型皆針對 Test Dataset 進行預測，並輸出 CSV：

### Plain CNN

```text
results/plain_gpu/evaluation/test_predictions.csv
```

### ResNet18

```text
results/resnet18_gpu/evaluation/test_predictions.csv
```

超參數實驗的預測結果也分別儲存在：

```text
results/hyper_plain_lr/lr_0001/evaluation/test_predictions.csv
results/hyper_plain_lr/lr_001/evaluation/test_predictions.csv

results/hyper_resnet18_lr/lr_00001/evaluation/test_predictions.csv
results/hyper_resnet18_lr/lr_001/evaluation/test_predictions.csv
```

每個模型皆保存 Test Dataset 的預測結果，可進一步分析錯誤分類案例。

---

# 12. Confusion Matrix

每次 Evaluation 均會產生：

```text
confusion_matrix.png
```

例如：

```text
results/plain_gpu/evaluation/confusion_matrix.png

results/resnet18_gpu/evaluation/confusion_matrix.png
```

Confusion Matrix 用於觀察不同 Wafer Defect 類別之間的錯誤分類情況。

特別可以觀察：

* Edge-Loc
* Edge-Ring
* Local
* Near-Full
* Random
* Scratch

等類別之間是否存在混淆。

---

# 13. ROC Curve

每次 Evaluation 均會產生：

```text
roc_curves.png
```

例如：

```text
results/plain_gpu/evaluation/roc_curves.png

results/resnet18_gpu/evaluation/roc_curves.png
```

本實驗共包含 9 個類別，因此 ROC Curve 分別計算：

```text
Center
Donut
Edge-Loc
Edge-Ring
Local
Near-Full
Normal
Random
Scratch
```

---

# 14. Per-Class AUC

## Plain CNN Baseline

| Class     |      AUC |
| --------- | -------: |
| Center    | 0.999972 |
| Donut     | 0.999986 |
| Edge-Loc  | 0.997977 |
| Edge-Ring | 0.998798 |
| Local     | 0.999986 |
| Near-Full | 0.999986 |
| Normal    | 1.000000 |
| Random    | 0.999986 |
| Scratch   | 1.000000 |

Macro-AUC：

```text
0.999632
```

## ResNet18 Baseline

| Class     |      AUC |
| --------- | -------: |
| Center    | 1.000000 |
| Donut     | 1.000000 |
| Edge-Loc  | 0.998783 |
| Edge-Ring | 0.998826 |
| Local     | 1.000000 |
| Near-Full | 0.999972 |
| Normal    | 1.000000 |
| Random    | 0.999972 |
| Scratch   | 1.000000 |

Macro-AUC：

```text
0.999728
```

---

# 15. Experiment Output Structure

目前 Quiz 2 的主要結果如下：

```text
week3/
└── quiz2/
    ├── train.py
    ├── evaluate.py
    ├── README.md
    │
    ├── results/
    │   │
    │   ├── plain_gpu/
    │   │   ├── best_model.pth
    │   │   ├── history.json
    │   │   ├── config.json
    │   │   └── evaluation/
    │   │       ├── test_predictions.csv
    │   │       ├── confusion_matrix.png
    │   │       ├── roc_curves.png
    │   │       └── metrics.json
    │   │
    │   ├── resnet18_gpu/
    │   │   ├── best_model.pth
    │   │   ├── history.json
    │   │   ├── config.json
    │   │   └── evaluation/
    │   │       ├── test_predictions.csv
    │   │       ├── confusion_matrix.png
    │   │       ├── roc_curves.png
    │   │       └── metrics.json
    │   │
    │   ├── hyper_plain_lr/
    │   │   ├── lr_0001/
    │   │   └── lr_001/
    │   │
    │   └── hyper_resnet18_lr/
    │       ├── lr_00001/
    │       └── lr_001/
```

Baseline 的 Learning Rate 實驗結果使用既有：

```text
Plain CNN:
results/plain_gpu/

ResNet18:
results/resnet18_gpu/
```

---

# 16. 執行方式

## 16.1 Train Plain CNN

```bash
python3 train.py \
    --model plain \
    --data-root ../dataset/wafer \
    --epochs 20 \
    --batch-size 64 \
    --lr 0.001 \
    --weight-decay 0.0001 \
    --num-workers 4 \
    --device cuda \
    --output-dir results/plain_gpu
```

## 16.2 Train ResNet18

```bash
python3 train.py \
    --model resnet18 \
    --data-root ../dataset/wafer \
    --epochs 20 \
    --batch-size 64 \
    --lr 0.0001 \
    --weight-decay 0.0001 \
    --num-workers 4 \
    --device cuda \
    --output-dir results/resnet18_gpu
```

## 16.3 Evaluation

```bash
python3 evaluate.py \
    --checkpoint results/plain_gpu/best_model.pth \
    --data-root ../dataset/wafer \
    --batch-size 64 \
    --num-workers 4 \
    --device cuda \
    --output-dir results/plain_gpu/evaluation
```

---

# 17. 結論

本實驗使用 Wafer Defect Dataset 建立兩種 CNN 影像分類模型，包括自行設計的 Plain CNN 與經典 ResNet18 Backbone。

Baseline 實驗結果顯示：

```text
Plain CNN
Parameters : 391,689
Test Top-1 : 98.23%
Test Top-5 : 100.00%
Macro-AUC  : 0.999632
```

```text
ResNet18
Parameters : 11,181,129
Test Top-1 : 98.70%
Test Top-5 : 100.00%
Macro-AUC  : 0.999728
```

此外，透過 Learning Rate 超參數實驗可以觀察不同初始 Learning Rate 對模型訓練與測試結果的影響。

Plain CNN 的三組實驗：

```text
LR = 0.0001 → Test Top-1 = 92.55%
LR = 0.001  → Test Top-1 = 98.23%
LR = 0.01   → Test Top-1 = 99.41%
```

ResNet18 的三組實驗：

```text
LR = 0.00001 → Test Top-1 = 98.35%
LR = 0.0001  → Test Top-1 = 98.70%
LR = 0.001   → Test Top-1 = 99.17%
```

所有實驗的 Test Top-5 Accuracy 均達到 100%。

整體而言，本次實驗完成了 CNN 多類別影像分類、Test Dataset 預測、Confusion Matrix、ROC/AUC 分析、模型參數量比較，以及 Learning Rate 超參數實驗，並可透過實驗結果分析模型架構與 Learning Rate 對分類效能的影響。

