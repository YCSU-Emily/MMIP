# Week 4 Quiz 3 — Vision Transformer

## 1. 題目

本作業使用 Wafer Defect 影像資料集進行多分類影像辨識，並完成：

* **Basic:** Fine-tune Vision Transformer (ViT-B/16) 進行影像分類
* **Advanced:** Fine-tune ResNet18，並比較 ViT 與 ResNet18 的 Macro-AUC

---

## 2. Dataset

本實驗使用 Week 3 Quiz 2 所建立的 Wafer Defect Dataset。

資料集共有 9 個類別：

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

資料分割如下：

| Split      | Images |
| ---------- | -----: |
| Train      |  3,924 |
| Validation |    837 |
| Test       |    846 |
| Total      |  5,607 |

資料透過 symbolic link 使用 Week 3 已整理完成的資料集。

---

## 3. Environment

| Item        | Specification           |
| ----------- | ----------------------- |
| OS          | Ubuntu                  |
| Python      | 3.10                    |
| PyTorch     | 2.14.0+cu130            |
| Torchvision | 0.29.0+cu130            |
| GPU         | NVIDIA GeForce RTX 5090 |
| CUDA        | Enabled                 |

---

# 4. Basic — ViT-B/16

## 4.1 Model

使用 torchvision 提供的 ImageNet pretrained **ViT-B/16**：

```python
from torchvision.models import vit_b_16, ViT_B_16_Weights

weights = ViT_B_16_Weights.DEFAULT
model = vit_b_16(weights=weights)
```

原本 ImageNet 的 1000-class classifier 替換為 9-class classifier：

```text
ViT-B/16
    │
    ▼
Transformer Encoder
    │
    ▼
Classification Head
    │
    ▼
9 Wafer Defect Classes
```

---

## 4.2 Training Configuration

| Parameter         | Setting                                  |
| ----------------- | ---------------------------------------- |
| Input Size        | 224 × 224                                |
| Batch Size        | 64                                       |
| Epochs            | 10                                       |
| Optimizer         | AdamW                                    |
| Learning Rate     | 1e-4                                     |
| Weight Decay      | 1e-4                                     |
| Scheduler         | CosineAnnealingLR                        |
| Data Augmentation | Random Horizontal Flip + Random Rotation |
| Loss              | Cross Entropy                            |
| Pretrained        | ImageNet                                 |

訓練過程中以 Validation Macro-AUC 作為最佳模型 checkpoint 的選擇依據。

---

## 4.3 ViT Results

Test set 結果：

| Metric              |   ViT-B/16 |
| ------------------- | ---------: |
| Parameters          | 85,805,577 |
| Accuracy            | **99.41%** |
| Precision Macro     | **99.41%** |
| Recall Macro        | **99.41%** |
| F1 Macro            | **99.41%** |
| Macro-AUC           | **1.0000** |
| Training Time       | 127.37 sec |
| Test Inference Time |   1.51 sec |
| Inference / Image   |  1.7882 ms |

最佳模型：

```text
models/best_vit_b16.pth
```

---

# 5. Advanced — ResNet18

## 5.1 Model

使用 torchvision 提供的 ImageNet pretrained **ResNet18**：

```python
from torchvision.models import resnet18, ResNet18_Weights

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
```

將原本 ImageNet 的 1000-class fully connected layer 替換為 9-class classifier：

```text
Input Image
    │
    ▼
ResNet18 Backbone
    │
    ▼
Global Average Pooling
    │
    ▼
Fully Connected Layer
    │
    ▼
9 Wafer Defect Classes
```

---

## 5.2 Training Configuration

為了讓兩種模型的比較具有一致性，ResNet18 使用與 ViT 相同的主要訓練設定：

| Parameter         | Setting                                  |
| ----------------- | ---------------------------------------- |
| Input Size        | 224 × 224                                |
| Batch Size        | 64                                       |
| Epochs            | 10                                       |
| Optimizer         | AdamW                                    |
| Learning Rate     | 1e-4                                     |
| Weight Decay      | 1e-4                                     |
| Scheduler         | CosineAnnealingLR                        |
| Data Augmentation | Random Horizontal Flip + Random Rotation |
| Loss              | Cross Entropy                            |
| Pretrained        | ImageNet                                 |

同樣以 Validation Macro-AUC 作為最佳 checkpoint 的選擇依據。

---

## 5.3 ResNet18 Results

Test set 結果：

| Metric              |   ResNet18 |
| ------------------- | ---------: |
| Parameters          | 11,181,129 |
| Accuracy            | **99.29%** |
| Precision Macro     | **99.32%** |
| Recall Macro        | **99.29%** |
| F1 Macro            | **99.29%** |
| Macro-AUC           | **0.9999** |
| Training Time       |  49.10 sec |
| Test Inference Time |   1.09 sec |
| Inference / Image   |  1.2910 ms |

最佳模型：

```text
models/best_resnet18.pth
```

最佳 checkpoint 出現在：

```text
Epoch 8
Validation Macro-AUC = 0.9998
```

---

# 6. ViT vs ResNet18 Comparison

兩種模型在相同 Wafer Defect 測試集上進行比較。

| Metric            |   ViT-B/16 |      ResNet18 |
| ----------------- | ---------: | ------------: |
| Parameters        | 85,805,577 |    11,181,129 |
| Accuracy          | **99.41%** |        99.29% |
| Precision Macro   | **99.41%** |        99.32% |
| Recall Macro      | **99.41%** |        99.29% |
| F1 Macro          | **99.41%** |        99.29% |
| **Macro-AUC**     | **1.0000** |    **0.9999** |
| Training Time     |   127.37 s |   **49.10 s** |
| Inference Time    |     1.51 s |    **1.09 s** |
| Inference / Image |  1.7882 ms | **1.2910 ms** |

---

# 7. Macro-AUC Comparison

本作業 Advanced 部分主要比較 ViT-B/16 與 ResNet18 的 Macro-AUC。

```text
ViT-B/16    ████████████████████  1.0000
ResNet18    ███████████████████▉  0.9999
```

兩種模型的 Macro-AUC 都非常接近 1.0，表示在本次 Wafer Defect 九分類測試集上，兩種模型皆具有非常高的分類能力。

實驗結果：

```text
ViT-B/16 Macro-AUC = 1.0000
ResNet18 Macro-AUC = 0.9999
```

兩者差異為：

```text
1.0000 - 0.9999 = 0.0001
```

---

# 8. Model Size and Runtime Comparison

本實驗也記錄模型參數量與執行時間。

ViT-B/16：

```text
85,805,577 parameters
Training: 127.37 sec
Inference: 1.51 sec
```

ResNet18：

```text
11,181,129 parameters
Training: 49.10 sec
Inference: 1.09 sec
```

在本次實驗設定下，ResNet18 使用較少的參數量，且量測到的訓練時間與測試推論時間也較短。

不過這些 runtime 結果會受到 GPU、batch size、資料載入與實作方式影響，因此主要用於本次實驗環境下的比較。

---

# 9. Discussion

從 Macro-AUC 來看：

```text
ViT-B/16 : 1.0000
ResNet18 : 0.9999
```

兩種模型皆取得非常高的 Macro-AUC。

ViT-B/16 使用較大的 Transformer 架構，總參數量為約 85.8M；ResNet18 則約 11.18M parameters。

在本次資料集與訓練設定下，ResNet18 的模型規模較小，同時訓練與推論時間也較短，而兩者的 Macro-AUC 差距僅為 0.0001。

因此，本實驗顯示在此 Wafer Defect dataset 上，CNN-based ResNet18 與 Transformer-based ViT-B/16 都能達到非常高的分類效能。

---

# 10. Output Files

本實驗產生以下結果：

```text
image_classification/
├── train_vit.py
├── train_resnet.py
│
├── models/
│   ├── best_vit_b16.pth
│   └── best_resnet18.pth
│
└── results/
    ├── vit/
    │   ├── config.json
    │   ├── history.json
    │   ├── metrics.json
    │   └── test_predictions.csv
    │
    └── resnet18/
        ├── config.json
        ├── history.json
        ├── metrics.json
        └── test_predictions.csv
```

模型 checkpoint (`.pth`) 與 dataset 不加入 Git repository，以避免 repository 過大。

---

# 11. How to Run

進入 image classification 目錄：

```bash
cd ~/MMIP/week4/quiz3/image_classification
```

執行 ViT：

```bash
python3 train_vit.py
```

執行 ResNet18：

```bash
python3 train_resnet.py
```

---

# 12. Conclusion

本 Quiz 完成 Vision Transformer 的影像分類 fine-tuning，並進一步使用 ResNet18 進行比較。

在相同 Wafer Defect 九分類測試集上：

```text
ViT-B/16
Macro-AUC = 1.0000

ResNet18
Macro-AUC = 0.9999
```

兩種模型皆達到非常高的分類效能。

此外，本次實驗中的 ResNet18 具有較少的參數量，並測得較短的 training time 與 inference time。這提供了 Transformer-based model 與 CNN-based model 在相同影像分類任務上的實驗比較。
