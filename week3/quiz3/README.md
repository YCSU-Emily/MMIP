# Week 3 Quiz 3：提升模型泛化能力

## 1. 題目說明

本題目主要探討 Data Augmentation 對影像分類模型泛化能力的影響，並進一步利用 CNN Kernel Visualization 與 Explainable AI（XAI）分析模型的特徵學習與預測依據。

本實驗使用 Wafer Defect Dataset，共包含 9 種晶圓缺陷類別：

* Center
* Donut
* Edge-Loc
* Edge-Ring
* Local
* Near-Full
* Normal
* Random
* Scratch

本題分為兩個基礎實驗與兩個進階實驗：

### 基礎實驗

1. Plain CNN 加入 Data Augmentation，並與原始模型比較。
2. ResNet18 加入 Data Augmentation，並與原始模型比較。

### 進階實驗

1. Visualize CNN Kernels，觀察訓練後的卷積核並分析其可能學習的特徵。
2. 使用 Grad-CAM 進行 XAI 分析，觀察模型預測時主要關注的影像區域。

---

# 2. Dataset

本實驗使用 Wafer Defect Dataset，共有 9 個類別。

| Class     | Description |
| --------- | ----------- |
| Center    | 晶圓中心附近的缺陷   |
| Donut     | 環狀分布缺陷      |
| Edge-Loc  | 晶圓邊緣局部缺陷    |
| Edge-Ring | 晶圓邊緣環狀缺陷    |
| Local     | 局部區域缺陷      |
| Near-Full | 接近整片晶圓範圍的缺陷 |
| Normal    | 正常晶圓        |
| Random    | 隨機分布缺陷      |
| Scratch   | 刮痕類缺陷       |

資料集切分如下：

| Dataset    | Images |
| ---------- | -----: |
| Train      |  3,924 |
| Validation |    837 |
| Test       |    846 |
| Total      |  5,607 |

---

# 3. Data Augmentation

本實驗只對 Training Dataset 使用 Data Augmentation。

Validation 與 Test Dataset 不使用隨機資料增強，以確保模型評估結果具有一致性。

使用的 augmentation 方法如下：

```text
Resize
    ↓
Random Horizontal Flip
    ↓
Random Vertical Flip
    ↓
Random Rotation ±15°
    ↓
Color Jitter
    ↓
ToTensor
    ↓
Normalize
```

實際設定：

```python
transforms.RandomHorizontalFlip(p=0.5)
transforms.RandomVerticalFlip(p=0.5)
transforms.RandomRotation(degrees=15)
transforms.ColorJitter(
    brightness=0.15,
    contrast=0.15
)
```

### Augmentation 策略說明

#### Random Horizontal Flip

以 50% 機率進行水平翻轉。

主要目的為增加不同方向的樣本，使模型降低對特定水平方向排列的依賴。

#### Random Vertical Flip

以 50% 機率進行垂直翻轉。

晶圓缺陷影像具有較強的空間結構，因此透過垂直翻轉可以增加缺陷位置與方向的變化。

#### Random Rotation

隨機旋轉角度設定為 ±15°。

目的為增加晶圓影像的方向變化，使模型降低對固定旋轉角度的依賴。

#### Color Jitter

Brightness 與 Contrast 均設定為 0.15。

目的為模擬不同影像亮度與對比度，降低模型對特定影像成像條件的依賴。

---

# 4. Baseline Results

Quiz 2 的原始模型結果作為本次實驗的 Baseline。

| Model     | Test Top-1 | Test Top-5 | Macro-AUC |
| --------- | ---------: | ---------: | --------: |
| Plain CNN |     98.23% |       100% |  0.999632 |
| ResNet18  |     98.70% |       100% |  0.999728 |

---

# 5. Plain CNN + Data Augmentation

## 5.1 Model

Plain CNN 使用四個 Convolution Block：

```text
Input
  ↓
Conv 3 → 32
  ↓
Conv 32 → 64
  ↓
Conv 64 → 128
  ↓
Conv 128 → 256
  ↓
Adaptive Average Pooling
  ↓
Fully Connected
  ↓
9 Classes
```

模型參數量：

```text
391,689 parameters
```

---

## 5.2 Training Result

加入 Data Augmentation 後：

* Best Validation Top-1 Accuracy：97.37%
* Final Validation Top-1 Accuracy：97.01%
* Training Top-1 Accuracy：98.42%
* Validation Top-5 Accuracy：100%

最佳模型位於：

```text
results/plain_aug/best_model.pth
```

---

## 5.3 Test Result

| Model                    | Test Top-1 | Test Top-5 |
| ------------------------ | ---------: | ---------: |
| Plain CNN Baseline       |     98.23% |       100% |
| Plain CNN + Augmentation |     98.94% |       100% |
| Improvement              |   +0.71 pp |       0 pp |

加入 Data Augmentation 後，Test Top-1 Accuracy 從 98.23% 提升至 98.94%，增加 **0.71 percentage points**。

這表示資料增強後，Plain CNN 在未看過的 Test Dataset 上具有較好的分類表現。

---

# 6. ResNet18 + Data Augmentation

## 6.1 Model

本實驗使用 ResNet18 作為經典 CNN 架構。

最後的 Fully Connected Layer 修改為 9 個輸出類別：

```text
ResNet18
    ↓
Global Average Pooling
    ↓
Fully Connected
    ↓
9 Classes
```

模型參數量：

```text
11,181,129 parameters
```

---

## 6.2 Training Result

加入 Data Augmentation 後：

* Best Validation Top-1 Accuracy：99.64%
* Final Validation Top-1 Accuracy：99.52%
* Training Top-1 Accuracy：99.85%
* Validation Top-5 Accuracy：100%

最佳模型位於：

```text
results/resnet18_aug/best_model.pth
```

---

## 6.3 Test Result

| Model                   | Test Top-1 | Test Top-5 |
| ----------------------- | ---------: | ---------: |
| ResNet18 Baseline       |     98.70% |       100% |
| ResNet18 + Augmentation |     99.29% |       100% |
| Improvement             |   +0.59 pp |       0 pp |

加入 Data Augmentation 後，Test Top-1 Accuracy 從 98.70% 提升至 99.29%，增加 **0.59 percentage points**。

---

# 7. Data Augmentation Comparison

整體比較如下：

| Model     | Baseline | +Augmentation | Improvement |
| --------- | -------: | ------------: | ----------: |
| Plain CNN |   98.23% |        98.94% |    +0.71 pp |
| ResNet18  |   98.70% |        99.29% |    +0.59 pp |

兩個模型加入 Data Augmentation 後，Test Top-1 Accuracy 都有所提升。

Plain CNN 的提升幅度為 0.71 percentage points，而 ResNet18 的提升幅度為 0.59 percentage points。

由於兩個模型的 Baseline Accuracy 已經相當高，因此提升幅度相對有限，但結果顯示本實驗設定的 Data Augmentation 對 Test Dataset 的分類表現具有正向影響。

---

# 8. Advanced 1：CNN Kernel Visualization

本實驗使用訓練完成的 Plain CNN：

```text
results/plain_aug/best_model.pth
```

分析第一層 Convolution Layer 的兩個 Kernel。

第一層卷積層：

```python
nn.Conv2d(
    3,
    32,
    kernel_size=3,
    padding=1
)
```

因此共有：

```text
32 kernels
```

每個 Kernel 的尺寸為：

```text
3 × 3 × 3
```

其中：

* 3：RGB 三個輸入通道
* 3 × 3：空間卷積範圍

---

## 8.1 Kernel 0

Kernel 0 的權重同時包含正值與負值。

其統計資訊：

```text
Minimum : -0.2272
Maximum :  0.1910
Mean    : -0.0017
Std     :  0.1143
```

正負權重的組合代表此 Kernel 可以對局部像素之間的差異產生不同程度的響應。

因此可以推測它可能與以下低階特徵有關：

* 局部亮度差異
* 色彩對比
* 局部紋理
* 缺陷邊界

Kernel 0：

```text
results/plain_aug/kernel_0.png
```

---

## 8.2 Kernel 1

Kernel 1 同樣具有正值與負值。

統計資訊：

```text
Minimum : -0.2124
Maximum :  0.1589
Mean    : -0.0039
Std     :  0.1204
```

其標準差略高於 Kernel 0，代表此 Kernel 的權重變化程度略大。

Kernel 1 可能對另一種局部紋理、亮度變化或方向性特徵產生不同響應。

Kernel 1：

```text
results/plain_aug/kernel_1.png
```

---

## 8.3 Kernel Comparison

兩個 Kernel 的比較：

```text
results/plain_aug/kernels_comparison.png
```

可以觀察到兩個 Kernel 都具有正負交錯的權重，而不是所有權重都集中於相同方向。

這表示不同卷積核可以學習不同的局部特徵。

需要注意的是，僅透過 Kernel 權重本身，無法嚴格證明某一個 Kernel 就是特定方向的 Edge Detector。

因此本實驗將其解釋為：

> 第一層 CNN Kernel 主要負責學習影像中的低階特徵，例如局部亮度變化、色彩對比、紋理與缺陷邊界等。

---

# 9. Advanced 2：XAI / Grad-CAM

為了理解模型進行分類時主要關注哪些影像區域，本實驗使用 Grad-CAM（Gradient-weighted Class Activation Mapping）進行 Explainable AI 分析。

使用的模型：

```text
ResNet18 + Data Augmentation
```

Checkpoint：

```text
results/resnet18_aug/best_model.pth
```

Grad-CAM Target Layer：

```python
model.layer4[-1]
```

Grad-CAM 會利用指定分類結果對最後高階卷積特徵的梯度，計算不同 Feature Map 的重要程度，再產生 Class Activation Map。

流程如下：

```text
Input Image
     ↓
ResNet18
     ↓
Layer4
     ↓
Feature Maps
     ↓
Backward Gradient
     ↓
Global Average Pooling
     ↓
Weighted Feature Maps
     ↓
ReLU
     ↓
Grad-CAM Heatmap
```

---

# 10. Grad-CAM Results

本實驗從 Test Dataset 每個類別選擇一張影像，共分析 9 張影像。

結果如下：

| Class     | Prediction | Confidence | Main Attention       |
| --------- | ---------- | ---------: | -------------------- |
| Center    | Center     |     99.99% | Upper-Left / Center  |
| Donut     | Donut      |    100.00% | Center               |
| Edge-Loc  | Edge-Loc   |     99.97% | Lower-Right          |
| Edge-Ring | Edge-Ring  |     99.15% | Center / Lower       |
| Local     | Local      |     99.99% | Upper-Right          |
| Near-Full | Near-Full  |    100.00% | Upper-Left           |
| Normal    | Normal     |     99.95% | Center / Lower-Right |
| Random    | Random     |     99.77% | Upper-Left / Center  |
| Scratch   | Scratch    |     99.99% | Lower region         |

9 張影像全部分類正確：

```text
Correct predictions: 9 / 9
Accuracy: 100%
```

---

# 11. Grad-CAM Spatial Analysis

本實驗進一步將 Grad-CAM 分成 3 × 3 spatial grid，觀察不同區域的 activation score。

例如：

### Edge-Loc

最高區域位於右下方：

```text
0.121  0.510  0.832
```

右下區域具有較高 activation。

### Edge-Ring

模型在中間與下方區域具有較高 activation：

```text
0.436  0.664  0.938
0.645  0.914  0.811
```

### Local

右上區域具有較高 activation：

```text
0.247  0.372  0.838
0.065  0.127  0.550
```

### Scratch

下方區域具有較高 activation：

```text
0.192  0.874  0.771
```

整體而言，不同類別的模型 activation 分布並不完全相同，顯示模型會根據輸入影像的特徵，在不同空間區域產生不同程度的響應。

---

# 12. XAI 分析注意事項

Grad-CAM 的結果主要用來說明模型預測時的高響應區域，而不是直接提供像素等級的缺陷 segmentation。

因此，本實驗將 Grad-CAM 視為：

```text
Classification Explanation
```

而不是：

```text
Pixel-level Defect Segmentation
```

此外，Grad-CAM 的結果會受到選擇的 Target Layer、模型架構與輸入影像影響，因此不能僅根據單一圖片推論模型對所有資料的行為。

---

# 13. Final Discussion

本實驗主要探討 Data Augmentation、CNN Kernel 與 XAI 三個面向。

首先，在 Data Augmentation 實驗中，Plain CNN 的 Test Top-1 Accuracy 從 98.23% 提升至 98.94%，ResNet18 則從 98.70% 提升至 99.29%。

這表示在本實驗設定下，透過水平翻轉、垂直翻轉、旋轉與亮度／對比度變化，可以增加訓練資料的變化程度，並提升 Test Dataset 的分類表現。

其次，透過 Kernel Visualization 可以觀察 CNN 第一層卷積核的實際權重。不同 Kernel 具有不同的正負權重組合，代表 CNN 可以從輸入影像中學習不同的局部特徵。

最後，Grad-CAM 顯示不同類別的影像會產生不同的空間 activation 分布。9 張測試影像皆被正確分類，且模型對各影像產生不同的高響應區域，因此可以利用 Grad-CAM 作為輔助工具，了解 CNN 預測時所使用的高階特徵位置。

---

# 14. Conclusion

本次 Quiz 3 完成以下工作：

* 建立 Plain CNN Data Augmentation 實驗
* 建立 ResNet18 Data Augmentation 實驗
* 比較 augmentation 前後的 Test Accuracy
* Visualize 第一層 CNN Kernel
* 分析 Kernel 的正負權重與可能學習的低階特徵
* 使用 Grad-CAM 進行 XAI 分析
* 分析不同類別影像的 spatial attention
* 建立完整的實驗結果與分析

最終結果：

```text
Plain CNN:
98.23% → 98.94%

ResNet18:
98.70% → 99.29%
```

兩種模型在加入 Data Augmentation 後，Test Top-1 Accuracy 均有所提升。

---

# 15. Output Files

## Plain CNN + Augmentation

```text
results/plain_aug/
├── best_model.pth
├── history.json
├── config.json
├── test_predictions.csv
├── confusion_matrix.png
├── roc_curves.png
├── metrics.json
├── kernel_0.png
├── kernel_1.png
└── kernels_comparison.png
```

## ResNet18 + Augmentation

```text
results/resnet18_aug/
├── best_model.pth
├── history.json
├── config.json
├── test_predictions.csv
├── confusion_matrix.png
├── roc_curves.png
├── metrics.json
└── gradcam/
    ├── sample_00_original.png
    ├── sample_00_heatmap.png
    ├── sample_00_overlay.png
    ├── sample_01_original.png
    ├── sample_01_heatmap.png
    ├── sample_01_overlay.png
    ├── ...
    └── gradcam_results.csv
```

---

# 16. How to Run

## Plain CNN + Augmentation

```bash
python train.py \
    --model plain \
    --device cuda \
    --augment \
    --output-dir results/plain_aug
```

Evaluation：

```bash
python evaluate.py \
    --checkpoint results/plain_aug/best_model.pth \
    --data-root ../dataset/wafer \
    --device cuda \
    --output-dir results/plain_aug
```

---

## ResNet18 + Augmentation

```bash
python train.py \
    --model resnet18 \
    --device cuda \
    --augment \
    --output-dir results/resnet18_aug
```

Evaluation：

```bash
python evaluate.py \
    --checkpoint results/resnet18_aug/best_model.pth \
    --data-root ../dataset/wafer \
    --device cuda \
    --output-dir results/resnet18_aug
```

---

## Kernel Visualization

```bash
python visualize_kernels.py
```

---

## Grad-CAM

```bash
python gradcam.py
```

---

## Grad-CAM Numerical Analysis

```bash
python analyze_gradcam.py
```

---

# 17. Environment

主要使用：

```text
Python
PyTorch
Torchvision
NumPy
Pillow
Matplotlib
Scikit-learn
CUDA
```

模型訓練使用 NVIDIA GPU 加速。
