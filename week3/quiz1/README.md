# Quiz 1：準備影像資料集

MMIP（Multi-Modality Image Processing）Week 3

---

## 1. 題目說明

本次實驗使用 Wafer Defect Dataset，建立晶圓缺陷影像分類任務。

實驗目標是將輸入的晶圓影像分類至 9 個不同的晶圓類別，並建立 Training、Validation 與 Testing Dataset，供後續 CNN 影像分類模型訓練與評估使用。

---

## 2. 資料集來源與內容

本實驗使用 Wafer Defect Dataset。

資料集中包含晶圓影像，每張影像皆屬於一個特定類別。

本資料集共有：

- 9 個分類
- 5,607 張影像
- 所有影像格式為 JPG
- 每個類別各有 623 張影像

### Dataset Classes

| Class | Number of Images |
|---|---:|
| Center | 623 |
| Donut | 623 |
| Edge-Loc | 623 |
| Edge-Ring | 623 |
| Local | 623 |
| Near-Full | 623 |
| Normal | 623 |
| Random | 623 |
| Scratch | 623 |
| **Total** | **5,607** |

由於每個類別皆具有相同數量的影像，因此本資料集屬於 balanced dataset。

---

## 3. 資料品質檢查

在進行資料切分前，先檢查所有影像的檔案格式與完整性。

資料集中共有：

```text
5,607 JPG images
