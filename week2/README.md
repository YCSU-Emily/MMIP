# Quiz 1: Machine Learning Classification

MMIP (Multi-Modality Image Processing) — Week 2

## 1. Task Description

| Requirement | Description                                                                            |
| ----------- | -------------------------------------------------------------------------------------- |
| Basic       | Select a binary classification dataset with at least 5 input features.                 |
| Basic       | Split the dataset into Training Dataset and Validation Dataset.                        |
| Basic       | Apply appropriate Feature Scaling.                                                     |
| Basic       | Train a Machine Learning Classification model using Scikit-Learn.                      |
| Basic       | Obtain classification probabilities and determine a Classification Threshold.          |
| Basic       | Generate a Confusion Matrix and calculate Accuracy, Precision, Recall, and F1-Score.   |
| Basic       | Adjust the Classification Threshold and evaluate the model again.                      |
| Advanced    | Train a second Machine Learning Classification model using the same dataset.           |
| Advanced    | Determine an appropriate Threshold for the second model.                               |
| Advanced    | Compare Accuracy, Precision, Recall, and F1-Score using the same Validation Dataset.   |
| Advanced    | Analyze the differences between Precision and Recall and identify the types of errors. |

---

## 2. Environment Setup (Anaconda + Git + Jupyter)

### Create Anaconda Environment

```bash
conda create -n mmip python=3.10 -y
conda activate mmip
```

### Install Required Packages

```bash
pip install numpy pandas scikit-learn matplotlib jupyter
```

### Clone the GitHub Repository

```bash
git clone https://github.com/YCSU-Emily/MMIP.git
cd MMIP/week2/quiz1
```

---

## 3. How to Run

### Run the Program

```bash
python3 quiz1.py
```

The program reads the binary classification dataset:

```text
churn_dataset.csv
```

The directory structure is:

```text
quiz1/
├── quiz1.py
├── churn_dataset.csv
├── confusion_matrices.png
├── roc_curve.png
├── metrics_comparison.png
└── model_comparison.csv
```

### Jupyter Notebook

The Python code can also be executed in Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
quiz1.py
```

or copy the code into a Jupyter Notebook cell.

---

## 4. Method

The overall processing pipeline is:

**Read Dataset → Train/Validation Split → Feature Scaling → Train Classification Models → Obtain Prediction Probabilities → Select Threshold → Generate Confusion Matrix → Calculate Metrics → Compare Models**

### 4.1 Dataset

A binary classification dataset named:

```text
churn_dataset.csv
```

is used.

The target column is:

```text
churn
```

The remaining columns are used as input features.

The dataset is divided into:

* Training Dataset: 75%
* Validation Dataset: 25%

The split uses:

```python
random_state = 42
stratify = y
```

Using `stratify=y` keeps the class distribution approximately consistent between the Training and Validation datasets.

---

### 4.2 Feature Scaling

Standardization is applied using:

```python
StandardScaler()
```

The scaler is fitted only on the Training Dataset:

```python
scaler.fit_transform(X_train)
```

The same scaler is then applied to the Validation Dataset:

```python
scaler.transform(X_val)
```

This prevents information from the Validation Dataset from being used during training.

---

### 4.3 Model 1: Logistic Regression

The first classification model is:

```python
LogisticRegression(random_state=42)
```

The model outputs the probability of the positive class using:

```python
predict_proba()
```

Two thresholds are evaluated:

```text
0.50
0.35
```

The purpose of changing the threshold is to observe the trade-off between Precision and Recall.

---

### 4.4 Model 2: Random Forest

The second classification model is:

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    random_state=42
)
```

The model also produces classification probabilities using:

```python
predict_proba()
```

The following thresholds are evaluated:

```text
0.50
0.40
```

---

### 4.5 Classification Threshold

The prediction is determined by:

```python
y_pred = (proba >= threshold).astype(int)
```

For example:

```text
Probability >= 0.50 → Class 1
Probability < 0.50  → Class 0
```

A lower threshold generally allows more samples to be classified as the positive class, which can increase Recall while potentially reducing Precision.

---

### 4.6 Evaluation Metrics

The following metrics are used:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

The confusion matrix contains:

```text
                 Predicted
                0       1
Actual 0       TN      FP
Actual 1       FN      TP
```

The four error types are:

* **True Positive (TP)**: Positive sample correctly classified.
* **True Negative (TN)**: Negative sample correctly classified.
* **False Positive (FP)**: Negative sample incorrectly classified as positive.
* **False Negative (FN)**: Positive sample incorrectly classified as negative.

---

## 5. Basic Task Results

The program generates four confusion matrices:

| Model               | Threshold |
| ------------------- | --------: |
| Logistic Regression |      0.50 |
| Logistic Regression |      0.35 |
| Random Forest       |      0.50 |
| Random Forest       |      0.40 |

The output is saved as:

```text
confusion_matrices.png
```

The numerical model comparison is stored in:

```text
model_comparison.csv
```

The CSV contains:

```text
Model
Accuracy
Precision
Recall
F1-Score
```

---

## 6. Advanced Task: Model and Threshold Comparison

### 6.1 Logistic Regression

Logistic Regression is evaluated using:

```text
Threshold = 0.50
Threshold = 0.35
```

Changing the threshold allows the effect on Precision and Recall to be observed.

A lower threshold classifies more samples as positive, which can increase the number of True Positives while also increasing False Positives.

---

### 6.2 Random Forest

Random Forest is evaluated using:

```text
Threshold = 0.50
Threshold = 0.40
```

Random Forest uses multiple decision trees and combines their predictions to produce the final classification probability.

---

### 6.3 Model Comparison

The generated comparison table is stored in:

```text
model_comparison.csv
```

The metrics are visualized in:

```text
metrics_comparison.png
```

The bar chart compares:

* Accuracy
* Precision
* Recall
* F1-Score

for the different model and threshold combinations.

---

### 6.4 Precision and Recall Analysis

Precision measures how many predicted positive samples are actually positive:

```text
Precision = TP / (TP + FP)
```

Recall measures how many actual positive samples are successfully detected:

```text
Recall = TP / (TP + FN)
```

Therefore:

* Increasing the threshold generally makes the model more conservative when predicting the positive class.
* Decreasing the threshold generally allows more positive predictions.
* A lower threshold may improve Recall but also increase False Positives and reduce Precision.

The actual metric values can be found in:

```text
model_comparison.csv
```

---

## 7. ROC Curve

Although ROC/AUC is formally introduced in Quiz 3, a ROC comparison is also generated in Quiz 1 for additional model analysis.

The program compares:

```text
Logistic Regression
Random Forest
```

using their validation-set prediction probabilities.

The output is:

```text
roc_curve.png
```

The axes are:

```text
X-axis: False Positive Rate (FPR)
Y-axis: True Positive Rate (TPR)
```

---

## 8. Output Files

| File                     | Description                                            |
| ------------------------ | ------------------------------------------------------ |
| `confusion_matrices.png` | Confusion matrices for different models and thresholds |
| `roc_curve.png`          | ROC Curve comparison                                   |
| `metrics_comparison.png` | Accuracy, Precision, Recall, and F1-Score comparison   |
| `model_comparison.csv`   | Numerical evaluation results                           |

---

## 9. Git Upload

From the repository root:

```bash
cd ~/MMIP
git status
git add week2/quiz1
git commit -m "Add Quiz 1: machine learning classification"
git push origin main
```

Check the GitHub repository:

```text
https://github.com/YCSU-Emily/MMIP
```

---

## 10. Using Google Colab

Install the required packages:

```python
!pip install numpy pandas scikit-learn matplotlib
```

Clone the repository:

```python
!git clone https://github.com/YCSU-Emily/MMIP.git
```

Enter the Quiz 1 directory:

```python
%cd MMIP/week2/quiz1
```

Run the program:

```python
!python3 quiz1.py
```

If the dataset is not included in the repository, upload:

```text
churn_dataset.csv
```

to the Quiz 1 directory before running the program.

---

## 11. Environment

* Python 3.10
* NumPy
* Pandas
* Scikit-Learn
* Matplotlib
* Jupyter Notebook
* Anaconda
* Ubuntu Linux
* Git / GitHub

---
# Quiz 2: Deep Learning Credit Card Default Prediction

MMIP (Multi-Modality Image Processing) — Week 2

## 1. Task Description

| Requirement | Description                                                                          |
| ----------- | ------------------------------------------------------------------------------------ |
| Basic       | Use the specified Kaggle/UCI Credit Card Default dataset.                            |
| Basic       | Apply appropriate Feature Scaling.                                                   |
| Basic       | Build a Multi-Layer Perceptron using PyTorch.                                        |
| Basic       | Train the model for at least 50 epochs.                                              |
| Basic       | Demonstrate the prediction of one Validation Dataset sample.                         |
| Basic       | Plot Training Loss and Validation Loss.                                              |
| Basic       | Explain the main hyperparameters used during training.                               |
| Advanced    | Improve the original MLP using at least one model improvement strategy.              |
| Advanced    | Retrain the improved model.                                                          |
| Advanced    | Compare Training Loss, Validation Loss, Accuracy, Precision, Recall, and F1-Score.   |
| Advanced    | Analyze whether the improvement reduces overfitting and makes the model more stable. |

---

## 2. Environment Setup (Anaconda + Git + Jupyter)

### Create Anaconda Environment

```bash
conda create -n mmip python=3.10 -y
conda activate mmip
```

### Install Required Packages

```bash
pip install numpy pandas scikit-learn matplotlib jupyter torch
```

### Clone the GitHub Repository

```bash
git clone https://github.com/YCSU-Emily/MMIP.git
cd MMIP/week2/quiz2
```

---

## 3. How to Run

The main program is:

```text
mlp_credit_card.py
```

Run:

```bash
python3 mlp_credit_card.py
```

The directory structure is:

```text
quiz2/
├── mlp_credit_card.py
├── UCI_Credit_Card.csv
├── loss_curves.png
├── mlp_metrics_comparison.png
└── mlp_comparison.csv
```

### Jupyter Notebook

Jupyter can also be used:

```bash
jupyter notebook
```

Then open or copy the Python code into a Notebook.

---

## 4. Method

The overall processing pipeline is:

**Read Dataset → Remove ID → Train/Validation Split → Feature Scaling → Convert to Tensor → Build MLP → Train Baseline → Evaluate → Improve MLP → Early Stopping → Threshold Analysis → Compare Results**

---

### 4.1 Dataset

The dataset is:

```text
UCI_Credit_Card.csv
```

The target column is:

```text
default.payment.next.month
```

where:

```text
0 = No default
1 = Default
```

The `ID` column is removed because it does not provide useful predictive information:

```python
df = df.drop(columns=["ID"])
```

The dataset contains:

```text
30,000 samples
23 features
```

The positive class ratio is:

```text
22.12%
```

---

### 4.2 Training / Validation Split

The dataset is divided into:

```text
Training Dataset:   24,000 samples
Validation Dataset:  6,000 samples
```

The split ratio is:

```text
80% Training
20% Validation
```

The split uses:

```python
random_state=42
stratify=y
```

This preserves a similar class distribution between the Training and Validation datasets.

---

### 4.3 Feature Scaling

Standardization is performed using:

```python
StandardScaler()
```

The scaler is fitted only using the Training Dataset:

```python
X_train_scaled = scaler.fit_transform(X_train)
```

The Validation Dataset is then transformed using the same scaler:

```python
X_val_scaled = scaler.transform(X_val)
```

This avoids information leakage from the Validation Dataset.

---

## 5. Baseline MLP

### 5.1 Network Architecture

The baseline MLP uses:

```text
Input Layer
    ↓
64 Neurons
    ↓
32 Neurons
    ↓
16 Neurons
    ↓
1 Output Logit
```

Each hidden layer uses:

```text
Linear → ReLU
```

The baseline model does not use Dropout or L2 regularization.

---

### 5.2 Hyperparameters

| Parameter     |             Value |
| ------------- | ----------------: |
| Epochs        |                80 |
| Batch Size    |               128 |
| Learning Rate |             0.001 |
| Optimizer     |              Adam |
| Hidden Layers |      64 → 32 → 16 |
| Loss Function | BCEWithLogitsLoss |
| Random Seed   |                42 |
| Dropout       |                 0 |
| Weight Decay  |                 0 |

The model was trained for 80 epochs, which satisfies the requirement of at least 50 epochs.

---

## 6. Basic Task Results

### 6.1 Baseline Training

The baseline model produced the following losses:

| Epoch | Training Loss | Validation Loss |
| ----: | ------------: | --------------: |
|     1 |        0.5233 |          0.4629 |
|    10 |        0.4224 |          0.4401 |
|    20 |        0.4158 |          0.4382 |
|    30 |        0.4104 |          0.4475 |
|    40 |        0.4054 |          0.4454 |
|    50 |        0.3993 |          0.4478 |
|    60 |        0.3947 |          0.4578 |
|    70 |        0.3892 |          0.4644 |
|    80 |        0.3852 |          0.4653 |

The Training Loss continuously decreases, while the Validation Loss starts increasing after approximately the middle of training.

This indicates a tendency toward overfitting in the later epochs.

---

### 6.2 Baseline Evaluation

At the default threshold of:

```text
0.50
```

the Baseline MLP achieved:

| Metric    | Baseline MLP |
| --------- | -----------: |
| Accuracy  |       0.8150 |
| Precision |       0.6468 |
| Recall    |       0.3602 |
| F1-Score  |       0.4627 |

Confusion Matrix:

```text
[[4412  261]
 [ 849  478]]
```

Therefore:

```text
TN = 4412
FP = 261
FN = 849
TP = 478
```

---

### 6.3 Single Validation Sample Prediction

The program also demonstrates one actual Validation Dataset sample.

For:

```text
Validation sample index = 0
```

the first five original features are:

```text
[50000, 1, 2, 2, 46]
```

The predicted default probability is:

```text
0.1238
```

Using:

```text
Threshold = 0.50
```

the prediction is:

```text
Predicted class = 0
Actual class    = 0
```

The sample is therefore correctly classified.

---

## 7. Advanced Task: Improved MLP

The improved model uses three regularization strategies:

```text
Dropout
L2 Regularization
Early Stopping
```

---

### 7.1 Dropout

The Dropout rate is:

```text
0.2
```

Dropout randomly disables a portion of neurons during training.

This helps reduce the dependence on individual neurons and can reduce overfitting.

---

### 7.2 L2 Regularization

L2 regularization is implemented through Adam's:

```python
weight_decay=5e-5
```

This penalizes excessively large model weights.

---

### 7.3 Early Stopping

The Early Stopping parameters are:

```text
Patience = 15
```

Training stops when the Validation Loss does not improve for the specified number of epochs.

The improved model stopped at:

```text
Epoch 40
```

with the best Validation Loss:

```text
0.4323
```

---

## 8. Improved MLP Results

### 8.1 Improved Model at Threshold = 0.50

| Metric          | Improved MLP |
| --------------- | -----------: |
| Accuracy        |       0.8217 |
| Precision       |       0.6748 |
| Recall          |       0.3738 |
| F1-Score        |       0.4811 |
| Validation Loss |       0.4333 |

Confusion Matrix:

```text
[[4434  239]
 [ 831  496]]
```

Compared with the Baseline MLP, the Validation Loss is lower and the Accuracy, Precision, Recall, and F1-Score are all higher at the 0.50 threshold.

---

### 8.2 Threshold Optimization

Because the dataset is imbalanced, the program also searches for a better classification threshold.

The threshold is scanned from:

```text
0.05 to 0.50
```

with:

```text
step = 0.01
```

The threshold that maximizes F1-Score is:

```text
Best Threshold = 0.30
```

The results are:

| Metric    | Improved MLP, Threshold = 0.30 |
| --------- | -----------------------------: |
| Accuracy  |                         0.8002 |
| Precision |                         0.5519 |
| Recall    |                         0.5124 |
| F1-Score  |                         0.5315 |

Confusion Matrix:

```text
[[4121  552]
 [ 647  680]]
```

The lower threshold increases Recall from:

```text
0.3738 → 0.5124
```

while Precision changes from:

```text
0.6748 → 0.5519
```

The F1-Score increases from:

```text
0.4811 → 0.5315
```

This demonstrates the trade-off between Precision and Recall when changing the classification threshold.

---

## 9. Baseline vs Improved Comparison

The complete comparison is saved in:

```text
mlp_comparison.csv
```

### Comparison

| Model               | Train Loss | Val Loss | Accuracy | Precision | Recall | F1-Score |
| ------------------- | ---------: | -------: | -------: | --------: | -----: | -------: |
| Baseline MLP (0.50) |     0.3852 |   0.4653 |   0.8150 |    0.6468 | 0.3602 |   0.4627 |
| Improved MLP (0.50) |     0.4274 |   0.4333 |   0.8217 |    0.6748 | 0.3738 |   0.4811 |
| Improved MLP (0.30) |     0.4274 |   0.4333 |   0.8002 |    0.5519 | 0.5124 |   0.5315 |

---

## 10. Loss Curve Analysis

The output file:

```text
loss_curves.png
```

contains:

1. Baseline MLP Training Loss vs Validation Loss
2. Improved MLP Training Loss vs Validation Loss

The Baseline MLP continues training until Epoch 80, while its Validation Loss begins to increase after reaching a lower point.

The Improved MLP uses Dropout, L2 Regularization, and Early Stopping. It stops at Epoch 40, with a lower Validation Loss.

This indicates that the regularization strategies help control overfitting and prevent unnecessary training after the validation performance stops improving.

---

## 11. Output Files

| File                         | Description                                          |
| ---------------------------- | ---------------------------------------------------- |
| `loss_curves.png`            | Training and Validation Loss comparison              |
| `mlp_metrics_comparison.png` | Accuracy, Precision, Recall, and F1-Score comparison |
| `mlp_comparison.csv`         | Numerical comparison of Baseline and Improved MLP    |

---

## 12. Git Upload

From the repository root:

```bash
cd ~/MMIP
git status
git add week2/quiz2
git commit -m "Add Quiz 2: credit card default prediction with PyTorch MLP"
git push origin main
```

---

## 13. Using Google Colab

Install the required packages:

```python
!pip install numpy pandas scikit-learn matplotlib torch
```

Clone the repository:

```python
!git clone https://github.com/YCSU-Emily/MMIP.git
```

Enter the Quiz 2 directory:

```python
%cd MMIP/week2/quiz2
```

Run the program:

```python
!python3 mlp_credit_card.py
```

Upload:

```text
UCI_Credit_Card.csv
```

if the dataset is not already available in the repository.

---

## 14. Environment

* Python 3.10
* PyTorch
* NumPy
* Pandas
* Scikit-Learn
* Matplotlib
* Jupyter Notebook
* Anaconda
* Ubuntu Linux
* CPU execution

Note: The current execution environment detected an older NVIDIA driver, so PyTorch automatically used the CPU.

---
# Quiz 3: Model Performance Evaluation — ROC Curve and AUC

MMIP (Multi-Modality Image Processing) — Week 2

## 1. Task Description

| Requirement | Description                                                                                                   |
| ----------- | ------------------------------------------------------------------------------------------------------------- |
| Basic       | Continue using the Credit Card Default Dataset and MLP model from Quiz 2.                                     |
| Basic       | Obtain prediction scores/probabilities from the MLP on the Validation Dataset.                                |
| Basic       | Plot the MLP ROC Curve with FPR on the X-axis and TPR on the Y-axis.                                          |
| Basic       | Calculate the MLP AUC and display the value on the ROC Curve.                                                 |
| Basic       | Explain the meaning of ROC Curve and AUC.                                                                     |
| Advanced    | Use the same Training Dataset and Validation Dataset to build a second Machine Learning Classification model. |
| Advanced    | Obtain prediction probabilities from the second model.                                                        |
| Advanced    | Plot both ROC Curves on the same figure.                                                                      |
| Advanced    | Calculate and display the AUC of both models.                                                                 |
| Advanced    | Compare the ROC Curves and AUC values.                                                                        |

---

## 2. Environment Setup (Anaconda + Git + Jupyter)

### Create Anaconda Environment

```bash
conda create -n mmip python=3.10 -y
conda activate mmip
```

### Install Required Packages

```bash
pip install numpy pandas scikit-learn matplotlib jupyter torch
```

### Clone the GitHub Repository

```bash
git clone https://github.com/YCSU-Emily/MMIP.git
cd MMIP/week2/quiz3
```

---

## 3. How to Run

The main program is:

```text
roc_auc_analysis.py
```

Run:

```bash
python3 roc_auc_analysis.py
```

The directory structure is:

```text
quiz3/
├── roc_auc_analysis.py
├── UCI_Credit_Card.csv
├── roc_curve_comparison.png
├── roc_curve_mlp_only.png
└── roc_auc_comparison.csv
```

### Jupyter Notebook

The code can also be executed using Jupyter Notebook:

```bash
jupyter notebook
```

---

## 4. Method

The overall processing pipeline is:

**Read Credit Card Dataset → Remove ID → Same Train/Validation Split as Quiz 2 → Feature Scaling → Train Improved MLP → Obtain Prediction Probability → Train Logistic Regression → Calculate ROC Curve → Calculate AUC → Compare Models**

---

### 4.1 Dataset and Data Split

The same dataset from Quiz 2 is used:

```text
UCI_Credit_Card.csv
```

The target is:

```text
default.payment.next.month
```

The same Training / Validation split is maintained:

```text
Training Dataset:   24,000 samples
Validation Dataset: 6,000 samples
```

The split uses:

```python
random_state=42
stratify=y
```

This ensures that the data partition is consistent with Quiz 2.

---

### 4.2 Feature Scaling

The same StandardScaler method from Quiz 2 is used.

The scaler is fitted using only the Training Dataset:

```python
scaler.fit_transform(X_train)
```

The Validation Dataset is then transformed:

```python
scaler.transform(X_val)
```

---

## 5. MLP Model

The MLP follows the improved model used in Quiz 2.

### Architecture

```text
Input
  ↓
64 Neurons
  ↓
32 Neurons
  ↓
16 Neurons
  ↓
Output Logit
```

Each hidden layer uses:

```text
Linear → ReLU → Dropout
```

The Dropout rate is:

```text
0.2
```

L2 regularization is implemented using:

```text
weight_decay = 5e-5
```

Early Stopping is configured with:

```text
patience = 15
```

Other training parameters:

| Parameter     |        Value |
| ------------- | -----------: |
| Epochs        |           80 |
| Batch Size    |          128 |
| Learning Rate |        0.001 |
| Optimizer     |         Adam |
| Dropout       |          0.2 |
| Weight Decay  |     5 × 10⁻⁵ |
| Hidden Layers | 64 → 32 → 16 |
| Random Seed   |           42 |

The MLP stopped early at:

```text
Epoch 48
```

with the best Validation Loss:

```text
0.4313
```

---

## 6. ROC Curve

### 6.1 Definition

The ROC Curve represents the relationship between:

```text
False Positive Rate (FPR)
```

and:

```text
True Positive Rate (TPR)
```

at different classification thresholds.

The formulas are:

```text
FPR = FP / (FP + TN)
```

```text
TPR = TP / (TP + FN)
```

The X-axis of the ROC Curve is:

```text
False Positive Rate (FPR)
```

The Y-axis is:

```text
True Positive Rate (TPR)
```

---

### 6.2 Prediction Probability

Unlike fixed-threshold classification, ROC analysis uses the continuous prediction probabilities generated by the model.

For the MLP:

```python
proba_mlp = torch.sigmoid(mlp_logits)
```

For Logistic Regression:

```python
proba_lr = log_reg.predict_proba(X_val_scaled)[:, 1]
```

These probabilities are used to calculate the ROC Curves.

---

## 7. AUC

AUC stands for:

```text
Area Under the ROC Curve
```

It summarizes the area under the ROC Curve.

A larger AUC indicates that the model's prediction scores provide better separation between positive and negative samples on the evaluated dataset.

An AUC of approximately:

```text
0.5
```

corresponds to discrimination close to random guessing.

An AUC closer to:

```text
1.0
```

indicates stronger separation between the two classes.

---

## 8. Basic Task Results

The MLP achieved:

```text
MLP AUC = 0.7734
```

The MLP ROC Curve is saved as:

```text
roc_curve_mlp_only.png
```

This figure contains:

* MLP ROC Curve
* Random Guess reference line
* FPR on the X-axis
* TPR on the Y-axis
* MLP AUC value

---

## 9. Advanced Task: MLP vs Logistic Regression

A second classification model is created using:

```text
Logistic Regression
```

The important point is that both models use:

```text
The same Training Dataset
The same Validation Dataset
The same Feature Scaling
```

This makes the ROC/AUC comparison more consistent.

---

### 9.1 Logistic Regression

The model is configured as:

```python
LogisticRegression(
    random_state=42,
    max_iter=1000
)
```

The model produces Validation Dataset probabilities using:

```python
predict_proba()
```

---

### 9.2 AUC Comparison

The final AUC values are:

| Model               |    AUC |
| ------------------- | -----: |
| MLP (Improved)      | 0.7734 |
| Logistic Regression | 0.7076 |

The complete numerical result is stored in:

```text
roc_auc_comparison.csv
```

---

## 10. ROC Curve Comparison

Both models are plotted on the same figure:

```text
roc_curve_comparison.png
```

The figure contains:

```text
MLP
Logistic Regression
Random Guess
```

The graph makes it possible to visually compare the True Positive Rate and False Positive Rate across different thresholds.

The corresponding AUC values are:

```text
MLP:
0.7734

Logistic Regression:
0.7076
```

The AUC values show the measured discrimination performance of the two models on the same Validation Dataset.

---

## 11. Interpretation

The MLP produces an AUC of:

```text
0.7734
```

while Logistic Regression produces:

```text
0.7076
```

Therefore, on this Validation Dataset, the ROC/AUC measurements indicate different levels of positive/negative sample discrimination between the two models.

The ROC Curve comparison should be considered together with the actual graph rather than relying only on a single threshold.

This is because ROC analysis evaluates model behavior across a range of classification thresholds.

---

## 12. Output Files

| File                       | Description                                              |
| -------------------------- | -------------------------------------------------------- |
| `roc_curve_mlp_only.png`   | ROC Curve for the MLP only                               |
| `roc_curve_comparison.png` | ROC Curve comparison between MLP and Logistic Regression |
| `roc_auc_comparison.csv`   | AUC values of both models                                |

---

## 13. Git Upload

From the repository root:

```bash
cd ~/MMIP
git status
git add week2/quiz3
git commit -m "Add Quiz 3: ROC curve and AUC model evaluation"
git push origin main
```

---

## 14. Using Google Colab

Install the required packages:

```python
!pip install numpy pandas scikit-learn matplotlib torch
```

Clone the repository:

```python
!git clone https://github.com/YCSU-Emily/MMIP.git
```

Enter the Quiz 3 directory:

```python
%cd MMIP/week2/quiz3
```

Make sure the following file is available:

```text
UCI_Credit_Card.csv
```

Run:

```python
!python3 roc_auc_analysis.py
```

---

## 15. Environment

* Python 3.10
* PyTorch
* NumPy
* Pandas
* Scikit-Learn
* Matplotlib
* Jupyter Notebook
* Anaconda
* Ubuntu Linux
* CPU execution

---
