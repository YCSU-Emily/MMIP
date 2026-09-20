import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

df = pd.read_csv("/home/kevin/MMIP/week2/quiz1/churn_dataset.csv")
X = df.drop(columns=["churn"])
y = df["churn"]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

log_reg = LogisticRegression(random_state=RANDOM_STATE).fit(X_train_scaled, y_train)
rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=RANDOM_STATE).fit(X_train_scaled, y_train)

proba_lr = log_reg.predict_proba(X_val_scaled)[:, 1]
proba_rf = rf.predict_proba(X_val_scaled)[:, 1]

def plot_cm(ax, y_true, proba, thr, title):
    y_pred = (proba >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=14)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["Actual 0", "Actual 1"])
    ax.set_title(title, fontsize=11)

fig, axes = plt.subplots(2, 2, figsize=(9, 8))
plot_cm(axes[0,0], y_val, proba_lr, 0.5, "Logistic Regression (thr=0.50)")
plot_cm(axes[0,1], y_val, proba_lr, 0.35, "Logistic Regression (thr=0.35)")
plot_cm(axes[1,0], y_val, proba_rf, 0.5, "Random Forest (thr=0.50)")
plot_cm(axes[1,1], y_val, proba_rf, 0.4, "Random Forest (thr=0.40)")
plt.tight_layout()
plt.savefig("/home/kevin/MMIP/week2/quiz1/confusion_matrices.png", dpi=150)
plt.close()

# ROC curves
fig, ax = plt.subplots(figsize=(6, 5))
for proba, label in [(proba_lr, "Logistic Regression"), (proba_rf, "Random Forest")]:
    fpr, tpr, _ = roc_curve(y_val, proba)
    ax.plot(fpr, tpr, label=label)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve Comparison")
ax.legend()
plt.tight_layout()
plt.savefig("/home/kevin/MMIP/week2/quiz1/roc_curve.png", dpi=150)
plt.close()

# Metrics bar chart
summary = pd.read_csv("/home/kevin/MMIP/week2/quiz1/model_comparison.csv")
fig, ax = plt.subplots(figsize=(9, 5))
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
x = np.arange(len(summary))
width = 0.2
for i, m in enumerate(metrics):
    ax.bar(x + i*width, summary[m], width, label=m)
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(summary["Model"], rotation=15, ha="right", fontsize=8)
ax.set_ylim(0, 1.05)
ax.legend()
ax.set_title("Model & Threshold Comparison")
plt.tight_layout()
plt.savefig("/home/kevin/MMIP/week2/quiz1/metrics_comparison.png", dpi=150)
plt.close()

print("plots saved")
