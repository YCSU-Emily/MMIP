"""
Quiz 3：ROC Curve / AUC 分析
======================================================
延續 Quiz 2 的信用卡違約預測資料集與 MLP 模型，
沿用「完全相同」的 Training/Validation 切分方式與 Feature Scaling。

基礎:
  - 取得 MLP 對 Validation Set 的預測機率
  - 繪製 ROC Curve (X 軸 FPR, Y 軸 TPR)，並標示 AUC
  - 說明 ROC / AUC 的意義

進階:
  - 使用相同 Training/Validation Set，建立第二個模型 (Logistic Regression)
  - 取得其預測機率，畫在同一張 ROC 圖上並標示 AUC
  - 比較兩個模型的區分能力 (discrimination ability)

資料集: UCI_Credit_Card.csv (需與本程式放在同一資料夾)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 0. 環境設定 (與 Quiz 2 完全一致，確保 Train/Val 切分相同)
# ------------------------------------------------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# 1. 讀取資料 + 前處理 (與 Quiz 2 相同)
# ------------------------------------------------------------------
df = pd.read_csv("./UCI_Credit_Card.csv")
df = df.drop(columns=["ID"])

TARGET_COL = "default.payment.next.month"
X = df.drop(columns=[TARGET_COL]).values.astype(np.float32)
y = df[TARGET_COL].values.astype(np.float32)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Training: {X_train.shape[0]} 筆, Validation: {X_val.shape[0]} 筆")

# ------------------------------------------------------------------
# 2. Feature Scaling (與 Quiz 2 相同：只用 Training data fit)
# ------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_val_scaled = scaler.transform(X_val).astype(np.float32)

BATCH_SIZE = 128
train_ds = TensorDataset(torch.tensor(X_train_scaled), torch.tensor(y_train))
val_ds = TensorDataset(torch.tensor(X_val_scaled), torch.tensor(y_val))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

INPUT_DIM = X_train_scaled.shape[1]

# ==================================================================
# 3. 重新訓練 Quiz 2 的 MLP (Improved 版本: Dropout + L2 + Early Stopping)
# ==================================================================
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 32, 16), dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(model, train_loader, val_loader, epochs, lr, weight_decay, patience):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_running_loss += criterion(model(xb), yb).item() * xb.size(0)
        val_loss = val_running_loss / len(val_loader.dataset)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  MLP Early stopping at epoch {epoch} (best val loss={best_val_loss:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


print("\n訓練 MLP (Dropout=0.2, weight_decay=5e-5, early stopping)...")
mlp_model = MLP(INPUT_DIM, hidden_dims=(64, 32, 16), dropout=0.2)
mlp_model = train_mlp(mlp_model, train_loader, val_loader,
                       epochs=80, lr=1e-3, weight_decay=5e-5, patience=15)

mlp_model.eval()
with torch.no_grad():
    mlp_logits = mlp_model(torch.tensor(X_val_scaled).to(device))
    proba_mlp = torch.sigmoid(mlp_logits).cpu().numpy()

# ==================================================================
# 4. 第二個模型: Logistic Regression (課程介紹過的 ML 演算法)
#    使用完全相同的 Training / Validation set 與 Scaled 特徵
# ==================================================================
print("訓練 Logistic Regression...")
log_reg = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
proba_lr = log_reg.predict_proba(X_val_scaled)[:, 1]

# ==================================================================
# 5. 計算 ROC Curve / AUC
# ==================================================================
fpr_mlp, tpr_mlp, _ = roc_curve(y_val, proba_mlp)
auc_mlp = roc_auc_score(y_val, proba_mlp)

fpr_lr, tpr_lr, _ = roc_curve(y_val, proba_lr)
auc_lr = roc_auc_score(y_val, proba_lr)

print(f"\nMLP AUC = {auc_mlp:.4f}")
print(f"Logistic Regression AUC = {auc_lr:.4f}")

# ==================================================================
# 6. 繪製 ROC Curve (兩模型畫在同一張圖)
# ==================================================================
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr_mlp, tpr_mlp, label=f"MLP (AUC = {auc_mlp:.3f})", linewidth=2)
ax.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {auc_lr:.3f})", linewidth=2)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess (AUC = 0.5)")
ax.set_xlabel("False Positive Rate (FPR)")
ax.set_ylabel("True Positive Rate (TPR)")
ax.set_title("ROC Curve: MLP vs Logistic Regression")
ax.legend(loc="lower right")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
plt.tight_layout()
plt.savefig("./roc_curve_comparison.png", dpi=150)
plt.close()

# 單獨一張只有 MLP 的 ROC 圖 (對應「基礎」部分要求)
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(fpr_mlp, tpr_mlp, color="tab:blue", linewidth=2, label=f"MLP (AUC = {auc_mlp:.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
ax.set_xlabel("False Positive Rate (FPR)")
ax.set_ylabel("True Positive Rate (TPR)")
ax.set_title("MLP ROC Curve")
ax.legend(loc="lower right")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
plt.tight_layout()
plt.savefig("./roc_curve_mlp_only.png", dpi=150)
plt.close()

# ------------------------------------------------------------------
# 7. 輸出比較表
# ------------------------------------------------------------------
summary = pd.DataFrame({
    "Model": ["MLP (Improved)", "Logistic Regression"],
    "AUC": [auc_mlp, auc_lr]
})
summary.to_csv("./roc_auc_comparison.csv", index=False)
print("\n=== AUC 比較表 ===")
print(summary.to_string(index=False))
print("\n完成。已輸出 roc_curve_comparison.png, roc_curve_mlp_only.png, roc_auc_comparison.csv")
