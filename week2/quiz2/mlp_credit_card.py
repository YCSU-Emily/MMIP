import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 0. 環境設定
# ------------------------------------------------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置: {device}")

# ------------------------------------------------------------------
# 1. 讀取資料 + 前處理
# ------------------------------------------------------------------
df = pd.read_csv("./UCI_Credit_Card.csv")

# 移除不具預測力的 ID 欄位
df = df.drop(columns=["ID"])

# 目標欄位: default.payment.next.month (1 = 下期會違約, 0 = 不會)
TARGET_COL = "default.payment.next.month"
X = df.drop(columns=[TARGET_COL]).values.astype(np.float32)
y = df[TARGET_COL].values.astype(np.float32)

print(f"資料筆數: {X.shape[0]}, 特徵數: {X.shape[1]}")
print(f"正類 (違約) 比例: {y.mean():.4f}")

# ------------------------------------------------------------------
# 2. 切分 Training / Validation
# ------------------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Training: {X_train.shape[0]} 筆, Validation: {X_val.shape[0]} 筆")

# ------------------------------------------------------------------
# 3. Feature Scaling (StandardScaler，僅用 Training data fit)
# ------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_val_scaled = scaler.transform(X_val).astype(np.float32)

# 轉成 PyTorch Tensor / DataLoader
BATCH_SIZE = 128

train_ds = TensorDataset(torch.tensor(X_train_scaled), torch.tensor(y_train))
val_ds = TensorDataset(torch.tensor(X_val_scaled), torch.tensor(y_val))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

INPUT_DIM = X_train_scaled.shape[1]

# ------------------------------------------------------------------
# 4. 定義 MLP 模型
#    baseline: 無 Dropout / 無正則化
#    improved: 加入 Dropout + (透過 optimizer 的 weight_decay 做 L2) + Early Stopping
# ------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 32, 16), dropout=0.0):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))  # 輸出 1 個 logit (搭配 BCEWithLogitsLoss)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ------------------------------------------------------------------
# 5. 訓練函式 (共用於 baseline / improved)
# ------------------------------------------------------------------
def train_model(model, train_loader, val_loader, epochs, lr, weight_decay=0.0,
                 use_early_stopping=False, patience=10):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    stopped_epoch = epochs

    for epoch in range(1, epochs + 1):
        # ---- training ----
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        # ---- validation ----
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_running_loss += loss.item() * xb.size(0)
        val_loss = val_running_loss / len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # ---- early stopping ----
        if use_early_stopping:
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"  Early stopping triggered at epoch {epoch} "
                          f"(best val loss = {best_val_loss:.4f} at earlier epoch)")
                    stopped_epoch = epoch
                    break

    if use_early_stopping and best_state is not None:
        model.load_state_dict(best_state)

    return model, train_losses, val_losses, stopped_epoch


def evaluate(model, X_val_scaled, y_val, threshold=0.5):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_val_scaled).to(device))
        proba = torch.sigmoid(logits).cpu().numpy()
    y_pred = (proba >= threshold).astype(int)
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    cm = confusion_matrix(y_val, y_pred)
    return proba, y_pred, acc, prec, rec, f1, cm


# ==================================================================
# 6. 訓練 Baseline MLP
# ==================================================================
print("\n========== Baseline MLP (無 Dropout / 無正則化) ==========")
EPOCHS = 80
LEARNING_RATE = 1e-3
HIDDEN_DIMS = (64, 32, 16)

baseline_model = MLP(INPUT_DIM, hidden_dims=HIDDEN_DIMS, dropout=0.0)
baseline_model, base_train_losses, base_val_losses, base_stopped = train_model(
    baseline_model, train_loader, val_loader,
    epochs=EPOCHS, lr=LEARNING_RATE, weight_decay=0.0, use_early_stopping=False
)

proba_base, pred_base, acc_b, prec_b, rec_b, f1_b, cm_b = evaluate(baseline_model, X_val_scaled, y_val)
print(f"\n[Baseline] Accuracy={acc_b:.4f} Precision={prec_b:.4f} Recall={rec_b:.4f} F1={f1_b:.4f}")
print("Confusion Matrix:\n", cm_b)

# ------------------------------------------------------------------
# 7. 單筆 Validation Sample 預測展示
# ------------------------------------------------------------------
sample_idx = 0
sample_x = torch.tensor(X_val_scaled[sample_idx:sample_idx+1]).to(device)
baseline_model.eval()
with torch.no_grad():
    sample_logit = baseline_model(sample_x)
    sample_proba = torch.sigmoid(sample_logit).item()
sample_pred = int(sample_proba >= 0.5)
sample_true = int(y_val[sample_idx])
print(f"\n=== 單筆 Validation Sample 預測展示 (index={sample_idx}) ===")
print(f"原始特徵 (前5個, 未標準化): {X_val[sample_idx][:5]}")
print(f"模型輸出機率 (違約機率): {sample_proba:.4f}")
print(f"預測類別: {sample_pred}  (threshold=0.5)")
print(f"實際類別: {sample_true}")

# ==================================================================
# 8. 訓練 Improved MLP (Dropout + L2 + Early Stopping)
# ==================================================================
print("\n========== Improved MLP (Dropout + L2 + Early Stopping) ==========")
DROPOUT = 0.2
WEIGHT_DECAY = 5e-5   # L2 regularization
PATIENCE = 15

improved_model = MLP(INPUT_DIM, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT)
improved_model, imp_train_losses, imp_val_losses, imp_stopped = train_model(
    improved_model, train_loader, val_loader,
    epochs=EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    use_early_stopping=True, patience=PATIENCE
)

proba_imp, pred_imp, acc_i, prec_i, rec_i, f1_i, cm_i = evaluate(improved_model, X_val_scaled, y_val)
print(f"\n[Improved @thr=0.5] Accuracy={acc_i:.4f} Precision={prec_i:.4f} Recall={rec_i:.4f} F1={f1_i:.4f}")
print("Confusion Matrix:\n", cm_i)

# 因為 Dropout + L2 + Early Stopping 讓模型輸出機率更保守(更接近資料集的
# 基準違約率 ~16%)，在資料不平衡且 threshold 固定為 0.5 時可能導致模型完全
# 不預測少數類別。這裡額外掃描 threshold，找出讓 F1 最大化的門檻，作為對照。
best_thr, best_f1 = 0.5, f1_i
for thr in np.arange(0.05, 0.51, 0.01):
    y_pred_thr = (proba_imp >= thr).astype(int)
    f1_thr = f1_score(y_val, y_pred_thr, zero_division=0)
    if f1_thr > best_f1:
        best_f1, best_thr = f1_thr, thr

_, pred_imp_best, acc_i_best, prec_i_best, rec_i_best, f1_i_best, cm_i_best = evaluate(
    improved_model, X_val_scaled, y_val, threshold=best_thr
)
print(f"\n[Improved @最佳threshold={best_thr:.2f}] Accuracy={acc_i_best:.4f} "
      f"Precision={prec_i_best:.4f} Recall={rec_i_best:.4f} F1={f1_i_best:.4f}")
print("Confusion Matrix:\n", cm_i_best)

# ==================================================================
# 9. 繪圖: Training / Validation Loss (baseline & improved)
# ==================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(range(1, len(base_train_losses)+1), base_train_losses, label="Training Loss")
axes[0].plot(range(1, len(base_val_losses)+1), base_val_losses, label="Validation Loss")
axes[0].set_title("Baseline MLP: Training vs Validation Loss")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("BCE Loss")
axes[0].legend()

axes[1].plot(range(1, len(imp_train_losses)+1), imp_train_losses, label="Training Loss")
axes[1].plot(range(1, len(imp_val_losses)+1), imp_val_losses, label="Validation Loss")
axes[1].set_title("Improved MLP: Training vs Validation Loss")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("BCE Loss")
axes[1].legend()

plt.tight_layout()
plt.savefig("./loss_curves.png", dpi=150)
plt.close()

# ==================================================================
# 10. Baseline vs Improved 指標比較表
# ==================================================================
comparison = pd.DataFrame({
    "Model": ["Baseline MLP (thr=0.5)",
              "Improved MLP (thr=0.5)",
              f"Improved MLP (best thr={best_thr:.2f})"],
    "Final Train Loss": [base_train_losses[-1], imp_train_losses[-1], imp_train_losses[-1]],
    "Final Val Loss": [base_val_losses[-1], imp_val_losses[-1], imp_val_losses[-1]],
    "Accuracy": [acc_b, acc_i, acc_i_best],
    "Precision": [prec_b, prec_i, prec_i_best],
    "Recall": [rec_b, rec_i, rec_i_best],
    "F1-Score": [f1_b, f1_i, f1_i_best],
    "Stopped Epoch": [base_stopped, imp_stopped, imp_stopped],
})
comparison.to_csv("./mlp_comparison.csv", index=False)
print("\n=== Baseline vs Improved 比較表 ===")
print(comparison.to_string(index=False))

# 指標長條圖
fig, ax = plt.subplots(figsize=(9, 5))
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
x = np.arange(len(metrics))
width = 0.25
ax.bar(x - width, [acc_b, prec_b, rec_b, f1_b], width, label="Baseline (thr=0.5)")
ax.bar(x, [acc_i, prec_i, rec_i, f1_i], width, label="Improved (thr=0.5)")
ax.bar(x + width, [acc_i_best, prec_i_best, rec_i_best, f1_i_best], width,
       label=f"Improved (best thr={best_thr:.2f})")
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.05)
ax.set_title("Baseline vs Improved MLP - Metrics")
ax.legend()
plt.tight_layout()
plt.savefig("./mlp_metrics_comparison.png", dpi=150)
plt.close()

print("\n完成。已輸出 loss_curves.png, mlp_metrics_comparison.png, mlp_comparison.csv")
