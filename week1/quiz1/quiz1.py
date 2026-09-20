import sys
import os
import csv
import time

import cv2
import numpy as np

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
IMG_PATH = sys.argv[1] if len(sys.argv) > 1 else "image.jpg"
N = 100                       # 計時重複次數
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------------------------------------
# 1. 讀取影像（OpenCV 讀進來的通道順序是 B, G, R）
# ------------------------------------------------------------
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(f"找不到或無法讀取影像：{IMG_PATH}")

print("=" * 60)
print("影像資訊")
print("=" * 60)
print("檔案 :", IMG_PATH)
print("形狀 :", img.shape, " dtype:", img.dtype)
cv2.imwrite(f"{RESULTS_DIR}/original.jpg", img)

# ------------------------------------------------------------
# 2. OpenCV 灰階轉換
# ------------------------------------------------------------
gray_cv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite(f"{RESULTS_DIR}/gray_opencv.png", gray_cv)

# ------------------------------------------------------------
# 3. NumPy 自行實作
#    Gray = 0.299 R + 0.587 G + 0.114 B（ITU-R BT.601 亮度公式）
#    因為 img 是 BGR，權重順序要寫成 [B, G, R]
# ------------------------------------------------------------
weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)


def gray_numpy(bgr_image):
    """NumPy 向量化灰階轉換：加權 → 四捨五入 → uint8"""
    return np.round(bgr_image.astype(np.float32) @ weights).astype(np.uint8)


def gray_numpy_core(bgr_float):
    """與上面相同，但輸入已是 float32（用來單獨量測核心運算成本）"""
    return np.round(bgr_float @ weights).astype(np.uint8)


gray_np = gray_numpy(img)
cv2.imwrite(f"{RESULTS_DIR}/gray_numpy.png", gray_np)

# ------------------------------------------------------------
# 4. 比較轉換結果
# ------------------------------------------------------------
diff = np.abs(gray_np.astype(np.int16) - gray_cv.astype(np.int16))
n_diff = int(np.count_nonzero(diff))

print("\n" + "=" * 60)
print("轉換結果比較（NumPy vs OpenCV）")
print("=" * 60)
print("最大差異        :", int(diff.max()))
print("平均差異        :", float(diff.mean()))
print(f"不同像素數      : {n_diff} / {diff.size} ({100 * n_diff / diff.size:.3f}%)")
for v in range(int(diff.max()) + 1):
    print(f"  差 {v} 個灰階   : {int((diff == v).sum())} 個像素")

# 差異圖放大 100 倍以便觀察（差 1 → 亮度 100）
cv2.imwrite(f"{RESULTS_DIR}/diff_x100.png", np.clip(diff * 100, 0, 255).astype(np.uint8))

# ------------------------------------------------------------
# 5. 計時（含 warm-up，逐次計時以取得標準差）
# ------------------------------------------------------------
def benchmark(fn, n=N):
    for _ in range(5):                      # warm-up
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    t = np.array(times) * 1000.0            # ms
    return t.mean(), t.std(), np.median(t)


img_float = img.astype(np.float32)          # 預先轉型，用於「僅核心運算」的量測

cv_mean, cv_std, cv_med = benchmark(lambda: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
core_mean, core_std, core_med = benchmark(lambda: gray_numpy_core(img_float))
full_mean, full_std, full_med = benchmark(lambda: gray_numpy(img))

print("\n" + "=" * 60)
print(f"執行速度比較（重複 {N} 次）")
print("=" * 60)
print(f"{'方法':<22}{'平均(ms)':>12}{'標準差(ms)':>14}{'中位數(ms)':>14}")
print(f"{'OpenCV':<22}{cv_mean:>12.4f}{cv_std:>14.4f}{cv_med:>14.4f}")
print(f"{'NumPy（僅運算）':<20}{core_mean:>12.4f}{core_std:>14.4f}{core_med:>14.4f}")
print(f"{'NumPy（含 astype）':<19}{full_mean:>12.4f}{full_std:>14.4f}{full_med:>14.4f}")
print(f"\nNumPy(含 astype) / OpenCV = {full_mean / cv_mean:.1f}x")
print(f"NumPy(僅運算)    / OpenCV = {core_mean / cv_mean:.1f}x")

# ------------------------------------------------------------
# 6. 輸出 CSV
# ------------------------------------------------------------
with open(f"{RESULTS_DIR}/quiz1_result.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Method", "Mean(ms)", "Std(ms)", "Median(ms)", "N"])
    w.writerow(["OpenCV", f"{cv_mean:.4f}", f"{cv_std:.4f}", f"{cv_med:.4f}", N])
    w.writerow(["NumPy (core only)", f"{core_mean:.4f}", f"{core_std:.4f}", f"{core_med:.4f}", N])
    w.writerow(["NumPy (with astype)", f"{full_mean:.4f}", f"{full_std:.4f}", f"{full_med:.4f}", N])
    w.writerow([])
    w.writerow(["Max difference", int(diff.max())])
    w.writerow(["Mean difference", f"{diff.mean():.6f}"])
    w.writerow(["Different pixels", n_diff])
    w.writerow(["Different pixels (%)", f"{100 * n_diff / diff.size:.4f}"])

print(f"\n完成！結果已存到 {RESULTS_DIR}/（original.jpg、gray_opencv.png、gray_numpy.png、diff_x100.png、quiz1_result.csv）")
