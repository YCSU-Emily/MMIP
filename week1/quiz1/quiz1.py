import cv2
import numpy as np
import time

# =========================================================
# Config
# =========================================================
IMG_PATH = "image.jpg"   # 改成你的檔名
N = 100                  # benchmark 重複次數

# =========================================================
# 1. Read Image
# =========================================================
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(f"Cannot find {IMG_PATH}")

# =========================================================
# 2. OpenCV Grayscale (一次)
# =========================================================
gray_cv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("gray_opencv.jpg", gray_cv)

# =========================================================
# 3. NumPy Grayscale (向量化、四捨五入)
# =========================================================
# OpenCV 輸入為 BGR uint8，所以 NumPy 計算權重要對應 B,G,R
arr = img.astype(np.float32)           # 只做一次轉型（在 benchmark 中也會共用）
weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)  # B, G, R

# 用 dot 向量化並做 round，然後回到 uint8
gray_numpy = np.round(np.dot(arr, weights)).astype(np.uint8)
cv2.imwrite("gray_numpy.jpg", gray_numpy)

# =========================================================
# 4. Compare Results
# =========================================================
diff = np.abs(gray_numpy.astype(np.int16) - gray_cv.astype(np.int16))

print("========== Result Comparison ==========")
print("Max Difference       :", int(diff.max()))
print("Mean Difference      :", float(diff.mean()))
print("Different Pixels     :", int(np.count_nonzero(diff)))

# =========================================================
# 5. Benchmark (含 warm-up)
# =========================================================

# Warm-up (避免第一次呼叫的額外延遲)
_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_ = np.round(np.dot(arr, weights)).astype(np.uint8)

# ---------- OpenCV ----------
start = time.perf_counter()
for _ in range(N):
    gray_cv_test = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
end = time.perf_counter()
opencv_avg = (end - start) / N

# ---------- NumPy ----------
start = time.perf_counter()
for _ in range(N):
    # 直接使用 precomputed float array 與 dot（不重做 astype 以免量測到 dtype 轉換成本）
    gray_numpy_test = np.round(np.dot(arr, weights)).astype(np.uint8)
end = time.perf_counter()
numpy_avg = (end - start) / N

# =========================================================
# 6. Print Speed
# =========================================================
print("\n========== Speed Comparison ==========")
print(f"OpenCV Average: {opencv_avg * 1000:.6f} ms")
print(f"NumPy Average : {numpy_avg * 1000:.6f} ms")
print(f"Speed Ratio   : {numpy_avg / opencv_avg:.2f}x (NumPy / OpenCV)")
