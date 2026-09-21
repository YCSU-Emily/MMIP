import sys
import os
import time

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")                  # 伺服器（無螢幕）也能存圖
import matplotlib.pyplot as plt
import pandas as pd

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
FILENAME = sys.argv[1] if len(sys.argv) > 1 else "images/dark.jpg"
N = 100
OUT = "results"
os.makedirs(OUT, exist_ok=True)


def save_fig(name):
    plt.tight_layout()
    plt.savefig(f"{OUT}/{name}", bbox_inches="tight", dpi=120)
    plt.close()


def hist_plot(ax, image, title):
    ax.hist(image.ravel(), bins=256, range=[0, 256])
    ax.set_title(title)
    ax.set_xlabel("Gray Level")
    ax.set_ylabel("Number of Pixels")
    ax.set_xlim([0, 256])


# 1. 讀取影像 → 灰階

img = cv2.imread(FILENAME)
if img is None:
    raise FileNotFoundError(f"找不到或無法讀取影像：{FILENAME}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("影像 :", FILENAME, img.shape)
print("原始灰階範圍 :", int(gray.min()), "~", int(gray.max()))

cv2.imwrite(f"{OUT}/original_color.jpg", img)
cv2.imwrite(f"{OUT}/original_gray.png", gray)


# 2. NumPy 自行實作 Histogram Equalization

def histogram_equalization_numpy(gray_image):
    hist = np.bincount(gray_image.ravel(), minlength=256)      # Step 1：Histogram
    cdf = hist.cumsum()                                        # Step 2：CDF

    nonzero = cdf[cdf > 0]
    if len(nonzero) == 0:
        return gray_image.copy()
    cdf_min = nonzero[0]                                       # Step 3：第一個非零 CDF
    total = gray_image.size                                    # Step 4：像素總數

    if cdf_min == total:                                       # 單一灰階值的影像，避免除以 0
        return gray_image.copy()

    lut = (cdf - cdf_min) / (total - cdf_min) * 255            # Step 5：均衡化公式
    lut = np.clip(np.round(lut), 0, 255).astype(np.uint8)      # Step 6：四捨五入並限制範圍
    return lut[gray_image]                                     # Step 7：查表


# 3. 執行兩種方法

opencv_result = cv2.equalizeHist(gray)
numpy_result = histogram_equalization_numpy(gray)

cv2.imwrite(f"{OUT}/equalized_opencv.png", opencv_result)
cv2.imwrite(f"{OUT}/equalized_numpy.png", numpy_result)

# 4. 圖表
# 4-1 原始彩色 / 灰階
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax[0].set_title("Original Color"); ax[0].axis("off")
ax[1].imshow(gray, cmap="gray", vmin=0, vmax=255); ax[1].set_title("Original Grayscale"); ax[1].axis("off")
save_fig("01_original.png")

# 4-2 處理前後影像（OpenCV）
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].imshow(gray, cmap="gray", vmin=0, vmax=255); ax[0].set_title("Before"); ax[0].axis("off")
ax[1].imshow(opencv_result, cmap="gray", vmin=0, vmax=255); ax[1].set_title("After (Histogram Equalization)"); ax[1].axis("off")
save_fig("02_before_after_image.png")

# 4-3 處理前後 Histogram
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
hist_plot(ax[0], gray, "Before Histogram")
hist_plot(ax[1], opencv_result, "After Histogram")
save_fig("03_before_after_histogram.png")

# 4-4 Original / NumPy / OpenCV 影像
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for a, im, t in zip(ax, [gray, numpy_result, opencv_result], ["Original", "NumPy", "OpenCV"]):
    a.imshow(im, cmap="gray", vmin=0, vmax=255); a.set_title(t); a.axis("off")
save_fig("04_compare_images.png")

# 4-5 三者 Histogram
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
for a, im, t in zip(ax, [gray, numpy_result, opencv_result], ["Original", "NumPy", "OpenCV"]):
    hist_plot(a, im, f"{t} Histogram")
save_fig("05_compare_histograms.png")

# 4-6 CDF 比較（均衡化後 CDF 應接近直線）
fig, ax = plt.subplots(figsize=(8, 5))
for im, t in zip([gray, opencv_result], ["Original", "After equalization"]):
    c = np.bincount(im.ravel(), minlength=256).cumsum()
    ax.plot(c / c[-1], label=t)
ax.set_title("Normalized CDF"); ax.set_xlabel("Gray Level"); ax.set_ylabel("Cumulative Ratio"); ax.legend()
save_fig("06_cdf.png")

# 5. NumPy 與 OpenCV 的差異

diff = cv2.absdiff(numpy_result, opencv_result)
n_diff = int(np.count_nonzero(diff))
print("\n" + "=" * 60)
print("NumPy vs OpenCV 差異")
print("=" * 60)
print("最大差異   :", int(diff.max()))
print("平均差異   :", float(diff.mean()))
print(f"不同像素數 : {n_diff} / {diff.size} ({100 * n_diff / diff.size:.4f}%)")

plt.figure(figsize=(8, 6))
plt.imshow(diff, cmap="gray"); plt.colorbar(); plt.title("Difference: NumPy vs OpenCV"); plt.axis("off")
save_fig("07_difference.png")

# 6. 影像增強效果統計

def stats(name, im):
    return {"Method": name, "Min": int(im.min()), "Max": int(im.max()),
            "Dynamic Range": int(im.max()) - int(im.min()),
            "Mean": round(float(im.mean()), 3), "Std": round(float(im.std()), 3)}


stat_table = pd.DataFrame([stats("Original", gray), stats("NumPy", numpy_result), stats("OpenCV", opencv_result)])
print("\n" + "=" * 60)
print("影像增強效果統計")
print("=" * 60)
print(stat_table.to_string(index=False))
stat_table.to_csv(f"{OUT}/quiz2_stats.csv", index=False)

plt.figure(figsize=(8, 5))
plt.bar(stat_table["Method"], stat_table["Std"])
plt.title("Standard Deviation (contrast) Comparison"); plt.ylabel("Standard Deviation")
save_fig("08_std_comparison.png")

# 7. 執行時間（含 warm-up）

def benchmark(fn, n=N):
    for _ in range(5):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts) * 1000.0
    return ts.mean(), ts.std(), np.median(ts)


np_mean, np_std, np_med = benchmark(lambda: histogram_equalization_numpy(gray))
cv_mean, cv_std, cv_med = benchmark(lambda: cv2.equalizeHist(gray))

time_table = pd.DataFrame({
    "Method": ["NumPy", "OpenCV"],
    "Mean (ms)": [round(np_mean, 4), round(cv_mean, 4)],
    "Std (ms)": [round(np_std, 4), round(cv_std, 4)],
    "Median (ms)": [round(np_med, 4), round(cv_med, 4)],
    "N": [N, N],
})
print("\n" + "=" * 60)
print(f"執行時間比較（重複 {N} 次）")
print("=" * 60)
print(time_table.to_string(index=False))
time_table.to_csv(f"{OUT}/quiz2_time.csv", index=False)

if np_mean > cv_mean:
    print(f"\nOpenCV 約比 NumPy 快 {np_mean / cv_mean:.2f} 倍")
else:
    print(f"\nNumPy 約比 OpenCV 快 {cv_mean / np_mean:.2f} 倍")

print(f"\n完成！所有圖表與 CSV 已存到 {OUT}/")
