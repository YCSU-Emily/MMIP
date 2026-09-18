# ============================================================
# 1. Import Libraries
# ============================================================

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import pandas as pd

print("Libraries imported successfully!")


# ============================================================
# 2. Select Image (Linux / Server)
# ============================================================

# 請把圖片放在 images/ 資料夾，並修改成你的實際檔名。
# 例如：images/dark.jpg
filename = "dark.jpg"

print("Image:", filename)


# ============================================================
# 3. Read Image
# ============================================================

img = cv2.imread(filename)

if img is None:
    raise ValueError("無法讀取圖片，請確認圖片格式是否正確。")

print("Image shape:", img.shape)


# ============================================================
# 4. Convert BGR -> RGB
# ============================================================

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# 顯示原始彩色圖片
plt.figure(figsize=(8, 6))
plt.imshow(img_rgb)
plt.title("Original Color Image")
plt.axis("off")
plt.savefig("result_1.png", bbox_inches="tight")
plt.close()


# ============================================================
# 5. Convert Image to Grayscale
# ============================================================

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("Grayscale shape:", gray.shape)
print("Gray level range:", gray.min(), "~", gray.max())


# 顯示灰階圖片
plt.figure(figsize=(8, 6))
plt.imshow(gray, cmap="gray")
plt.title("Original Grayscale Image")
plt.axis("off")
plt.savefig("result_2.png", bbox_inches="tight")
plt.close()


# ============================================================
# 6. Original Histogram
# ============================================================

plt.figure(figsize=(10, 5))

plt.hist(
    gray.ravel(),
    bins=256,
    range=[0, 256]
)

plt.title("Original Grayscale Histogram")
plt.xlabel("Gray Level")
plt.ylabel("Number of Pixels")
plt.xlim([0, 256])

plt.savefig("result_3.png", bbox_inches="tight")
plt.close()


# ============================================================
# 7. OpenCV Histogram Equalization
# ============================================================

opencv_result = cv2.equalizeHist(gray)

print("OpenCV Histogram Equalization completed.")


# ============================================================
# 8. Display Original vs OpenCV Result
# ============================================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(gray, cmap="gray")
plt.title("Before - Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(opencv_result, cmap="gray")
plt.title("After - OpenCV")
plt.axis("off")

plt.tight_layout()
plt.savefig("result_4.png", bbox_inches="tight")
plt.close()


# ============================================================
# 9. Compare Histograms Before and After OpenCV
# ============================================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.hist(
    gray.ravel(),
    bins=256,
    range=[0, 256]
)

plt.title("Before Histogram")
plt.xlabel("Gray Level")
plt.ylabel("Number of Pixels")
plt.xlim([0, 256])


plt.subplot(1, 2, 2)

plt.hist(
    opencv_result.ravel(),
    bins=256,
    range=[0, 256]
)

plt.title("After Histogram - OpenCV")
plt.xlabel("Gray Level")
plt.ylabel("Number of Pixels")
plt.xlim([0, 256])


plt.tight_layout()
plt.savefig("result_5.png", bbox_inches="tight")
plt.close()


# ============================================================
# 10. NumPy Histogram Equalization
# ============================================================

def histogram_equalization_numpy(gray_image):

    # --------------------------------------------------------
    # Step 1: Calculate Histogram
    # --------------------------------------------------------

    hist = np.bincount(
        gray_image.ravel(),
        minlength=256
    )


    # --------------------------------------------------------
    # Step 2: Calculate CDF
    # --------------------------------------------------------

    cdf = hist.cumsum()


    # --------------------------------------------------------
    # Step 3: Find the first non-zero CDF
    # --------------------------------------------------------

    non_zero_cdf = cdf[cdf > 0]

    if len(non_zero_cdf) == 0:
        return gray_image.copy()

    cdf_min = non_zero_cdf[0]


    # --------------------------------------------------------
    # Step 4: Total number of pixels
    # --------------------------------------------------------

    total_pixels = gray_image.size


    # --------------------------------------------------------
    # Step 5: Histogram Equalization Formula
    #
    # s = ((CDF(r) - CDF_min)
    #      / (N - CDF_min)) * 255
    # --------------------------------------------------------

    lookup_table = (
        (cdf - cdf_min)
        / (total_pixels - cdf_min)
        * 255
    )


    # --------------------------------------------------------
    # Step 6: Round and limit range to 0~255
    # --------------------------------------------------------

    lookup_table = np.round(lookup_table)

    lookup_table = np.clip(
        lookup_table,
        0,
        255
    ).astype(np.uint8)


    # --------------------------------------------------------
    # Step 7: Apply lookup table to image
    # --------------------------------------------------------

    result = lookup_table[gray_image]


    return result


# 執行 NumPy Histogram Equalization
numpy_result = histogram_equalization_numpy(gray)

print("NumPy Histogram Equalization completed.")


# ============================================================
# 11. Display Original / NumPy / OpenCV
# ============================================================

plt.figure(figsize=(15, 5))


plt.subplot(1, 3, 1)

plt.imshow(gray, cmap="gray")
plt.title("Original")
plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(numpy_result, cmap="gray")
plt.title("NumPy")
plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(opencv_result, cmap="gray")
plt.title("OpenCV")
plt.axis("off")


plt.tight_layout()
plt.savefig("result_6.png", bbox_inches="tight")
plt.close()


# ============================================================
# 12. Compare Histograms
# ============================================================

plt.figure(figsize=(15, 5))


plt.subplot(1, 3, 1)

plt.hist(
    gray.ravel(),
    bins=256,
    range=[0, 256]
)

plt.title("Original Histogram")
plt.xlabel("Gray Level")
plt.ylabel("Pixels")
plt.xlim([0, 256])


plt.subplot(1, 3, 2)

plt.hist(
    numpy_result.ravel(),
    bins=256,
    range=[0, 256]
)

plt.title("NumPy Histogram")
plt.xlabel("Gray Level")
plt.ylabel("Pixels")
plt.xlim([0, 256])


plt.subplot(1, 3, 3)

plt.hist(
    opencv_result.ravel(),
    bins=256,
    range=[0, 256]
)

plt.title("OpenCV Histogram")
plt.xlabel("Gray Level")
plt.ylabel("Pixels")
plt.xlim([0, 256])


plt.tight_layout()
plt.savefig("result_7.png", bbox_inches="tight")
plt.close()


# ============================================================
# 13. Difference Between NumPy and OpenCV
# ============================================================

difference = cv2.absdiff(
    numpy_result,
    opencv_result
)

print("Maximum difference:", difference.max())
print("Mean difference:", difference.mean())


# 顯示 Difference
plt.figure(figsize=(8, 6))

plt.imshow(
    difference,
    cmap="gray"
)

plt.title("Difference: NumPy vs OpenCV")
plt.colorbar()
plt.axis("off")

plt.savefig("result_8.png", bbox_inches="tight")
plt.close()


# ============================================================
# 14. Compare Image Statistics
# ============================================================

print("=" * 60)
print("Image Enhancement Statistics")
print("=" * 60)


print("\nOriginal")
print("Minimum gray level :", gray.min())
print("Maximum gray level :", gray.max())
print("Mean                :", gray.mean())
print("Standard deviation  :", gray.std())


print("\nNumPy")
print("Minimum gray level :", numpy_result.min())
print("Maximum gray level :", numpy_result.max())
print("Mean                :", numpy_result.mean())
print("Standard deviation  :", numpy_result.std())


print("\nOpenCV")
print("Minimum gray level :", opencv_result.min())
print("Maximum gray level :", opencv_result.max())
print("Mean                :", opencv_result.mean())
print("Standard deviation  :", opencv_result.std())


# ============================================================
# 15. Execution Time Test
# ============================================================

# 重複執行 N 次
N = 100

print("\n" + "=" * 60)
print("Execution Time Test")
print("=" * 60)

print("Number of repetitions:", N)


# ------------------------------------------------------------
# NumPy
# ------------------------------------------------------------

numpy_times = []

for i in range(N):

    start = time.perf_counter()

    result = histogram_equalization_numpy(gray)

    end = time.perf_counter()

    numpy_times.append(end - start)


numpy_average_time = np.mean(numpy_times)
numpy_std_time = np.std(numpy_times)


# ------------------------------------------------------------
# OpenCV
# ------------------------------------------------------------

opencv_times = []

for i in range(N):

    start = time.perf_counter()

    result = cv2.equalizeHist(gray)

    end = time.perf_counter()

    opencv_times.append(end - start)


opencv_average_time = np.mean(opencv_times)
opencv_std_time = np.std(opencv_times)


# ============================================================
# 16. Print Execution Time
# ============================================================

print("\nNumPy")
print("Average time :", numpy_average_time * 1000, "ms")
print("Std time     :", numpy_std_time * 1000, "ms")


print("\nOpenCV")
print("Average time :", opencv_average_time * 1000, "ms")
print("Std time     :", opencv_std_time * 1000, "ms")


# ============================================================
# 17. Calculate Speed Difference
# ============================================================

if opencv_average_time > 0:

    speed_ratio = (
        numpy_average_time
        / opencv_average_time
    )

    print("\nNumPy / OpenCV time ratio:",
          speed_ratio)

    if numpy_average_time > opencv_average_time:
        print(
            "OpenCV is approximately",
            round(speed_ratio, 2),
            "times faster than NumPy."
        )

    else:
        print(
            "NumPy is approximately",
            round(1 / speed_ratio, 2),
            "times faster than OpenCV."
        )


# ============================================================
# 18. Create Result Table
# ============================================================

data = {
    "Method": [
        "Original",
        "NumPy",
        "OpenCV"
    ],

    "Min Gray Level": [
        gray.min(),
        numpy_result.min(),
        opencv_result.min()
    ],

    "Max Gray Level": [
        gray.max(),
        numpy_result.max(),
        opencv_result.max()
    ],

    "Mean": [
        gray.mean(),
        numpy_result.mean(),
        opencv_result.mean()
    ],

    "Standard Deviation": [
        gray.std(),
        numpy_result.std(),
        opencv_result.std()
    ]
}


result_table = pd.DataFrame(data)

print("\n")
print("=" * 60)
print("Image Comparison Table")
print("=" * 60)

print(result_table.to_string(index=False))


# ============================================================
# 19. Execution Time Table
# ============================================================

time_table = pd.DataFrame({
    "Method": [
        "NumPy",
        "OpenCV"
    ],

    "Average Time (ms)": [
        numpy_average_time * 1000,
        opencv_average_time * 1000
    ],

    "Std Time (ms)": [
        numpy_std_time * 1000,
        opencv_std_time * 1000
    ]
})


print("\n")
print("=" * 60)
print("Execution Time Table")
print("=" * 60)

print(time_table.to_string(index=False))


# ============================================================
# 20. Final Comparison Plot
# ============================================================

methods = ["Original", "NumPy", "OpenCV"]

std_values = [
    gray.std(),
    numpy_result.std(),
    opencv_result.std()
]


plt.figure(figsize=(8, 5))

plt.bar(
    methods,
    std_values
)

plt.title("Standard Deviation Comparison")
plt.xlabel("Method")
plt.ylabel("Standard Deviation")

plt.savefig("result_9.png", bbox_inches="tight")
plt.close()


# ============================================================
# 21. Final Summary
# ============================================================

print("\n")
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print("\n[Image Enhancement]")
print(
    "Original gray range:",
    gray.min(),
    "~",
    gray.max()
)

print(
    "NumPy gray range:",
    numpy_result.min(),
    "~",
    numpy_result.max()
)

print(
    "OpenCV gray range:",
    opencv_result.min(),
    "~",
    opencv_result.max()
)


print("\n[Execution Time]")
print(
    "NumPy average:",
    round(numpy_average_time * 1000, 4),
    "ms"
)

print(
    "OpenCV average:",
    round(opencv_average_time * 1000, 4),
    "ms"
)


print("\n[Difference]")
print(
    "Maximum difference:",
    difference.max()
)

print(
    "Mean difference:",
    round(difference.mean(), 4)
)


print("\nExperiment completed!")
