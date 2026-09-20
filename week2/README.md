# Quiz 1: Color Image to Grayscale

MMIP (Multi-Modality Image Processing) — Week 1

## 1. Task Description

| Item       | Requirement                                                                                                 |
| ---------- | ----------------------------------------------------------------------------------------------------------- |
| Basic      | Select a color image and convert the RGB color image into a grayscale image.                                |
| Advanced 1 | Implement the grayscale conversion algorithm using NumPy and compare it with the OpenCV implementation.     |
| Advanced 2 | Repeat the conversion N times and compare (1) execution speed, (2) conversion results, and (3) differences. |

## 2. Environment Setup (Anaconda + Git + Jupyter)

```bash
# Create and activate the virtual environment
conda create -n mmip python=3.11 -y
conda activate mmip

# Install required packages (including Jupyter Notebook)
pip install opencv-python numpy jupyter

# Clone the repository from GitHub
git clone https://github.com/<your-account>/MMIP.git
cd MMIP/week1/quiz_1
```

## 3. How to Run

```bash
python3 quiz1.py
```

The default input image is:

```text
image.jpg
```

A different image can also be specified:

```bash
python3 quiz1.py my_photo.jpg
```

Directory structure:

```text
quiz_1/
├── quiz1.py                # Main program
├── image.jpg               # Input color image
└── results/                # Generated results
    ├── original.jpg
    ├── gray_opencv.png
    ├── gray_numpy.png
    ├── diff_x100.png
    └── quiz1_result.csv
```

The program can also be executed in Jupyter Notebook:

```bash
jupyter notebook
```

Create a new `.ipynb` file and import the functions from `quiz1.py`:

```python
from quiz1 import *
```

Individual functions such as `gray_numpy()` and the benchmark function can then be tested interactively.

## 4. Method

The overall pipeline is:

**Read Color Image → BGR to Grayscale → NumPy / OpenCV Conversion → Difference Analysis → Execution Time Benchmark**

### OpenCV Method

OpenCV provides the built-in `cv2.cvtColor()` function:

```python
gray_cv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

Since OpenCV loads color images in **BGR** order, the conversion uses:

```python
cv2.COLOR_BGR2GRAY
```

### NumPy Method

The NumPy implementation uses the ITU-R BT.601 luminance formula:

$$
Gray = 0.299R + 0.587G + 0.114B
$$

Because the image is stored as BGR, the corresponding NumPy weights are:

```python
weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)
```

The grayscale image is generated using vectorized matrix multiplication:

```python
def gray_numpy(bgr_image):
    return np.round(
        bgr_image.astype(np.float32) @ weights
    ).astype(np.uint8)
```

### Result Difference

The absolute difference between the two grayscale images is calculated:

```python
diff = np.abs(
    gray_np.astype(np.int16) -
    gray_cv.astype(np.int16)
)
```

The following metrics are evaluated:

* Maximum grayscale difference
* Mean grayscale difference
* Number of different pixels
* Percentage of different pixels

### Execution Time

Each method is executed **100 times**.

A 5-iteration warm-up is performed before timing to reduce the influence of initial execution overhead.

Three cases are measured:

* OpenCV grayscale conversion
* NumPy core computation
* NumPy conversion including `astype()`

## 5. Basic Task Results

The input image is:

```text
image.jpg
```

Image information:

| Item         | Value      |
| ------------ | ---------- |
| Image Size   | 1000 × 703 |
| Channels     | 3          |
| Data Type    | uint8      |
| Total Pixels | 703,000    |

The generated grayscale images are:

| Original Image | OpenCV Result     | NumPy Result     |
| -------------- | ----------------- | ---------------- |
| `original.jpg` | `gray_opencv.png` | `gray_numpy.png` |

## 6. Advanced Task: NumPy vs OpenCV

### 6.1 Conversion Result Comparison

| Metric               |        Result |
| -------------------- | ------------: |
| Maximum Difference   |             1 |
| Mean Difference      |     0.0001735 |
| Different Pixels     | 122 / 703,000 |
| Different Pixels (%) |        0.017% |

The maximum difference is only **1 grayscale level**.

Only 122 pixels out of 703,000 pixels are different, corresponding to **0.017%** of the image.

Therefore, the NumPy and OpenCV grayscale conversion results are almost identical.

### 6.2 Execution Speed Comparison

The conversion was repeated 100 times.

| Method                | Mean (ms) | Std (ms) | Median (ms) |
| --------------------- | --------: | -------: | ----------: |
| OpenCV                |    0.1980 |   0.0508 |      0.2199 |
| NumPy (core only)     |    0.5921 |   0.0216 |      0.5825 |
| NumPy (with `astype`) |    0.7885 |   0.0209 |      0.7943 |

The measured execution-time ratios are:

```text
NumPy (with astype) / OpenCV ≈ 4.0×
NumPy (core only) / OpenCV ≈ 3.0×
```

OpenCV is therefore faster than the NumPy implementation in this experiment.

### 6.3 Discussion

The NumPy implementation performs floating-point conversion and matrix multiplication:

```python
bgr_image.astype(np.float32) @ weights
```

The additional data type conversion contributes to the total execution time.

OpenCV's `cv2.cvtColor()` uses an optimized low-level implementation, allowing it to perform the same grayscale conversion more efficiently.

The experiment demonstrates that a manually implemented NumPy algorithm can achieve nearly identical numerical results while having a higher execution cost than an optimized image-processing library.

## 7. Output Files

The program generates the following files:

| File               | Description                                          |
| ------------------ | ---------------------------------------------------- |
| `original.jpg`     | Original color image                                 |
| `gray_opencv.png`  | Grayscale result generated by OpenCV                 |
| `gray_numpy.png`   | Grayscale result generated by NumPy                  |
| `diff_x100.png`    | Difference image amplified by 100× for visualization |
| `quiz1_result.csv` | Numerical comparison and execution-time results      |

The difference image uses an amplification factor of 100 so that small grayscale differences can be visually inspected.

## 8. Git Upload

```bash
cd ~/MMIP
git status
git add week1/quiz_1
git commit -m "Add Quiz 1: grayscale conversion with NumPy and OpenCV"
git push origin main
```

## 9. Using Google Colab

This assignment only requires CPU computation, so the local environment is sufficient. If Google Colab is used instead:

1. Install the required packages and clone the repository:

```python
!pip install opencv-python-headless numpy
!git clone https://github.com/<your-account>/MMIP.git
%cd MMIP/week1/quiz_1
```

2. Run the program:

```python
!python quiz1.py
```

3. Upload a custom image if needed:

```python
from google.colab import files
files.upload()
```

4. Files generated by the program can be downloaded from the Colab file panel.

5. Colab files are temporary and may be deleted when the runtime ends. Therefore, important results should be copied to Google Drive or pushed back to GitHub.

When pushing to GitHub, use a **Personal Access Token (PAT)** and do not hard-code the token into the source code.

## 10. Environment

* Python 3.11
* OpenCV
* NumPy
* Jupyter Notebook
* Ubuntu
* Anaconda environment: `mmip`
