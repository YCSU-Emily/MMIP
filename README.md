MMIP — Multi-Modality Image Processing

Week 1: Basic Image Processing

This repository contains four image processing quizzes implemented using Python, NumPy, and OpenCV.

---

Environment

* Python 3.10
* NumPy
* OpenCV
* Matplotlib
* Pandas
* Jupyter Notebook

Install dependencies:

bash
pip install -r requirements.txt




Quiz 1 — Color Image to Grayscale

Objective

Convert an RGB/BGR color image into a grayscale image.

Methods

* OpenCV `cv2.cvtColor()`
* NumPy vectorized implementation

Comparison

* Execution time
* Pixel-level difference
* Conversion result

---

Quiz 2 — Histogram Equalization

Objective

Improve image contrast and dynamic range using Histogram Equalization.

Methods

* OpenCV `cv2.equalizeHist()`
* NumPy implementation

Comparison

* Original vs enhanced image
* Histogram distribution
* Execution time
* Image statistics
* Difference between NumPy and OpenCV

---


# Quiz 3: Perspective Correction and Perspective Transformation

MMIP (Multi-Modality Image Processing) — Week 1

## 1. Task Description

| Item       | Requirement                                                                                                                                                           |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Basic      | Select an obliquely captured image and use Perspective Transformation to correct it into a frontal-view image.                                                        |
| Advanced 1 | Apply the algorithm to multiple images captured under different conditions and use the same pipeline to automatically perform the correction.                         |
| Advanced 2 | Investigate the correction performance under different viewing angles, identify the angles or conditions at which the method begins to fail, and explain the reasons. |

## 2. Environment Setup (Anaconda + Git + Jupyter)

```bash
# Create and activate the virtual environment
conda create -n mmip python=3.11 -y
conda activate mmip

# Install required packages (including Jupyter Notebook)
pip install opencv-python numpy jupyter

# Clone the repository from GitHub
git clone https://github.com/<your-account>/MMIP.git
cd MMIP/week1/quiz_3
```

## 3. How to Run

```bash
python3 quiz3.py
```

Directory structure:

```text
quiz_3/
├── quiz3.py               # Main program
├── images/                # Obliquely captured images (test0~test5, test1 is the basic task image)
├── test_images/           # 47 automatically generated test images + ground_truth.json
├── results/               # Corrected results, contour visualization, contact_sheet.jpg
├── summary.csv / .md      # Complete results of the 47 test cases
└── summary_real.csv       # Results for images in images/
```

The program can also be executed in Jupyter Notebook:

```bash
jupyter notebook
```

Create a new `.ipynb` file, then import the functions from `quiz3.py`:

```python
from quiz3 import *
```

You can then call `main()` or individual functions such as `correct_image()`.

## 4. Method

The overall pipeline is:

**Grayscale → Gaussian Blur → Multi-strategy Binarization/Edge Detection → Contour Detection → Convex Hull → Quadrilateral Approximation → Corner Ordering → getPerspectiveTransform → warpPerspective**

### Multi-strategy Detection

To improve the level of automation, four preprocessing strategies are attempted simultaneously:

* Canny
* Low-threshold Canny
* Otsu
* Inverted Otsu

The quadrilateral with the highest **area ratio × fitting quality** is selected.

### False Detection Rejection

To reduce incorrect detections:

* Contours with an area smaller than 3% of the image are ignored.
* Contours corresponding to the outer boundary of the entire image are excluded.
* The detected quadrilateral must be convex.

### Corner Ordering

The four corners are first sorted according to their angles relative to the centroid, and then the point with the smallest `x + y` value is selected as the top-left corner.

This approach is more robust under planar rotation than simply using the conventional `sum/difference` method.

### Output Size

The output width and height are automatically calculated from the detected edge lengths, so no manual output size needs to be specified.

### Fully Automated Processing

The same program is used to process all images in `images/` as well as the 47 automatically generated test images, without manual parameter tuning.

## 5. Basic Task Results

| Original Image       | Detected Contour    | Corrected Result      |
| -------------------- | ------------------- | --------------------- |
| `basic_original.jpg` | `basic_contour.jpg` | `basic_corrected.jpg` |

`images/test1.jpg` was successfully detected using the Otsu strategy, producing an output image of **545 × 844 pixels**.

## 6. Advanced Task: Automatic Correction of Multiple Images

### 6.1 Images in `images/`

There are six images in `images/`, and all of them were successfully detected and corrected using the same pipeline.

| File  | Detection Strategy | Output Size |
| ----- | ------------------ | ----------- |
| test0 | Canny              | 793 × 725   |
| test1 | Otsu               | 545 × 844   |
| test2 | Canny              | 348 × 863   |
| test3 | Canny_low          | 861 × 497   |
| test4 | Canny              | 485 × 853   |
| test5 | Otsu               | 539 × 840   |

> **Note:** `test1`–`test5` are synthetically generated oblique-view images simulating different documents, backgrounds, and viewing angles. They were not captured using a real camera. `test0` is a ✏️ **[fill in: real photograph / source]**.

### 6.2 Automatically Generated 47 Test Images with Ground Truth

The program uses a pinhole camera model to project a planar document under different imaging conditions. The **ground-truth corner coordinates** are also saved, allowing the correction error to be quantitatively evaluated.

The evaluation metrics include:

* **Mean corner error (pixels)**
* **Similarity between the corrected image and the frontal-view document**, measured using normalized grayscale correlation

The classification criteria are:

* **OK:** corner error < 20 px and similarity ≥ 0.5
* **WARN:** corner error < 50 px
* **FAIL:** otherwise, or when no valid quadrilateral can be detected

Test conditions include:

* Vertical-axis tilt
* Horizontal-axis tilt
* Dual-axis tilt
* In-plane rotation
* Five different backgrounds
* Low contrast
* Side lighting and shadows
* Noise
* Blur
* Occlusion
* Image truncation
* Reduced image scale / distant capture

**Overall result: 37 OK, 1 WARN, and 9 FAIL cases out of 47 tests.**

`contact_sheet.jpg` contains the visual summary of all test cases.

## 7. Failure Boundary Analysis

| Condition                     | Result                                                                                                      |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Vertical-axis tilt            | 10°–80° successful; **85° and 88° failed**                                                                  |
| Horizontal-axis tilt          | 10°–85° successful; **88° failed**                                                                          |
| Dual-axis tilt                | (30°, 30°), (45°, 30°), and (60°, 45°) all successful                                                       |
| In-plane rotation             | 15°, 45°, and 90° all successful                                                                            |
| Low contrast                  | Successful with up to 95% contrast reduction; **98% failed**                                                |
| Noise σ                       | 25 and 50 successful                                                                                        |
| Blur kernel                   | 5 and 11 successful                                                                                         |
| Distant / reduced-scale image | 0.5× successful; **0.25× failed**                                                                           |
| Background                    | Wood texture, light background, and gradient background successful; **2 cluttered-background cases failed** |
| Occlusion / truncation        | Corner occlusion failed; finger occlusion resulted in WARN; **document truncation failed**                  |

### Possible Reasons for Failure

1. **Large viewing angles (≥85°)**
   The document becomes an extremely narrow strip in the image. Its area ratio falls below the 3% threshold, while the boundary becomes difficult to distinguish from the background. Although detection still succeeds at 80°, the similarity decreases to approximately 0.51. This indicates severe stretching of the corrected text and that the practical usefulness is already close to its limit.

   **Successful detection does not necessarily mean that the corrected result is practically usable.**

2. **Very low contrast (98% reduction)**
   The grayscale intensity of the document becomes nearly identical to the background, making it difficult for both Canny and Otsu thresholding to identify a stable boundary.

3. **Very small document (0.25× scale)**
   The document area falls below the 3% threshold and is therefore treated as noise. Even if the threshold were relaxed, the number of available pixels would be too small, resulting in poor output resolution after perspective correction.

4. **Cluttered background**
   Background objects may become connected with the document and form a larger outer contour, which can then be incorrectly detected as the document itself.

5. **Occlusion / truncation**
   When the true corners are outside the visible image area, the assumption that the document forms a complete quadrilateral no longer holds. As a result, the detected corners may be shifted from their actual positions.

### Limitations

* The test images are idealized simulations with relatively clean boundaries. Real photographs may fail at smaller viewing angles due to lens distortion, reflections, and paper deformation.
* The method assumes that the document is a single, convex quadrilateral and is the most visually prominent object in the image.

## 8. Git Upload

```bash
cd ~/MMIP
git status
git add week1/quiz_3
git commit -m "Add Quiz 3: perspective correction with batch tests"
git push origin main
```

## 9. Using Google Colab

This assignment only requires CPU computation, so the local environment is sufficient. If Google Colab is used instead:

1. Open a new notebook and install the required packages and clone the repository:

```python
!pip install opencv-python-headless numpy
!git clone https://github.com/<your-account>/MMIP.git
%cd MMIP/week1/quiz_3
```

2. Run the program:

```python
!python quiz3.py
```

3. To display images, use the Colab-specific function because `cv2.imshow()` does not work directly in Colab:

```python
import cv2
from google.colab.patches import cv2_imshow

cv2_imshow(cv2.imread("results/contact_sheet.jpg"))
```

4. Upload your own images using the file panel on the left, or:

```python
from google.colab import files
files.upload()
```

5. Note that files in Colab are deleted when the runtime ends. Therefore, use Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

or push the results back to GitHub.

When pushing to GitHub, use a **Personal Access Token (PAT)** and do not hard-code the token into the source code.

## 10. Environment

* Python 3.11
* OpenCV
* NumPy
* Ubuntu
* Anaconda environment: `mmip`

---

# Quiz 4: Image Stitching

MMIP (Multi-Modality Image Processing) — Week 1

## 1. Task Description

| Item       | Requirement                                                                                                                                 |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Basic      | Select two images with an overlapping region and use SIFT or another feature detection and matching method to perform image stitching.      |
| Advanced 1 | Adjust brightness, viewing angle, or overlap ratio to identify the conditions under which stitching begins to fail and explain the reasons. |
| Advanced 2 | Apply appropriate preprocessing before stitching to improve feature detection, matching, and stitching stability.                           |

## 2. Environment Setup (Anaconda + Git + Jupyter)

```bash
# Create and activate the virtual environment
conda create -n mmip python=3.11 -y
conda activate mmip

# Install required packages (including Jupyter Notebook)
pip install opencv-python numpy jupyter

# Clone the repository from GitHub
git clone https://github.com/<your-account>/MMIP.git
cd MMIP/week1/quiz_4
```

## 3. How to Run

```bash
python3 stitch.py
```

Place `image1.jpg` (left image) and `image2.jpg` (right image) in the same directory. The two images should contain an overlapping region.

```text
quiz_4/
├── stitch.py               # Main program
├── image1.jpg / image2.jpg # Input images
├── matches.jpg             # All good matches
├── matches_inliers.jpg     # RANSAC inliers
├── stitched.jpg            # Stitching result
├── stitched_cropped.jpg    # Result after removing black borders
├── variants/               # 28 automatically generated test variants
├── stitched_variants/      # Stitching results for each condition × preprocessing method
├── match_debug/            # Matching visualizations for failed / borderline cases
└── summary.csv / .md       # Complete results for 112 experiments
```

The program can also be interactively executed in Jupyter Notebook:

```bash
jupyter notebook
```

Create a new `.ipynb` file and import the functions:

```python
from stitch import *
```

Then call `basic_stitch()` or `batch_test(img1, img2)`.

## 4. Method

The overall pipeline is:

**Grayscale (optional preprocessing) → SIFT → BFMatcher kNN → Lowe Ratio Test → RANSAC Homography → Perspective Transformation → Feather Blending → Crop Black Borders**

| Parameter        |                                Value | Description                                                        |
| ---------------- | -----------------------------------: | ------------------------------------------------------------------ |
| Lowe ratio       |                                 0.75 | Filters out ambiguous matches                                      |
| MIN_MATCHES      |                                   10 | Fewer than this number of matches is directly considered a failure |
| RANSAC threshold |                               5.0 px | Maximum allowed reprojection error                                 |
| OK condition     | Inliers ≥ 15 and Inlier Ratio ≥ 0.30 | Cases below this threshold are classified as WARN                  |

### Homography

A **Homography with 8 degrees of freedom** is used instead of an affine transformation because the two images may contain perspective differences.

### Homography Validity Check

A geometric validity check is added:

* The transformed image boundary must form a convex quadrilateral.
* The transformed area must remain within a reasonable range.

This prevents cases where the number of matches appears sufficient but the estimated geometry is invalid.

### Image Blending

Distance-weighted feather blending is used to reduce visible seams between the two images.

## 5. Basic Task Results

| Item                    |                   Value |
| ----------------------- | ----------------------: |
| Image size              | 781 × 561 and 777 × 565 |
| Number of keypoints     |             3225 / 3133 |
| Good matches            |                     686 |
| Inliers                 |             681 (99.3%) |
| Mean reprojection error |                 0.97 px |

### Observation of the Stitching Result

The matched inliers and final stitching results are shown in:

* `matches_inliers.jpg`
* `stitched_cropped.jpg`

>  **Observation:** Please add comments regarding whether the seam is noticeable, whether there is distortion in distant regions, and whether there are visible differences in brightness or color.

## 6. Advanced Task: Testing Different Conditions

The program automatically generates 27 variants from `image2`, plus the original image, resulting in **28 variants**.

Each variant is tested using four preprocessing methods, resulting in:

**28 × 4 = 112 experiments**

| Category                                          | Parameters                  |
| ------------------------------------------------- | --------------------------- |
| Brightness (β)                                    | −100, −60, +60, +100        |
| Contrast (α)                                      | 0.5, 0.3                    |
| In-plane rotation                                 | 5°, 15°, 30°, 45°, 60°, 90° |
| Perspective distortion (right-side compression)   | 10%, 20%, 30%, 40%          |
| Overlap range (crop from the left side of image2) | 20%, 40%, 50%, 60%, 70%     |
| Noise σ                                           | 10, 25, 40                  |
| Blur kernel                                       | 5, 11, 21                   |

### 6.1 Failure Boundary

| Condition           | None           | CLAHE          | Norm+CLAHE                      | Blur+CLAHE     |
| ------------------- | -------------- | -------------- | ------------------------------- | -------------- |
| Brightness ±100     | All successful | All successful | All successful                  | All successful |
| Contrast            | **0.3 failed** | All successful | All successful                  | All successful |
| Rotation 5°–90°     | All successful | All successful | All successful                  | All successful |
| Perspective 10%–40% | All successful | All successful | All successful                  | All successful |
| Overlap             | **40% failed** | **40% failed** | **50% failed** (40% borderline) | **40% failed** |
| Noise σ ≤ 40        | All successful | All successful | All successful                  | All successful |
| Blur kernel ≤ 21    | All successful | All successful | All successful                  | All successful |

### Most Significant Failure Factor: Insufficient Overlap

The overlap ratio is the most obvious factor affecting stitching failure.

When 20% of the left side of `image2` is cropped:

* Good Matches remain around 320.

When 40% is cropped:

* Good Matches decrease to approximately 27.
* Inliers decrease to only 10–16.

At 50% or more:

* All cases either fail or become borderline.
* Inliers are typically ≤ 5.

> **Note:** The "crop ratio" refers to the percentage of the left side of `image2` that is removed. It does **not** directly represent the actual overlap percentage.
>
> For `overlap_crop40`, only Norm+CLAHE reaches the OK threshold, with 15 inliers exactly at the threshold. Therefore, it should be considered **borderline rather than stable success**.
>
> In `overlap_crop60`, the None preprocessing case is classified as WARN while the others fail. This difference is caused by the evaluation threshold; the five inliers obtained in these cases are not reliable enough for robust stitching.

### 6.2 Brightness

* All brightness variations within ±100 were successfully stitched.
* SIFT demonstrates a certain degree of robustness to brightness changes.
* **Overexposure has a greater impact than underexposure.**

For `bright_+100`:

* Good Matches decrease from 686 to 151 with no preprocessing.

For `bright_-100`:

* Good Matches remain at 633.

A possible explanation is that severe overexposure causes bright regions to become saturated, resulting in irreversible loss of image details. Underexposure, on the other hand, still preserves relatively more structural information.

### 6.3 Viewing Angle

* All planar rotations from 5° to 90° were successfully stitched.
* This is consistent with the rotation invariance of SIFT.
* The rotated variants contain black borders at the corners, which may have a minor influence on matching.

For perspective compression:

* Stitching remains successful up to 40% compression.
* However, Good Matches decrease gradually from approximately 636 at 10% compression to 326 at 40%.

This indicates that more severe perspective differences may eventually reach a critical point.

### 6.4 Possible Reasons for Failure

1. **Insufficient overlap**

   As the common field of view becomes smaller, the absolute number of corresponding features decreases. RANSAC then lacks enough inliers to reliably estimate the 8 degrees of freedom of the Homography. The remaining matches are more likely to be incorrect.

2. **Concentrated feature distribution**

   If the overlapping region is only a narrow strip or contains a large flat area such as the sky, the remaining features may be concentrated in a small region. In this situation, the estimated Homography can become inaccurate in areas far away from the detected features.

3. **Low contrast (0.3)**

   Lowering the overall image contrast weakens the Difference-of-Gaussian responses used by SIFT. Some features may therefore be filtered out by the contrast threshold.

   The number of Good Matches decreases from approximately 686 to 72.

   The exact reason for failure—whether insufficient matches or failure of the Homography validity check—should be verified using the `reason` column in `summary.csv`.

## 7. Advanced Task: Preprocessing Comparison

| Preprocessing  |     OK |  WARN |  FAIL | Avg. Good Matches | Avg. Inliers |
| -------------- | -----: | ----: | ----: | ----------------: | -----------: |
| None           |     23 |     1 |     4 |             377.7 |        366.1 |
| CLAHE          |     24 |     0 |     4 |             531.5 |        511.8 |
| **Norm+CLAHE** | **25** | **0** | **3** |         **549.3** |    **528.0** |
| Blur+CLAHE     |     24 |     0 |     4 |             447.8 |        437.0 |

### Preprocessing Methods

**CLAHE (Contrast Limited Adaptive Histogram Equalization)**

Enhances local contrast, allowing more features to become detectable in dark or low-contrast regions.

**Norm+CLAHE**

First normalizes the grayscale image to a fixed mean and standard deviation, followed by CLAHE. This helps handle images that are globally overexposed or underexposed.

**Blur+CLAHE**

Applies a light Gaussian blur before CLAHE to reduce noise and prevent noise from being excessively amplified by contrast enhancement.

### Conclusion

* The CLAHE-based methods produce approximately **18%–45% more Good Matches on average** than the None preprocessing method.
* Among them, **Norm+CLAHE achieves the most stable overall performance**.
* The clearest example is `contrast_0.3`: None fails with only 72 Good Matches, while all three CLAHE-based methods succeed with approximately 308–881 Good Matches.
* Under noisy conditions, **Blur+CLAHE performs best**. For `noise_40`, it obtains 308 Good Matches, compared with 199 for None and 227 for CLAHE.
* **Preprocessing cannot compensate for insufficient overlap.** Most preprocessing methods still fail when `overlap_crop40` or more severe overlap reduction is applied.
* The averages include failed cases and are intended only for comparing preprocessing methods; they should not be interpreted as absolute performance guarantees.

## 8. Limitations

* Only one pair of source images was tested. Results may differ for other scenes with different textures, repetitive patterns, or dynamic objects.
* Rotated variants contain black borders, and overlap reduction was simulated through cropping. These conditions are not exactly equivalent to real-world camera capture.
* The decision thresholds (15 inliers and 0.30 inlier ratio) are empirical values. Results near the boundary may change if different thresholds are used.

## 9. Git Upload

```bash
cd ~/MMIP
git status
git add week1/quiz_4
git commit -m "Add Quiz 4: SIFT + Homography stitching with condition experiments"
git push origin main
```

## 10. Using Google Colab

This assignment only requires CPU computation, so the local environment is sufficient. If Google Colab is used instead:

1. Install the required packages and clone the repository:

```python
!pip install opencv-python-headless numpy
!git clone https://github.com/<your-account>/MMIP.git
%cd MMIP/week1/quiz_4
```

2. Run the program:

```python
!python stitch.py
```

3. Display images using `cv2_imshow`, because `cv2.imshow()` does not work directly in Colab:

```python
import cv2
from google.colab.patches import cv2_imshow

cv2_imshow(cv2.imread("stitched_cropped.jpg"))
```

4. Upload your own images using the file panel on the left, or:

```python
from google.colab import files
files.upload()
```

5. Files in Colab are deleted when the runtime ends. Therefore, mount Google Drive or push the results back to GitHub.

6. If the images are very large, SIFT may take more time to process. The `MAX_DIM` parameter in the program can be reduced to improve processing speed.

When pushing to GitHub, use a **Personal Access Token (PAT)** and never hard-code the token into the source code.

## 11. Environment

* Python 3.11
* OpenCV with SIFT support
* NumPy
* Ubuntu
* Anaconda environment: `mmip`
