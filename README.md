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
Quiz 3 — Perspective Transformation

Objective

Correct a tilted document/image using Perspective Transformation.

Processing Pipeline

text
Input Image
     ↓
Grayscale
     ↓
Gaussian Blur
     ↓
Canny Edge Detection
     ↓
Contour Detection
     ↓
Quadrilateral Detection
     ↓
Perspective Transformation
     ↓
Corrected Image


Advanced Experiment

Different camera angles are tested to determine the conditions under which the automatic correction fails.



Quiz 4 — Image Stitching

Objective

Combine two overlapping images into a single panoramic image.

Processing Pipeline

text
Image 1 ──→ SIFT ──→ Feature Matching ──┐
                                        ↓
Image 2 ──→ SIFT ──→ Feature Matching ──→ RANSAC
                                        ↓
                                  Transformation
                                        ↓
                                  Image Stitching


Advanced Experiment

The effects of the following conditions are investigated:

* Different overlap ratios
* Different brightness levels
* Different shooting angles
* Image preprocessing

---

Project Structure

text
MMIP/
├── quiz1/
├── quiz2/
├── quiz3/
├── quiz4/
├── requirements.txt
└── README.md






Each quiz contains its own input images, source code, and generated results.

