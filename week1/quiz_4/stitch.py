# ============================================================
# Quiz4：影像拼接（Image Stitching）
# 基礎：兩張影像的 SIFT 特徵偵測、匹配與拼接
# 進階：自動生成不同亮度/角度/重疊範圍的測試影像，批次測試 +
#       比較有無 CLAHE 前處理的效果，找出拼接失敗的邊界
# ============================================================

import cv2
import numpy as np
import os
import glob
import csv

IMG1_PATH = "image1.jpg"
IMG2_PATH = "image2.jpg"

VARIANTS_DIR = "variants"
os.makedirs(VARIANTS_DIR, exist_ok=True)

RATIO_THRESH = 0.6
MIN_MATCHES = 4


# ============================================================
# 共用函式：SIFT 特徵匹配 + Affine 估計
# ============================================================

def apply_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def match_features(img1, img2, use_clahe=False):
    """回傳 (keypoints1, keypoints2, good_matches)"""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    if use_clahe:
        gray1 = apply_clahe(gray1)
        gray2 = apply_clahe(gray2)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return kp1 or [], kp2 or [], []

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = [m for m, n in matches if m.distance < RATIO_THRESH * n.distance]
    return kp1, kp2, good


def estimate_affine(kp1, kp2, good_matches):
    """回傳 (M, mask)，估計失敗時 M 為 None"""
    if len(good_matches) < MIN_MATCHES:
        return None, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.estimateAffinePartial2D(
        src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    return M, mask


def warp_and_merge(img1, img2, M):
    """依 affine 矩陣把 img1 warp 過去並與 img2 合成，回傳拼接結果"""
    H = np.vstack([M, [0, 0, 1]])

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    warped_corners1 = cv2.perspectiveTransform(corners1, H)
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

    all_corners = np.concatenate((warped_corners1, corners2), axis=0)
    xmin, ymin = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    xmax, ymax = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]], dtype=np.float64)

    output_size = (xmax - xmin, ymax - ymin)
    M_translated = Ht.dot(H)[:2, :]

    warped = cv2.warpAffine(img1, M_translated, output_size)
    warped[t[1]:t[1] + h2, t[0]:t[0] + w2] = img2

    return warped


# ============================================================
# 基礎題：兩張影像的完整拼接流程
# ============================================================

def basic_stitch():
    img1 = cv2.imread(IMG1_PATH)
    img2 = cv2.imread(IMG2_PATH)

    if img1 is None or img2 is None:
        print("Error: Cannot load image1.jpg / image2.jpg")
        return

    print("Image 1:", img1.shape)
    print("Image 2:", img2.shape)

    kp1, kp2, good_matches = match_features(img1, img2, use_clahe=False)
    print("Keypoints image 1:", len(kp1))
    print("Keypoints image 2:", len(kp2))
    print("Good matches (after ratio test):", len(good_matches))

    match_image = cv2.drawMatches(
        img1, kp1, img2, kp2, good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite("matches.jpg", match_image)

    if len(good_matches) < MIN_MATCHES:
        print("Error: Not enough matches.")
        return

    M, mask = estimate_affine(kp1, kp2, good_matches)
    if M is None:
        print("Error: RANSAC failed to estimate transform.")
        return

    print("Affine matrix:")
    print(M)

    num_inliers = int(mask.sum())
    print(f"Inliers: {num_inliers} / {len(good_matches)} "
          f"({100 * num_inliers / len(good_matches):.1f}%)")

    inlier_matches = [m for i, m in enumerate(good_matches) if mask[i]]
    inlier_match_image = cv2.drawMatches(
        img1, kp1, img2, kp2, inlier_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite("matches_inliers.jpg", inlier_match_image)
    print("Saved inlier-only match visualization: matches_inliers.jpg")

    warped = warp_and_merge(img1, img2, M)
    cv2.imwrite("stitched.jpg", warped)
    print("Stitching completed!")
    print("Output: stitched.jpg")

    gray_result = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_result, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)

    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = warped[y:y + h, x:x + w]
        cv2.imwrite("stitched_cropped.jpg", cropped)
        print("Saved cropped version: stitched_cropped.jpg")

    return img1, img2


# ============================================================
# 進階題 Step 1：自動生成不同拍攝條件的 image2 變體
# ============================================================

def generate_test_variants(img2):
    h, w = img2.shape[:2]
    generated = []

    # 1. 亮度變化
    for beta in [-100, -60, 60, 100]:
        variant = cv2.convertScaleAbs(img2, alpha=1.0, beta=beta)
        name = f"bright_{'+' if beta > 0 else ''}{beta}.jpg"
        cv2.imwrite(os.path.join(VARIANTS_DIR, name), variant)
        generated.append(name)

    # 2. 角度變化（原地旋轉）
    for angle in [5, 15, 30, 45]:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        variant = cv2.warpAffine(img2, M, (w, h), borderValue=(0, 0, 0))
        name = f"rotate_{angle}.jpg"
        cv2.imwrite(os.path.join(VARIANTS_DIR, name), variant)
        generated.append(name)

    # 3. 縮小重疊範圍（裁掉左側）
    for crop_pct in [20, 40, 60]:
        crop_px = int(w * crop_pct / 100)
        variant = img2[:, crop_px:]
        name = f"overlap_crop{crop_pct}.jpg"
        cv2.imwrite(os.path.join(VARIANTS_DIR, name), variant)
        generated.append(name)

    return generated


# ============================================================
# 進階題 Step 2：批次測試（含 CLAHE 前處理對照）
# ============================================================

def evaluate_condition(img1, img2v, use_clahe):
    kp1, kp2, good = match_features(img1, img2v, use_clahe=use_clahe)

    if len(kp1) < 2 or len(kp2) < 2:
        return {
            "keypoints1": len(kp1), "keypoints2": len(kp2),
            "good_matches": 0, "inliers": 0, "inlier_ratio": 0.0,
            "status": "FAIL", "reason": "特徵點過少"
        }

    if len(good) < MIN_MATCHES:
        return {
            "keypoints1": len(kp1), "keypoints2": len(kp2),
            "good_matches": len(good), "inliers": 0, "inlier_ratio": 0.0,
            "status": "FAIL", "reason": f"good matches 過少 (<{MIN_MATCHES})"
        }

    M, mask = estimate_affine(kp1, kp2, good)
    if M is None:
        return {
            "keypoints1": len(kp1), "keypoints2": len(kp2),
            "good_matches": len(good), "inliers": 0, "inlier_ratio": 0.0,
            "status": "FAIL", "reason": "RANSAC 無法估計轉換矩陣"
        }

    num_inliers = int(mask.sum())
    inlier_ratio = num_inliers / len(good)
    status = "OK" if (num_inliers >= 4 and inlier_ratio >= 0.15) else "WARN"
    reason = "成功拼接" if status == "OK" else "inlier 數量過少/比例過低，結果可能不可靠"

    return {
        "keypoints1": len(kp1), "keypoints2": len(kp2),
        "good_matches": len(good), "inliers": num_inliers,
        "inlier_ratio": round(inlier_ratio, 3), "status": status, "reason": reason
    }


def batch_test(img1):
    variant_paths = sorted(glob.glob(os.path.join(VARIANTS_DIR, "*.jpg")))
    rows = []

    img2_orig = cv2.imread(IMG2_PATH)
    for use_clahe in [False, True]:
        r = evaluate_condition(img1, img2_orig, use_clahe)
        r["file"] = "image2_original"
        r["preprocess"] = "CLAHE" if use_clahe else "None"
        rows.append(r)

    for path in variant_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        img2v = cv2.imread(path)

        for use_clahe in [False, True]:
            r = evaluate_condition(img1, img2v, use_clahe)
            r["file"] = name
            r["preprocess"] = "CLAHE" if use_clahe else "None"
            rows.append(r)

        print(f"{name}: done")

    fieldnames = ["file", "preprocess", "keypoints1", "keypoints2",
                  "good_matches", "inliers", "inlier_ratio", "status", "reason"]

    with open("summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open("summary.md", "w", encoding="utf-8") as f:
        f.write("| 檔案 | 前處理 | KP1 | KP2 | Good Matches | Inliers | Inlier比例 | 結果 | 說明 |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['file']} | {r['preprocess']} | {r['keypoints1']} | {r['keypoints2']} | "
                f"{r['good_matches']} | {r['inliers']} | {r['inlier_ratio']} | {r['status']} | {r['reason']} |\n"
            )

    print("\n進階測試結果已存到 summary.csv / summary.md")


# ============================================================
# Main
# ============================================================

def main():
    # ---------- 基礎題 ----------
    result = basic_stitch()
    if result is None:
        return
    img1, img2 = result

    # ---------- 進階題 ----------
    print("\n" + "=" * 60)
    print("進階題：自動生成不同拍攝條件測試影像並批次測試")
    print("=" * 60)

    names = generate_test_variants(img2)
    print("已生成測試影像：", names)

    batch_test(img1)


if __name__ == "__main__":
    main()
