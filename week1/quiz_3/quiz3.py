# ============================================================
# Quiz3：梯形校正與透視轉換
# 基礎：單張影像自動偵測四邊形 + 透視校正
# 進階：自動生成多種拍攝條件的測試影像，批次測試 + 失敗邊界分析
# ============================================================

import cv2
import numpy as np
import os
import glob
import csv
import math

IMAGE_PATH = "images/test1.jpg"   # 基礎題使用的單張測試影像
FLAT_REF_PATH = "results/test1_corrected.jpg"  # 進階題用來合成測試影像的「地面真相」平面

VARIANTS_DIR = "images_variants"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VARIANTS_DIR, exist_ok=True)


# ============================================================
# 共用函式：角點排序 / 透視轉換 / 四邊形偵測
# ============================================================

def order_points(pts):
    """將四個角點排序成：左上、右上、右下、左下"""
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下

    return rect


def perspective_transform(image, points):
    """依四個角點執行 Perspective Transformation"""
    rect = order_points(points)
    tl, tr, br, bl = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    dst = np.float32([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ])

    matrix = cv2.getPerspectiveTransform(rect, dst)
    result = cv2.warpPerspective(image, matrix, (max_width, max_height))

    return result, rect


def detect_quadrilateral(image, max_contours_checked=10):
    """
    偵測影像中面積最大、且能被近似成四邊形的輪廓。
    回傳 (document, edges, checked_count)；找不到時 document 為 None。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    document = None
    checked = 0

    for contour in contours[:max_contours_checked]:
        checked += 1
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4:
            document = approx
            break

    return document, edges, checked


# ============================================================
# 基礎題：處理單張影像
# ============================================================

def correct_single_image(image_path):
    print(f"\nProcessing: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        print("ERROR: Cannot read image.")
        return

    document, edges, checked = detect_quadrilateral(image)

    if document is None:
        print(f"ERROR: Cannot find document (checked {checked} contours).")
        return

    print("Found quadrilateral!")

    result, _ = perspective_transform(image, document)

    filename = os.path.basename(image_path)
    name, _ = os.path.splitext(filename)

    corrected_path = f"{RESULTS_DIR}/{name}_corrected.jpg"
    edges_path = f"{RESULTS_DIR}/{name}_edges.jpg"

    cv2.imwrite(corrected_path, result)
    cv2.imwrite(edges_path, edges)

    print(f"Saved: {corrected_path}")
    print(f"Saved: {edges_path}")

    return result


# ============================================================
# 進階題 Step 1：自動生成不同拍攝條件的測試影像
# ============================================================

CANVAS_W, CANVAS_H = 1000, 1000
BG_COLOR = (35, 30, 28)
FOCAL = 900.0
DIST = 1300.0


def project_plane(doc_w, doc_h, theta_deg, tilt_axis="y"):
    """
    簡化的針孔相機模型：將文件平面繞垂直軸(y)或水平軸(x)旋轉 theta 度，
    計算四個角點投影到畫布上的位置，回傳可用於 getPerspectiveTransform 的座標。
    """
    theta = math.radians(theta_deg)
    half_w, half_h = doc_w / 2.0, doc_h / 2.0

    pts_2d = []
    for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:  # TL, TR, BR, BL
        x, y, z = sx * half_w, sy * half_h, 0.0

        if tilt_axis == "y":
            x2, z2, y2 = x * math.cos(theta), -x * math.sin(theta), y
        else:
            y2, z2, x2 = y * math.cos(theta), -y * math.sin(theta), x

        depth = DIST + z2
        pts_2d.append((FOCAL * x2 / depth, FOCAL * y2 / depth))

    dst = np.array(pts_2d, dtype=np.float32) + np.array(
        [CANVAS_W / 2.0, CANVAS_H / 2.0], dtype=np.float32
    )
    return dst


def warp_onto_canvas(doc_img, dst_pts):
    h, w = doc_img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, dst_pts)

    canvas = np.full((CANVAS_H, CANVAS_W, 3), BG_COLOR, dtype=np.uint8)
    warped = cv2.warpPerspective(
        doc_img, M, (CANVAS_W, CANVAS_H),
        dst=canvas.copy(), borderMode=cv2.BORDER_TRANSPARENT
    )
    return warped


def generate_test_variants(flat_doc):
    """產生不同角度 / 低對比 / 遮擋條件的測試影像，存到 images_variants/"""
    h, w = flat_doc.shape[:2]
    generated = []

    # 1. 不同角度（繞垂直軸，模擬左右側拍）
    for angle in [10, 20, 30, 40, 50, 60, 70, 80]:
        dst = project_plane(w, h, angle, tilt_axis="y")
        warped = warp_onto_canvas(flat_doc, dst)
        name = f"tilt_y_{angle:02d}.jpg"
        cv2.imwrite(os.path.join(VARIANTS_DIR, name), warped)
        generated.append(name)

    # 2. 不同角度（繞水平軸，模擬俯視角度）
    for angle in [20, 40, 60]:
        dst = project_plane(w, h, angle, tilt_axis="x")
        warped = warp_onto_canvas(flat_doc, dst)
        name = f"tilt_x_{angle:02d}.jpg"
        cv2.imwrite(os.path.join(VARIANTS_DIR, name), warped)
        generated.append(name)

    # 3. 低對比（固定 30 度，降低文件與背景的對比）
    dst = project_plane(w, h, 30, tilt_axis="y")
    dark_doc = cv2.convertScaleAbs(flat_doc, alpha=0.35, beta=20)
    warped = warp_onto_canvas(dark_doc, dst)
    name = "low_contrast_30.jpg"
    cv2.imwrite(os.path.join(VARIANTS_DIR, name), warped)
    generated.append(name)

    # 4. 邊界遮擋（固定 30 度，用背景色蓋住一角）
    dst = project_plane(w, h, 30, tilt_axis="y")
    warped = warp_onto_canvas(flat_doc, dst)
    pts = dst.astype(int)
    x_max, y_min = pts[:, 0].max(), pts[:, 1].min()
    box_w = int((pts[:, 0].max() - pts[:, 0].min()) * 0.35)
    box_h = int((pts[:, 1].max() - pts[:, 1].min()) * 0.35)
    cv2.rectangle(warped, (x_max - box_w, y_min), (x_max + 20, y_min + box_h), BG_COLOR, -1)
    name = "occluded_30.jpg"
    cv2.imwrite(os.path.join(VARIANTS_DIR, name), warped)
    generated.append(name)

    return generated


# ============================================================
# 進階題 Step 2：批次測試 + 失敗原因記錄
# ============================================================

def batch_test():
    variant_paths = sorted(glob.glob(os.path.join(VARIANTS_DIR, "*.jpg")))
    rows = []

    for path in variant_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        image = cv2.imread(path)

        document, edges, checked = detect_quadrilateral(image)

        if document is None:
            cv2.imwrite(f"{RESULTS_DIR}/{name}_edges.jpg", edges)
            row = {
                "file": name, "status": "FAIL",
                "reason": f"在前 {checked} 大輪廓中找不到四邊形（可能邊緣糾纏或角度過大/遮擋）"
            }
        else:
            result, _ = perspective_transform(image, document)
            out_w, out_h = result.shape[1], result.shape[0]
            aspect_ok = 0.3 < (out_w / max(out_h, 1)) < 3.0

            cv2.imwrite(f"{RESULTS_DIR}/{name}_corrected.jpg", result)

            vis = image.copy()
            cv2.drawContours(vis, [document], -1, (0, 255, 0), 3)
            cv2.imwrite(f"{RESULTS_DIR}/{name}_contour.jpg", vis)

            status = "OK" if aspect_ok else "WARN"
            reason = "成功偵測並校正" if aspect_ok else "偵測到四邊形，但長寬比異常，結果可能不可靠"
            row = {"file": name, "status": status, "reason": reason}

        print(f"{name}: {row['status']} - {row['reason']}")
        rows.append(row)

    with open("summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "status", "reason"])
        writer.writeheader()
        writer.writerows(rows)

    with open("summary.md", "w", encoding="utf-8") as f:
        f.write("| 檔案 | 結果 | 說明 |\n|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['file']} | {r['status']} | {r['reason']} |\n")

    n_ok = sum(1 for r in rows if r["status"] == "OK")
    n_warn = sum(1 for r in rows if r["status"] == "WARN")
    n_fail = sum(1 for r in rows if r["status"] == "FAIL")

    print(f"\n=== 進階測試總結：共 {len(rows)} 組，OK {n_ok}，WARN {n_warn}，FAIL {n_fail} ===")
    print("完整結果已存到 summary.csv / summary.md")


# ============================================================
# Main
# ============================================================

def main():
    # ---------- 基礎題 ----------
    correct_single_image(IMAGE_PATH)

    # ---------- 進階題 ----------
    # 用基礎題已校正出來的近似正視影像，作為生成測試變體的「地面真相」平面
    flat_doc = cv2.imread(FLAT_REF_PATH)

    if flat_doc is None:
        print(f"\n[進階題跳過] 找不到 {FLAT_REF_PATH}，"
              f"請先確認基礎題已成功執行並產生校正結果。")
        return

    print("\n" + "=" * 60)
    print("進階題：自動生成多種拍攝條件測試影像並批次測試")
    print("=" * 60)

    names = generate_test_variants(flat_doc)
    print("已生成測試影像：", names)

    batch_test()


if __name__ == "__main__":
    main()
