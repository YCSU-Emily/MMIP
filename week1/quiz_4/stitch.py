
import cv2
import numpy as np
import os
import glob
import csv
import time

IMG1_PATH = "image1.jpg"
IMG2_PATH = "image2.jpg"

VARIANTS_DIR = "variants"
STITCH_DIR = "stitched_variants"
DEBUG_DIR = "match_debug"

RATIO_THRESH = 0.75       # Lowe ratio test（0.6 太嚴、會讓臨界條件提早失敗；0.75 較常用）
MIN_MATCHES = 10          # 少於此數量直接判定失敗（Homography 理論最少 4 點，但太不穩）
RANSAC_THRESH = 5.0       # RANSAC 重投影誤差門檻（px）
OK_MIN_INLIERS = 15       # OK 需要的最少 inliers
OK_MIN_RATIO = 0.30       # OK 需要的最低 inlier 比例
MAX_DIM = 1600            # 影像最長邊超過就縮小（加速）
MAX_CANVAS = 12000        # 拼接畫布單邊上限（避免 Homography 退化時爆記憶體）

PREPROCESSES = ["None", "CLAHE", "Norm+CLAHE", "Blur+CLAHE"]

for d in (VARIANTS_DIR, STITCH_DIR, DEBUG_DIR):
    os.makedirs(d, exist_ok=True)


# ============================================================
# 一、前處理 + SIFT 特徵匹配
# ============================================================

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    m = max(img.shape[:2])
    if m > MAX_DIM:
        s = MAX_DIM / m
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


def preprocess_gray(img, mode):
    """回傳前處理後的灰階圖"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if mode == "None":
        return gray

    if mode == "Norm+CLAHE":
        # 先把整體亮度/對比拉到固定的平均與標準差，對抗整體過亮/過暗
        g = gray.astype(np.float32)
        std = g.std() if g.std() > 1e-3 else 1.0
        g = (g - g.mean()) / std * 50.0 + 128.0
        gray = np.clip(g, 0, 255).astype(np.uint8)
    elif mode == "Blur+CLAHE":
        # 先輕微去雜訊，避免 CLAHE 把雜訊一起放大
        gray = cv2.GaussianBlur(gray, (5, 5), 1.2)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


_SIFT = cv2.SIFT_create()
_feature_cache = {}


def detect_features(img, mode, cache_key=None):
    if cache_key is not None and (cache_key, mode) in _feature_cache:
        return _feature_cache[(cache_key, mode)]
    gray = preprocess_gray(img, mode)
    kp, des = _SIFT.detectAndCompute(gray, None)
    kp = list(kp) if kp is not None else []
    if cache_key is not None:
        _feature_cache[(cache_key, mode)] = (kp, des)
    return kp, des


def match_descriptors(des1, des2):
    """kNN + Lowe ratio test（安全處理少於 2 個近鄰的情況）"""
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return []
    bf = cv2.BFMatcher(cv2.NORM_L2)
    pairs = bf.knnMatch(des1, des2, k=2)
    good = []
    for pair in pairs:
        if len(pair) == 2 and pair[0].distance < RATIO_THRESH * pair[1].distance:
            good.append(pair[0])
    return good


# ============================================================
# 二、Homography + 合理性檢查
# ============================================================

def estimate_homography(kp1, kp2, good):
    """回傳 (H, mask)；失敗回傳 (None, None)。H 把 img1 座標映射到 img2 座標"""
    if len(good) < 4:
        return None, None
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESH)
    if H is None or mask is None:
        return None, None
    return H, mask.ravel().astype(bool)


def homography_sane(H, shape1):
    """檢查 H 是否退化：有限值、轉換後的 img1 外框為凸四邊形、面積比例合理"""
    if not np.all(np.isfinite(H)):
        return False, "H 含非有限值"
    h, w = shape1[:2]
    c = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    wc = cv2.perspectiveTransform(c, H).reshape(4, 2)
    if not np.all(np.isfinite(wc)):
        return False, "轉換後角點含非有限值"
    if not cv2.isContourConvex(wc.astype(np.float32).reshape(-1, 1, 2)):
        return False, "轉換後外框非凸四邊形（H 扭曲）"
    ratio = cv2.contourArea(wc.astype(np.float32)) / float(w * h)
    if ratio < 0.05 or ratio > 20:
        return False, f"轉換後面積比例異常 ({ratio:.2f})"
    return True, ""


def reprojection_error(H, kp1, kp2, good, mask):
    inl = [m for m, k in zip(good, mask) if k]
    if not inl:
        return float("nan")
    src = np.float32([kp1[m.queryIdx].pt for m in inl]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in inl]).reshape(-1, 2)
    proj = cv2.perspectiveTransform(src, H).reshape(-1, 2)
    return float(np.linalg.norm(proj - dst, axis=1).mean())


# ============================================================
# 三、拼接（Homography warp + 羽化融合）
# ============================================================

def warp_and_merge(img1, img2, H):
    """把 img1 透視轉換到 img2 座標系，重疊區用距離權重羽化融合。失敗回傳 None"""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    c1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    c2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    allc = np.concatenate((cv2.perspectiveTransform(c1, H), c2), axis=0)
    xmin, ymin = np.floor(allc.min(axis=0).ravel()).astype(int)
    xmax, ymax = np.ceil(allc.max(axis=0).ravel()).astype(int)
    W, Hh = int(xmax - xmin), int(ymax - ymin)

    if W <= 0 or Hh <= 0 or W > MAX_CANVAS or Hh > MAX_CANVAS:
        return None

    Ht = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float64)

    warped1 = cv2.warpPerspective(img1, Ht @ H, (W, Hh)).astype(np.float32)
    mask1 = cv2.warpPerspective(np.full((h1, w1), 255, np.uint8), Ht @ H, (W, Hh))

    canvas2 = np.zeros((Hh, W, 3), np.float32)
    mask2 = np.zeros((Hh, W), np.uint8)
    ox, oy = -xmin, -ymin
    canvas2[oy:oy + h2, ox:ox + w2] = img2
    mask2[oy:oy + h2, ox:ox + w2] = 255

    d1 = cv2.distanceTransform((mask1 > 0).astype(np.uint8), cv2.DIST_L2, 3)
    d2 = cv2.distanceTransform((mask2 > 0).astype(np.uint8), cv2.DIST_L2, 3)
    wsum = d1 + d2
    wsum[wsum == 0] = 1.0
    out = (warped1 * d1[..., None] + canvas2 * d2[..., None]) / wsum[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def crop_black(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero((gray > 0).astype(np.uint8))
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y + h, x:x + w]


# ============================================================
# 四、基礎題
# ============================================================

def basic_stitch():
    print("=" * 60)
    print("基礎題：SIFT + Lowe ratio + RANSAC Homography 拼接")
    print("=" * 60)

    img1, img2 = load_image(IMG1_PATH), load_image(IMG2_PATH)
    if img1 is None or img2 is None:
        print(f"錯誤：讀不到 {IMG1_PATH} / {IMG2_PATH}")
        return None

    print("Image 1:", img1.shape, " Image 2:", img2.shape)
    kp1, des1 = detect_features(img1, "None", "img1")
    kp2, des2 = detect_features(img2, "None")
    good = match_descriptors(des1, des2)
    print(f"Keypoints: {len(kp1)} / {len(kp2)}，Good matches: {len(good)}")

    cv2.imwrite("matches.jpg", cv2.drawMatches(img1, kp1, img2, kp2, good, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS))

    if len(good) < MIN_MATCHES:
        print("錯誤：good matches 不足")
        return None

    H, mask = estimate_homography(kp1, kp2, good)
    if H is None:
        print("錯誤：RANSAC 無法估計 Homography")
        return None
    ok, why = homography_sane(H, img1.shape)
    if not ok:
        print("錯誤：Homography 不合理 —", why)
        return None

    n_in = int(mask.sum())
    print(f"Inliers: {n_in}/{len(good)} ({100 * n_in / len(good):.1f}%)，"
          f"平均重投影誤差 {reprojection_error(H, kp1, kp2, good, mask):.2f}px")
    print("Homography:\n", np.round(H, 4))

    inl = [m for m, k in zip(good, mask) if k]
    cv2.imwrite("matches_inliers.jpg", cv2.drawMatches(img1, kp1, img2, kp2, inl, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS))

    pano = warp_and_merge(img1, img2, H)
    if pano is None:
        print("錯誤：拼接畫布過大（Homography 退化）")
        return None
    cv2.imwrite("stitched.jpg", pano)
    cv2.imwrite("stitched_cropped.jpg", crop_black(pano))
    print("已輸出 matches.jpg / matches_inliers.jpg / stitched.jpg / stitched_cropped.jpg")
    return img1, img2


# ============================================================
# 五、進階題 Step 1：生成測試變體
# ============================================================

def generate_variants(img2):
    """回傳 list[(name, group, param, image)]，同時寫入 variants/"""
    h, w = img2.shape[:2]
    out = []

    def add(name, group, param, img):
        cv2.imwrite(os.path.join(VARIANTS_DIR, name + ".jpg"), img)
        out.append((name, group, param, img))

    for beta in (-100, -60, 60, 100):                       # 亮度
        add(f"bright_{beta:+d}", "bright", beta, cv2.convertScaleAbs(img2, alpha=1.0, beta=beta))

    for alpha in (0.5, 0.3):                                # 對比降低
        add(f"contrast_{alpha}", "contrast", alpha, cv2.convertScaleAbs(img2, alpha=alpha, beta=(1 - alpha) * 128))

    for ang in (5, 15, 30, 45, 60):                         # 平面旋轉（四角會有黑邊）
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        add(f"rotate_{ang}", "rotate", ang, cv2.warpAffine(img2, M, (w, h)))
    add("rotate_90", "rotate", 90, cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE))

    for pct in (10, 20, 30, 40):                            # 透視（模擬拍攝角度：右側被壓縮）
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        d = h * pct / 200.0
        dst = np.float32([[0, 0], [w, d], [w, h - d], [0, h]])
        Mp = cv2.getPerspectiveTransform(src, dst)
        add(f"persp_{pct}", "persp", pct, cv2.warpPerspective(img2, Mp, (w, h)))

    for pct in (20, 40, 50, 60, 70):                        # 縮小重疊範圍（裁掉 image2 左側）
        add(f"overlap_crop{pct}", "overlap", pct, img2[:, int(w * pct / 100):])

    rng = np.random.default_rng(0)
    for s in (10, 25, 40):                                  # 雜訊
        noisy = np.clip(img2.astype(np.float32) + rng.normal(0, s, img2.shape), 0, 255).astype(np.uint8)
        add(f"noise_{s}", "noise", s, noisy)

    for k in (5, 11, 21):                                   # 模糊
        add(f"blur_{k}", "blur", k, cv2.GaussianBlur(img2, (k, k), 0))

    return out


# ============================================================
# 六、進階題 Step 2：批次評估
# ============================================================

def evaluate(img1, img2v, mode, name):
    t0 = time.time()
    row = {"file": name, "preprocess": mode, "keypoints1": 0, "keypoints2": 0,
           "good_matches": 0, "inliers": 0, "inlier_ratio": 0.0, "reproj_err": "",
           "status": "FAIL", "reason": "", "time_s": 0.0}

    kp1, des1 = detect_features(img1, mode, "img1")
    kp2, des2 = detect_features(img2v, mode)
    row["keypoints1"], row["keypoints2"] = len(kp1), len(kp2)

    if len(kp1) < 2 or len(kp2) < 2:
        row["reason"] = "特徵點過少"
        row["time_s"] = round(time.time() - t0, 2)
        return row

    good = match_descriptors(des1, des2)
    row["good_matches"] = len(good)

    if len(good) < MIN_MATCHES:
        row["reason"] = f"good matches 過少 (<{MIN_MATCHES})"
        row["time_s"] = round(time.time() - t0, 2)
        _save_debug(img1, kp1, img2v, kp2, good, name, mode)
        return row

    H, mask = estimate_homography(kp1, kp2, good)
    if H is None:
        row["reason"] = "RANSAC 無法估計 Homography"
        row["time_s"] = round(time.time() - t0, 2)
        _save_debug(img1, kp1, img2v, kp2, good, name, mode)
        return row

    n_in = int(mask.sum())
    row["inliers"] = n_in
    row["inlier_ratio"] = round(n_in / len(good), 3)
    row["reproj_err"] = round(reprojection_error(H, kp1, kp2, good, mask), 2)

    ok, why = homography_sane(H, img1.shape)
    if not ok:
        row["reason"] = "Homography 不合理：" + why
        row["time_s"] = round(time.time() - t0, 2)
        _save_debug(img1, kp1, img2v, kp2, good, name, mode, mask)
        return row

    pano = warp_and_merge(img1, img2v, H)
    if pano is None:
        row["reason"] = "拼接畫布過大（Homography 退化）"
        row["time_s"] = round(time.time() - t0, 2)
        return row
    cv2.imwrite(os.path.join(STITCH_DIR, f"{name}__{mode.replace('+', '_')}.jpg"), crop_black(pano))

    if n_in >= OK_MIN_INLIERS and row["inlier_ratio"] >= OK_MIN_RATIO:
        row["status"], row["reason"] = "OK", "成功拼接"
    else:
        row["status"] = "WARN"
        row["reason"] = f"臨界：inliers<{OK_MIN_INLIERS} 或比例<{OK_MIN_RATIO}，結果可能不可靠"
        _save_debug(img1, kp1, img2v, kp2, good, name, mode, mask)

    row["time_s"] = round(time.time() - t0, 2)
    return row


def _save_debug(img1, kp1, img2, kp2, good, name, mode, mask=None):
    """失敗 / 臨界條件存匹配圖，方便報告說明原因"""
    if mask is not None:
        good = [m for m, k in zip(good, mask) if k]
    if not good:
        return
    vis = cv2.drawMatches(img1, kp1, img2, kp2, good[:200], None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{name}__{mode.replace('+', '_')}.jpg"), vis)


def first_non_ok(rows, group, mode):
    rs = sorted([r for r in rows if r["group"] == group and r["preprocess"] == mode],
                key=lambda r: r["param"])
    return rs


def batch_test(img1, img2):
    print("\n" + "=" * 60)
    print("進階題：不同條件 × 不同前處理 批次測試")
    print("=" * 60)

    variants = [("image2_original", "original", 0, img2)] + generate_variants(img2)
    print(f"共 {len(variants)} 種條件 × {len(PREPROCESSES)} 種前處理 = {len(variants) * len(PREPROCESSES)} 組")

    rows = []
    for name, group, param, img2v in variants:
        for mode in PREPROCESSES:
            r = evaluate(img1, img2v, mode, name)
            r["group"], r["param"] = group, param
            rows.append(r)
        line = " | ".join(f"{r['preprocess']}:{r['status']}({r['good_matches']}/{r['inliers']})"
                          for r in rows[-len(PREPROCESSES):])
        print(f"{name:<18} {line}")

    fields = ["file", "group", "param", "preprocess", "keypoints1", "keypoints2", "good_matches",
              "inliers", "inlier_ratio", "reproj_err", "status", "reason", "time_s"]
    with open("summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    md = ["# Quiz 4 進階測試結果\n",
          f"參數：ratio={RATIO_THRESH}、MIN_MATCHES={MIN_MATCHES}、RANSAC={RANSAC_THRESH}px；"
          f"OK 需 inliers ≥ {OK_MIN_INLIERS} 且比例 ≥ {OK_MIN_RATIO}\n",
          "Good Matches / Inliers 格式為 `good/inliers`\n",
          "| 條件 | " + " | ".join(PREPROCESSES) + " |",
          "|---|" + "---|" * len(PREPROCESSES)]
    for name, *_ in variants:
        cells = []
        for mode in PREPROCESSES:
            r = next(x for x in rows if x["file"] == name and x["preprocess"] == mode)
            cells.append(f"{r['status']} ({r['good_matches']}/{r['inliers']})")
        md.append(f"| {name} | " + " | ".join(cells) + " |")

    # ---- 前處理效果統計 ----
    print("\n=== 前處理效果統計（所有條件）===")
    md.append("\n## 前處理效果統計\n")
    md.append("| 前處理 | OK | WARN | FAIL | 平均 Good Matches | 平均 Inliers |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for mode in PREPROCESSES:
        rs = [r for r in rows if r["preprocess"] == mode]
        n = {s: sum(r["status"] == s for r in rs) for s in ("OK", "WARN", "FAIL")}
        gm = np.mean([r["good_matches"] for r in rs])
        inl = np.mean([r["inliers"] for r in rs])
        print(f"{mode:<12} OK {n['OK']:>2}  WARN {n['WARN']:>2}  FAIL {n['FAIL']:>2}  "
              f"avg good={gm:.1f}  avg inliers={inl:.1f}")
        md.append(f"| {mode} | {n['OK']} | {n['WARN']} | {n['FAIL']} | {gm:.1f} | {inl:.1f} |")

    # ---- 失效邊界 ----
    print("\n=== 失效邊界（各條件第一個非 OK 的參數）===")
    md.append("\n## 失效邊界（第一個非 OK 的參數）\n")
    md.append("| 條件類型 | " + " | ".join(PREPROCESSES) + " |")
    md.append("|---|" + "---|" * len(PREPROCESSES))
    for g in ("bright", "contrast", "rotate", "persp", "overlap", "noise", "blur"):
        cells = []
        for mode in PREPROCESSES:
            rs = first_non_ok(rows, g, mode)
            bad = [r for r in rs if r["status"] != "OK"]
            if not rs:
                cells.append("-")
            elif not bad:
                cells.append("全部成功")
            else:
                cells.append(f"{bad[0]['param']}（{bad[0]['status']}）")
        print(f"{g:<9}" + "  ".join(f"{m}: {c}" for m, c in zip(PREPROCESSES, cells)))
        md.append(f"| {g} | " + " | ".join(cells) + " |")
    md.append("\n> 註：亮度為 beta 值，可能同時有正負值，請對照上方明細表；旋轉變體含黑色邊角。")

    with open("summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print("\n已輸出 summary.csv、summary.md、stitched_variants/、match_debug/")


# ============================================================
# Main
# ============================================================

def main():
    result = basic_stitch()
    if result is None:
        return
    img1, img2 = result
    batch_test(img1, img2)


if __name__ == "__main__":
    main()
