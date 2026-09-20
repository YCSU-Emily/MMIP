import cv2
import numpy as np
import os
import glob
import csv
import json
import math

IMAGES_DIR = "images"            # 放你自己的真實照片（例如 test1.jpg）
TEST_DIR = "test_images"         # 程式自動生成的測試影像
RESULTS_DIR = "results"
REAL_RESULTS_DIR = os.path.join(RESULTS_DIR, "real")

BASIC_IMAGE = os.path.join(IMAGES_DIR, "test1.jpg")

# 判定門檻（單位：像素，畫布 1000x1000）
ERR_OK = 20.0        # 角點平均誤差 < 20px → OK
ERR_WARN = 50.0      # 20~50px → WARN；>50px 或找不到 → FAIL

for d in (IMAGES_DIR, TEST_DIR, RESULTS_DIR, REAL_RESULTS_DIR):
    os.makedirs(d, exist_ok=True)


# ============================================================
# 一、核心演算法
# ============================================================

def order_points(pts):
    """把四個點排成 左上、右上、右下、左下（依質心角度排序，比 sum/diff 法在旋轉時更穩）"""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]              # 影像座標 y 向下 → 角度遞增 = 順時針
    start = int(np.argmin(pts.sum(axis=1)))  # 左上 = x+y 最小
    return np.roll(pts, -start, axis=0)


def perspective_transform(image, points):
    """依四個角點做 Perspective Transformation，回傳 (正視影像, 排序後角點)"""
    rect = order_points(points)
    tl, tr, br, bl = rect

    max_width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    max_height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    max_width, max_height = max(max_width, 2), max(max_height, 2)

    dst = np.float32([[0, 0], [max_width - 1, 0],
                      [max_width - 1, max_height - 1], [0, max_height - 1]])
    matrix = cv2.getPerspectiveTransform(rect, dst)
    result = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return result, rect


def _binary_maps(gray):
    """產生多種二值化 / 邊緣圖，讓偵測不依賴單一參數（提高自動化程度）"""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    maps = []

    for name, lo, hi in [("canny", 50, 150), ("canny_low", 15, 45)]:
        e = cv2.Canny(blur, lo, hi)
        e = cv2.dilate(e, np.ones((3, 3), np.uint8), iterations=1)
        e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        maps.append((name, e))

    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = np.ones((15, 15), np.uint8)
    for name, b in [("otsu", otsu), ("otsu_inv", 255 - otsu)]:
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k)
        b = cv2.morphologyEx(b, cv2.MORPH_OPEN, k)
        maps.append((name, b))

    return maps


def _quad_from_contour(contour):
    """輪廓 → 凸包 → 逐步放寬 epsilon 近似成四邊形；找不到就回傳 None"""
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    for eps in (0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32), "approx"
    # 最後手段：最小外接矩形（對梯形不精確，所以分數會被打折）
    box = cv2.boxPoints(cv2.minAreaRect(hull))
    return box.astype(np.float32), "minAreaRect"


def detect_quadrilateral(image, min_area_ratio=0.03, max_area_ratio=0.95, top_n=8):
    """
    多策略文件偵測。回傳 dict：
      quad(4x2 or None), method, score, area_ratio, touches_border, debug_map, checked
    評分 = 面積比例 × 貼合度(凸包面積/四邊形面積) × 方法係數
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    img_area = float(h * w)

    best = None
    checked = 0
    first_map = None

    for mname, bmap in _binary_maps(gray):
        if first_map is None:
            first_map = bmap
        contours, _ = cv2.findContours(bmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:top_n]

        for c in contours:
            checked += 1
            ratio = cv2.contourArea(c) / img_area
            if ratio < min_area_ratio or ratio > max_area_ratio:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            if bw >= 0.97 * w and bh >= 0.97 * h:   # 整個畫面外框（背景/影像邊界），不是文件
                continue

            quad, how = _quad_from_contour(c)
            quad_area = cv2.contourArea(quad)
            hull_area = cv2.contourArea(cv2.convexHull(c))
            if quad_area < 1:
                continue

            fit = min(hull_area, quad_area) / max(hull_area, quad_area)
            score = ratio * fit * (1.0 if how == "approx" else 0.6)

            if best is None or score > best["score"]:
                margin = 3
                touches = bool(np.any(quad[:, 0] < margin) or np.any(quad[:, 1] < margin) or
                               np.any(quad[:, 0] > w - 1 - margin) or np.any(quad[:, 1] > h - 1 - margin))
                best = {"quad": quad, "method": f"{mname}/{how}", "score": score,
                        "area_ratio": ratio, "touches_border": touches, "debug_map": bmap}

    if best is None:
        best = {"quad": None, "method": "none", "score": 0.0, "area_ratio": 0.0,
                "touches_border": False, "debug_map": first_map}
    best["checked"] = checked
    return best


def correct_image(image):
    """完整流程：偵測 → 透視轉換。回傳 (result or None, detection dict)"""
    det = detect_quadrilateral(image)
    if det["quad"] is None:
        return None, det
    result, rect = perspective_transform(image, det["quad"])
    det["quad"] = rect
    return result, det


# ============================================================
# 二、基礎題：單張影像
# ============================================================

def draw_overlay(image, quad, color=(0, 255, 0)):
    vis = image.copy()
    if quad is not None:
        cv2.polylines(vis, [quad.astype(np.int32).reshape(-1, 1, 2)], True, color, 3)
        for i, p in enumerate(quad):
            cv2.circle(vis, tuple(int(v) for v in p), 8, (0, 0, 255), -1)
            cv2.putText(vis, "TL TR BR BL".split()[i], (int(p[0]) + 8, int(p[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return vis


def basic_task(fallback_image=None):
    print("=" * 60)
    print("基礎題：單張斜拍影像 → 正視影像")
    print("=" * 60)

    if os.path.exists(BASIC_IMAGE):
        image = cv2.imread(BASIC_IMAGE)
        src_name = BASIC_IMAGE
    elif fallback_image is not None:
        image = fallback_image
        src_name = "（找不到 images/test1.jpg，改用自動生成的 tilt_y_30）"
    else:
        print("找不到基礎題影像。")
        return
    print("來源：", src_name)

    result, det = correct_image(image)
    if result is None:
        print("失敗：找不到四邊形")
        cv2.imwrite(f"{RESULTS_DIR}/basic_edges.jpg", det["debug_map"])
        return

    cv2.imwrite(f"{RESULTS_DIR}/basic_original.jpg", image)
    cv2.imwrite(f"{RESULTS_DIR}/basic_contour.jpg", draw_overlay(image, det["quad"]))
    cv2.imwrite(f"{RESULTS_DIR}/basic_corrected.jpg", result)
    print(f"偵測方法：{det['method']}，輸出尺寸：{result.shape[1]}x{result.shape[0]}")
    print(f"已儲存 {RESULTS_DIR}/basic_original.jpg / basic_contour.jpg / basic_corrected.jpg")


# ============================================================
# 三、自動生成測試影像（模擬不同拍攝條件）
# ============================================================

CW, CH = 1000, 1000
FOCAL, DIST = 900.0, 1300.0
DARK_BG = (35, 30, 28)


def make_document(w=800, h=1100, seed=7):
    """程式生成一張「假文件」：標題、文字行、表格、圖片區、方塊圖案"""
    rng = np.random.default_rng(seed)
    doc = np.full((h, w, 3), 245, np.uint8)
    cv2.rectangle(doc, (0, 0), (w - 1, h - 1), (170, 170, 170), 4)

    cv2.putText(doc, "PERSPECTIVE TEST", (60, 110), cv2.FONT_HERSHEY_DUPLEX, 1.7, (30, 30, 30), 3)
    cv2.line(doc, (60, 140), (w - 60, 140), (60, 60, 200), 4)

    y = 200
    for _ in range(9):
        x = 60
        while x < w - 120:
            wl = int(rng.integers(50, 150))
            cv2.putText(doc, "x" * max(2, wl // 18), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (50, 50, 50), 2)
            x += wl + 20
        y += 42

    # 表格
    ty = 620
    for r in range(5):
        for c in range(4):
            cv2.rectangle(doc, (60 + c * 165, ty + r * 45), (60 + (c + 1) * 165, ty + (r + 1) * 45),
                          (90, 90, 90), 2)
            cv2.putText(doc, str(int(rng.integers(10, 999))), (75 + c * 165, ty + 32 + r * 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2)

    # 圖片區（漸層 + 圓）
    grad = np.tile(np.linspace(60, 200, 300, dtype=np.uint8), (200, 1))
    pic = cv2.merge([grad, grad[:, ::-1], np.full_like(grad, 120)])
    cv2.circle(pic, (150, 100), 60, (30, 160, 230), -1)
    doc[880:1080, 60:360] = pic

    # 方塊圖案（QR 風）
    for i in range(10):
        for j in range(10):
            if rng.random() > 0.5:
                cv2.rectangle(doc, (500 + i * 20, 880 + j * 20), (520 + i * 20, 900 + j * 20),
                              (20, 20, 20), -1)
    return doc


def make_background(kind, rng):
    if kind == "dark":
        bg = np.full((CH, CW, 3), DARK_BG, np.float32)
        bg += rng.normal(0, 3, (CH, CW, 1))
    elif kind == "light":
        bg = np.full((CH, CW, 3), (215, 215, 210), np.float32)
        bg += rng.normal(0, 3, (CH, CW, 1))
    elif kind == "gradient":
        g = np.tile(np.linspace(40, 170, CW, dtype=np.float32), (CH, 1))
        bg = np.dstack([g * 0.8, g * 0.9, g])
    else:  # wood / cluttered
        n = rng.normal(0, 1, (CH, CW)).astype(np.float32)
        n = cv2.blur(cv2.blur(n, (121, 1)), (61, 1)) * 400
        bg = np.dstack([np.full((CH, CW), 45, np.float32) + n * 0.6,
                        np.full((CH, CW), 85, np.float32) + n * 0.9,
                        np.full((CH, CW), 135, np.float32) + n * 1.2])
        bg += rng.normal(0, 4, (CH, CW, 1))
        if kind == "cluttered":
            bg = np.clip(bg, 0, 255).astype(np.uint8)
            for _ in range(10):
                cx, cy = int(rng.integers(0, CW)), int(rng.integers(0, CH))
                col = tuple(int(v) for v in rng.integers(30, 250, 3))
                if rng.random() < 0.5:
                    rect = ((cx, cy), (int(rng.integers(60, 240)), int(rng.integers(40, 160))),
                            float(rng.integers(0, 90)))
                    cv2.fillPoly(bg, [cv2.boxPoints(rect).astype(np.int32)], col)
                else:
                    cv2.circle(bg, (cx, cy), int(rng.integers(25, 90)), col, -1)
            for _ in range(6):
                cv2.line(bg, (int(rng.integers(0, CW)), int(rng.integers(0, CH))),
                         (int(rng.integers(0, CW)), int(rng.integers(0, CH))),
                         tuple(int(v) for v in rng.integers(0, 255, 3)), int(rng.integers(3, 9)))
            bg = bg.astype(np.float32)
    return np.clip(bg, 0, 255).astype(np.uint8)


def project_corners(w, h, ay=0.0, ax=0.0, roll=0.0, scale=1.0, shift=(0, 0)):
    """針孔相機模型：文件先繞 y 軸轉 ay、再繞 x 軸轉 ax，投影後再做平面旋轉 roll / 縮放 / 平移"""
    ty, tx, tr = math.radians(ay), math.radians(ax), math.radians(roll)
    pts = []
    for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:      # TL TR BR BL
        x, y = sx * w / 2.0, sy * h / 2.0
        x1, z1, y1 = x * math.cos(ty), -x * math.sin(ty), y
        y2 = y1 * math.cos(tx) - z1 * math.sin(tx)
        z2 = y1 * math.sin(tx) + z1 * math.cos(tx)
        depth = DIST + z2
        px, py = FOCAL * x1 / depth, FOCAL * y2 / depth
        px, py = (px * math.cos(tr) - py * math.sin(tr), px * math.sin(tr) + py * math.cos(tr))
        pts.append((px, py))
    pts = np.float32(pts)
    # 自動置中並縮放到畫面內（避免文件被畫布截斷而干擾角度分析），再套用 scale / shift
    pts -= (pts.max(axis=0) + pts.min(axis=0)) / 2.0
    extent = (pts.max(axis=0) - pts.min(axis=0)).max()
    fit = min(1.0, 0.85 * min(CW, CH) / extent)
    return pts * fit * scale + np.float32([CW / 2 + shift[0], CH / 2 + shift[1]])


def render_scene(doc, ay=0, ax=0, roll=0, scale=1.0, shift=(0, 0), bg="dark",
                 fade=0.0, shadow=False, noise=0.0, blur=0, occlude=None, seed=0):
    rng = np.random.default_rng(seed)
    h, w = doc.shape[:2]
    dst = project_corners(w, h, ay, ax, roll, scale, shift)
    bgimg = make_background(bg, rng).astype(np.float32)

    d = doc.astype(np.float32)
    if fade > 0:   # 讓文件顏色往背景平均色靠近 → 低對比
        d = d * (1 - fade) + bgimg.mean(axis=(0, 1)) * fade

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(d, M, (CW, CH))
    mask = cv2.warpPerspective(np.full((h, w), 1.0, np.float32), M, (CW, CH))[..., None]
    img = warped * mask + bgimg * (1 - mask)

    if shadow:     # 側光：由左到右逐漸變暗
        light = np.tile(np.linspace(1.0, 0.3, CW, dtype=np.float32), (CH, 1))[..., None]
        img = img * light

    img = np.clip(img, 0, 255).astype(np.uint8)

    if occlude == "corner":      # 用背景色蓋住右上角
        p = dst.astype(int)
        bw = int((p[:, 0].max() - p[:, 0].min()) * 0.3)
        bh = int((p[:, 1].max() - p[:, 1].min()) * 0.3)
        cv2.rectangle(img, (p[:, 0].max() - bw, p[:, 1].min() - 5),
                      (p[:, 0].max() + 20, p[:, 1].min() + bh),
                      tuple(int(v) for v in bgimg.mean(axis=(0, 1))), -1)
    elif occlude == "finger":    # 手指壓住右下角
        p = dst[2].astype(int)
        cv2.ellipse(img, (int(p[0]) - 20, int(p[1]) - 10), (90, 45), -40, 0, 360, (120, 150, 220), -1)

    if blur > 0:
        k = blur | 1
        img = cv2.GaussianBlur(img, (k, k), 0)
    if noise > 0:
        img = np.clip(img.astype(np.float32) + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)

    return img, dst


def generate_test_images(doc):
    """回傳 list[dict(name, group, param, image, gt)]，同時把影像寫入 test_images/"""
    items = []

    def add(name, group, param, **kw):
        img, gt = render_scene(doc, seed=len(items), **kw)
        cv2.imwrite(os.path.join(TEST_DIR, name + ".jpg"), img)
        items.append({"name": name, "group": group, "param": param, "image": img, "gt": gt})

    for a in list(range(10, 90, 10)) + [85, 88]:
        add(f"tilt_y_{a:02d}", "tilt_y", a, ay=a)
    for a in list(range(10, 90, 10)) + [85, 88]:
        add(f"tilt_x_{a:02d}", "tilt_x", a, ax=a)
    for ay, ax in [(30, 30), (45, 30), (60, 45)]:
        add(f"tilt_xy_{ay}_{ax}", "tilt_xy", ay + ax, ay=ay, ax=ax)
    for r in [15, 45, 90]:
        add(f"roll_{r:02d}_y30", "roll", r, ay=30, roll=r)

    for bg in ["wood", "cluttered", "light", "gradient"]:
        add(f"bg_{bg}_y30", "background", bg, ay=30, bg=bg)
    add("bg_cluttered_y50", "background", "cluttered_y50", ay=50, bg="cluttered")

    for f in [0.6, 0.8, 0.9, 0.95, 0.98]:
        add(f"lowcontrast_{int(f * 100)}_y30", "contrast", f, ay=30, fade=f, noise=8)
    for a in [30, 50]:
        add(f"shadow_y{a}", "shadow", a, ay=a, shadow=True)
    for s in [25, 50]:
        add(f"noise_{s}_y30", "noise", s, ay=30, noise=s)
    for b in [5, 11]:
        add(f"blur_{b}_y30", "blur", b, ay=30, blur=b)

    add("occluded_corner_y30", "occlusion", "corner", ay=30, occlude="corner")
    add("occluded_finger_y30", "occlusion", "finger", ay=30, occlude="finger")
    add("cutoff_y30", "cutoff", "edge", ay=30, shift=(380, 0))
    add("far_scale_0.5_y30", "scale", 0.5, ay=30, scale=0.5)
    add("far_scale_0.25_y30", "scale", 0.25, ay=30, scale=0.25)
    return items


# ============================================================
# 四、評估
# ============================================================

def similarity_to_flat(result, flat):
    """校正結果與正視文件的相似度（灰階正規化相關；容許 90° 倍數旋轉）"""
    size = (160, 220)
    f = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(flat, size), cv2.COLOR_BGR2GRAY), (5, 5), 0)
    best = -1.0
    for k in range(4):
        r = np.ascontiguousarray(np.rot90(result, k))
        r = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(r, size), cv2.COLOR_BGR2GRAY), (5, 5), 0)
        best = max(best, float(cv2.matchTemplate(r, f, cv2.TM_CCOEFF_NORMED)[0, 0]))
    return best


def evaluate(item, flat):
    img, gt = item["image"], item["gt"]
    result, det = correct_image(img)
    row = {"file": item["name"], "group": item["group"], "param": item["param"],
           "method": det["method"], "corner_err_px": "", "similarity": "",
           "status": "FAIL", "reason": ""}

    if result is None:
        row["reason"] = "找不到符合條件的四邊形（文件面積過小 <3%、或邊緣/二值化未形成封閉外輪廓）"
        cv2.imwrite(f"{RESULTS_DIR}/{item['name']}_edges.jpg", det["debug_map"])
        return row, None, det

    dq, gq = order_points(det["quad"]), order_points(gt)
    # 平面旋轉接近 45° 時「左上角」定義會模糊，所以取四種循環排序中誤差最小者
    err = min(float(np.linalg.norm(np.roll(dq, k, axis=0) - gq, axis=1).mean()) for k in range(4))
    sim = similarity_to_flat(result, flat)
    row["corner_err_px"] = round(err, 1)
    row["similarity"] = round(sim, 3)

    if err < ERR_OK and sim >= 0.5:
        row["status"], row["reason"] = "OK", "角點準確，校正成功"
    elif err < ERR_WARN:
        row["status"], row["reason"] = "WARN", "角點有偏移，校正結果有變形或相似度偏低"
    else:
        row["status"] = "FAIL"
        row["reason"] = "偵測到錯誤的四邊形（角點嚴重偏離真實文件）"
    if det["touches_border"]:
        row["reason"] += "；四邊形貼近影像邊界（文件可能被截斷）"

    cv2.imwrite(f"{RESULTS_DIR}/{item['name']}_corrected.jpg", result)
    vis = draw_overlay(img, det["quad"])
    for p in gt:  # 黃色 = 真實角點
        cv2.circle(vis, tuple(int(v) for v in p), 6, (0, 255, 255), 2)
    cv2.imwrite(f"{RESULTS_DIR}/{item['name']}_contour.jpg", vis)
    return row, vis, det


def boundary(rows, group):
    rs = sorted([r for r in rows if r["group"] == group], key=lambda r: r["param"])
    last_ok = first_bad = None
    for r in rs:
        if first_bad is None:
            if r["status"] == "OK":
                last_ok = r["param"]
            else:
                first_bad = r["param"]
    return rs, last_ok, first_bad


def make_contact_sheet(items_vis, path, cols=6, cell=(250, 250)):
    if not items_vis:
        return
    rows_img = []
    for i in range(0, len(items_vis), cols):
        chunk = items_vis[i:i + cols]
        cells = []
        for name, status, vis in chunk:
            t = cv2.resize(vis, cell)
            color = {"OK": (0, 180, 0), "WARN": (0, 200, 255), "FAIL": (0, 0, 220)}[status]
            cv2.rectangle(t, (0, 0), (cell[0] - 1, cell[1] - 1), color, 6)
            cv2.putText(t, f"{name[:22]} {status}", (6, cell[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cells.append(t)
        while len(cells) < cols:
            cells.append(np.zeros((cell[1], cell[0], 3), np.uint8))
        rows_img.append(np.hstack(cells))
    cv2.imwrite(path, np.vstack(rows_img))


def batch_test(items, flat):
    print("\n" + "=" * 60)
    print(f"進階題：批次測試 {len(items)} 張自動生成的測試影像")
    print("=" * 60)

    rows, sheet = [], []
    for it in items:
        row, vis, _ = evaluate(it, flat)
        rows.append(row)
        if vis is None:
            vis = it["image"]
        sheet.append((it["name"], row["status"], vis))
        print(f"{row['file']:<24} {row['status']:<5} err={row['corner_err_px']!s:<6} "
              f"sim={row['similarity']!s:<6} {row['reason']}")

    make_contact_sheet(sheet, f"{RESULTS_DIR}/contact_sheet.jpg")

    fields = ["file", "group", "param", "method", "corner_err_px", "similarity", "status", "reason"]
    with open("summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    n = {s: sum(r["status"] == s for r in rows) for s in ("OK", "WARN", "FAIL")}

    lines = ["# Quiz 3 進階測試結果\n",
             f"共 {len(rows)} 組：OK {n['OK']}、WARN {n['WARN']}、FAIL {n['FAIL']}\n",
             f"判定：角點平均誤差 < {ERR_OK}px 且相似度 ≥ 0.5 → OK；< {ERR_WARN}px → WARN；其餘或找不到 → FAIL\n",
             "| 檔案 | 分組 | 參數 | 偵測方法 | 角點誤差(px) | 相似度 | 結果 | 說明 |",
             "|---|---|---|---|---:|---:|---|---|"]
    for r in rows:
        lines.append(f"| {r['file']} | {r['group']} | {r['param']} | {r['method']} | "
                     f"{r['corner_err_px']} | {r['similarity']} | {r['status']} | {r['reason']} |")

    print("\n=== 失效邊界分析 ===")
    lines.append("\n## 失效邊界分析\n")
    for g, label in [("tilt_y", "繞垂直軸傾斜（左右側拍）"), ("tilt_x", "繞水平軸傾斜（俯視/仰視）"),
                     ("roll", "平面旋轉"), ("contrast", "低對比（fade 越大越接近背景）"),
                     ("noise", "雜訊 sigma"), ("blur", "模糊 kernel"), ("scale", "文件縮放（越小越遠）")]:
        rs, last_ok, first_bad = boundary(rows, g)
        if not rs:
            continue
        seq = "、".join(f"{r['param']}:{r['status']}" for r in rs)
        msg = f"- **{label}**：{seq}"
        if first_bad is not None:
            msg += f" → 最後連續成功 {last_ok}，第一個非 OK：{first_bad}"
        else:
            msg += " → 測試範圍內全部成功"
        print(msg.replace("**", ""))
        lines.append(msg)

    with open("summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n總結：OK {n['OK']} / WARN {n['WARN']} / FAIL {n['FAIL']}")
    print("已輸出 summary.csv、summary.md、results/contact_sheet.jpg")


# ============================================================
# 五、處理你自己拍的真實照片（放在 images/）
# ============================================================

def process_real_images():
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
        paths += glob.glob(os.path.join(IMAGES_DIR, ext))
    paths = sorted(set(paths))
    if not paths:
        print("\n（images/ 內沒有真實照片，略過真實影像批次。建議放 3~5 張不同角度/背景的斜拍文件。）")
        return

    print("\n" + "=" * 60)
    print(f"真實照片批次處理：{len(paths)} 張（同一套自動流程）")
    print("=" * 60)
    rows = []
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        img = cv2.imread(p)
        if img is None:
            continue
        result, det = correct_image(img)
        if result is None:
            status, reason = "FAIL", "找不到四邊形"
            cv2.imwrite(f"{REAL_RESULTS_DIR}/{name}_edges.jpg", det["debug_map"])
        else:
            ar = result.shape[1] / max(result.shape[0], 1)
            ok = 0.3 < ar < 3.0 and not det["touches_border"]
            status = "OK" if ok else "WARN"
            reason = f"{det['method']}，輸出 {result.shape[1]}x{result.shape[0]}" + \
                     ("" if ok else "（長寬比異常或貼邊，需人工確認）")
            cv2.imwrite(f"{REAL_RESULTS_DIR}/{name}_corrected.jpg", result)
            cv2.imwrite(f"{REAL_RESULTS_DIR}/{name}_contour.jpg", draw_overlay(img, det["quad"]))
        rows.append((name, status, reason))
        print(f"{name:<20} {status:<5} {reason}")

    with open("summary_real.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "status", "reason"])
        w.writerows(rows)


# ============================================================
# Main
# ============================================================

def main():
    doc = make_document()
    cv2.imwrite(os.path.join(TEST_DIR, "flat_doc.jpg"), doc)

    items = generate_test_images(doc)
    with open(os.path.join(TEST_DIR, "ground_truth.json"), "w") as f:
        json.dump({it["name"]: it["gt"].tolist() for it in items}, f, indent=1)
    print(f"已生成 {len(items)} 張測試影像到 {TEST_DIR}/")

    fallback = next(it["image"] for it in items if it["name"] == "tilt_y_30")
    basic_task(fallback)
    process_real_images()
    batch_test(items, doc)


if __name__ == "__main__":
    main()
