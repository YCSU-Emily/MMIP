import json
from pathlib import Path
from PIL import Image, ImageDraw

BASE = Path("Week7/Assignment_B_VLM")
IMAGE_DIR = BASE / "images" / "benchmark"
JSON_PATH = BASE / "benchmarks" / "visual_benchmark.json"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

W, H = 640, 400
BG = "white"
BLACK = "black"

items = []

def new_canvas():
    img = Image.new("RGB", (W, H), BG)
    return img, ImageDraw.Draw(img)

def add(item_id, category, prompt, answer, draw_fn):
    img, draw = new_canvas()
    draw_fn(img, draw)
    filename = f"{item_id:02d}.png"
    img.save(IMAGE_DIR / filename)
    items.append({
        "id": item_id,
        "image": f"images/benchmark/{filename}",
        "category": category,
        "prompt": prompt,
        "reference_answer": answer,
    })

def text(draw, xy, value, fill=BLACK):
    draw.text(xy, value, fill=fill)

# 1. Shape and color identification
def d01(img, d):
    d.ellipse((70, 100, 220, 250), fill="red", outline=BLACK, width=3)
    d.rectangle((390, 100, 540, 250), fill="blue", outline=BLACK, width=3)
add(1, "shape_color", "圖片中有哪些形狀？各自是什麼顏色？", "一個紅色圓形和一個藍色正方形。", d01)

# 2. Object counting
def d02(img, d):
    for x, y in [(70, 80), (230, 80), (390, 80)]:
        d.ellipse((x, y, x+100, y+100), fill="red", outline=BLACK, width=2)
    d.rectangle((160, 240, 260, 340), fill="blue", outline=BLACK, width=2)
    d.rectangle((360, 240, 460, 340), fill="blue", outline=BLACK, width=2)
add(2, "counting", "圖片中總共有幾個物件？紅色圓形和藍色正方形各有幾個？", "總共 5 個物件：3 個紅色圓形和 2 個藍色正方形。", d02)

# 3. Relative positions
def d03(img, d):
    d.rectangle((60, 140, 170, 250), fill="red", outline=BLACK, width=2)
    d.rectangle((265, 140, 375, 250), fill="green", outline=BLACK, width=2)
    d.rectangle((470, 140, 580, 250), fill="blue", outline=BLACK, width=2)
add(3, "spatial_relation", "由左到右，三個正方形的顏色依序是什麼？", "由左到右依序是紅色、綠色、藍色。", d03)

# 4. Largest shape
def d04(img, d):
    d.ellipse((50, 150, 130, 230), fill="green", outline=BLACK, width=2)
    d.ellipse((220, 100, 370, 250), fill="red", outline=BLACK, width=2)
    d.ellipse((430, 60, 600, 230), fill="blue", outline=BLACK, width=2)
add(4, "comparison", "哪一個圓形最大？請指出它的顏色。", "藍色圓形最大。", d04)

# 5. Shape types
def d05(img, d):
    d.rectangle((55, 130, 155, 230), fill="red", outline=BLACK, width=2)
    d.ellipse((270, 130, 370, 230), fill="yellow", outline=BLACK, width=2)
    d.polygon([(490, 230), (550, 120), (610, 230)], fill="blue", outline=BLACK)
add(5, "shape_color", "圖片中有哪三種幾何形狀？請說出各自的顏色。", "紅色正方形、黃色圓形和藍色三角形。", d05)

# 6. Count triangles
def d06(img, d):
    for x in [60, 210, 360, 510]:
        d.polygon([(x, 280), (x+50, 170), (x+100, 280)], fill="green", outline=BLACK)
add(6, "counting", "圖片中有幾個綠色三角形？", "有 4 個綠色三角形。", d06)

# 7. Top and bottom
def d07(img, d):
    d.ellipse((260, 40, 360, 140), fill="red", outline=BLACK, width=2)
    d.rectangle((260, 260, 360, 360), fill="blue", outline=BLACK, width=2)
add(7, "spatial_relation", "哪個形狀在上方？哪個形狀在下方？", "紅色圓形在上方，藍色正方形在下方。", d07)

# 8. Odd one out
def d08(img, d):
    for x in [50, 190, 330]:
        d.rectangle((x, 130, x+90, 220), fill="blue", outline=BLACK, width=2)
    d.ellipse((470, 130, 560, 220), fill="blue", outline=BLACK, width=2)
add(8, "visual_reasoning", "哪一個物件的形狀和其他三個不同？", "最右邊的藍色圓形不同；其他三個是正方形。", d08)

# 9. Color counting
def d09(img, d):
    colors = ["red", "blue", "red", "green", "red", "blue"]
    for i, c in enumerate(colors):
        x = 30 + i * 100
        d.ellipse((x, 140, x+70, 210), fill=c, outline=BLACK, width=2)
add(9, "counting", "圖片中紅色圓形有幾個？", "有 3 個紅色圓形。", d09)

# 10. Object alignment
def d10(img, d):
    for x in [70, 250, 430]:
        d.rectangle((x, 140, x+100, 240), fill="orange", outline=BLACK, width=2)
    text(d, (225, 300), "Three orange squares")
add(10, "spatial_relation", "三個橘色正方形排列成什麼方向？", "三個橘色正方形由左至右水平排列。", d10)

# 11. OCR: single word
def d11(img, d):
    text(d, (190, 160), "HELLO", fill="blue")
add(11, "ocr", "圖片中寫了什麼英文單字？", "HELLO。", d11)

# 12. OCR: number
def d12(img, d):
    text(d, (250, 150), "2026", fill="black")
add(12, "ocr", "圖片中顯示的數字是什麼？", "2026。", d12)

# 13. OCR: colored labels
def d13(img, d):
    text(d, (90, 100), "CAT", fill="red")
    text(d, (390, 230), "DOG", fill="blue")
add(13, "ocr", "圖片中有哪兩個英文單字？", "CAT 和 DOG。", d13)

# 14. Object-color matching
def d14(img, d):
    d.ellipse((70, 120, 180, 230), fill="yellow", outline=BLACK, width=2)
    d.rectangle((265, 120, 375, 230), fill="purple", outline=BLACK, width=2)
    d.polygon([(460, 230), (515, 120), (570, 230)], fill="green", outline=BLACK)
add(14, "shape_color", "請分別說出圓形、正方形和三角形的顏色。", "圓形是黃色，正方形是紫色，三角形是綠色。", d14)

# 15. Relative size
def d15(img, d):
    d.rectangle((70, 150, 150, 230), fill="red", outline=BLACK, width=2)
    d.rectangle((240, 110, 360, 230), fill="blue", outline=BLACK, width=2)
    d.rectangle((430, 70, 590, 230), fill="green", outline=BLACK, width=2)
add(15, "comparison", "三個正方形中，哪個最大、哪個最小？", "綠色正方形最大，紅色正方形最小。", d15)

# 16. Count mixed shapes
def d16(img, d):
    d.ellipse((50, 70, 130, 150), fill="red", outline=BLACK, width=2)
    d.ellipse((180, 70, 260, 150), fill="red", outline=BLACK, width=2)
    d.rectangle((320, 70, 400, 150), fill="blue", outline=BLACK, width=2)
    d.polygon([(480, 150), (520, 70), (560, 150)], fill="green", outline=BLACK)
add(16, "counting", "圖片中一共有幾個物件？", "一共有 4 個物件。", d16)

# 17. Left-right relation
def d17(img, d):
    d.ellipse((100, 130, 200, 230), fill="yellow", outline=BLACK, width=2)
    d.rectangle((400, 130, 500, 230), fill="blue", outline=BLACK, width=2)
add(17, "spatial_relation", "藍色正方形位於黃色圓形的哪一側？", "藍色正方形位於黃色圓形的右側。", d17)

# 18. Simple arithmetic from image
def d18(img, d):
    for x in [70, 170, 270]:
        d.ellipse((x, 120, x+65, 185), fill="red", outline=BLACK, width=2)
    for x in [390, 490]:
        d.ellipse((x, 120, x+65, 185), fill="blue", outline=BLACK, width=2)
add(18, "visual_reasoning", "如果把所有紅色和藍色圓形合在一起，總共有幾個圓形？", "總共有 5 個圓形。", d18)

# 19. Object above another
def d19(img, d):
    d.rectangle((250, 230, 350, 330), fill="green", outline=BLACK, width=2)
    d.ellipse((250, 70, 350, 170), fill="orange", outline=BLACK, width=2)
add(19, "spatial_relation", "綠色正方形的上方是什麼形狀、什麼顏色？", "綠色正方形上方是橘色圓形。", d19)

# 20. Multi-attribute reasoning
def d20(img, d):
    d.ellipse((50, 130, 130, 210), fill="red", outline=BLACK, width=2)
    d.rectangle((190, 130, 270, 210), fill="blue", outline=BLACK, width=2)
    d.polygon([(350, 210), (390, 130), (430, 210)], fill="red", outline=BLACK)
    d.rectangle((490, 130, 570, 210), fill="red", outline=BLACK, width=2)
add(20, "visual_reasoning", "圖片中紅色物件有幾個？其中有哪些形狀？", "有 3 個紅色物件：圓形、三角形和正方形。", d20)

JSON_PATH.write_text(
    json.dumps(items, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"Created {len(items)} benchmark images in: {IMAGE_DIR}")
print(f"Created benchmark JSON: {JSON_PATH}")
print("\nCategory counts:")
from collections import Counter
for category, count in sorted(Counter(x["category"] for x in items).items()):
    print(f"  {category}: {count}")
