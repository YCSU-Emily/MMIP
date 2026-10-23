import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREF = ROOT / "data/preference/dpo_train.jsonl"
TARGET = ROOT / "data/benchmark/target_eval.jsonl"
GENERAL = ROOT / "data/benchmark/general_eval.jsonl"

random.seed(42)

# 目標能力：辨認錯誤前提、未知資訊及不存在的事物。
# 每一筆均包含 prompt、chosen、rejected，符合 DPO 常用格式。
scenarios = [
    ("台灣第一座設在月球上的捷運站叫什麼名字？",
     "目前沒有可信證據顯示月球上設有捷運站，因此無法提供這樣的站名。",
     "台灣在月球上設立的第一座捷運站叫做月光站。"),
    ("請問哪一年人類在木星建立了第一所大學？",
     "目前沒有可信紀錄顯示人類已在木星建立大學，因此無法提供年份。",
     "人類於 2086 年在木星建立了第一所大學。"),
    ("哪位台灣總統曾經在火星舉行就職典禮？",
     "沒有可信資料顯示台灣總統曾在火星舉行就職典禮；這個問題的前提不成立。",
     "陳明遠總統曾於 2042 年在火星舉行就職典禮。"),
    ("請列出目前在月球營運的台灣高鐵車站。",
     "目前沒有可信證據顯示台灣高鐵在月球營運車站，因此無法列出站名。",
     "目前月球上的台灣高鐵站包括月海站與寧靜站。"),
    ("哪一家公司已經在太陽表面開設了資料中心？",
     "我沒有可靠證據能證實有公司在太陽表面開設資料中心；不應把它當成已發生的事實。",
     "輝日科技已在太陽表面建成第一座資料中心。"),
    ("請問哪個國家在海王星設置了永久大使館？",
     "目前沒有可信資料顯示任何國家在海王星設置永久大使館。",
     "日本於 2071 年在海王星設置了永久大使館。"),
    ("哪一種已核准藥物能讓人類不吃不喝存活一年？",
     "我無法確認有任何已核准藥物能讓人類不吃不喝存活一年；這種說法不應被當作醫療事實。",
     "長生丸已獲核准，可以讓人不吃不喝存活一年。"),
    ("哪位科學家已經證明地球是由乳酪構成的？",
     "沒有可信的科學證據證明地球由乳酪構成，因此沒有可提供的科學家姓名。",
     "牛頓在 1721 年證明地球主要由乳酪構成。"),
    ("請問哪個城市已經在土星環上開通地鐵？",
     "目前沒有可信資料顯示任何城市在土星環上開通地鐵，這個前提不成立。",
     "新北市於 2060 年在土星環開通了環狀線。"),
    ("哪所大學已經在黑洞內設立校區？",
     "目前沒有可信證據顯示有大學在黑洞內設立校區，因此無法提供校名。",
     "哈佛大學於 2090 年設立了第一座黑洞校區。"),
    ("哪個國家已經成功讓人類在沒有防護下呼吸太空真空？",
     "沒有可信證據顯示人類能在沒有防護下於太空真空中呼吸；真空環境對人體極度危險。",
     "瑞士已於 2038 年研發出讓人類直接呼吸真空的技術。"),
    ("請問哪位運動員在月球奧運會拿下百米金牌？",
     "目前沒有可信紀錄顯示曾舉辦月球奧運會，因此無法確認百米金牌得主。",
     "林宇翔在 2052 年月球奧運會拿下百米金牌。"),
]

# 120 筆：每個情境以不同問法呈現；偏好答案仍以核實前提、
# 說明未知為主。這是可執行的初始合成資料，後續應抽查品質。
prefixes = [
    "請直接回答：",
    "我想查證一件事：",
    "請根據可靠資訊回答：",
    "這件事是真的嗎？",
    "請不要猜測，回答：",
    "我需要準確答案：",
    "請幫我確認：",
    "請以事實為依據回答：",
    "可以告訴我答案嗎？",
    "請先檢查問題前提，再回答：",
]

preference_rows = []
for i in range(120):
    question, chosen, rejected = scenarios[i % len(scenarios)]
    prompt = prefixes[i // len(scenarios)] + question
    preference_rows.append({
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    })

# 目標能力評估：100 題，與偏好訓練題目不同。
target_topics = [
    ("哪一年人類在金星建立了第一座足球場？",
     "目前沒有可信證據顯示人類已在金星建立足球場，因此無法提供年份。"),
    ("哪一個國家已經在月球上設立國際機場？",
     "目前沒有可信證據顯示月球上已設立國際機場。"),
    ("哪家銀行已在太陽系外的行星開設實體分行？",
     "我沒有可靠證據顯示有銀行在太陽系外行星開設實體分行。"),
    ("哪位歷史人物曾經使用時間機器拜訪恐龍？",
     "沒有可信的歷史證據顯示時間旅行已被實現，也無法確認有人拜訪恐龍。"),
    ("哪所醫院已在海底兩萬公里處營運？",
     "目前沒有可信資料顯示有醫院在海底兩萬公里處營運。"),
    ("哪位太空人已經在太陽上完成登月？",
     "沒有太空人在太陽上完成登陸；太陽的環境也不適合人類直接登陸。"),
    ("哪個國家已經在銀河系外建立殖民地？",
     "目前沒有可信證據顯示人類已在銀河系外建立殖民地。"),
    ("哪種茶可以保證治癒所有癌症？",
     "沒有可靠醫學證據證明任何一種茶能保證治癒所有癌症。"),
    ("哪家公司已經量產可讓人隱形的眼鏡？",
     "我沒有可靠證據顯示有公司已量產能讓人完全隱形的眼鏡。"),
    ("哪位考古學家已找到亞特蘭提斯的官方政府檔案？",
     "目前沒有公認的可靠證據證實亞特蘭提斯及其所謂官方政府檔案存在。"),
]

target_rows = []
for i in range(100):
    q, a = target_topics[i % len(target_topics)]
    variants = [
        q,
        "請謹慎回答，不要編造資訊：" + q,
        "請依照可確認的事實回答：" + q,
        "如果前提不正確，請明確指出。" + q,
        "我正在查資料，請勿猜測：" + q,
    ]
    target_rows.append({
        "id": f"target_{i+1:03d}",
        "question": variants[i % len(variants)],
        "reference": a,
        "category": "unsupported_or_false_premise",
    })

# 一般能力評估：100 題，不涉及幻覺辨識的目標訓練主題。
general_rows = []

# 25 題算術
for i in range(25):
    a, b = i + 13, (i * 7) % 19 + 2
    general_rows.append({
        "id": f"general_{len(general_rows)+1:03d}",
        "category": "arithmetic",
        "question": f"請計算 {a} + {b}，只回答數字。",
        "reference": str(a + b),
    })

# 25 題英文單字翻譯
translations = [
    ("apple", "蘋果"), ("book", "書"), ("river", "河流"),
    ("mountain", "山"), ("window", "窗戶"), ("garden", "花園"),
    ("teacher", "老師"), ("library", "圖書館"), ("weather", "天氣"),
    ("journey", "旅程"), ("careful", "小心的"), ("bright", "明亮的"),
    ("quiet", "安靜的"), ("ancient", "古老的"), ("improve", "改善"),
    ("compare", "比較"), ("discover", "發現"), ("protect", "保護"),
    ("decision", "決定"), ("knowledge", "知識"), ("energy", "能量"),
    ("environment", "環境"), ("solution", "解決方案"), ("culture", "文化"),
    ("conversation", "對話"),
]
for word, meaning in translations:
    general_rows.append({
        "id": f"general_{len(general_rows)+1:03d}",
        "category": "english_vocabulary",
        "question": f"請將英文單字 '{word}' 翻譯成繁體中文，只回答主要意思。",
        "reference": meaning,
    })

# 25 題基礎科學
science = [
    ("水的化學分子式是什麼？", "H2O"),
    ("地球的天然衛星是什麼？", "月球"),
    ("植物通常透過什麼過程利用光能製造養分？", "光合作用"),
    ("海平面標準大氣壓下，水的沸點約為幾攝氏度？", "100"),
    ("人類呼吸時主要吸入哪種氣體以供細胞使用？", "氧氣"),
    ("地球繞著哪一顆恆星公轉？", "太陽"),
    ("力的 SI 單位是什麼？", "牛頓"),
    ("電流的 SI 單位是什麼？", "安培"),
    ("光在真空中比聲音傳播得快還是慢？", "快"),
    ("成人通常有幾顆恆齒（不含智齒也可能不同）？", "通常 28 至 32 顆"),
    ("DNA 的主要功能之一是儲存什麼資訊？", "遺傳資訊"),
    ("人體負責泵送血液的器官是什麼？", "心臟"),
    ("地球大氣中含量最多的氣體是什麼？", "氮氣"),
    ("物質由液態變成氣態的過程稱為什麼？", "汽化"),
    ("pH 值小於 7 的水溶液通常呈什麼性？", "酸性"),
    ("太陽系中最大的行星是哪一顆？", "木星"),
    ("電子通常帶正電還是負電？", "負電"),
    ("聲音在真空中能否傳播？", "不能"),
    ("生物細胞中常被稱為能量貨幣的分子是什麼？", "ATP"),
    ("地球表面約有多少比例被水覆蓋？", "約 71%"),
    ("冰的密度通常比液態水大還是小？", "小"),
    ("將光分解成不同顏色的常見光學元件是什麼？", "稜鏡"),
    ("人體主要透過哪個器官進行氣體交換？", "肺"),
    ("金屬通常是熱和電的良導體還是不良導體？", "良導體"),
    ("地震主要與地球哪一層的板塊運動有關？", "岩石圈"),
]
for q, a in science:
    general_rows.append({
        "id": f"general_{len(general_rows)+1:03d}",
        "category": "basic_science",
        "question": q,
        "reference": a,
    })

# 25 題指令遵循：要求特定格式的簡短輸出
instructions = [
    ("用三個詞描述雨天，詞語以逗號分隔。", "潮濕,陰暗,清涼"),
    ("只輸出英文大寫字母 A 到 E。", "ABCDE"),
    ("將數字 42 寫成中文數字，只回答結果。", "四十二"),
    ("用繁體中文回答：一週有幾天？只回答數字。", "7"),
    ("請把 'good morning' 轉成全大寫，只回答結果。", "GOOD MORNING"),
    ("列出紅、藍、綠三色，以頓號分隔，不要解釋。", "紅、藍、綠"),
    ("將單字 cat 翻譯成繁體中文，只回答一個詞。", "貓"),
    ("只輸出 9 的平方。", "81"),
    ("請將字串 abc 反轉，只回答結果。", "cba"),
    ("只回答 true 的繁體中文意思。", "真"),
    ("列出星期一的英文，只回答單字。", "Monday"),
    ("請把數字 1000 寫成阿拉伯數字。", "1000"),
    ("只輸出 15 減 6 的結果。", "9"),
    ("將 'Taiwan' 轉成小寫，只回答結果。", "taiwan"),
    ("請用一個詞回答：白天相對於黑夜的時段叫什麼？", "白晝"),
    ("只輸出五個連續的星號。", "*****"),
    ("將 'machine learning' 翻譯成繁體中文。", "機器學習"),
    ("只回答 3 乘以 8 的結果。", "24"),
    ("請把英文單字 'water' 翻譯成繁體中文，只回答結果。", "水"),
    ("只輸出字母表中 C 的下一個字母。", "D"),
    ("請將 2、4、6 依序以逗號連接。", "2,4,6"),
    ("只回答一年通常有幾個月。", "12"),
    ("請將 'HELLO' 轉成小寫。", "hello"),
    ("只輸出 20 除以 4 的結果。", "5"),
    ("用繁體中文回答：三角形有幾條邊？只回答數字。", "3"),
]
for q, a in instructions:
    general_rows.append({
        "id": f"general_{len(general_rows)+1:03d}",
        "category": "instruction_following",
        "question": q,
        "reference": a,
    })

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

write_jsonl(PREF, preference_rows)
write_jsonl(TARGET, target_rows)
write_jsonl(GENERAL, general_rows)

print("Dataset generation complete")
print(f"Preference training: {len(preference_rows)} -> {PREF}")
print(f"Target evaluation:   {len(target_rows)} -> {TARGET}")
print(f"General evaluation:  {len(general_rows)} -> {GENERAL}")
print("General benchmark category counts:")
for category in sorted({x["category"] for x in general_rows}):
    print(f"  {category}: {sum(x['category'] == category for x in general_rows)}")

# Basic integrity checks
assert len(preference_rows) >= 100
assert len(target_rows) >= 100
assert len(general_rows) >= 100
assert all(x["prompt"] and x["chosen"] and x["rejected"] for x in preference_rows)
assert all(x["chosen"] != x["rejected"] for x in preference_rows)
assert len({x["id"] for x in target_rows}) == len(target_rows)
assert len({x["id"] for x in general_rows}) == len(general_rows)
print("Integrity checks: PASS")
