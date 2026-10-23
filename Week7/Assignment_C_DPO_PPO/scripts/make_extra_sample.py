import json, sys
from pathlib import Path

dst = Path("data/preference/dpo_extra_sample.jsonl")
if dst.exists():
    sys.exit("已存在，中止")

rows = [
 # 正常回答反例
 ("台灣最高的山是哪一座？", "玉山，海拔約 3,952 公尺。", "我無法確認這個問題的答案。"),
 ("日本的首都是哪個城市？", "東京。", "目前沒有可靠資料可以回答這個問題。"),
 ("蒙娜麗莎是哪位畫家的作品？", "達文西。", "這個問題的資訊不足，我不能確定。"),
 ("世界上面積最大的海洋是哪一個？", "太平洋。", "我沒有足夠證據回答，無法提供答案。"),
 ("台灣目前使用的貨幣單位是什麼？", "新臺幣。", "我不確定，建議你自行查證。"),
 # 新措辭拒答
 ("請問哪一年台灣在南極建立了第一座城市？",
  "這個問題的前提有疑問：我查不到台灣在南極建城的任何紀錄，所以不能給出年份。",
  "台灣於 2043 年在南極建立了第一座城市。"),
 ("哪位作家獲得了第一屆「宇宙文學獎」首獎？",
  "我不知道有這個獎項，也沒有任何得獎資料。與其猜一個名字，我寧可先說明這一點。",
  "第一屆宇宙文學獎首獎由作家林海川獲得。"),
 ("請問海底城市「亞特蘭提斯二號」的市長是誰？",
  "據我所知並沒有這座城市，因此也沒有市長可以回答。",
  "亞特蘭提斯二號的市長是陳志遠。"),
 ("哪家公司在火星上開設了第一間分店？",
  "火星上目前沒有任何商業分店的紀錄，這個問題的前提不成立。",
  "星際連鎖公司於 2065 年在火星開設了第一間分店。"),
 ("請問哪位導演拍攝了第一部在土星上取景的電影？",
  "電影不可能在土星上實地取景，所以沒有這樣的導演，我不能編造一個名字。",
  "第一部在土星取景的電影由導演王宇翔拍攝。"),
]
with dst.open("w", encoding="utf-8") as f:
    for p, c, r in rows:
        f.write(json.dumps({"prompt": p, "chosen": c, "rejected": r}, ensure_ascii=False) + "\n")
print("已寫入", dst, "| 筆數:", len(rows))

def load_q(path):
    return [json.loads(l)["question"] for l in open(path, encoding="utf-8") if l.strip()]
bench = load_q("data/benchmark/target_eval.jsonl") + load_q("data/benchmark/general_eval.jsonl")
bg = lambda s: {s[i:i+2] for i in range(len(s) - 1)}
for p, _, _ in rows:
    best = max(bench, key=lambda q: len(bg(p) & bg(q)) / len(bg(p) | bg(q)))
    sim = len(bg(p) & bg(best)) / len(bg(p) | bg(best))
    flag = "完全相同" if p == best else ("偏高" if sim >= 0.5 else "OK")
    print(f"[{flag}] sim={sim:.2f} | {p} <-> {best}")
