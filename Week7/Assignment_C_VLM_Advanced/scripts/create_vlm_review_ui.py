import base64
import csv
import json
import mimetypes
from pathlib import Path
from html import escape

ROOT = Path.cwd()
INPUT = ROOT / "results/auto_screen_v1/manual_review_sample_20.csv"
OUTPUT = ROOT / "results/auto_screen_v1/manual_review_interface.html"

RATING_FIELDS = [
    "baseline_overall_grounded",
    "dpo_overall_grounded",
    "baseline_hallucinated_object",
    "dpo_hallucinated_object",
    "baseline_unsupported_action_or_attribute",
    "dpo_unsupported_action_or_attribute",
    "baseline_important_object_omitted",
    "dpo_important_object_omitted",
    "baseline_severity",
    "dpo_severity",
    "which_is_better",
    "review_notes",
]

def main():
    with INPUT.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 20:
        raise ValueError(f"預期 20 筆資料，實際為 {len(rows)} 筆")

    for row in rows:
        image_path = Path(row["image"])
        if not image_path.is_file():
            raise FileNotFoundError(f"圖片不存在：{image_path}")

        mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        row["embedded_image"] = f"data:{mime};base64,{encoded}"

        # 只把原本已有的評分帶入介面；空白欄位仍可評分。
        for key in RATING_FIELDS:
            row.setdefault(key, "")

    data = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    fields = json.dumps(RATING_FIELDS, ensure_ascii=False)

    html = r'''<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VLM Baseline vs DPO 人工審查</title>
<style>
body{font-family:system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;line-height:1.55;color:#222}
h1{font-size:1.5rem}
.panel{border:1px solid #ccc;border-radius:12px;padding:16px;margin:14px 0}
img{display:block;max-width:100%;max-height:480px;object-fit:contain;margin:auto}
.predictions{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.pred{background:#f5f5f5;border-radius:8px;padding:12px;white-space:pre-wrap}
label{display:block;margin:10px 0 4px;font-weight:600}
select,textarea{width:100%;box-sizing:border-box;padding:8px;font:inherit}
textarea{min-height:65px}
button{padding:9px 14px;margin:5px 5px 5px 0;cursor:pointer}
.muted{color:#666;font-size:.9rem}
.progress{font-weight:600}
@media(max-width:700px){.predictions{grid-template-columns:1fr}}
</style>
</head>
<body>
<h1>VLM 人工審查：Baseline vs DPO</h1>
<p>請根據原圖評估兩個描述。完成後按「儲存此筆」，最後按「匯出 CSV」。本頁不會上傳圖片或評分。</p>
<div class="panel">
  <div class="progress" id="progress"></div>
  <p id="group"></p>
  <p id="objects"></p>
  <label>參考描述（COCO captions）</label>
  <div id="refs" class="pred"></div>
</div>
<div class="panel">
  <label>原始圖片</label>
  <img id="photo" alt="待審查圖片">
</div>
<div class="predictions">
  <div class="panel"><h2>Baseline</h2><div id="baseline" class="pred"></div></div>
  <div class="panel"><h2>DPO</h2><div id="dpo" class="pred"></div></div>
</div>
<div class="panel">
  <h2>人工評分</h2>
  <label for="bg">Baseline：整體描述是否符合圖片</label>
  <select id="bg"><option value="">請選擇</option><option>good</option><option>minor_issue</option><option>major_issue</option></select>
  <label for="dg">DPO：整體描述是否符合圖片</label>
  <select id="dg"><option value="">請選擇</option><option>good</option><option>minor_issue</option><option>major_issue</option></select>
  <label for="bh">Baseline：是否有虛構物件</label>
  <select id="bh"><option value="">請選擇</option><option>no</option><option>yes</option></select>
  <label for="dh">DPO：是否有虛構物件</label>
  <select id="dh"><option value="">請選擇</option><option>no</option><option>yes</option></select>
  <label for="ba">Baseline：是否有不受支持的動作或屬性</label>
  <select id="ba"><option value="">請選擇</option><option>no</option><option>yes</option></select>
  <label for="da">DPO：是否有不受支持的動作或屬性</label>
  <select id="da"><option value="">請選擇</option><option>no</option><option>yes</option></select>
  <label for="bo">Baseline：是否遺漏重要物件</label>
  <select id="bo"><option value="">請選擇</option><option>no</option><option>yes</option></select>
  <label for="do">DPO：是否遺漏重要物件</label>
  <select id="do"><option value="">請選擇</option><option>no</option><option>yes</option></select>
  <label for="bs">Baseline：錯誤嚴重程度</label>
  <select id="bs"><option value="">請選擇</option><option value="0">0 — 無</option><option value="1">1 — 輕微</option><option value="2">2 — 嚴重</option></select>
  <label for="ds">DPO：錯誤嚴重程度</label>
  <select id="ds"><option value="">請選擇</option><option value="0">0 — 無</option><option value="1">1 — 輕微</option><option value="2">2 — 嚴重</option></select>
  <label for="winner">整體比較</label>
  <select id="winner"><option value="">請選擇</option><option value="baseline">Baseline 較好</option><option value="dpo">DPO 較好</option><option value="tie">平手</option></select>
  <label for="notes">備註（可留白）</label>
  <textarea id="notes" placeholder="簡述判斷理由"></textarea>
  <button onclick="saveCurrent()">儲存此筆</button>
  <button onclick="previous()">上一張</button>
  <button onclick="next()">下一張</button>
  <button onclick="exportCSV()">匯出 CSV</button>
  <p id="status" class="muted"></p>
</div>
<script>
const rows = __DATA__;
const fields = __FIELDS__;
const ids = {
 baseline_overall_grounded:"bg", dpo_overall_grounded:"dg",
 baseline_hallucinated_object:"bh", dpo_hallucinated_object:"dh",
 baseline_unsupported_action_or_attribute:"ba",
 dpo_unsupported_action_or_attribute:"da",
 baseline_important_object_omitted:"bo",
 dpo_important_object_omitted:"do",
 baseline_severity:"bs", dpo_severity:"ds",
 which_is_better:"winner", review_notes:"notes"
};
let index = 0;
const $ = id => document.getElementById(id);

function loadCurrent(){
 const r = rows[index];
 $("progress").textContent = `第 ${index+1} / ${rows.length} 張｜ID ${r.image_id}`;
 $("group").textContent = `抽樣組別：${r.sample_group}｜初篩優先度：${r.screening_priority}`;
 $("objects").textContent = `COCO 物件標註：${r.coco_objects || "無資料"}`;
 $("refs").textContent = r.reference_captions || "無參考描述";
 $("photo").src = r.embedded_image;
 $("baseline").textContent = r.baseline_prediction;
 $("dpo").textContent = r.dpo_prediction;
 for(const [field,id] of Object.entries(ids)) $(id).value = r[field] || "";
 $("status").textContent = "";
}
function saveCurrent(){
 const r = rows[index];
 for(const [field,id] of Object.entries(ids)) r[field] = $(id).value;
 $("status").textContent = `已將 ID ${r.image_id} 的評分保留在目前頁面記憶體中。請最後匯出 CSV 保存。`;
}
function next(){ saveCurrent(); if(index < rows.length-1){index++;loadCurrent();} }
function previous(){ saveCurrent(); if(index > 0){index--;loadCurrent();} }
function csvEscape(v){return '"' + String(v ?? "").replace(/"/g,'""') + '"';}
function exportCSV(){
 saveCurrent();
 const columns = Object.keys(rows[0]).filter(k => k !== "embedded_image");
 const lines = [columns.map(csvEscape).join(",")];
 for(const r of rows) lines.push(columns.map(k => csvEscape(r[k])).join(","));
 const blob = new Blob(["\ufeff" + lines.join("\r\n")],{type:"text/csv;charset=utf-8"});
 const url = URL.createObjectURL(blob);
 const a = document.createElement("a");
 a.href = url; a.download = "manual_review_sample_20_scored.csv"; a.click();
 URL.revokeObjectURL(url);
 $("status").textContent = "已觸發 CSV 匯出。請確認瀏覽器下載資料夾中有 manual_review_sample_20_scored.csv。";
}
loadCurrent();
</script>
</body>
</html>'''

    html = html.replace("__DATA__", data).replace("__FIELDS__", fields)
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"Created: {OUTPUT}")
    print(f"Embedded images: {len(rows)}")
    print(f"File size: {OUTPUT.stat().st_size / (1024*1024):.1f} MB")
    print("\nOpen this file in your browser:")
    print(OUTPUT)

if __name__ == "__main__":
    main()
