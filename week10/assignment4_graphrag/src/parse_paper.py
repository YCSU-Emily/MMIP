import fitz
import json
import re
from pathlib import Path
from datetime import datetime, timezone

BASE = Path(__file__).resolve().parents[1]
PDF_PATH = BASE / "papers" / "graphrag_paper.pdf"
OUTPUT_DIR = BASE / "results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not PDF_PATH.exists():
    raise FileNotFoundError(f"Paper not found: {PDF_PATH}")

print("=" * 60)
print("GRAPH RAG PAPER PARSING")
print("=" * 60)

doc = fitz.open(PDF_PATH)

pages = []
for i, page in enumerate(doc):
    text = page.get_text("text")
    text = re.sub(r'\n{3,}', '\n\n', text)
    pages.append({
        "page": i + 1,
        "text": text.strip()
    })

full_text = "\n\n".join(
    f"[PAGE {p['page']}]\n{p['text']}"
    for p in pages
)

metadata = {
    "title": "From Local to Global: A Graph RAG Approach to Query-Focused Summarization",
    "authors": ["Darren Edge"],
    "arxiv_id": "2404.16130",
    "version": "v2",
    "pdf_filename": PDF_PATH.name,
    "total_pages": len(doc),
    "total_characters": len(full_text),
    "extracted_at": datetime.now(timezone.utc).isoformat(),
    "pdf_metadata": doc.metadata
}

text_path = OUTPUT_DIR / "paper_text.txt"
metadata_path = OUTPUT_DIR / "paper_metadata.json"

text_path.write_text(full_text, encoding="utf-8")

metadata_path.write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print(f"\nPages: {len(doc)}")
print(f"Characters: {len(full_text):,}")
print(f"Text saved: {text_path}")
print(f"Metadata saved: {metadata_path}")

print("\n========== PAPER PREVIEW ==========")
print(full_text[:4000])

print("\n========== OUTPUT FILES ==========")
print(text_path.stat().st_size, "bytes")
print(metadata_path.stat().st_size, "bytes")

doc.close()
