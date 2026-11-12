import json
import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"

INPUT_FILE = RESULTS / "paper_text.txt"
OUTPUT_FILE = RESULTS / "paper_chunks.json"

CHUNK_SIZE = 1200
OVERLAP = 150


def split_long_text(text):
    """Split an oversized paragraph into overlapping character windows."""
    pieces = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        piece = text[start:end].strip()

        if piece:
            pieces.append(piece)

        if end >= len(text):
            break

        start = end - OVERLAP

    return pieces


def make_page_chunks(page_number, page_text):
    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", page_text)
        if p.strip()
    ]

    chunks = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > CHUNK_SIZE:
            if current:
                chunks.append(current)
                current = ""

            chunks.extend(split_long_text(paragraph))
            continue

        candidate = f"{current}\n\n{paragraph}" if current else paragraph

        if len(candidate) <= CHUNK_SIZE:
            current = candidate
        else:
            if current:
                chunks.append(current)

            overlap_text = current[-OVERLAP:] if current else ""
            current = (
                f"{overlap_text}\n{paragraph}"
                if overlap_text else paragraph
            )

    if current:
        chunks.append(current)

    return chunks


def main():
    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Output already exists; refusing to overwrite: {OUTPUT_FILE}"
        )

    text = INPUT_FILE.read_text(encoding="utf-8")

    matches = list(
        re.finditer(r"(?m)^\[PAGE (\d+)\]\s*$", text)
    )

    if len(matches) != 26:
        raise ValueError(
            f"Expected 26 page markers, found {len(matches)}"
        )

    all_chunks = []

    for i, match in enumerate(matches):
        page_number = int(match.group(1))
        start = match.end()
        end = (
            matches[i + 1].start()
            if i + 1 < len(matches)
            else len(text)
        )

        page_text = text[start:end].strip()
        page_chunks = make_page_chunks(page_number, page_text)

        for chunk_index, content in enumerate(page_chunks, 1):
            all_chunks.append({
                "chunk_id": f"p{page_number:02d}_c{chunk_index:03d}",
                "page": page_number,
                "chunk_index": chunk_index,
                "text": content,
                "char_count": len(content),
            })

    if not all_chunks:
        raise ValueError("No chunks were generated.")

    ids = [item["chunk_id"] for item in all_chunks]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs detected.")

    invalid_pages = [
        item["chunk_id"]
        for item in all_chunks
        if not 1 <= item["page"] <= 26
    ]

    if invalid_pages:
        raise ValueError(f"Invalid page numbers: {invalid_pages}")

    output = {
        "schema_version": "1.0",
        "source": "From Local to Global: A GraphRAG Approach to Query-Focused Summarization",
        "source_file": str(INPUT_FILE),
        "total_pages": 26,
        "chunk_size_target": CHUNK_SIZE,
        "overlap": OVERLAP,
        "total_chunks": len(all_chunks),
        "chunks": all_chunks,
    }

    OUTPUT_FILE.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 65)
    print("PAPER CHUNKING COMPLETE")
    print("=" * 65)
    print("Pages:", 26)
    print("Total chunks:", len(all_chunks))
    print("Average characters:", round(
        sum(x["char_count"] for x in all_chunks) / len(all_chunks), 1
    ))
    print("Minimum characters:", min(x["char_count"] for x in all_chunks))
    print("Maximum characters:", max(x["char_count"] for x in all_chunks))
    print("Unique IDs:", len(set(ids)))
    print("Output:", OUTPUT_FILE)

    print("\nChunks per page:")
    for page in range(1, 27):
        count = sum(x["page"] == page for x in all_chunks)
        print(f"Page {page:02d}: {count} chunks")


if __name__ == "__main__":
    main()
