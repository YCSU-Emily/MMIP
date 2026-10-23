import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path.home() / "MMIP/week4/quiz4/image_captioning"
OUT = Path.home() / "MMIP/Week7/Assignment_C_VLM_Advanced/data"
SEED = 3407
N = 120

random.seed(SEED)

annotation_path = ROOT / "data/captions/train.json"
image_dir = ROOT / "data/images/train"

data = json.loads(annotation_path.read_text())

images = {x["id"]: x["file_name"] for x in data["images"]}
captions = defaultdict(list)

for ann in data["annotations"]:
    text = ann.get("caption", "").strip()
    image_id = ann.get("image_id")
    if text and image_id in images:
        captions[image_id].append(text)

eligible = [
    image_id for image_id, texts in captions.items()
    if texts and (image_dir / images[image_id]).is_file()
]
random.shuffle(eligible)

# Use a fixed list of plausible but deliberately unsupported claims.
# These are synthetic negatives and must be manually audited.
false_claims = [
    "A giraffe is clearly visible in the image.",
    "A zebra is clearly visible in the image.",
    "A helicopter is clearly visible in the image.",
    "A refrigerator is clearly visible in the image.",
    "A surfboard is clearly visible in the image.",
    "A horse is clearly visible in the image.",
    "A stop sign is clearly visible in the image.",
    "A baseball bat is clearly visible in the image.",
    "A dining table is clearly visible in the image.",
    "A suitcase is clearly visible in the image.",
    "A sheep is clearly visible in the image.",
    "A fire hydrant is clearly visible in the image.",
]

rows = []
for image_id in eligible:
    reference = captions[image_id][0]
    claim = random.choice(false_claims)

    # Avoid exact object-name overlap with the selected reference caption.
    claim_object = claim.split(" is clearly visible")[0].replace(
        "A ", "", 1
    ).replace("An ", "", 1).lower()

    if claim_object.rstrip("s") in reference.lower():
        continue

    rows.append({
        "image_id": image_id,
        "image": str(image_dir / images[image_id]),
        "prompt": (
            "Describe this image in one concise sentence. "
            "Mention only objects and actions supported by the image."
        ),
        "chosen": reference,
        "rejected": reference.rstrip(".") + ". " + claim,
        "reference_caption": reference,
        "negative_type": "synthetic_object_insertion",
        "label_status": "needs_manual_audit",
    })

    if len(rows) >= N:
        break

out_file = OUT / "vlm_preferences_candidates.jsonl"
with out_file.open("w") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("Output:", out_file)
print("Candidate pairs:", len(rows))
print("Unique images:", len({r['image_id'] for r in rows}))
print("Missing image paths:", sum(
    not Path(r["image"]).is_file() for r in rows
))
print("All labels marked for manual audit:", all(
    r["label_status"] == "needs_manual_audit" for r in rows
))
print("\nFirst 3 candidates:")
for row in rows[:3]:
    print(json.dumps(row, ensure_ascii=False, indent=2))
