import json
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

manifest = PROJECT_ROOT / "data" / "paper_reference" / "manifests" / "test.jsonl"
output_dir = PROJECT_ROOT / "data" / "coco" / "val2014"

output_dir.mkdir(parents=True, exist_ok=True)

image_ids = set()

with open(manifest, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            item = json.loads(line)

            if "image_id" in item:
                image_ids.add(int(item["image_id"]))

print(f"Found {len(image_ids)} unique image IDs.")

downloaded = 0
failed = 0

for image_id in sorted(image_ids):

    filename = f"COCO_val2014_{image_id:012d}.jpg"
    destination = output_dir / filename

    if destination.exists():
        downloaded += 1
        continue

    url = (
        "http://images.cocodataset.org/"
        f"val2014/{filename}"
    )

    try:
        print(f"Downloading {filename}...")

        urllib.request.urlretrieve(
            url,
            destination
        )

        downloaded += 1

    except Exception as e:
        failed += 1
        print(f"FAILED: {filename}")
        print(e)

print()
print("================================")
print("Gallery download completed")
print("================================")
print(f"Available: {downloaded}")
print(f"Failed:    {failed}")
print(f"Location:  {output_dir}")