import json
from pathlib import Path


GT_PATH = "annotations/test.json"

ROOT_DIRS = [
    "TEST_subsets_by_band_kde",
    "TEST_subsets_by_band_knn",
]

BANDS = ["low", "med", "high", "very_high"]


def normalize_image_id(image_id):
    try:
        return int(image_id)
    except:
        return image_id


def load_gt_annotations():
    with open(GT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = data["annotations"]

    # normalize and add "id"
    processed = []
    for ann in annotations:
        new_ann = {
            "image_id": normalize_image_id(ann["image_id"]),
            "sentence_id": ann["sentence_id"],
            "caption": ann["caption"],
            "id": int(ann["sentence_id"]),
        }
        processed.append(new_ann)

    return processed


def main():
    gt_annotations = load_gt_annotations()

    for root in ROOT_DIRS:
        root_path = Path(root)

        for band in BANDS:
            band_dir = root_path / band
            gen_path = band_dir / "gen.json"

            if not gen_path.exists():
                continue

            # Load generated subset
            with open(gen_path, "r", encoding="utf-8") as f:
                gen_data = json.load(f)

            image_ids = set([normalize_image_id(x["image_id"]) for x in gen_data])

            # Filter GT
            subset_gt = [ann for ann in gt_annotations if ann["image_id"] in image_ids]

            # Save
            gt_out = {"annotations": subset_gt}

            out_path = band_dir / "gt.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(gt_out, f, indent=2, ensure_ascii=False)

            print(
                f"{root}/{band}: {len(image_ids)} images → {len(subset_gt)} GT captions"
            )


if __name__ == "__main__":
    main()
