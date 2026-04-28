import json
from pathlib import Path


INPUT_JSON = "captions_with_kde_knn_uncertainty_envisage_test.json"

METHODS = {
    "kde": {
        "band_key": "sequence_uncertainty_band_kde",
        "out_dir": "TEST_subsets_by_band_kde",
    },
    "knn": {
        "band_key": "sequence_uncertainty_band_knn",
        "out_dir": "TEST_subsets_by_band_knn",
    },
}

BANDS = ["low", "med", "high", "very_high"]


def normalize_image_id(image_id):
    """
    Converts image_id like '00000002' to 2.
    Keeps non-numeric IDs unchanged.
    """
    try:
        return int(image_id)
    except (ValueError, TypeError):
        return image_id


def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    for method_name, cfg in METHODS.items():
        band_key = cfg["band_key"]
        out_root = Path(cfg["out_dir"])

        subsets = {band: [] for band in BANDS}

        for item in data:
            band = item.get(band_key)

            if band not in subsets:
                print(f"Skipping unknown band: {band}")
                continue

            subsets[band].append(
                {
                    "image_id": normalize_image_id(item["image_id"]),
                    "caption": item["caption"],
                }
            )

        for band, records in subsets.items():
            band_dir = out_root / band
            band_dir.mkdir(parents=True, exist_ok=True)

            out_path = band_dir / "gen.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)

            print(
                f"{method_name.upper()} | {band}: {len(records)} captions -> {out_path}"
            )


if __name__ == "__main__":
    main()
