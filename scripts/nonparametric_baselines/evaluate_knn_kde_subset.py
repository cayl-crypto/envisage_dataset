#!/usr/bin/env python3
# eval_kde_knn_band_subsets.py

import json
import os
from typing import Any, Dict, List
from statistics import mean

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


ROOT_DIRS = [
    "TEST_subsets_by_band_kde",
    "TEST_subsets_by_band_knn",
]

BANDS = ["low", "med", "high", "very_high"]


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def normalize_image_id_to_int(x: Any) -> int:
    if isinstance(x, int):
        return x
    s = str(x).strip()
    s = "".join(ch for ch in s if ch.isdigit())
    if s == "":
        raise ValueError(f"Invalid image_id: {x!r}")
    return int(s)


def clean_caption_one_line(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    for ln in text.splitlines():
        ln = ln.strip()
        if ln:
            return ln
    return ""


def fix_gen_for_coco(
    gen_data: List[Dict[str, Any]], valid_img_ids: set
) -> List[Dict[str, Any]]:
    fixed = []
    seen = set()

    for r in gen_data:
        img_id = normalize_image_id_to_int(r["image_id"])

        if img_id not in valid_img_ids:
            continue

        # COCO loadRes expects one prediction per image
        if img_id in seen:
            continue

        fixed.append(
            {
                "image_id": img_id,
                "caption": clean_caption_one_line(r.get("caption", "")),
            }
        )
        seen.add(img_id)

    return fixed


def ensure_gt_coco_format(gt_path: str) -> None:
    """
    Ensures gt.json has:
    - annotations
    - images
    - integer image_id
    - annotation id
    """
    gt = load_json(gt_path)

    anns = []
    image_ids = set()

    for idx, ann in enumerate(gt.get("annotations", [])):
        img_id = normalize_image_id_to_int(ann["image_id"])
        image_ids.add(img_id)

        sentence_id = ann.get("sentence_id", str(idx))

        try:
            ann_id = int("".join(ch for ch in str(sentence_id) if ch.isdigit()))
        except Exception:
            ann_id = idx

        anns.append(
            {
                "image_id": img_id,
                "sentence_id": sentence_id,
                "caption": clean_caption_one_line(ann.get("caption", "")),
                "id": ann.get("id", ann_id),
            }
        )

    gt_fixed = {
        "annotations": anns,
        "images": [{"id": img_id} for img_id in sorted(image_ids)],
    }

    save_json(gt_path, gt_fixed)


def _get_metric(metrics: Dict[str, float], *keys: str) -> float:
    for k in keys:
        if k in metrics and isinstance(metrics[k], (int, float)):
            return float(metrics[k])
    return 0.0


def harmonic_mean(values: List[float]) -> float:
    non_zero = [v for v in values if v > 0]
    if not non_zero:
        return 0.0
    return len(non_zero) / sum(1.0 / v for v in non_zero)


def add_final_score(metrics: Dict[str, float]) -> Dict[str, float]:
    bleu1 = _get_metric(metrics, "Bleu_1", "BLEU-1", "BLEU_1")
    bleu2 = _get_metric(metrics, "Bleu_2", "BLEU-2", "BLEU_2")
    bleu3 = _get_metric(metrics, "Bleu_3", "BLEU-3", "BLEU_3")
    bleu4 = _get_metric(metrics, "Bleu_4", "BLEU-4", "BLEU_4")
    cider = _get_metric(metrics, "CIDEr", "Cider")
    rouge = _get_metric(metrics, "ROUGE_L", "ROUGE-L", "Rouge_L", "ROUGE")
    meteor = _get_metric(metrics, "METEOR", "Meteor")
    spice = _get_metric(metrics, "SPICE", "Spice")

    bleu_hm = harmonic_mean([bleu1, bleu2, bleu3, bleu4])
    final = harmonic_mean([bleu_hm, cider, rouge, meteor, spice])

    metrics["BLEU_HM"] = bleu_hm
    metrics["final_score"] = final
    return metrics


def evaluate_subset(
    gt_path: str, gen_path: str, metrics_out_path: str
) -> Dict[str, float]:
    print(f"\nEvaluating:\n  GT : {gt_path}\n  GEN: {gen_path}")

    ensure_gt_coco_format(gt_path)

    coco = COCO(gt_path)
    valid_img_ids = set(coco.getImgIds())

    gen_data = load_json(gen_path)
    gen_fixed = fix_gen_for_coco(gen_data, valid_img_ids)

    temp_gen_path = gen_path.replace(".json", "_coco_fixed.json")
    save_json(temp_gen_path, gen_fixed)

    if len(gen_fixed) == 0:
        raise RuntimeError(f"No valid generated captions found for {gen_path}")

    coco_res = coco.loadRes(temp_gen_path)
    coco_eval = COCOEvalCap(coco, coco_res)
    coco_eval.evaluate()

    metrics = dict(coco_eval.eval)
    metrics = add_final_score(metrics)

    metrics["num_images"] = len(gen_fixed)
    metrics["num_gt_annotations"] = len(coco.dataset.get("annotations", []))

    save_json(metrics_out_path, metrics)

    summary_keys = ["Bleu_4", "CIDEr", "SPICE", "final_score"]
    summary = ", ".join(f"{k}: {metrics.get(k, 0):.4f}" for k in summary_keys)
    print(f" -> {summary}")

    return metrics


def main():
    all_results = {}

    for root in ROOT_DIRS:
        method_name = root.replace("TEST_subsets_by_band_", "")
        all_results[method_name] = {}

        print(f"\n==============================")
        print(f"Evaluating method: {method_name.upper()}")
        print(f"==============================")

        for band in BANDS:
            band_dir = os.path.join(root, band)
            gt_path = os.path.join(band_dir, "gt.json")
            gen_path = os.path.join(band_dir, "gen.json")
            metrics_path = os.path.join(band_dir, "metrics.json")

            if not os.path.exists(gt_path) or not os.path.exists(gen_path):
                print(f"Skipping missing folder/file: {band_dir}")
                continue

            try:
                metrics = evaluate_subset(gt_path, gen_path, metrics_path)
                all_results[method_name][band] = metrics
            except Exception as e:
                print(f"⚠️ Evaluation failed for {method_name}/{band}: {e}")

    save_json("kde_knn_band_metrics_summary.json", all_results)

    print("\nSaved summary: kde_knn_band_metrics_summary.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
