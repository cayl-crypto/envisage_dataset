#!/usr/bin/env python3
# make_and_eval_subsets.py (final_score version)

import os
import json
from typing import Any, Dict, List, Set
from collections import defaultdict

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

# ---------- utils ----------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def normalize_image_id_to_int(x: Any) -> int:
    if isinstance(x, int):
        return x
    s = str(x).strip()
    s = "".join(ch for ch in s if ch.isdigit())
    if s == "":
        raise ValueError(f"Invalid image_id: {x!r}")
    return int(s)

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def clean_caption_one_line(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    for ln in text.splitlines():
        ln = ln.strip()
        if ln:
            return ln
    return ""

# ---------- loaders ----------
def load_generated_preds(gen_preds_path: str) -> List[Dict[str, Any]]:
    arr = load_json(gen_preds_path)
    out = []
    for r in arr:
        try:
            img_id = normalize_image_id_to_int(r.get("image_id"))
        except Exception:
            continue
        out.append({"image_id": img_id, "caption": clean_caption_one_line(r.get("caption", ""))})
    return out

def build_tag_index(test_tags_path: str) -> Dict[str, Set[int]]:
    data = load_json(test_tags_path)
    tag_to_ids: Dict[str, Set[int]] = defaultdict(set)
    for row in data.get("annotations", []):
        img_int = normalize_image_id_to_int(row.get("image_id"))
        for tag in row.get("tags", []):
            tag_to_ids[tag].add(img_int)
    return tag_to_ids

def build_band_index(uncertainty_path: str) -> Dict[str, Set[int]]:
    arr = load_json(uncertainty_path)
    band_to_ids: Dict[str, Set[int]] = defaultdict(set)
    for item in arr:
        band = str(item.get("sequence_uncertainty_band", "")).strip().lower()
        if not band:
            continue
        img_int = normalize_image_id_to_int(item.get("image_id"))
        band_to_ids[band].add(img_int)
    return band_to_ids

# ---------- filters & fixes ----------
def filter_gen_by_ids(gen_all: List[Dict[str, Any]], keep_ids: Set[int]) -> List[Dict[str, Any]]:
    return [r for r in gen_all if r["image_id"] in keep_ids]

def fix_and_filter_gt_for_coco(gt_full: Dict[str, Any], keep_ids: Set[int]) -> Dict[str, Any]:
    """
    Produce a COCO-caption-compatible GT json:
      - only annotations whose image_id ∈ keep_ids
      - ensure each annotation has an integer image_id and a unique 'id'
      - include an 'images' list with {'id': image_id} entries
    """
    out = {k: v for k, v in gt_full.items() if k not in ("annotations", "images")}

    # Filter & fix annotations
    filt_anns: List[Dict[str, Any]] = []
    next_auto_id = 0
    for a in gt_full.get("annotations", []):
        try:
            img_id_int = normalize_image_id_to_int(a.get("image_id"))
        except Exception:
            continue
        if img_id_int not in keep_ids:
            continue

        aa = dict(a)
        aa["image_id"] = img_id_int
        aa["caption"] = clean_caption_one_line(aa.get("caption", ""))

        # ensure 'id'
        if "id" in aa:
            pass
        elif "sentence_id" in aa:
            sid = "".join(ch for ch in str(aa["sentence_id"]) if ch.isdigit())
            if sid != "":
                aa["id"] = int(sid)
            else:
                aa["id"] = next_auto_id
                next_auto_id += 1
        else:
            aa["id"] = next_auto_id
            next_auto_id += 1

        filt_anns.append(aa)

    out["annotations"] = filt_anns

    # minimal images list
    image_ids_sorted = sorted(keep_ids)
    out["images"] = [{"id": img_id} for img_id in image_ids_sorted]

    return out

# ---------- metrics helpers ----------
def _get_metric(metrics: Dict[str, float], *keys: str) -> float:
    """Fetch a metric value by trying several common JSON keys; return 0.0 if not found."""
    for k in keys:
        if k in metrics and isinstance(metrics[k], (int, float)):
            return float(metrics[k])
    return 0.0

from statistics import mean

def harmonic_mean(values: List[float]) -> float:
    """Compute harmonic mean safely, ignoring zeros. If all zero -> return 0."""
    non_zero = [v for v in values if v > 0]
    if not non_zero:
        return 0.0
    return len(non_zero) / sum(1.0 / v for v in non_zero)

def add_final_score(metrics: Dict[str, float]) -> Dict[str, float]:
    # Fetch metrics
    bleu1 = _get_metric(metrics, "Bleu_1", "BLEU-1", "BLEU_1")
    bleu2 = _get_metric(metrics, "Bleu_2", "BLEU-2", "BLEU_2")
    bleu3 = _get_metric(metrics, "Bleu_3", "BLEU-3", "BLEU_3")
    bleu4 = _get_metric(metrics, "Bleu_4", "BLEU-4", "BLEU_4")
    cider  = _get_metric(metrics, "CIDEr", "Cider")
    rouge  = _get_metric(metrics, "ROUGE_L", "ROUGE-L", "Rouge_L", "ROUGE")
    meteor = _get_metric(metrics, "METEOR", "Meteor")
    spice  = _get_metric(metrics, "SPICE", "Spice")

    # Step 1: harmonic mean of BLEU scores
    bleu_hm = harmonic_mean([bleu1, bleu2, bleu3, bleu4])

    # Step 2: harmonic mean of BLEU_HM + other 4 metrics
    final = harmonic_mean([bleu_hm, cider, rouge, meteor, spice])

    metrics["BLEU_HM"] = bleu_hm
    metrics["final_score"] = final
    return metrics


# ---------- evaluation ----------
def evaluate_subset(gt_path: str, gen_path: str, metrics_out_path: str) -> None:
    print(f"Evaluating\n  GT : {gt_path}\n  GEN: {gen_path}")
    coco = COCO(gt_path)

    with open(gen_path, "r", encoding="utf-8") as f:
        gen_all = json.load(f)

    split_img_ids = set(coco.getImgIds())
    gen_filt = [r for r in gen_all if int(r["image_id"]) in split_img_ids]

    coco_res = coco.loadRes(gen_filt)
    coco_eval = COCOEvalCap(coco, coco_res)
    coco_eval.evaluate()

    metrics = dict(coco_eval.eval)
    metrics = add_final_score(metrics)
    save_json(metrics_out_path, metrics)

    summary = ", ".join(f"{k}:{v:.3f}" for k, v in metrics.items() if isinstance(v, (int, float)))
    print(" ->", summary)

# ---------- pipeline ----------
def main():
    # Inputs
    GEN_PREDS = "generated_results/test_epochbest_test_preds.json"
    GT_TEST_JSON = "annotations/test.json"
    TEST_TAGS_JSON = "annotations/test_tags.json"
    UNCERTAINTY_JSON = "captions_with_uncertainty_envisage_test.json"

    # Outputs
    TAG_ROOT = "subsets_by_tag"
    BAND_ROOT = "subsets_by_band"
    ensure_dir(TAG_ROOT)
    ensure_dir(BAND_ROOT)

    # Load
    print("Loading inputs...")
    gen_all = load_generated_preds(GEN_PREDS)
    gt_full = load_json(GT_TEST_JSON)

    # --- by TAG ---
    print("\n=== Building subsets by TAG ===")
    tag_map = build_tag_index(TEST_TAGS_JSON)
    for tag, idset in sorted(tag_map.items(), key=lambda x: x[0].lower()):
        out_dir = os.path.join(TAG_ROOT, tag.replace("/", "-"))
        ensure_dir(out_dir)

        gen_subset = filter_gen_by_ids(gen_all, idset)
        gen_path = os.path.join(out_dir, "gen.json")
        save_json(gen_path, gen_subset)

        gt_subset = fix_and_filter_gt_for_coco(gt_full, idset)
        gt_path = os.path.join(out_dir, "gt.json")
        save_json(gt_path, gt_subset)

        metrics_path = os.path.join(out_dir, "metrics.json")
        try:
            evaluate_subset(gt_path, gen_path, metrics_path)
        except Exception as e:
            print(f"⚠️ Eval failed for tag '{tag}': {e}")

    # --- by BAND ---
    print("\n=== Building subsets by BAND ===")
    band_map = build_band_index(UNCERTAINTY_JSON)
    desired_order = ["low", "med", "high", "very_high"]
    bands_sorted = [b for b in desired_order if b in band_map] + [b for b in band_map if b not in desired_order]

    for band in bands_sorted:
        idset = band_map[band]
        out_dir = os.path.join(BAND_ROOT, band)
        ensure_dir(out_dir)

        gen_subset = filter_gen_by_ids(gen_all, idset)
        gen_path = os.path.join(out_dir, "gen.json")
        save_json(gen_path, gen_subset)

        gt_subset = fix_and_filter_gt_for_coco(gt_full, idset)
        gt_path = os.path.join(out_dir, "gt.json")
        save_json(gt_path, gt_subset)

        metrics_path = os.path.join(out_dir, "metrics.json")
        try:
            evaluate_subset(gt_path, gen_path, metrics_path)
        except Exception as e:
            print(f"⚠️ Eval failed for band '{band}': {e}")

    print("\nAll subsets created, evaluated, and scored with final_score.")

if __name__ == "__main__":
    main()
