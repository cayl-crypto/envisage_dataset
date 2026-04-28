#!/usr/bin/env python3
# calibrate_conformal_blip2.py
import os, sys, json, argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

# Your local LAVIS path
sys.path.insert(0, "/home/white-shark/Desktop/Ozkan/envisage_captioning_lavis/envisage_lavis_models")
import lavis
from lavis.models import load_model_and_preprocess

def set_seed(seed=123):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    os.environ["PYTHONHASHSEED"]=str(seed)

def load_val_annotations(val_json_path: str) -> Dict[str, Any]:
    with open(val_json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def id_to_path(image_id: int, images_root: str) -> Path:
    # ENVISAGE names: 8-digit zero-padded + .jpg
    name = f"{int(image_id):08d}.jpg"
    p = Path(images_root) / name
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    return p

def get_llm_and_tokenizer(model):
    # Try common LAVIS BLIP-2 attributes
    cand = []
    for attr in ["llm_model","llm","opt_model","text_decoder","language_model"]:
        if hasattr(model, attr):
            cand.append(getattr(model, attr))
    llm = cand[0] if cand else None

    tok = None
    for attr in ["llm_tokenizer","tokenizer","opt_tokenizer","text_tokenizer"]:
        if hasattr(model, attr):
            tok = getattr(model, attr); break
    return llm, tok

def build_inputs_for_llm(model, image_tensor, device):
    """
    Create the image-conditioned prompt for the LM using model internals.
    We rely on LAVIS generate to produce sequences, but we also try to
    get scores from the underlying HF LM by re-running generate with output_scores.
    """
    # Many LAVIS BLIP-2 models accept {"image": image_tensor} directly in .generate
    return {"image": image_tensor.to(device)}

def softmax(x, dim=-1):
    return torch.softmax(x, dim=dim)

def collect_token_scores_from_generate_output(out, tokenizer, selected_seq_ids):
    """
    Robustly align per-step logits (out.scores) with the chosen sequence token IDs.
    Works whether sequences include the prompt or not, and when scores are longer
    than the selected sequence (e.g., other beams kept going).
    """
    scores = out.scores  # list of logits tensors, length = T (max generated steps)
    if not isinstance(scores, (list, tuple)) or len(scores) == 0:
        raise RuntimeError("No per-step scores found in generation output.")

    # Tokens we want to evaluate (the chosen final sequence)
    chosen_ids = list(selected_seq_ids)  # make a copy
    L = len(chosen_ids)
    T = len(scores)

    # If sequences include a prompt, keep only the generated suffix
    # Heuristic: if T <= L, assume last T tokens are generated tokens
    # If T > L (rare with inputs_embeds), we will trim scores down to L anyway.
    if T <= L:
        chosen = chosen_ids[-T:]
        effective_scores = scores  # use all T score steps
    else:
        # More score steps than tokens in the chosen sequence (can happen with beams of unequal length).
        # Trim scores to the last L steps so lengths match.
        chosen = chosen_ids  # length L
        effective_scores = scores[-L:]  # keep only last L score tensors
        T = L

    token_probs = []
    for t in range(T):
        logits_t = effective_scores[t]              # [num_cands, vocab] or [1, vocab]
        probs_t = torch.softmax(logits_t, dim=-1)
        chosen_col = chosen[t]

        if probs_t.shape[0] > 1:
            # Pick the row (candidate) that places the most mass on the chosen token
            idx = int(torch.argmax(probs_t[:, chosen_col]).item())
            p_t = probs_t[idx, chosen_col].item()
        else:
            p_t = probs_t[0, chosen_col].item()

        # clamp to avoid log(0)
        p_t = float(max(min(p_t, 1.0), 1e-12))
        token_probs.append(p_t)

    return token_probs


def quantiles(values: List[float], levels=(0.68,0.95,0.997)):
    arr = np.asarray(values, dtype=np.float64)
    qs = {}
    for q in levels:
        # finite-sample conformal quantile: ceil((n+1)*q)
        n = len(arr)
        k = int(np.ceil((n+1)*q)) - 1
        k = np.clip(k, 0, n-1)
        qs[str(q)] = float(np.partition(arr, k)[k])
    return qs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--images_root", required=True)
    ap.add_argument("--val_json", required=True, help="envisage/annotations_eval/val.json")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--decode", choices=["beam","sample"], default="beam")
    ap.add_argument("--beams", type=int, default=5)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=20)
    ap.add_argument("--min_new_tokens", type=int, default=5)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--limit", type=int, default=-1, help="limit number of calibration images")
    ap.add_argument("--out", default="conformal_blip2_token_seq_quantiles.json")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    Image.MAX_IMAGE_PIXELS = None

    print("Using LAVIS from:", lavis.__file__)
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_opt", model_type="caption_coco_opt6.7b", is_eval=True, device=device
    )
    # Load FT weights
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval(); torch.set_grad_enabled(False)

    llm, tokenizer = get_llm_and_tokenizer(model)
    if llm is None or tokenizer is None:
        print("⚠️ Could not find underlying HF LM or tokenizer. We'll rely purely on LAVIS.generate outputs.")

    # Build generation args
    if args.decode == "beam":
        gen_args = dict(
            use_nucleus_sampling=False,
            num_beams=args.beams,
            max_length=args.max_new_tokens,
            max_new_tokens=args.max_new_tokens,
            min_length=args.min_new_tokens,
            num_captions=1,
            return_dict_in_generate=True,   # many LAVIS forward these to HF
            output_scores=True
        )
    else:
        gen_args = dict(
            use_nucleus_sampling=True,
            top_p=args.top_p,
            temperature=args.temperature,
            num_beams=1,
            max_length=args.max_new_tokens,
            max_new_tokens=args.max_new_tokens,
            min_length=args.min_new_tokens,
            num_captions=1,
            return_dict_in_generate=True,
            output_scores=True
        )

    val = load_val_annotations(args.val_json)
    # unique image IDs from val
    image_ids = sorted({ann["image_id"] for ann in val["annotations"]})
    if args.limit and args.limit > 0:
        image_ids = image_ids[:args.limit]

    token_nlls = []
    seq_avg_nlls = []

    for idx, img_id in tqdm(list(enumerate(image_ids, 1)), total=len(image_ids), desc="Calibrating"):
        img_path = id_to_path(img_id, args.images_root)
        try:
            raw = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"skip {img_path}: {e}")
            continue
        image = vis_processors["eval"](raw).unsqueeze(0).to(device)

        # Run generation with scores; some LAVIS builds don't pass kwargs → retry without unsupported keys
        try:
            with torch.inference_mode():
                out = model.generate({"image": image}, **gen_args)
        except TypeError:
            safe = dict(gen_args)
            for k in ["return_dict_in_generate","output_scores","max_new_tokens"]:
                safe.pop(k, None)
            with torch.inference_mode():
                out = model.generate({"image": image}, **safe)
            # If we cannot get scores from LAVIS, we cannot calibrate. Stop here with a clear message.
            raise RuntimeError("Your LAVIS build did not return scores. Expose HF generate with output_scores=True.")

        # Normalize output structure
        # Expect HF-like structure if return_dict_in_generate=True
        sequences = getattr(out, "sequences", None)
        texts = getattr(out, "sequences_text", None)
        scores = getattr(out, "scores", None)
        if sequences is None:
            # Some LAVIS return dict-like. Try common keys:
            sequences = out.get("sequences", None) if isinstance(out, dict) else None
            scores = out.get("scores", None) if isinstance(out, dict) else None

        if sequences is None or scores is None:
            raise RuntimeError("No sequences/scores in output; cannot calibrate token-level thresholds.")

        seq_ids = sequences[0].tolist()
        probs = collect_token_scores_from_generate_output(out, tokenizer, seq_ids)
        nlls = [-float(np.log(max(p, 1e-12))) for p in probs]
        token_nlls.extend(nlls)
        seq_avg_nlls.append(float(np.mean(nlls)))

    token_q = quantiles(token_nlls, levels=(0.68,0.95,0.997))
    seq_q   = quantiles(seq_avg_nlls, levels=(0.68,0.95,0.997))

    payload = {
        "token_nll_quantiles": token_q,
        "seq_avg_nll_quantiles": seq_q,
        "decoding_config": {
            "decode": args.decode,
            "beams": args.beams,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.min_new_tokens
        },
        "counts": {"tokens": len(token_nlls), "captions": len(seq_avg_nlls)}
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"✅ Saved thresholds to {args.out}")
    print("Token NLL quantiles:", token_q)
    print("Seq avg NLL quantiles:", seq_q)

if __name__ == "__main__":
    main()
