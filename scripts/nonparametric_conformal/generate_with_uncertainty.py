#!/usr/bin/env python3
# generate_with_uncertainty.py
import os, sys, json, argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

sys.path.insert(0, "/home/white-shark/Desktop/Ozkan/envisage_captioning_lavis/envisage_lavis_models")
import lavis
from lavis.models import load_model_and_preprocess

def set_seed(seed=123):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    os.environ["PYTHONHASHSEED"]=str(seed)

def list_images_from_dir(images_root: str, limit: int=-1) -> List[Path]:
    exts = {".jpg",".jpeg",".png",".bmp",".webp"}
    files = [p for p in Path(images_root).rglob("*") if p.suffix.lower() in exts]
    files = sorted(files)
    return files if (limit is None or limit<0) else files[:limit]

# def get_llm_and_tokenizer(model):
#     cand = []
#     for attr in ["llm_model","llm","opt_model","text_decoder","language_model"]:
#         if hasattr(model, attr): cand.append(getattr(model, attr))
#     llm = cand[0] if cand else None
#     tok = None
#     for attr in ["llm_tokenizer","tokenizer","opt_tokenizer","text_tokenizer"]:
#         if hasattr(model, attr): tok = getattr(model, attr); break
#     return llm, tok
def get_llm_and_tokenizer(model):
    # Use the exact components used for generation
    llm = getattr(model, "opt_model", None)
    tok = getattr(model, "opt_tokenizer", None)
    return llm, tok


def softmax(x, dim=-1): return torch.softmax(x, dim=dim)

# def collect_token_probs(out, selected_seq_ids):
#     scores = out.scores
#     gen_len = len(scores)
#     chosen = selected_seq_ids[-gen_len:]
#     token_probs = []
#     for t, logits_t in enumerate(scores):
#         probs_t = softmax(logits_t, dim=-1)
#         if probs_t.shape[0] > 1:
#             chosen_col = chosen[t]
#             col = probs_t[:, chosen_col]
#             idx = int(torch.argmax(col).item())
#             p_t = probs_t[idx, chosen_col].item()
#         else:
#             p_t = probs_t[0, chosen[t]].item()
#         token_probs.append(max(min(p_t, 1.0), 1e-12))
#     return token_probs

def collect_token_probs(out, selected_seq_ids):
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

# def merge_subwords(tokens: List[str], bands: List[str]) -> Tuple[List[str], List[str]]:
#     """Merge BPE/byte-level tokens into words; propagate the MAX (worst) band to the word."""
#     words, word_bands = [], []
#     cur, cur_band = "", "low"
#     def worse(a,b):
#         order = {"low":0,"med":1,"high":2,"very_high":3}
#         return a if order[a] >= order[b] else b

#     for tok, band in zip(tokens, bands):
#         st = tok.replace("▁"," ").replace("Ġ"," ")  # common sentencepiece/BPE markers
#         piece = st.strip()
#         if tok.startswith(("▁","Ġ")) or (cur=="" and words):  # new word boundary
#             if cur:
#                 words.append(cur); word_bands.append(cur_band)
#             cur, cur_band = piece, band
#         else:
#             # continuation subword
#             joiner = "" if piece.startswith("'") else ""
#             cur = (cur + joiner + piece)
#             cur_band = worse(cur_band, band)
#     if cur:
#         words.append(cur); word_bands.append(cur_band)
#     return words, word_bands

def merge_subwords(tokens: List[str], bands: List[str]) -> Tuple[List[str], List[str]]:
    words, word_bands = [], []
    cur, cur_band = "", "low"
    order = {"low":0,"med":1,"high":2,"very_high":3}

    def worse(a,b): return a if order[a] >= order[b] else b
    def norm(tok: str) -> str:
        s = tok.replace("Ġ"," ").replace("▁"," ")
        # strip a few common byte artifacts if they appear
        for junk in ["Â", "Ċ", "ĉ", "Š", "¤"]:
            s = s.replace(junk, "")
        return s

    for tok, band in zip(tokens, bands):
        st = norm(tok)
        piece = st.strip()
        new_word_boundary = (st.startswith(" ") or tok.startswith(("Ġ","▁")))
        if new_word_boundary:
            if cur:
                words.append(cur); word_bands.append(cur_band)
            cur, cur_band = piece, band
        else:
            cur = cur + piece
            cur_band = worse(cur_band, band)
    if cur:
        words.append(cur); word_bands.append(cur_band)
    return words, word_bands


def band_from_nll(nll, q68, q95, q997):
    if nll <= q68: return "low"
    if nll <= q95: return "med"
    if nll <= q997: return "high"
    return "very_high"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--images_or_dir", required=True)
    ap.add_argument("--quantiles_json", required=True, help="conformal_blip2_token_seq_quantiles.json")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--decode", choices=["beam","sample"], default="beam")
    ap.add_argument("--beams", type=int, default=5)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=20)
    ap.add_argument("--min_new_tokens", type=int, default=5)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--num_images", type=int, default=-1)
    ap.add_argument("--out_json", default="captions_with_uncertainty.json")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    Image.MAX_IMAGE_PIXELS = None

    with open(args.quantiles_json, "r", encoding="utf-8") as f:
        q = json.load(f)
    q68 = q["token_nll_quantiles"]["0.68"]
    q95 = q["token_nll_quantiles"]["0.95"]
    q997 = q["token_nll_quantiles"]["0.997"]
    Q68 = q["seq_avg_nll_quantiles"]["0.68"]
    Q95 = q["seq_avg_nll_quantiles"]["0.95"]
    Q997 = q["seq_avg_nll_quantiles"]["0.997"]

    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_opt", model_type="caption_coco_opt6.7b", is_eval=True, device=device
    )
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval(); torch.set_grad_enabled(False)

    llm, tokenizer = get_llm_and_tokenizer(model)

    # Inputs
    target = args.images_or_dir
    paths = [Path(target)] if Path(target).is_file() else list_images_from_dir(target, args.num_images)
    if not paths:
        raise FileNotFoundError(f"No images at {target}")

    # Decoding args
    if args.decode == "beam":
        gen_args = dict(
            use_nucleus_sampling=False,
            num_beams=args.beams,
            max_length=args.max_new_tokens,
            max_new_tokens=args.max_new_tokens,
            min_length=args.min_new_tokens,
            num_captions=1,
            return_dict_in_generate=True,
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

    outputs = []
    for i, img_path in tqdm(list(enumerate(paths, 1)), total=len(paths), desc="Generating"):
        raw = Image.open(img_path).convert("RGB")
        image = vis_processors["eval"](raw).unsqueeze(0).to(device)
        try:
            with torch.inference_mode():
                out = model.generate({"image": image}, **gen_args)
        except TypeError:
            safe = dict(gen_args)
            for k in ["return_dict_in_generate","output_scores","max_new_tokens"]:
                safe.pop(k, None)
            with torch.inference_mode():
                out = model.generate({"image": image}, **safe)
            raise RuntimeError("No scores returned; rebuild with HF generate exposure to proceed.")

        sequences = getattr(out, "sequences", None)
        texts = getattr(out, "sequences_text", None)
        scores = getattr(out, "scores", None)
        if sequences is None or scores is None:
            raise RuntimeError("No sequences/scores in output.")

        seq_ids = sequences[0].tolist()
        token_probs = collect_token_probs(out, seq_ids)
        token_nll = [-float(np.log(max(p, 1e-12))) for p in token_probs]
        seq_avg_nll = float(np.mean(token_nll))

        # # Get token strings (best effort)
        # toks = None
        # if hasattr(tokenizer, "convert_ids_to_tokens"):
        #     toks = tokenizer.convert_ids_to_tokens(seq_ids[-len(token_probs):])
        # else:
        #     # fallback: just lengths
        #     toks = [f"T{i}" for i in range(len(token_probs))]

        # bands = [band_from_nll(n, q68, q95, q997) for n in token_nll]
        # words, word_bands = merge_subwords(toks, bands)
        # Map generated token ids -> OPT tokens, then drop special tokens so you
        # don't see [PAD], <s>, </s>, etc. in the "words" array.
        gen_ids = seq_ids[-len(token_probs):]

        bands = [band_from_nll(n, q68, q95, q997) for n in token_nll]

        # Convert ids -> tokens with the OPT tokenizer
        toks = tokenizer.convert_ids_to_tokens(gen_ids, skip_special_tokens=False)

        # Remove special tokens and keep bands aligned
        special_mask = tokenizer.get_special_tokens_mask(gen_ids, already_has_special_tokens=True)
        toks  = [t for t, m in zip(toks, special_mask) if m == 0]
        bands = [b for b, m in zip(bands, special_mask) if m == 0]

        # Merge BPE pieces into words; uses Ġ/▁ (leading-space markers) to detect boundaries
        words, word_bands = merge_subwords(toks, bands)


        # Text string
        if texts is not None and len(texts) > 0:
            caption = texts[0]
        else:
            # If not provided, try decoding with tokenizer
            if hasattr(tokenizer, "decode"):
                caption = tokenizer.decode(seq_ids, skip_special_tokens=True)
            else:
                caption = ""

        seq_band = ("low" if seq_avg_nll <= Q68 else
                    "med" if seq_avg_nll <= Q95 else
                    "high" if seq_avg_nll <= Q997 else
                    "very_high")

        outputs.append({
            "image_id": img_path.stem,
            "caption": caption,
            "sequence_uncertainty_band": seq_band,
            "sequence_avg_nll": seq_avg_nll,
            "words": [{"word": w, "band": b} for w,b in zip(words, word_bands)],
            "tokenwise": [{"nll": n, "p": float(np.exp(-n)), "band": b} for n,b in zip(token_nll, bands)]
        })

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    print(f"✅ Wrote {len(outputs)} results to {args.out_json}")

if __name__ == "__main__":
    main()
