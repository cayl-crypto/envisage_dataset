#!/usr/bin/env python3
# blip2_caption_eval_batched.py
import os, sys, json, argparse, random, re
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

# ensure your local LAVIS path is first if needed
sys.path.insert(0, "/home/white-shark/Desktop/Ozkan/envisage_captioning_lavis/envisage_lavis_models")
import lavis
from lavis.models import load_model_and_preprocess

def set_seed(seed: int = 123, deterministic_cudnn: bool = True):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def list_images(path: str, limit: int = -1, recursive: bool = True) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    p = Path(path)
    if p.is_file():
        return [p] if p.suffix.lower() in exts else []
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {p}")
    imgs = [x for x in (p.rglob("*") if recursive else p.iterdir()) if x.is_file() and x.suffix.lower() in exts]
    imgs = sorted(imgs)
    if limit and limit > 0: imgs = imgs[:limit]
    return imgs

def load_checkpoint_into_model(model: torch.nn.Module, ckpt_path: str):
    print(f"📦 Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    load_out = model.load_state_dict(state, strict=False)
    mk = getattr(load_out, "missing_keys", []); uk = getattr(load_out, "unexpected_keys", [])
    print(f"✅ Weights loaded. missing_keys={len(mk)} unexpected_keys={len(uk)}")
    if mk: print("  missing_keys (first 10):", mk[:10])
    if uk: print("  unexpected_keys (first 10):", uk[:10])

def clean_caption(text: str) -> str:
    """
    Keep only the top hypothesis, remove extra newlines, collapse spaces.
    """
    if not isinstance(text, str): return text
    # Some LAVIS builds return multiple lines for top-k; keep first non-empty line.
    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    cleaned = re.sub(r"\s+", " ", first_line).strip()
    return cleaned or text.strip()

def batch(iterable, n):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf

def main():
    parser = argparse.ArgumentParser(description="BLIP-2 OPT 6.7B batched captioning")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--image_or_dir", required=True)
    parser.add_argument("--device", default="cuda:2" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch images for faster throughput")

    # decoding
    parser.add_argument("--decode", choices=["beam", "sample"], default="beam")
    parser.add_argument("--beams", type=int, default=3)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.9)

    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--min_new_tokens", type=int, default=5)
    parser.add_argument("--num_captions", type=int, default=1)

    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", default="")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    set_seed(args.seed)
    Image.MAX_IMAGE_PIXELS = None
    device = torch.device(args.device)

    print("Using LAVIS from:", lavis.__file__)
    print("🔧 Loading BLIP-2 OPT 6.7B (COCO init)...")
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_opt",
        model_type="caption_coco_opt6.7b",
        is_eval=True,
        device=device,
    )
    load_checkpoint_into_model(model, args.ckpt)
    model.eval()
    torch.set_grad_enabled(False)

    imgs = list_images(args.image_or_dir, limit=args.num_samples, recursive=args.recursive)
    if not imgs: raise FileNotFoundError(f"No images found in {args.image_or_dir}")
    print(f"🖼️ Found {len(imgs)} image(s). Batch size = {args.batch_size}. Decode = {args.decode}")

    # generation args
    if args.decode == "beam":
        gen_args = dict(
            use_nucleus_sampling=False,
            num_beams=args.beams,
            max_length=args.max_new_tokens,
            max_new_tokens=args.max_new_tokens,
            min_length=args.min_new_tokens,
            num_captions=args.num_captions,
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
            num_captions=args.num_captions,
        )

    # output filename
    if args.out:
        out_file = args.out
    else:
        mode = "beam" if args.decode == "beam" else f"sample_p{args.top_p}_t{args.temperature}"
        tag = f"_{args.tag}" if args.tag else ""
        out_file = f"blip2_opt6p7_ft_envisage_{mode}_max{args.max_new_tokens}_min{args.min_new_tokens}_bs{args.batch_size}{tag}.json"

    results = []
    total = len(imgs)
    pbar = tqdm(total=total)

    # process in batches
    for batch_paths in batch(imgs, args.batch_size):
        # load and preprocess images on CPU, then stack and move once
        raw_list = []
        for p in batch_paths:
            try:
                raw = Image.open(p).convert("RGB")
            except Exception as e:
                print(f"⚠️  Failed to open {p}: {e}")
                raw = None
            raw_list.append((p, raw))

        valid = [(p, raw) for p, raw in raw_list if raw is not None]
        if not valid:
            pbar.update(len(batch_paths))
            continue

        tensors = [vis_processors["eval"](raw).unsqueeze(0) for _, raw in valid]
        images_tensor = torch.cat(tensors, dim=0).to(device, non_blocking=True)

        try:
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                outs = model.generate({"image": images_tensor}, **gen_args)
        except TypeError:
            safe_args = dict(gen_args); safe_args.pop("max_new_tokens", None)
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                outs = model.generate({"image": images_tensor}, **safe_args)

        # Normalize outputs to a list of strings (one per input)
        # LAVIS typically returns a list length == batch_size
        if isinstance(outs, (list, tuple)):
            out_list = list(outs)
        else:
            # single string for entire batch (unexpected); replicate
            out_list = [str(outs)] * len(valid)

        for (p, _), cap in zip(valid, out_list):
            # if a hypothesis blob came with multiple lines, keep only the first non-empty line
            cap = cap[0] if isinstance(cap, (list, tuple)) and cap else cap
            cap = clean_caption(cap)
            results.append({"image_id": p.stem, "caption": cap})
            pbar.set_postfix_str(p.name[:24] + " → " + cap[:40])
            pbar.update(1)

        # account for any failed openings in this batch
        failed = len(batch_paths) - len(valid)
        if failed > 0:
            pbar.update(failed)

    pbar.close()
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(results)} captions to {out_file}")

if __name__ == "__main__":
    main()
