#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use Gemini to OCR images into plain text files.
"""

import argparse
import os
import time
from typing import List

import google.generativeai as genai
from PIL import Image


IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp")


def iter_image_files(root: str) -> List[str]:
    files: List[str] = []
    for r, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(IMG_EXTS):
                files.append(os.path.join(r, fn))
    files.sort()
    return files


def infer_rel(fp: str, roots: List[str]) -> str:
    f_abs = os.path.abspath(fp)
    for root in roots:
        if not root:
            continue
        r_abs = os.path.abspath(root)
        try:
            rel = os.path.relpath(f_abs, r_abs)
        except Exception:
            continue
        if rel and not rel.startswith(".."):
            return rel.replace("\\", "/")
    return os.path.basename(fp)


_OCR_PROMPTS = [
    (
        "You are OCR for historical city directory page images.\n"
        "Return plain UTF-8 text only.\n"
        "Preserve line breaks as much as possible.\n"
        "Do not add explanations, markdown, or code fences."
    ),
    (
        "Transcribe every word visible in this image exactly as printed.\n"
        "Include all columns, headings, and advertisements.\n"
        "Return raw text only, preserving line breaks. No explanations."
    ),
    (
        "Read and output all text in this scanned page image.\n"
        "Return plain text only."
    ),
]


def call_ocr(model: genai.GenerativeModel, image_path: str, retries: int, sleep_s: float) -> str:
    last_err = None
    total_attempts = retries + 1
    for attempt in range(total_attempts):
        prompt = _OCR_PROMPTS[min(attempt, len(_OCR_PROMPTS) - 1)]
        try:
            with Image.open(image_path) as raw:
                img = raw.convert("RGB")
            resp = model.generate_content([prompt, img])
            text = (getattr(resp, "text", "") or "").strip()
            finish = resp.candidates[0].finish_reason if resp.candidates else "N/A"
            print(f"[debug] call_ocr attempt={attempt+1}/{total_attempts} text_len={len(text)} finish={finish}", flush=True)
            if text:
                return text
            # Empty response — log safety info and retry
            if resp.candidates:
                sr = getattr(resp.candidates[0], "safety_ratings", [])
                if sr:
                    print(f"[debug] safety_ratings={sr}", flush=True)
            pf = getattr(resp, "prompt_feedback", None)
            if pf:
                print(f"[debug] prompt_feedback={pf}", flush=True)
            last_err = RuntimeError(f"Empty OCR response (finish={finish})")
            time.sleep(sleep_s)
        except Exception as e:
            last_err = e
            print(f"[debug] call_ocr attempt={attempt+1}/{total_attempts} exception: {e}", flush=True)
            time.sleep(sleep_s)
    print(f"[warn] OCR returned empty for {image_path} after {total_attempts} attempts", flush=True)
    return ""


def merge_ocr_candidates(model: genai.GenerativeModel, candidates: List[str]) -> str:
    nonempty = [c for c in candidates if c.strip()]
    if not nonempty:
        return ""
    if len(nonempty) == 1:
        return nonempty[0]
    n = len(candidates)
    parts = "\n\n".join(
        f"=== Candidate {i+1} ===\n{c}" for i, c in enumerate(candidates)
    )
    prompt = (
        f"You have {n} OCR outputs of the same historical city directory page.\n"
        "Produce one best version by majority vote:\n"
        "- Prefer text that appears in 2+ candidates (exact or near-exact).\n"
        "- Preserve original line breaks, spacing, and all details.\n"
        "- Return plain text only — no markdown, no explanations.\n\n"
        f"{parts}"
    )
    resp = model.generate_content(prompt)
    return (getattr(resp, "text", "") or "").strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Gemini OCR images to txt")
    ap.add_argument("--input", default="", help="Single image path")
    ap.add_argument("--input_roots", default="", help="Comma-separated image roots")
    ap.add_argument("--output_dir", default="data/txt", help="Output txt root")
    ap.add_argument("--model", default="gemini-2.0-flash", help="Gemini model name")
    ap.add_argument("--api_key", default="", help="Gemini API key or env GOOGLE_API_KEY")
    ap.add_argument("--skip_existing", action="store_true", help="Skip if output txt exists")
    ap.add_argument("--max_files", type=int, default=0, help="Limit file count (0=all)")
    ap.add_argument("--num_votes", type=int, default=1, help="OCR 投票次数（>1时多次调用后合并）")
    ap.add_argument("--retry", type=int, default=2, help="Retry times on failure")
    ap.add_argument("--sleep_seconds", type=float, default=1.0, help="Sleep between retries")
    args = ap.parse_args()

    api_key = args.api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Missing API key: set --api_key or GOOGLE_API_KEY")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model)

    roots = [r.strip() for r in args.input_roots.split(",") if r.strip()]
    targets: List[str] = []
    if args.input:
        targets = [args.input]
    else:
        for root in roots:
            if os.path.isdir(root):
                targets.extend(iter_image_files(root))
    if not targets:
        raise SystemExit("No image files found")

    if args.max_files > 0:
        targets = targets[: args.max_files]

    os.makedirs(args.output_dir, exist_ok=True)

    ok = 0
    fail = 0
    for i, fp in enumerate(targets, start=1):
        rel = infer_rel(fp, roots)
        rel_no_ext, _ = os.path.splitext(rel)
        out_txt = os.path.join(args.output_dir, rel_no_ext + ".txt")
        os.makedirs(os.path.dirname(out_txt), exist_ok=True)

        if args.skip_existing and os.path.isfile(out_txt):
            print(f"[{i}/{len(targets)}] skip existing: {out_txt}")
            continue

        try:
            num_votes = max(1, args.num_votes)
            if num_votes == 1:
                text = call_ocr(model, fp, args.retry, args.sleep_seconds)
            else:
                candidates = []
                for v in range(num_votes):
                    c = call_ocr(model, fp, args.retry, args.sleep_seconds)
                    candidates.append(c)
                    print(f"  vote {v+1}/{num_votes}: {len(c)} chars")
                text = merge_ocr_candidates(model, candidates)
                print(f"  merged -> {len(text)} chars")
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write(text + ("\n" if text else ""))
            print(f"[{i}/{len(targets)}] ok: {fp} -> {out_txt}")
            ok += 1
        except Exception as e:
            print(f"[{i}/{len(targets)}] fail: {fp} ({e})")
            fail += 1

    print(f"done ok={ok} fail={fail}")
    if fail > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
