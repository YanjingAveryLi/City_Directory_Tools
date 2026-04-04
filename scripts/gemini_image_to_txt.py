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


def call_ocr(model: genai.GenerativeModel, image_path: str, retries: int, sleep_s: float) -> str:
    prompt = (
        "You are OCR for historical city directory page images.\n"
        "Return plain UTF-8 text only.\n"
        "Preserve line breaks as much as possible.\n"
        "Do not add explanations, markdown, or code fences."
    )
    last_err = None
    for _ in range(retries + 1):
        try:
            with Image.open(image_path) as img:
                resp = model.generate_content([prompt, img])
            text = (getattr(resp, "text", "") or "").strip()
            return text
        except Exception as e:
            last_err = e
            time.sleep(sleep_s)
    raise RuntimeError(f"OCR failed after retries: {last_err}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Gemini OCR images to txt")
    ap.add_argument("--input", default="", help="Single image path")
    ap.add_argument("--input_roots", default="", help="Comma-separated image roots")
    ap.add_argument("--output_dir", default="txt_data", help="Output txt root")
    ap.add_argument("--model", default="gemini-2.0-flash", help="Gemini model name")
    ap.add_argument("--api_key", default="", help="Gemini API key or env GOOGLE_API_KEY")
    ap.add_argument("--skip_existing", action="store_true", help="Skip if output txt exists")
    ap.add_argument("--max_files", type=int, default=0, help="Limit file count (0=all)")
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
            text = call_ocr(model, fp, args.retry, args.sleep_seconds)
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
