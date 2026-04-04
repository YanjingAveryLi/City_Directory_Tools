#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 Gemini 将原始目录文本整理为“一行一个组织”
"""

import argparse
import json
import os
import re
from typing import Dict, List

import google.generativeai as genai


def _iter_txt_files(root: str) -> List[str]:
    files: List[str] = []
    for r, _, fns in os.walk(root):
        for fn in fns:
            if fn.endswith(".txt"):
                files.append(os.path.join(r, fn))
    files.sort()
    return files


def _extract_json(text: str) -> Dict:
    if not text:
        return {}
    s = text.strip()
    s = re.sub(r"^```json\s*", "", s, flags=re.I)
    s = re.sub(r"^```\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try:
        data = json.loads(s)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _lines_with_numbers(lines: List[str]) -> str:
    return "\n".join([f"[{i+1:04d}] {ln}" for i, ln in enumerate(lines)])


def _build_prompt(lines: List[str]) -> str:
    return f"""
You normalize city directory OCR text into one organization per line.

Goal:
- Merge hard-wrapped lines that belong to the same organization entry.
- Output one line per organization entry, preserving order.

Rules:
- Use ONLY text from input lines (copy exact substrings).
- You may join pieces with a single space when merging lines.
- Do NOT add new words/symbols or normalize casing/punctuation.
- Remove section headers, page numbers, and decorative leader lines.

Output STRICT JSON:
{{
  "org_lines": ["..."]
}}

Input lines:
{_lines_with_numbers(lines)}
""".strip()


def _call_model(model: genai.GenerativeModel, prompt: str, temperature: float) -> Dict:
    resp = model.generate_content(
        prompt,
        generation_config={
            "temperature": temperature,
            "response_mime_type": "application/json",
        },
    )
    return _extract_json(getattr(resp, "text", "") or "")


def _write_txt(path: str, lines: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write((ln or "").strip() + "\n")


def _infer_output_rel(fp: str, roots: List[str]) -> str:
    """Prefer a path relative to one of input roots; fallback to basename."""
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
            # Keep the state-level root folder name in output hierarchy, e.g.
            # txt_data/ca_txt/... -> ca_txt/...
            root_name = os.path.basename(r_abs.rstrip(os.sep))
            rel_norm = rel.replace("\\", "/")
            return f"{root_name}/{rel_norm}" if root_name else rel_norm
    return os.path.basename(fp)


def main() -> None:
    ap = argparse.ArgumentParser(description="Gemini 输出一行一个组织")
    ap.add_argument("--input", default="", help="单文件路径 .txt")
    ap.add_argument("--input_roots", default="", help="逗号分隔 roots")
    ap.add_argument("--output_dir", default="out_org_lines", help="输出目录")
    ap.add_argument("--model", default="gemini-2.0-flash", help="Gemini 模型名")
    ap.add_argument("--api_key", default="", help="API key（或环境变量 GOOGLE_API_KEY）")
    ap.add_argument("--temperature", type=float, default=0.2, help="温度")
    ap.add_argument("--skip_existing", action="store_true", help="已存在输出则跳过")
    args = ap.parse_args()

    api_key = args.api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("❌ 缺少 API key: --api_key 或 GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model)

    targets: List[str] = []
    roots: List[str] = [r.strip() for r in args.input_roots.split(",") if r.strip()]
    if args.input:
        targets = [args.input]
    else:
        for root in roots:
            if os.path.isdir(root):
                targets.extend(_iter_txt_files(root))
    if not targets:
        raise SystemExit("❌ 未指定输入文件或 roots")

    os.makedirs(args.output_dir, exist_ok=True)

    for fp in targets:
        rel = _infer_output_rel(fp, roots)
        json_path = os.path.join(args.output_dir, rel + ".org_lines.json")
        txt_path = os.path.join(args.output_dir, rel + ".org_lines.txt")
        if args.skip_existing and os.path.isfile(json_path) and os.path.isfile(txt_path):
            print(f"⏭️ skip existing: {txt_path}")
            continue

        with open(fp, "r", encoding="utf-8") as f:
            raw_lines = [ln.rstrip("\n") for ln in f]
        data = _call_model(model, _build_prompt(raw_lines), args.temperature) or {"org_lines": []}
        org_lines = data.get("org_lines") or []

        os.makedirs(os.path.dirname(json_path), exist_ok=True)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"file": rel, "org_lines": org_lines}, f, ensure_ascii=False, indent=2)
        _write_txt(txt_path, org_lines)

        print(f"✅ org_lines: {fp} -> {txt_path}")


if __name__ == "__main__":
    main()
