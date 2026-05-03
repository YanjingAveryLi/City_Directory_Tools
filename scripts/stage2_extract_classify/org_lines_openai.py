#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use OpenAI to normalize raw city-directory OCR into one organization per line.
"""

import argparse
import json
import os
import re
import time
from typing import Dict, List

try:
    from openai import OpenAI  # pyright: ignore[reportMissingImports]
except ImportError:
    OpenAI = None


ORG_LINES_SCHEMA = {
    "name": "organization_lines",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "org_lines": {
                "type": "array",
                "items": {"type": "string"},
            }
        },
        "required": ["org_lines"],
        "additionalProperties": False,
    },
}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip())
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except Exception:
        return default


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


def _extract_openai_output_text(resp) -> str:
    txt = getattr(resp, "output_text", "")
    if isinstance(txt, str) and txt.strip():
        return txt
    output = getattr(resp, "output", None)
    if isinstance(output, list):
        chunks: List[str] = []
        for item in output:
            content = getattr(item, "content", None)
            if not isinstance(content, list):
                continue
            for c in content:
                if getattr(c, "type", "") in ("output_text", "text"):
                    t = getattr(c, "text", "")
                    if isinstance(t, str) and t.strip():
                        chunks.append(t)
        if chunks:
            return "\n".join(chunks)
    return ""


def _lines_with_numbers(lines: List[str]) -> str:
    return "\n".join([f"[{i+1:04d}] {ln}" for i, ln in enumerate(lines)])


def _developer_prompt() -> str:
    return """
You normalize city directory OCR text into one organization per line.

Goal:
- Merge hard-wrapped lines that belong to the same organization entry.
- Output one line per organization entry, preserving order.

Rules:
- Use ONLY text from input lines (copy exact substrings).
- You may join pieces with a single space when merging lines.
- Do NOT add new words/symbols or normalize casing/punctuation.
- Remove section headers, page numbers, and decorative leader lines.
- Return STRICT JSON only matching the provided schema.
""".strip()


def _call_model(client: "OpenAI", model_name: str, lines: List[str]) -> Dict:
    resp = client.responses.create(
        model=model_name,
        input=[
            {"role": "developer", "content": _developer_prompt()},
            {"role": "user", "content": _lines_with_numbers(lines)},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": ORG_LINES_SCHEMA["name"],
                "schema": ORG_LINES_SCHEMA["schema"],
                "strict": True,
            }
        },
    )
    return _extract_json(_extract_openai_output_text(resp))


def _call_model_with_retry(client: "OpenAI", model_name: str, lines: List[str]) -> Dict:
    attempts = max(1, _env_int("OPENAI_REQUEST_ATTEMPTS", 4))
    base_sleep = max(0.0, _env_float("OPENAI_RETRY_BACKOFF_SECONDS", 5.0))
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return _call_model(client, model_name, lines)
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts:
                raise
            sleep_s = min(60.0, base_sleep * (2 ** (attempt - 1)))
            print(
                f"[openai] org_lines attempt {attempt}/{attempts} failed: {exc.__class__.__name__}; retry in {sleep_s:.1f}s",
                flush=True,
            )
            time.sleep(sleep_s)
    if last_exc:
        raise last_exc
    return {}


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
            root_name = os.path.basename(r_abs.rstrip(os.sep))
            rel_norm = rel.replace("\\", "/")
            return f"{root_name}/{rel_norm}" if root_name else rel_norm
    return os.path.basename(fp)


def main() -> None:
    ap = argparse.ArgumentParser(description="OpenAI 输出一行一个组织")
    ap.add_argument("--input", default="", help="单文件路径 .txt")
    ap.add_argument("--input_roots", default="", help="逗号分隔 roots")
    ap.add_argument("--output_dir", default="output/org_lines", help="输出目录")
    ap.add_argument("--model", default="gpt-4.1-nano", help="OpenAI 模型名")
    ap.add_argument("--api_key", default="", help="API key（或环境变量 OPENAI_API_KEY）")
    ap.add_argument("--skip_existing", action="store_true", help="已存在输出则跳过")
    args = ap.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("❌ 缺少 API key: --api_key 或 OPENAI_API_KEY")
    if OpenAI is None:
        raise SystemExit("❌ openai package 未安装，请先安装 requirements.txt")
    client = OpenAI(
        api_key=api_key,
        timeout=_env_float("OPENAI_TIMEOUT_SECONDS", 180.0),
        max_retries=max(0, _env_int("OPENAI_MAX_RETRIES", 5)),
    )

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
        data = _call_model_with_retry(client, args.model, raw_lines) or {"org_lines": []}
        org_lines = data.get("org_lines") or []

        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"file": rel, "org_lines": org_lines}, f, ensure_ascii=False, indent=2)
        _write_txt(txt_path, org_lines)

        print(f"✅ org_lines: {fp} -> {txt_path}")


if __name__ == "__main__":
    main()
