#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract social organization names from reflow.txt using Gemini (names only).
"""

import argparse
import json
import os
import re
import csv
import sys
import time
from typing import Dict, List, Optional

try:
    from PIL import Image
except ImportError:
    Image = None

# Allow importing configs from project root or scripts dir
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from configs.org_entry_schema import (
    get_name_patterns_for_rule_based,
    get_llm_category_descriptions,
    normalize_to_allowed_category,
    is_excluded_organization,
)

import google.generativeai as genai
try:
    from openai import OpenAI  # pyright: ignore[reportMissingImports]
except ImportError:
    OpenAI = None


OPENAI_CATEGORY_ENUM = [
    "Churches",
    "Clubs",
    "Hospitals",
    "Libraries",
    "Organizations > Civic",
    "Organizations > Labor",
    "Organizations > Benevolent and Fraternal",
    "Organizations > Patriotic and Veterans",
    "Organizations > Welfare and Relief",
    "Organizations > Young People",
    "Organizations > Miscellaneous",
    "Secret Societies",
    "Unclear",
]

OPENAI_DEV_PROMPT = """
You are classifying rows from a historical city directory.

Choose exactly one final_category from:
- Churches
- Clubs
- Hospitals
- Libraries
- Organizations > Civic
- Organizations > Labor
- Organizations > Benevolent and Fraternal
- Organizations > Patriotic and Veterans
- Organizations > Welfare and Relief
- Organizations > Young People
- Organizations > Miscellaneous
- Secret Societies
- Unclear

Rules:
1. Use current row plus previous/next rows as context.
2. If current row is clearly a continuation fragment, set is_continuation_line=true and should_merge_with_previous=true.
3. Business/company/non-social entries should be final_category="Unclear".
4. confidence must be in [0,1].
5. Return valid JSON only.
""".strip()

OPENAI_SCHEMA = {
    "name": "organization_classification",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "row_id": {"type": "string"},
            "social_organization": {"type": "string"},
            "final_category": {"type": "string", "enum": OPENAI_CATEGORY_ENUM},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "is_continuation_line": {"type": "boolean"},
            "should_merge_with_previous": {"type": "boolean"},
            "normalized_name": {"type": "string"},
            "reason_short": {"type": "string"},
            "reason_detailed": {"type": "string"},
            "evidence": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "row_id",
            "social_organization",
            "final_category",
            "confidence",
            "is_continuation_line",
            "should_merge_with_previous",
            "normalized_name",
            "reason_short",
            "reason_detailed",
            "evidence",
        ],
        "additionalProperties": False,
    },
}

OPENAI_PAGE_SCHEMA = {
    "name": "organization_classification_page",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "line_no": {"type": "string"},
                        "social_organization": {"type": "string"},
                        "final_category": {"type": "string", "enum": OPENAI_CATEGORY_ENUM},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "is_continuation_line": {"type": "boolean"},
                        "should_merge_with_previous": {"type": "boolean"},
                        "normalized_name": {"type": "string"},
                        "reason_short": {"type": "string"},
                        "reason_detailed": {"type": "string"},
                        "evidence": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "line_no",
                        "social_organization",
                        "final_category",
                        "confidence",
                        "is_continuation_line",
                        "should_merge_with_previous",
                        "normalized_name",
                        "reason_short",
                        "reason_detailed",
                        "evidence",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["rows"],
        "additionalProperties": False,
    },
}


def _iter_txt_files_by_ext(root: str, ext: str) -> List[str]:
    files: List[str] = []
    for r, _, fns in os.walk(root):
        for fn in fns:
            if fn.endswith(ext):
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


def _build_prompt(text: str) -> str:
    return f"""
You extract ONLY organization names from a city directory reflow text.
Do NOT output any fields other than org_names.
Do NOT include people, roles, addresses, times, or categories.
Do NOT add or normalize text. Use contiguous substrings from the input.

Output STRICT JSON:
{{
  "org_names": ["..."]
}}

Text:
{text}
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


def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def resolve_image_path(image_root: str, base_name: str) -> Optional[str]:
    """
    Resolve page image path from a pipeline base name (e.g. from txt or CSV filename).
    Expects base_name to contain __{scans_split}__{year_folder}__{page_side} like
    out_org_lines__al_txt__Birmingham_scans_split__Birmingham -1899__11_left.txt.org_lines.txt
    -> image at image_root/Birmingham_scans_split/Birmingham -1899/11_left.png (or .jpg).
    """
    if not image_root or not os.path.isdir(image_root):
        return None
    s = base_name.replace("\\", "/")
    for suffix in (".social_org_lines.category.csv", ".org_lines.txt", ".reflow.txt", ".txt"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    parts = [p for p in s.split("__") if p]
    scans_split = ""
    year_folder = ""
    page_side = ""
    if len(parts) >= 3:
        page_side = parts[-1]
        year_folder = parts[-2]
        scans_split = parts[-3]
    else:
        path_parts = [p for p in s.split("/") if p]
        idx = -1
        for i, p in enumerate(path_parts):
            if "scans_split" in p:
                idx = i
                break
        if idx >= 0 and len(path_parts) >= idx + 3:
            scans_split = path_parts[idx]
            year_folder = path_parts[idx + 1]
            page_side = path_parts[idx + 2]
    if not (scans_split and year_folder and page_side):
        return None
    for suffix in (".social_org_lines.category", ".org_lines", ".reflow", ".txt", ".json", ".csv"):
        if page_side.endswith(suffix):
            page_side = page_side[: -len(suffix)]
    if not page_side:
        return None
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"):
        candidate = os.path.join(image_root, scans_split, year_folder, page_side + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


def _write_txt(path: str, org_names: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for name in org_names:
            f.write(name + "\n")


def _restore_flat_rel_to_hierarchy(rel_or_base: str) -> str:
    """Convert flat '__' encoded filename into nested relative path if possible."""
    s = (rel_or_base or "").replace("\\", "/")
    if "/" in s:
        return s
    parts = [p for p in s.split("__") if p]
    if len(parts) >= 4:
        head = parts[:-1]
        tail = parts[-1]
        return "/".join(head + [tail])
    return s


def _shorten_rel_leaf(rel_path: str) -> str:
    """Shorten leaf name while keeping directory hierarchy."""
    s = (rel_path or "").replace("\\", "/")
    if not s:
        return s
    parts = [p for p in s.split("/") if p]
    if not parts:
        return s
    leaf = parts[-1]
    for suf in (".org_lines.txt", ".reflow.txt", ".txt"):
        if leaf.endswith(suf):
            leaf = leaf[: -len(suf)]
    parts[-1] = leaf or parts[-1]
    return "/".join(parts)


def _build_reason_path(out_csv: str, reason_suffix: str) -> str:
    sfx = reason_suffix or ".reasons.csv"
    if not sfx.startswith("."):
        sfx = "." + sfx
    if not sfx.endswith(".csv"):
        sfx = sfx + ".csv"
    if out_csv.endswith(".csv"):
        return out_csv[: -len(".csv")] + sfx
    return out_csv + sfx


def _infer_output_rel(fp: str, input_dir: str) -> str:
    """Prefer path relative to input_dir; fallback to parsed basename hierarchy."""
    if input_dir:
        try:
            rel = os.path.relpath(os.path.abspath(fp), os.path.abspath(input_dir))
            if rel and not rel.startswith(".."):
                return _shorten_rel_leaf(_restore_flat_rel_to_hierarchy(rel))
        except Exception:
            pass
    return _shorten_rel_leaf(_restore_flat_rel_to_hierarchy(os.path.basename(fp)))


def _match_org_name(line: str, names_sorted: List[str]) -> str:
    for name in names_sorted:
        if name and name in line:
            return name
    return ""


def _classify_org_name(name: str) -> str:
    """Rule-based classification; keywords from configs.org_entry_schema. Insurance/B&L etc. -> Uncategorized."""
    if not name:
        return ""
    if is_excluded_organization(name):
        return "Uncategorized"
    t = name.lower()
    for label, keys in get_name_patterns_for_rule_based():
        if any(k in t for k in keys):
            return label
    return "Uncategorized"


def _build_category_prompt(name: str) -> str:
    """LLM classification; only allowed categories count as social organizations."""
    cat_lines = "\n".join(get_llm_category_descriptions())
    return f"""
You classify ONE organization name into one category. Only the categories below are valid social organization types.
Return STRICT JSON: {{ "organization_category": "..." }} only.
Use ONLY the category name (e.g. "Churches"), not the description in parentheses.

These are NOT social organizations — always return "Uncategorized" for them:
- Insurance companies, underwriters, indemnity associations
- Mutual fire insurance, mutual life insurance, or any "Mutual ... Insurance"
- Building & Loan associations, Building and Loan associations
- Any financial/insurance business that is not a charity or fraternal benefit society
- Government or public bodies (e.g. Board of Education, Board of Health, city/county departments)
- Schools (e.g. Music School, dance school, art school — not social organizations)

Classify as "Benevolent and Fraternal" (not Patriotic/Veterans/Uncategorized): lodge-style bodies, tribes (e.g. Tecumseh Tribe, Red Men), encampments, Triple Link, Rebekah lodges, I.O.O.F., Grand Lodge, mystic circles, G.A.R. posts, Sons of Veterans camps, Catholic Knights, Hebrew lodges (I.O.B.R.), B.P.O.E. Elks, and similar fraternal or veterans-benefit bodies.

Valid categories (choose exactly one; only these count as social organizations):
{cat_lines}

If the name is not clearly one of these social organization types, return "Uncategorized". Prefer specific categories over Miscellaneous.

Organization name:
{name}
""".strip()


def _build_refine_category_prompt(
    org_name: str,
    line_text: str,
    prev_line: str = "",
    next_line: str = "",
    current_category: str = "",
) -> str:
    """LLM refinement prompt with local line context."""
    cat_lines = "\n".join(get_llm_category_descriptions())
    return f"""
You are performing second-pass category refinement for one historical city-directory row.
Use organization name plus nearby lines as context.
Return STRICT JSON only: {{ "organization_category": "..." }}.
Use only the category name (no parenthetical descriptions).

Current predicted category:
{current_category or "Uncategorized"}

Organization name:
{org_name}

Previous line:
{prev_line or "(none)"}

Current line:
{line_text}

Next line:
{next_line or "(none)"}

These are NOT social organizations — always return "Uncategorized":
- Insurance companies / underwriters / indemnity associations
- Mutual insurance entities
- Building & Loan entities
- Government/public agencies and schools
- General businesses that are not social organizations

Valid categories (choose exactly one):
{cat_lines}

If uncertain, return "Uncategorized".
""".strip()


def _normalize_category(cat: str) -> str:
    """Strip trailing parenthetical (e.g. 'Churches (Religious institutions)' -> 'Churches')."""
    if not cat:
        return cat
    s = cat.strip()
    i = s.find(" (")
    if i > 0 and s.endswith(")"):
        s = s[:i].strip()
    return s


def _classify_org_name_llm(model: genai.GenerativeModel, name: str) -> str:
    data = _call_model(model, _build_category_prompt(name), 0.0) or {}
    cat = (data.get("organization_category") or "").strip()
    cat = _normalize_category(cat)
    return cat if cat else "Uncategorized"


def _build_category_prompt_vision(line_text: str, org_name: str) -> str:
    """Prompt for vision model: use page image + this line to choose category."""
    cat_lines = "\n".join(get_llm_category_descriptions())
    return f"""
You see one page of a historical city directory. The following line of text appears on this page:

"{line_text}"

The organization name extracted from this line is: {org_name!r}

Use the page layout and any section headers visible on the image (e.g. CHURCHES, LODGES, FRATERNAL ORGANIZATIONS, INSURANCE) to choose the correct organization category. If the line clearly falls under a section header on the page, use that section. If it is clearly not a social organization (e.g. insurance, school, government), return "Uncategorized".

Return STRICT JSON only: {{ "organization_category": "..." }}
Use ONLY one of these category names (no parentheses or extra text):
{cat_lines}

Organization name: {org_name}
""".strip()


def _classify_org_name_llm_vision(
    model: genai.GenerativeModel, name: str, line_text: str, image_path: str
) -> str:
    """Classify using page image + line text so section headers can disambiguate."""
    if not image_path or not os.path.isfile(image_path) or Image is None:
        return _classify_org_name_llm(model, name)
    try:
        img = Image.open(image_path)
    except Exception:
        return _classify_org_name_llm(model, name)
    prompt = _build_category_prompt_vision(line_text, name)
    resp = model.generate_content(
        [prompt, img],
        generation_config={
            "temperature": 0.0,
            "response_mime_type": "application/json",
        },
    )
    data = _extract_json(getattr(resp, "text", "") or "") or {}
    cat = (data.get("organization_category") or "").strip()
    cat = _normalize_category(cat)
    return cat if cat else "Uncategorized"


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


def _to_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _truncate_text(s: str, limit: int) -> str:
    t = (s or "").strip()
    if limit <= 0 or len(t) <= limit:
        return t
    return t[: max(0, limit - 1)].rstrip() + "."


def _map_openai_category_to_pipeline(cat: str) -> str:
    c = (cat or "").strip()
    mapping = {
        "Churches": "Churches",
        "Clubs": "Clubs",
        "Hospitals": "Hospitals",
        "Libraries": "Libraries",
        "Organizations > Civic": "Civic Organizations",
        "Organizations > Labor": "Labor Organizations",
        "Organizations > Benevolent and Fraternal": "Benevolent and Fraternal",
        "Organizations > Patriotic and Veterans": "Patriotic Organizations",
        "Organizations > Welfare and Relief": "Welfare Organizations",
        "Organizations > Young People": "Youth Organizations",
        "Organizations > Miscellaneous": "Miscellaneous",
        "Secret Societies": "Secret Societies",
        "Unclear": "Uncategorized",
    }
    return mapping.get(c, "Uncategorized")


def _openai_payload_for_row(rows: List[Dict[str, str]], i: int) -> Dict:
    def pack(row: Optional[Dict[str, str]]) -> Dict[str, str]:
        if not row:
            return {"line_no": "", "social_organization": "", "line": "", "organization_category_input": ""}
        return {
            "line_no": str(row.get("line_no", "")).strip(),
            "social_organization": str(row.get("social_organization", "")).strip(),
            "line": str(row.get("line", "")).strip(),
            "organization_category_input": str(row.get("organization_category", "")).strip(),
        }

    prev_row = rows[i - 1] if i - 1 >= 0 else None
    cur_row = rows[i]
    next_row = rows[i + 1] if i + 1 < len(rows) else None
    return {"previous_row": pack(prev_row), "current_row": pack(cur_row), "next_row": pack(next_row)}


def _classify_row_openai(client: OpenAI, model_name: str, rows: List[Dict[str, str]], i: int) -> Dict:
    payload = _openai_payload_for_row(rows, i)
    resp = client.responses.create(
        model=model_name,
        input=[
            {"role": "developer", "content": OPENAI_DEV_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": OPENAI_SCHEMA["name"],
                "schema": OPENAI_SCHEMA["schema"],
                "strict": True,
            }
        },
    )
    raw = _extract_openai_output_text(resp).strip()
    raw = re.sub(r"^```json\s*", "", raw, flags=re.I)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw) if raw else {}


def _classify_page_openai(
    client: OpenAI,
    model_name: str,
    rows: List[Dict[str, str]],
    developer_prompt: str = OPENAI_DEV_PROMPT,
) -> Dict[str, Dict]:
    payload_rows = []
    for r in rows:
        payload_rows.append(
            {
                "line_no": str(r.get("line_no", "")).strip(),
                "social_organization_input": str(r.get("social_organization", "")).strip(),
                "line": str(r.get("line", "")).strip(),
                "organization_category_input": str(r.get("organization_category", "")).strip(),
            }
        )
    page_prompt = (
        developer_prompt
        + "\n\nClassify ALL rows in one response. Return one output item per input row."
        + "\nPreserve line_no and align outputs strictly to provided rows."
    )
    resp = client.responses.create(
        model=model_name,
        input=[
            {"role": "developer", "content": page_prompt},
            {"role": "user", "content": json.dumps({"rows": payload_rows}, ensure_ascii=False)},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": OPENAI_PAGE_SCHEMA["name"],
                "schema": OPENAI_PAGE_SCHEMA["schema"],
                "strict": True,
            }
        },
    )
    raw = _extract_openai_output_text(resp).strip()
    raw = re.sub(r"^```json\s*", "", raw, flags=re.I)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    data = json.loads(raw) if raw else {}
    out: Dict[str, Dict] = {}
    for item in (data.get("rows") or []):
        if not isinstance(item, dict):
            continue
        ln = str(item.get("line_no", "")).strip()
        if ln:
            out[ln] = item
    return out


def _build_line_rows_openai(
    text: str,
    org_names: List[str],
    client: OpenAI,
    model_name: str,
    refine_model_name: str = "",
    refine_below: float = 0.8,
    max_evidence: int = 2,
    reason_short_max_chars: int = 140,
    reason_detailed_max_chars: int = 220,
    page_chunk_size: int = 120,
    sleep_seconds: float = 0.0,
) -> (List[Dict[str, str]], List[Dict[str, str]]):
    names_sorted = sorted([n for n in org_names if n], key=len, reverse=True)
    rows_raw: List[Dict[str, str]] = []
    for idx, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        org = _match_org_name(line, names_sorted)
        rows_raw.append(
            {
                "organization_category": "",
                "social_organization": org,
                "line_no": str(idx),
                "line": line,
            }
        )

    out_rows: List[Dict[str, str]] = []
    reason_rows: List[Dict[str, str]] = []
    results_by_line: Dict[str, Dict] = {}
    chunk_size = max(1, int(page_chunk_size or 120))
    base_prompt = (
        OPENAI_DEV_PROMPT
        + f"\nKeep reason_short concise (<= {max(20, reason_short_max_chars)} chars)."
        + f"\nKeep reason_detailed to one short sentence (<= {max(40, reason_detailed_max_chars)} chars)."
        + f"\nReturn at most {max(1, max_evidence)} evidence snippets."
    )
    for start in range(0, len(rows_raw), chunk_size):
        chunk = rows_raw[start : start + chunk_size]
        chunk_res = _classify_page_openai(client, model_name, chunk, developer_prompt=base_prompt)
        # Second pass: only low-confidence rows go to the stronger model.
        if refine_model_name and refine_model_name != model_name:
            low_rows: List[Dict[str, str]] = []
            for r in chunk:
                ln = str(r.get("line_no", "")).strip()
                conf = _to_float((chunk_res.get(ln) or {}).get("confidence"), 0.0)
                if conf < refine_below:
                    low_rows.append(r)
            if low_rows:
                refine_prompt = base_prompt + "\nSecond-pass review with higher accuracy on uncertain rows."
                refined = _classify_page_openai(client, refine_model_name, low_rows, developer_prompt=refine_prompt)
                chunk_res.update(refined)
        results_by_line.update(chunk_res)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    for i in range(len(rows_raw)):
        r = rows_raw[i]
        result = results_by_line.get(str(r.get("line_no", "")).strip(), {}) or {}
        final_cat = _map_openai_category_to_pipeline((result.get("final_category") or "").strip())
        final_cat = normalize_to_allowed_category(final_cat)
        out_rows.append(
            {
                "organization_category": final_cat,
                "social_organization": r.get("social_organization", ""),
                "line_no": r.get("line_no", ""),
                "line": r.get("line", ""),
            }
        )
        ev = result.get("evidence", [])
        if not isinstance(ev, list):
            ev = []
        ev_out = [str(x).strip() for x in ev if str(x).strip()][: max(1, max_evidence)]
        reason_rows.append(
            {
                "line_no": r.get("line_no", ""),
                "social_organization": r.get("social_organization", ""),
                "line": r.get("line", ""),
                "final_category_openai": (result.get("final_category") or "").strip(),
                "final_category_pipeline": final_cat,
                "confidence": str(result.get("confidence", "")),
                "is_continuation_line": str(bool(result.get("is_continuation_line", False))),
                "should_merge_with_previous": str(bool(result.get("should_merge_with_previous", False))),
                "normalized_name": str(result.get("normalized_name", "") or ""),
                "reason_short": _truncate_text(str(result.get("reason_short", "") or ""), reason_short_max_chars),
                "reason_detailed": _truncate_text(str(result.get("reason_detailed", "") or ""), reason_detailed_max_chars),
                "evidence": " | ".join(ev_out),
            }
        )

    return _fix_sandwich(out_rows), reason_rows


def _build_line_rows(
    text: str,
    org_names: List[str],
    category_mode: str,
    model: genai.GenerativeModel,
    cache: Dict[str, str],
    image_path: Optional[str] = None,
) -> List[Dict[str, str]]:
    names_sorted = sorted([n for n in org_names if n], key=len, reverse=True)
    rows: List[Dict[str, str]] = []
    for idx, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        org = _match_org_name(line, names_sorted)
        category = _classify_org_name(org) if org else ""
        if org and category in ("", "Uncategorized", "Miscellaneous"):
            if category_mode == "llm":
                if org in cache:
                    category = cache[org]
                else:
                    category = _classify_org_name_llm(model, org)
                    cache[org] = category
            elif category_mode == "llm_vision" and image_path:
                cache_key = (org, line)
                if cache_key in cache:
                    category = cache[cache_key]
                else:
                    category = _classify_org_name_llm_vision(model, org, line, image_path)
                    cache[cache_key] = category
            elif category_mode == "llm_vision":
                if org in cache:
                    category = cache[org]
                else:
                    category = _classify_org_name_llm(model, org)
                    cache[org] = category
        category = normalize_to_allowed_category(category)
        rows.append(
            {
                "organization_category": category,
                "social_organization": org,
                "line_no": str(idx),
                "line": line,
            }
        )
    return _fix_sandwich(rows)


def _fix_sandwich(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """1) If above and below are Uncategorized but middle is not -> set middle to Uncategorized.
    2) If above and below are the same non-Uncategorized category but middle is Uncategorized -> set middle to that category."""
    if len(rows) < 3:
        return rows
    out = [dict(r) for r in rows]
    for i in range(1, len(rows) - 1):
        prev_cat = (rows[i - 1].get("organization_category") or "").strip()
        curr_cat = (rows[i].get("organization_category") or "").strip()
        next_cat = (rows[i + 1].get("organization_category") or "").strip()
        if prev_cat == "Uncategorized" and next_cat == "Uncategorized" and curr_cat and curr_cat != "Uncategorized":
            out[i]["organization_category"] = "Uncategorized"
        elif prev_cat == next_cat and prev_cat and prev_cat != "Uncategorized" and curr_cat == "Uncategorized":
            out[i]["organization_category"] = prev_cat
    return out


def _refine_rows_with_openai(
    text: str,
    rows: List[Dict[str, str]],
    client: Optional["OpenAI"],
    model_name: str,
    target: str = "uncategorized",
    page_chunk_size: int = 120,
    sleep_seconds: float = 0.0,
) -> List[Dict[str, str]]:
    """
    Optional second-pass refinement after Gemini classification.
    Currently targets Uncategorized rows to be re-reviewed by OpenAI model.
    """
    if not client or not model_name:
        return rows
    target = (target or "uncategorized").strip().lower()

    target_orgs: List[str] = []
    seen: set = set()
    for r in rows:
        cat = (r.get("organization_category") or "").strip()
        org = (r.get("social_organization") or "").strip()
        if not org:
            continue
        if target == "uncategorized" and cat != "Uncategorized":
            continue
        if org not in seen:
            seen.add(org)
            target_orgs.append(org)
    if not target_orgs:
        return rows

    refined_rows, _ = _build_line_rows_openai(
        text=text,
        org_names=target_orgs,
        client=client,
        model_name=model_name,
        refine_model_name="",
        refine_below=0.0,
        max_evidence=0,
        reason_short_max_chars=0,
        reason_detailed_max_chars=0,
        page_chunk_size=page_chunk_size,
        sleep_seconds=sleep_seconds,
    )
    cat_by_line: Dict[str, str] = {}
    for rr in refined_rows:
        ln = str(rr.get("line_no", "")).strip()
        cat = normalize_to_allowed_category(rr.get("organization_category") or "")
        if ln and cat:
            cat_by_line[ln] = cat

    out = [dict(r) for r in rows]
    for i, r in enumerate(out):
        ln = str(r.get("line_no", "")).strip()
        old_cat = (r.get("organization_category") or "").strip()
        if target == "uncategorized" and old_cat != "Uncategorized":
            continue
        new_cat = cat_by_line.get(ln, "")
        if new_cat:
            out[i]["organization_category"] = new_cat
    return _fix_sandwich(out)


def _refine_rows_with_gemini(
    rows: List[Dict[str, str]],
    model: Optional[genai.GenerativeModel],
    target: str = "uncategorized",
    sleep_seconds: float = 0.0,
) -> List[Dict[str, str]]:
    """
    Optional second-pass refinement using Gemini model.
    Currently targets Uncategorized rows by default.
    """
    if model is None:
        return rows

    target = (target or "uncategorized").strip().lower()
    out = [dict(r) for r in rows]
    cache: Dict[str, str] = {}
    refine_indexes: List[int] = []
    for i, r in enumerate(out):
        old_cat = (r.get("organization_category") or "").strip()
        if target == "uncategorized" and old_cat != "Uncategorized":
            continue
        if (r.get("social_organization") or "").strip():
            refine_indexes.append(i)

    total_refine = len(refine_indexes)
    if total_refine == 0:
        return _fix_sandwich(out)
    print(f"Gemini refine start: target_rows={total_refine}", flush=True)
    refine_t0 = time.time()

    for n, i in enumerate(refine_indexes, start=1):
        r = out[i]
        old_cat = (r.get("organization_category") or "").strip()
        if target == "uncategorized" and old_cat != "Uncategorized":
            continue

        org = (r.get("social_organization") or "").strip()
        line = (r.get("line") or "").strip()
        if not org:
            continue

        prev_line = (out[i - 1].get("line") or "").strip() if i - 1 >= 0 else ""
        next_line = (out[i + 1].get("line") or "").strip() if i + 1 < len(out) else ""
        cache_key = "\n".join([org, line, prev_line, next_line, old_cat])
        if cache_key in cache:
            new_cat = cache[cache_key]
        else:
            prompt = _build_refine_category_prompt(
                org_name=org,
                line_text=line,
                prev_line=prev_line,
                next_line=next_line,
                current_category=old_cat,
            )
            data = _call_model(model, prompt, 0.0) or {}
            new_cat = _normalize_category((data.get("organization_category") or "").strip())
            new_cat = normalize_to_allowed_category(new_cat or "Uncategorized")
            cache[cache_key] = new_cat

        if new_cat:
            out[i]["organization_category"] = new_cat
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        if n % 20 == 0 or n == total_refine:
            elapsed = time.time() - refine_t0
            avg = elapsed / max(1, n)
            remain = total_refine - n
            eta = int(remain * avg)
            print(
                f"Gemini refine progress: {n}/{total_refine} "
                f"avg={avg:.2f}s/row eta={eta}s",
                flush=True,
            )

    return _fix_sandwich(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract organization names from reflow/org_lines txt (Gemini)")
    ap.add_argument("--input", default="", help="Single input file path")
    ap.add_argument("--input_dir", default="out_reflow", help="Input directory")
    ap.add_argument("--input_ext", default=".reflow.txt", help="Input file suffix, e.g. .reflow.txt or .org_lines.txt")
    ap.add_argument("--output", default="", help="Output file path (.json or .txt)")
    ap.add_argument("--output_dir", default="out_org_names", help="Batch output directory")
    ap.add_argument("--emit_lines", action="store_true", help="Emit per-line social_organization csv/json")
    ap.add_argument("--lines_output", default="", help="Single-file per-line output path (.csv or .json)")
    ap.add_argument("--lines_output_dir", default="out_org_lines_rows", help="Batch per-line output directory")
    ap.add_argument("--lines_output_dir_cat", default="out_org_lines_cat", help="Per-line output dir with categories")
    ap.add_argument("--lines_suffix", default=".cat.csv", help="Per-line category output suffix")
    ap.add_argument("--reason_suffix", default=".reasons.csv", help="Per-line reasons output suffix")
    ap.add_argument("--category_mode", default="rule", help="rule=rules only; llm=Gemini fallback; llm_vision=Gemini+image; openai=Responses API page-batch classification")
    ap.add_argument("--image_root", default="", help="Root dir for page images (required for llm_vision); e.g. path to Birmingham_scans_split parent")
    ap.add_argument("--max_files", type=int, default=0, help="Max files to process (0=all)")
    ap.add_argument("--skip_existing", action="store_true", help="Skip files that already have output CSV in lines_output_dir_cat")
    ap.add_argument("--progress", action="store_true", help="Show progress")
    ap.add_argument("--model", default="gemini-2.0-flash", help="Gemini model name")
    ap.add_argument("--api_key", default="", help="API key (or env GOOGLE_API_KEY)")
    ap.add_argument("--openai_model", default="gpt-4.1-mini", help="OpenAI primary model for category_mode=openai")
    ap.add_argument("--openai_refine_model", default="gpt-5.4", help="OpenAI second-pass model for low-confidence rows")
    ap.add_argument("--openai_refine_below", type=float, default=0.8, help="Second-pass threshold (confidence < this value)")
    ap.add_argument("--openai_max_evidence", type=int, default=2, help="Max evidence items in reasons output")
    ap.add_argument("--openai_reason_short_max_chars", type=int, default=140, help="Max chars for reason_short")
    ap.add_argument("--openai_reason_detailed_max_chars", type=int, default=220, help="Max chars for reason_detailed")
    ap.add_argument("--openai_api_key", default="", help="OpenAI API key (or env OPENAI_API_KEY)")
    ap.add_argument("--openai_page_chunk_size", type=int, default=120, help="Rows per OpenAI page-batch request in category_mode=openai")
    ap.add_argument("--openai_sleep_seconds", type=float, default=0.0, help="Sleep between OpenAI page-batch calls")
    ap.add_argument("--llm_refine_with_openai", action="store_true", help="For category_mode=llm/llm_vision, run OpenAI second-pass refinement")
    ap.add_argument("--llm_refine_target", default="uncategorized", help="Refine target for llm modes: uncategorized")
    ap.add_argument("--llm_refine_with_gemini", action="store_true", help="For category_mode=llm/llm_vision, run Gemini second-pass refinement")
    ap.add_argument("--gemini_refine_model", default="", help="Gemini model for second-pass refinement (e.g. gemini-3.1-pro-preview)")
    ap.add_argument("--gemini_refine_target", default="uncategorized", help="Refine target for Gemini second pass")
    ap.add_argument("--gemini_refine_sleep_seconds", type=float, default=0.0, help="Sleep between Gemini refinement calls")
    ap.add_argument("--temperature", type=float, default=0.2, help="Temperature")
    args = ap.parse_args()

    api_key = args.api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Missing API key: set --api_key or GOOGLE_API_KEY")
    if args.category_mode in ("llm", "llm_vision") and not api_key:
        raise SystemExit("Missing API key: set --api_key or GOOGLE_API_KEY")
    if args.category_mode == "llm_vision" and not args.image_root:
        raise SystemExit("llm_vision requires --image_root (path to folder containing e.g. Birmingham_scans_split)")
    openai_client = None
    if args.category_mode == "openai" or args.llm_refine_with_openai:
        openai_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise SystemExit("OpenAI refinement requires --openai_api_key or OPENAI_API_KEY")
        if OpenAI is None:
            raise SystemExit("openai package not installed. Please install dependencies from requirements.txt")
        openai_client = OpenAI(api_key=openai_key)
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model)
    gemini_refine_model = None
    if args.llm_refine_with_gemini:
        gm = (args.gemini_refine_model or "").strip()
        if not gm:
            raise SystemExit("Gemini refinement requires --gemini_refine_model")
        gemini_refine_model = genai.GenerativeModel(gm)

    targets: List[str] = []
    if args.input:
        targets = [args.input]
    else:
        if os.path.isdir(args.input_dir):
            targets = _iter_txt_files_by_ext(args.input_dir, args.input_ext)
    if not targets:
        raise SystemExit("No input files found")
    if args.max_files and args.max_files > 0:
        targets = targets[: args.max_files]

    if args.output and len(targets) == 1:
        text = _load_text(targets[0])
        data = _call_model(model, _build_prompt(text), args.temperature) or {"org_names": []}
        names = data.get("org_names") or []
        if args.output.endswith(".txt"):
            _write_txt(args.output, names)
        else:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump({"file": targets[0], "org_names": names}, f, ensure_ascii=False, indent=2)
        print(f"Output: {args.output}")
        if args.emit_lines:
            if args.lines_output:
                out_path = args.lines_output
            else:
                os.makedirs(args.lines_output_dir_cat, exist_ok=True)
                rel = _infer_output_rel(targets[0], args.input_dir)
                out_path = os.path.join(args.lines_output_dir_cat, rel + args.lines_suffix)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
            if args.skip_existing and os.path.isfile(out_path):
                print(f"Skipped (exists): {out_path}")
                return
            cache: Dict[str, str] = {}
            rel = _infer_output_rel(targets[0], args.input_dir)
            image_path = resolve_image_path(args.image_root, rel) if args.image_root else None
            reason_rows: List[Dict[str, str]] = []
            if args.category_mode == "openai":
                rows, reason_rows = _build_line_rows_openai(
                    text=text,
                    org_names=names,
                    client=openai_client,
                    model_name=args.openai_model,
                    refine_model_name=args.openai_refine_model,
                    refine_below=args.openai_refine_below,
                    max_evidence=args.openai_max_evidence,
                    reason_short_max_chars=args.openai_reason_short_max_chars,
                    reason_detailed_max_chars=args.openai_reason_detailed_max_chars,
                    page_chunk_size=args.openai_page_chunk_size,
                    sleep_seconds=args.openai_sleep_seconds,
                )
            else:
                rows = _build_line_rows(text, names, args.category_mode, model, cache, image_path=image_path)
                if args.llm_refine_with_openai:
                    rows = _refine_rows_with_openai(
                        text=text,
                        rows=rows,
                        client=openai_client,
                        model_name=args.openai_refine_model,
                        target=args.llm_refine_target,
                        page_chunk_size=args.openai_page_chunk_size,
                        sleep_seconds=args.openai_sleep_seconds,
                    )
                if args.llm_refine_with_gemini:
                    rows = _refine_rows_with_gemini(
                        rows=rows,
                        model=gemini_refine_model,
                        target=args.gemini_refine_target,
                        sleep_seconds=args.gemini_refine_sleep_seconds,
                    )
            if out_path.endswith(".json"):
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump({"file": targets[0], "rows": rows}, f, ensure_ascii=False, indent=2)
            else:
                with open(out_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=["organization_category", "social_organization", "line_no", "line"],
                    )
                    writer.writeheader()
                    for r in rows:
                        writer.writerow(
                            {
                                "organization_category": r.get("organization_category", ""),
                                "social_organization": r.get("social_organization", ""),
                                "line_no": r.get("line_no", ""),
                                "line": r.get("line", ""),
                            }
                        )
            if reason_rows and out_path.endswith(".csv"):
                reason_path = _build_reason_path(out_path, args.reason_suffix)
                with open(reason_path, "w", newline="", encoding="utf-8") as rf:
                    rw = csv.DictWriter(
                        rf,
                        fieldnames=[
                            "line_no",
                            "social_organization",
                            "line",
                            "final_category_openai",
                            "final_category_pipeline",
                            "confidence",
                            "is_continuation_line",
                            "should_merge_with_previous",
                            "normalized_name",
                            "reason_short",
                            "reason_detailed",
                            "evidence",
                        ],
                    )
                    rw.writeheader()
                    rw.writerows(reason_rows)
                print(f"Per-line reasons: {reason_path}")
            print(f"Per-line output: {out_path}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    if args.emit_lines:
        os.makedirs(args.lines_output_dir_cat, exist_ok=True)
    if args.skip_existing and args.emit_lines:
        targets = [
            fp
            for fp in targets
            if not os.path.isfile(
                os.path.join(args.lines_output_dir_cat, _infer_output_rel(fp, args.input_dir) + args.lines_suffix)
            )
        ]
        if not targets:
            print("All files already have output CSV. Nothing to do.")
            return
        print(f"Processing {len(targets)} files (skipping existing CSVs).")
    total = len(targets)
    for idx, fp in enumerate(targets, start=1):
        rel = _infer_output_rel(fp, args.input_dir)
        if args.progress:
            sys.stdout.write(f"\rProgress: {idx}/{total}")
            sys.stdout.flush()
        if args.emit_lines and args.skip_existing:
            out_csv = os.path.join(args.lines_output_dir_cat, rel + args.lines_suffix)
            if os.path.isfile(out_csv):
                continue
        text = _load_text(fp)
        data = _call_model(model, _build_prompt(text), args.temperature) or {"org_names": []}
        names = data.get("org_names") or []
        out_json = os.path.join(args.output_dir, rel + ".org_names.json")
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"file": fp, "org_names": names}, f, ensure_ascii=False, indent=2)
        print(f"Output: {out_json}")
        if args.emit_lines:
            cache = {}
            image_path = resolve_image_path(args.image_root, rel) if args.image_root else None
            reason_rows: List[Dict[str, str]] = []
            if args.category_mode == "openai":
                rows, reason_rows = _build_line_rows_openai(
                    text=text,
                    org_names=names,
                    client=openai_client,
                    model_name=args.openai_model,
                    refine_model_name=args.openai_refine_model,
                    refine_below=args.openai_refine_below,
                    max_evidence=args.openai_max_evidence,
                    reason_short_max_chars=args.openai_reason_short_max_chars,
                    reason_detailed_max_chars=args.openai_reason_detailed_max_chars,
                    page_chunk_size=args.openai_page_chunk_size,
                    sleep_seconds=args.openai_sleep_seconds,
                )
            else:
                rows = _build_line_rows(text, names, args.category_mode, model, cache, image_path=image_path)
                if args.llm_refine_with_openai:
                    rows = _refine_rows_with_openai(
                        text=text,
                        rows=rows,
                        client=openai_client,
                        model_name=args.openai_refine_model,
                        target=args.llm_refine_target,
                        page_chunk_size=args.openai_page_chunk_size,
                        sleep_seconds=args.openai_sleep_seconds,
                    )
                if args.llm_refine_with_gemini:
                    rows = _refine_rows_with_gemini(
                        rows=rows,
                        model=gemini_refine_model,
                        target=args.gemini_refine_target,
                        sleep_seconds=args.gemini_refine_sleep_seconds,
                    )
            out_csv = os.path.join(args.lines_output_dir_cat, rel + args.lines_suffix)
            os.makedirs(os.path.dirname(out_csv), exist_ok=True)
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["organization_category", "social_organization", "line_no", "line"],
                )
                writer.writeheader()
                for r in rows:
                    writer.writerow(
                        {
                            "organization_category": r.get("organization_category", ""),
                            "social_organization": r.get("social_organization", ""),
                            "line_no": r.get("line_no", ""),
                            "line": r.get("line", ""),
                        }
                    )
            if reason_rows:
                reason_path = _build_reason_path(out_csv, args.reason_suffix)
                with open(reason_path, "w", newline="", encoding="utf-8") as rf:
                    rw = csv.DictWriter(
                        rf,
                        fieldnames=[
                            "line_no",
                            "social_organization",
                            "line",
                            "final_category_openai",
                            "final_category_pipeline",
                            "confidence",
                            "is_continuation_line",
                            "should_merge_with_previous",
                            "normalized_name",
                            "reason_short",
                            "reason_detailed",
                            "evidence",
                        ],
                    )
                    rw.writeheader()
                    rw.writerows(reason_rows)
                print(f"Per-line reasons: {reason_path}")
            print(f"Per-line output: {out_csv}")
    if args.progress:
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
