#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
import time
from typing import List, Tuple


def iter_txt_files(root: str) -> List[str]:
    files: List[str] = []
    for r, _, fns in os.walk(root):
        for fn in fns:
            if fn.endswith(".txt"):
                files.append(os.path.join(r, fn))
    files.sort()
    return files


def rel_to_project(base_dir: str, file_path: str) -> str:
    rel = os.path.relpath(os.path.abspath(file_path), os.path.abspath(base_dir))
    return rel.replace("\\", "/")


def infer_rel_from_roots(base_dir: str, file_path: str, roots: List[str]) -> str:
    f_abs = os.path.abspath(file_path)
    for root in roots:
        if not root:
            continue
        r_abs = os.path.abspath(root)
        try:
            rel = os.path.relpath(f_abs, r_abs)
        except Exception:
            continue
        if rel and not rel.startswith(".."):
            # Keep state-level root folder in output path, e.g.
            # txt_data/ca_txt/... -> ca_txt/...
            root_name = os.path.basename(r_abs.rstrip(os.sep))
            rel_norm = rel.replace("\\", "/")
            return f"{root_name}/{rel_norm}" if root_name else rel_norm
    return rel_to_project(base_dir, file_path)


def strip_txt_suffix(path_rel: str) -> str:
    s = path_rel.replace("\\", "/")
    if s.endswith(".txt"):
        return s[: -len(".txt")]
    return s


def expected_org_lines_txt(base_dir: str, raw_fp: str, roots: List[str]) -> str:
    rel = infer_rel_from_roots(base_dir, raw_fp, roots)
    return os.path.join(base_dir, "out_org_lines", rel + ".org_lines.txt")


def expected_cat_csv(base_dir: str, raw_fp: str, roots: List[str]) -> str:
    rel = strip_txt_suffix(infer_rel_from_roots(base_dir, raw_fp, roots))
    return os.path.join(base_dir, "out_org_lines_cat", rel + ".cat.csv")


def collect_targets(base_dir: str, roots: List[str], skip_prefix: str) -> List[Tuple[str, str]]:
    targets: List[Tuple[str, str]] = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for fp in iter_txt_files(root):
            rel = rel_to_project(base_dir, fp)
            seg = rel.split("/")
            city_year = seg[-2] if len(seg) >= 2 else "UNKNOWN"
            if skip_prefix and city_year.startswith(skip_prefix):
                continue
            targets.append((city_year, fp))
    targets.sort(key=lambda x: (x[0], x[1]))
    return targets


def run(cmd: List[str], cwd: str) -> int:
    return subprocess.run(cmd, cwd=cwd).returncode


def main() -> None:
    ap = argparse.ArgumentParser(description="Run full pipeline from raw *_txt roots (slow, resumable).")
    ap.add_argument("--base_dir", default=".", help="Project root")
    ap.add_argument(
        "--roots",
        default="",
        help="Comma-separated absolute/relative roots. Empty means auto-detect txt_data/*_txt first, then base_dir/*_txt.",
    )
    ap.add_argument("--max_new", type=int, default=120, help="Max files to process in this run")
    ap.add_argument("--sleep_between", type=float, default=1.0, help="Seconds between files")
    ap.add_argument("--skip_city_prefix", default="", help="Skip city-year prefix (e.g. Birmingham)")
    ap.add_argument("--dry_run", action="store_true", help="Print plan only")
    ap.add_argument("--org_model", default="gemini-2.0-flash")
    ap.add_argument("--category_model", default="gemini-2.5-flash")
    ap.add_argument("--openai_model", default="gpt-4.1-mini")
    ap.add_argument("--openai_refine_model", default="gpt-5.4")
    args = ap.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    if args.roots.strip():
        roots = [os.path.abspath(r.strip()) for r in args.roots.split(",") if r.strip()]
    else:
        roots = []
        txt_data_root = os.path.join(base_dir, "txt_data")
        if os.path.isdir(txt_data_root):
            for name in sorted(os.listdir(txt_data_root)):
                p = os.path.join(txt_data_root, name)
                if os.path.isdir(p) and name.endswith("_txt"):
                    roots.append(p)
        if not roots:
            for name in sorted(os.listdir(base_dir)):
                p = os.path.join(base_dir, name)
                if os.path.isdir(p) and name.endswith("_txt"):
                    roots.append(p)

    if not roots:
        raise SystemExit("No roots found. Use --roots or ensure *_txt folders exist under base_dir.")

    targets = collect_targets(base_dir, roots, args.skip_city_prefix)
    if not targets:
        print("No raw txt files found for current filters.")
        return

    planned: List[Tuple[str, str, bool, bool]] = []
    for city_year, raw_fp in targets:
        need_stage1 = not os.path.isfile(expected_org_lines_txt(base_dir, raw_fp, roots))
        need_stage2 = not os.path.isfile(expected_cat_csv(base_dir, raw_fp, roots))
        if need_stage1 or need_stage2:
            planned.append((city_year, raw_fp, need_stage1, need_stage2))

    planned = planned[: args.max_new]
    print(f"targets={len(planned)} (max_new={args.max_new})")
    print(f"roots={len(roots)}")
    for i, (city_year, raw_fp, need1, need2) in enumerate(planned[:10], start=1):
        print(f"[{i}] {city_year} :: {raw_fp} | stage1={need1} stage2={need2}")
    if len(planned) > 10:
        print(f"... ({len(planned) - 10} more)")

    if args.dry_run:
        return

    roots_csv = ",".join(roots)
    ok = 0
    fail = 0

    for idx, (city_year, raw_fp, need1, need2) in enumerate(planned, start=1):
        print(f"\n[{idx}/{len(planned)}] {city_year}")
        print(f"raw: {raw_fp}")
        org_fp = expected_org_lines_txt(base_dir, raw_fp, roots)

        if need1:
            cmd1 = [
                "python",
                "scripts/org_lines_gemini.py",
                "--input",
                raw_fp,
                "--input_roots",
                roots_csv,
                "--output_dir",
                "out_org_lines",
                "--model",
                args.org_model,
                "--skip_existing",
            ]
            rc1 = run(cmd1, base_dir)
            if rc1 != 0:
                print(f"  !! stage1 failed rc={rc1}")
                fail += 1
                time.sleep(args.sleep_between)
                continue

        if need2:
            cmd2 = [
                "python",
                "scripts/extract_org_names_from_reflow.py",
                "--input",
                org_fp,
                "--input_dir",
                "out_org_lines",
                "--emit_lines",
                "--lines_output_dir_cat",
                "out_org_lines_cat",
                "--output_dir",
                "out_org_names",
                "--category_mode",
                "llm",
                "--model",
                args.category_model,
                "--llm_refine_with_openai",
                "--llm_refine_target",
                "uncategorized",
                "--openai_refine_model",
                args.openai_refine_model,
                "--openai_page_chunk_size",
                "160",
                "--openai_sleep_seconds",
                "0.8",
                "--skip_existing",
            ]
            rc2 = run(cmd2, base_dir)
            if rc2 != 0:
                print(f"  !! stage2 failed rc={rc2}")
                fail += 1
                time.sleep(args.sleep_between)
                continue

        ok += 1
        time.sleep(args.sleep_between)

    print(f"\ndone ok={ok} fail={fail}")
    if fail > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
