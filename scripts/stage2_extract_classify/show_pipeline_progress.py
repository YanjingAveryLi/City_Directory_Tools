#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Dict, List


def _format_hms(seconds: float) -> str:
    s = max(0, int(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _read_jsonl(path: str) -> List[Dict]:
    if not os.path.isfile(path):
        return []
    out: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Show concise pipeline progress from JSONL logs.")
    ap.add_argument("--log", default="logs/pipeline_progress.jsonl", help="Path to progress jsonl")
    ap.add_argument("--recent", type=int, default=8, help="Show N most recent file records")
    args = ap.parse_args()

    rows = _read_jsonl(args.log)
    if not rows:
        print(f"No progress records found: {args.log}")
        return

    file_rows = [r for r in rows if r.get("event") == "file_done"]
    if not file_rows:
        print(f"No file_done records found yet in: {args.log}")
        return

    last = file_rows[-1]
    processed = int(last.get("processed", 0) or 0)
    total = int(last.get("total_planned", 0) or 0)
    ok = int(last.get("ok", 0) or 0)
    fail = int(last.get("fail", 0) or 0)
    elapsed = float(last.get("elapsed_sec", 0.0) or 0.0)
    avg = float(last.get("avg_sec_per_file", 0.0) or 0.0)
    eta = float(last.get("eta_sec", -1.0) or -1.0)
    pct = (100.0 * processed / total) if total > 0 else 0.0

    print("=== Pipeline Progress ===")
    print(f"log: {args.log}")
    print(f"processed: {processed}/{total} ({pct:.1f}%)")
    print(f"ok/fail: {ok}/{fail}")
    print(f"elapsed: {_format_hms(elapsed)}")
    print(f"avg per file: {avg:.2f}s")
    print(f"eta: {_format_hms(eta) if eta >= 0 else '--:--:--'}")

    recent_n = max(1, int(args.recent))
    print(f"\n=== Recent {recent_n} files ===")
    for r in file_rows[-recent_n:]:
        idx = int(r.get("idx", 0) or 0)
        city = str(r.get("city_year", ""))
        status = str(r.get("status", ""))
        fsec = float(r.get("file_sec", 0.0) or 0.0)
        s1 = float(r.get("stage1_sec", 0.0) or 0.0)
        s2 = float(r.get("stage2_sec", 0.0) or 0.0)
        print(f"[{idx}] {city} | status={status} | file={fsec:.2f}s (s1={s1:.2f}s, s2={s2:.2f}s)")


if __name__ == "__main__":
    main()
