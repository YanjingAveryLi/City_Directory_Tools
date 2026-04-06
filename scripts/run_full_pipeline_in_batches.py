#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import subprocess
import sys
import time
from typing import List


TARGETS_RE = re.compile(r"^targets=(\d+)\b")


def _run_capture(cmd: List[str], cwd: str) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)


def _run_stream(cmd: List[str], cwd: str) -> int:
    return subprocess.run(cmd, cwd=cwd).returncode


def _parse_targets(stdout_text: str) -> int:
    for line in stdout_text.splitlines():
        m = TARGETS_RE.match(line.strip())
        if m:
            return int(m.group(1))
    return -1


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run full pipeline in repeated batches until targets=0."
    )
    ap.add_argument("--base_dir", default=".", help="Project root")
    ap.add_argument(
        "--roots",
        default="",
        help="Comma-separated roots. Empty means auto-detect *_txt under base_dir.",
    )
    ap.add_argument("--batch_size", type=int, default=120, help="Files per batch")
    ap.add_argument("--sleep_between_files", type=float, default=1.0, help="Pass-through to per-file sleep")
    ap.add_argument("--sleep_between_batches", type=float, default=60.0, help="Sleep seconds between batches")
    ap.add_argument("--max_batches", type=int, default=0, help="0 means unlimited")
    ap.add_argument("--continue_on_error", action="store_true", help="Continue next batch even if current batch has failures")
    ap.add_argument("--org_model", default="gemini-2.0-flash")
    ap.add_argument("--category_model", default="gemini-2.5-flash")
    ap.add_argument("--openai_model", default="gpt-4.1-mini")
    ap.add_argument("--openai_refine_model", default="gpt-5.4")
    ap.add_argument("--gemini_refine_model", default="", help="Gemini second-pass refinement model for llm mode")
    args = ap.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    runner = os.path.join("scripts", "run_full_pipeline_from_roots_slow.py")

    batch_idx = 0
    while True:
        batch_idx += 1
        if args.max_batches > 0 and batch_idx > args.max_batches:
            print(f"Reached max_batches={args.max_batches}, stop.")
            break

        dry_cmd = [
            "python",
            runner,
            "--base_dir",
            base_dir,
            "--max_new",
            str(args.batch_size),
            "--sleep_between",
            str(args.sleep_between_files),
            "--org_model",
            args.org_model,
            "--category_model",
            args.category_model,
            "--openai_model",
            args.openai_model,
            "--openai_refine_model",
            args.openai_refine_model,
            "--gemini_refine_model",
            args.gemini_refine_model,
            "--dry_run",
        ]
        if args.roots.strip():
            dry_cmd.extend(["--roots", args.roots])
        dry = _run_capture(dry_cmd, base_dir)
        if dry.returncode != 0:
            sys.stdout.write(dry.stdout or "")
            sys.stderr.write(dry.stderr or "")
            raise SystemExit(f"Dry-run failed with exit code {dry.returncode}")

        targets = _parse_targets(dry.stdout)
        print(f"\n=== Batch {batch_idx} check: targets={targets} ===")
        if targets == 0:
            print("All done. No pending targets.")
            break
        if targets < 0:
            print("Cannot parse targets from dry-run output, stop for safety.")
            print(dry.stdout)
            raise SystemExit(2)

        run_cmd = [
            "python",
            runner,
            "--base_dir",
            base_dir,
            "--max_new",
            str(args.batch_size),
            "--sleep_between",
            str(args.sleep_between_files),
            "--org_model",
            args.org_model,
            "--category_model",
            args.category_model,
            "--openai_model",
            args.openai_model,
            "--openai_refine_model",
            args.openai_refine_model,
            "--gemini_refine_model",
            args.gemini_refine_model,
        ]
        if args.roots.strip():
            run_cmd.extend(["--roots", args.roots])
        rc = _run_stream(run_cmd, base_dir)
        if rc != 0:
            print(f"Batch {batch_idx} finished with non-zero exit code: {rc}")
            if not args.continue_on_error:
                raise SystemExit(rc)

        print(f"Batch {batch_idx} complete. Sleep {args.sleep_between_batches}s ...")
        time.sleep(args.sleep_between_batches)


if __name__ == "__main__":
    main()
