# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo does

A two-stage LLM pipeline for digitizing historical city directory data:

1. **Stage 1 (org_lines):** OCR page images to plain text, then reformat raw OCR text into one-organization-per-line using Gemini (or OpenAI). Produces `output/org_lines/**/*.org_lines.txt`.
2. **Stage 2 (categorization):** For each org_lines file, extract organization names and classify each line into a category (Churches, Clubs, Fraternal, etc.). Produces `output/org_names/**/*.org_names.json` and `output/org_lines_cat/**/*.cat.csv` (+ `.cat.reasons.csv`).

## Environment setup

```bash
pip install -r requirements.txt
set -a && source .env && set +a   # loads GOOGLE_API_KEY and OPENAI_API_KEY
```

`.env` holds `GOOGLE_API_KEY` and `OPENAI_API_KEY`. These are never committed (see `.gitignore`).

## Running the pipeline

**OCR images first (if starting from raw scans):**
```bash
python scripts/gemini_image_to_txt.py \
  --input_roots "data/raw" \
  --output_dir "data/txt" \
  --skip_existing
```

**Standard run (Gemini stage 1 + Gemini/OpenAI stage 2):**
```bash
python scripts/run_full_pipeline_in_batches.py \
  --base_dir "." \
  --roots "data/txt/al_txt,data/txt/ar_txt,..." \
  --batch_size 120 --sleep_between_files 1.0 --sleep_between_batches 300 \
  --continue_on_error \
  --category_model "gemini-2.5-flash" \
  --openai_refine_model "gpt-5.4"
```

**All-OpenAI run:**
```bash
python scripts/run_full_pipeline_in_batches_openai.py \
  --base_dir "." \
  --roots "data/txt/al_txt,..." \
  --batch_size 120 --sleep_between_files 1.0 --sleep_between_batches 300 \
  --continue_on_error \
  --org_model "gpt-4.1-nano" --name_extraction_model "gpt-4.1-nano" \
  --category_model "gpt-4.1-nano" --category_refine_model "gpt-5.4"
```

**Check progress:**
```bash
python scripts/show_pipeline_progress.py --log logs/pipeline_progress.jsonl
```

**Run a single file manually (stage 1):**
```bash
python scripts/org_lines_gemini.py \
  --input data/txt/al_txt/Birmingham_scans_split/Birmingham\ -1899/11_left.txt \
  --input_roots data/txt/al_txt \
  --output_dir output/org_lines \
  --model gemini-2.0-flash
```

**Run a single file manually (stage 2):**
```bash
python scripts/extract_org_names_from_reflow.py \
  --input output/org_lines/al_txt/... \
  --input_dir output/org_lines \
  --emit_lines \
  --lines_output_dir_cat output/org_lines_cat \
  --output_dir output/org_names \
  --category_mode llm \
  --model gemini-2.5-flash \
  --skip_existing
```

## Architecture

### Script roles

| Script | Role |
|---|---|
| `gemini_image_to_txt.py` | OCR: image → `.txt` in `data/txt/` |
| `org_lines_gemini.py` | Stage 1: raw `.txt` → `output/org_lines/**/*.org_lines.txt` (Gemini) |
| `org_lines_openai.py` | Stage 1: same, using OpenAI instead |
| `extract_org_names_from_reflow.py` | Stage 2: org_lines → `.org_names.json` + `.cat.csv` + `.cat.reasons.csv`; supports Gemini, OpenAI, or hybrid |
| `run_full_pipeline_from_roots_slow.py` | Orchestrates both stages per file with resumability, progress logging to `logs/pipeline_progress.jsonl` |
| `run_full_pipeline_in_batches.py` | Loops `run_full_pipeline_from_roots_slow.py` in batches until no pending targets remain |
| `run_full_pipeline_in_batches_openai.py` | Same loop but delegates to the OpenAI-native pipeline runner |
| `show_pipeline_progress.py` | Reads `logs/pipeline_progress.jsonl` and prints a human-readable summary |

### Classification logic (`extract_org_names_from_reflow.py`)

Stage 2 classification is layered:
1. **Rule-based pass** (`_classify_org_name`): keyword matching against `CATEGORY_TAXONOMY` name_patterns in priority order (from `configs/org_entry_schema.py`).
2. **LLM primary pass** (`category_mode=llm` or `openai`): Gemini or OpenAI classifies names that the rule pass left as `Uncategorized`.
3. **Second-pass refinement** (`--llm_refine_with_openai` or `--llm_refine_with_gemini`): a stronger/different model re-reviews rows still `Uncategorized` after the primary pass.
4. **Sandwich fix** (`_fix_sandwich`): post-hoc heuristic — if a row is flanked on both sides by `Uncategorized` it is corrected to `Uncategorized`; if flanked on both sides by the same category it is corrected to that category.

### Configs

- `configs/org_entry_schema.py` — single source of truth for: allowed categories (`ALLOWED_ORGANIZATION_CATEGORIES`), category taxonomy with name patterns and sub-types (`CATEGORY_TAXONOMY`), exclusion patterns for non-social-orgs (insurance, schools, etc.), and helper functions used by `extract_org_names_from_reflow.py`.
- `configs/org_keywords.yaml` — keyword lists used to decide which sections of a raw page are org-section headings (`headings_include/exclude`, `content_signals_include/exclude`).

### Data directories (all git-ignored)

| Dir | Content |
|---|---|
| `data/txt/*_txt/` | OCR output; state-keyed subdirs (e.g. `al_txt/`) |
| `output/org_lines/` | Stage 1 outputs: `.org_lines.txt` + `.org_lines.json` per input file |
| `output/org_names/` | Stage 2 name extraction: `.org_names.json` |
| `output/org_lines_cat/` | Stage 2 classification: `.cat.csv` and `.cat.reasons.csv` |
| `logs/` | `pipeline_progress.jsonl` — append-only JSONL run log |

### Output path convention

Output paths mirror input directory structure. For input `data/txt/al_txt/Birmingham_scans_split/Birmingham -1899/11_left.txt`, stage 1 writes to `output/org_lines/al_txt/Birmingham_scans_split/Birmingham -1899/11_left.txt.org_lines.txt`. The state root folder name (`al_txt`) is preserved as the first segment so outputs are namespaced by state.

### Resumability

`run_full_pipeline_from_roots_slow.py` skips files that already have their expected output (checks for `.org_lines.txt` for stage 1 and `.cat.csv` for stage 2). A `--dry_run` flag prints how many files are pending without running anything — `run_full_pipeline_in_batches.py` uses this to decide when to stop looping.
