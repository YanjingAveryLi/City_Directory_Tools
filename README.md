## Civic Associations Pipeline

This repository implements a city directory processing pipeline for extracting organization-related lines and structured category outputs from historical OCR text. The pipeline starts from raw images, normalizes page content into one-organization-per-line intermediates, then classifies and exports final artifacts for downstream analysis. 

Current production flow keeps these scripts:
- `scripts/run_full_pipeline_in_batches.py`
- `scripts/run_full_pipeline_from_roots_slow.py`
- `scripts/org_lines_gemini.py`
- `scripts/extract_org_names_from_reflow.py`
- `scripts/gemini_image_to_txt.py` 

### What it does
1. Read raw files under `txt_data/*_txt` roots (auto-detected), or custom roots via `--roots`.
2. Stage1: generate `out_org_lines/.../*.org_lines.txt`.
3. Stage2: generate:
   - `out_org_names/.../*.org_names.json`
   - `out_org_lines_cat/.../*.cat.csv`
   - `out_org_lines_cat/.../*.cat.reasons.csv`

### Run 

**OCR images to txt_data first**
```
python scripts/gemini_image_to_txt.py \
  --input_roots "images_data" \
  --output_dir "txt_data" \
  --skip_existing
```

**Batch run (explicit roots)**
```
python scripts/run_full_pipeline_in_batches.py \
  --base_dir "." \
  --roots "txt_data/al_txt,txt_data/ar_txt,txt_data/ca_txt,txt_data/ia_txt,txt_data/ks_txt,txt_data/ky_txt,txt_data/la_txt,txt_data/ma_txt,txt_data/mi_txt,txt_data/mn_txt,txt_data/mo_txt" \
  --batch_size 120 \
  --sleep_between_files 1.0 \
  --sleep_between_batches 300 \
  --continue_on_error \
  --category_model "gemini-2.5-flash" \
  --openai_refine_model "gpt-5.4"
```