## Civic Associations Pipeline

This repository implements a city directory processing pipeline for extracting organization-related lines and structured category outputs from historical OCR text. The pipeline starts from raw images, normalizes page content into one-organization-per-line intermediates, then classifies and exports final artifacts for downstream analysis. 

### Run 

**stage 1: OCR**
```
python scripts/stage0_ocr/gemini_image_to_txt.py \
  --input_roots "images_data" \
  --output_dir "txt_data" \
  --skip_existing
```

**stage 2: Extract & Classify**
```
cd /path/to/city-directory-tools
set -a && source .env && set +a
python scripts/pipeline/run_full_pipeline_in_batches.py \
  --base_dir "." \
  --roots "txt_data/al_txt,txt_data/ar_txt,txt_data/ca_txt,txt_data/ia_txt,txt_data/ks_txt,txt_data/ky_txt,txt_data/la_txt,txt_data/ma_txt,txt_data/mi_txt,txt_data/mn_txt,txt_data/mo_txt" \
  --batch_size 120 \
  --sleep_between_files 1.0 \
  --sleep_between_batches 300 \
  --continue_on_error \
  --category_model "gemini-2.5-flash" \
  --openai_refine_model "gpt-5.4"
```

**All Gemini version**

```
python scripts/pipeline/run_full_pipeline_in_batches.py \
  --base_dir "." \
  --roots "txt_data/al_txt,txt_data/ar_txt,txt_data/ca_txt,txt_data/ia_txt,txt_data/ks_txt,txt_data/ky_txt,txt_data/la_txt,txt_data/ma_txt,txt_data/mi_txt,txt_data/mn_txt,txt_data/mo_txt" \
  --batch_size 120 \
  --sleep_between_files 1.0 \
  --sleep_between_batches 300 \
  --continue_on_error \
  --category_model "gemini-3-flash-preview" \
  --gemini_refine_model "gemini-3.1-pro-preview"
  ```

**All GPT / OpenAI version**

```
cd /path/to/city-directory-tools
set -a && source .env && set +a
python scripts/pipeline/run_full_pipeline_in_batches_openai.py \
  --base_dir "." \
  --roots "txt_data/al_txt,txt_data/ar_txt,txt_data/ca_txt,txt_data/ia_txt,txt_data/ks_txt,txt_data/ky_txt,txt_data/la_txt,txt_data/ma_txt,txt_data/mi_txt,txt_data/mn_txt,txt_data/mo_txt" \
  --batch_size 120 \
  --sleep_between_files 1.0 \
  --sleep_between_batches 300 \
  --continue_on_error \
  --org_model "gpt-4.1-nano" \
  --name_extraction_model "gpt-4.1-nano" \
  --category_model "gpt-4.1-nano" \
  --category_refine_model "gpt-5.4"
```

**stage 3: Merge**
```
python3 scripts/utils/merge_city_to_single_csv.py \
  --root output/org_lines_cat \
  --outdir output/city_merged
```