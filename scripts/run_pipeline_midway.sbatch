#!/bin/bash
#SBATCH --job-name=city-pipeline
#SBATCH --output=logs/city_pipeline_%j.out
#SBATCH --error=logs/city_pipeline_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=standard

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs

# Optional local overrides. Copy midway.local.env.example -> midway.local.env.
if [[ -f "${SCRIPT_DIR}/midway.local.env" ]]; then
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/midway.local.env"
fi

# ---- Default config (override in scripts/midway.local.env) ----
: "${CONDA_INIT_SCRIPT:=}"
: "${CONDA_ENV_NAME:=}"
: "${PIPELINE_ROOTS:=txt_data/al_txt,txt_data/ar_txt,txt_data/ca_txt,txt_data/ia_txt,txt_data/ks_txt,txt_data/ky_txt,txt_data/la_txt,txt_data/ma_txt,txt_data/mi_txt,txt_data/mn_txt,txt_data/mo_txt}"
: "${PIPELINE_BATCH_SIZE:=120}"
: "${PIPELINE_SLEEP_BETWEEN_FILES:=1.0}"
: "${PIPELINE_SLEEP_BETWEEN_BATCHES:=120}"
: "${PIPELINE_MAX_BATCHES_PER_JOB:=6}"
: "${PIPELINE_ORG_MODEL:=gemini-2.0-flash}"
: "${PIPELINE_CATEGORY_MODEL:=gemini-3-flash-preview}"
: "${PIPELINE_GEMINI_REFINE_MODEL:=gemini-3.1-pro-preview}"
: "${PIPELINE_OPENAI_REFINE_MODEL:=gpt-5.4}"
: "${MAX_CHAIN_DEPTH:=200}"

CHAIN_DEPTH="${CHAIN_DEPTH:-0}"
echo "[midway] chain depth: ${CHAIN_DEPTH}/${MAX_CHAIN_DEPTH}"
if (( CHAIN_DEPTH >= MAX_CHAIN_DEPTH )); then
  echo "[midway] reached MAX_CHAIN_DEPTH=${MAX_CHAIN_DEPTH}, stop resubmitting."
  exit 0
fi

# Optional conda activation (recommended).
if [[ -n "${CONDA_INIT_SCRIPT}" && -f "${CONDA_INIT_SCRIPT}" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_INIT_SCRIPT}"
  if [[ -n "${CONDA_ENV_NAME}" ]]; then
    conda activate "${CONDA_ENV_NAME}"
  fi
elif [[ -n "${CONDA_ENV_NAME}" ]]; then
  echo "[midway] CONDA_ENV_NAME set but CONDA_INIT_SCRIPT missing. Please set both in midway.local.env."
  exit 2
else
  echo "[midway] CONDA config not provided; assuming python env is already active."
fi

if [[ ! -f "${REPO_ROOT}/.env" ]]; then
  echo "[midway] missing .env in repo root."
  exit 2
fi
set -a
source "${REPO_ROOT}/.env"
set +a

echo "[midway] starting batch runner..."
CMD=(
  python scripts/run_full_pipeline_in_batches.py
  --base_dir "."
  --roots "${PIPELINE_ROOTS}"
  --batch_size "${PIPELINE_BATCH_SIZE}"
  --sleep_between_files "${PIPELINE_SLEEP_BETWEEN_FILES}"
  --sleep_between_batches "${PIPELINE_SLEEP_BETWEEN_BATCHES}"
  --max_batches "${PIPELINE_MAX_BATCHES_PER_JOB}"
  --continue_on_error
  --org_model "${PIPELINE_ORG_MODEL}"
  --category_model "${PIPELINE_CATEGORY_MODEL}"
  --openai_refine_model "${PIPELINE_OPENAI_REFINE_MODEL}"
)

if [[ -n "${PIPELINE_GEMINI_REFINE_MODEL}" ]]; then
  CMD+=(--gemini_refine_model "${PIPELINE_GEMINI_REFINE_MODEL}")
fi

echo "[midway] command: ${CMD[*]}"
set +e
"${CMD[@]}"
RUN_RC=$?
set -e
echo "[midway] batch runner exit code: ${RUN_RC}"

# Check remaining targets and auto-resubmit if needed.
echo "[midway] checking remaining targets via dry-run..."
set +e
DRY_OUT="$(
  python scripts/run_full_pipeline_from_roots_slow.py \
    --base_dir "." \
    --roots "${PIPELINE_ROOTS}" \
    --max_new "${PIPELINE_BATCH_SIZE}" \
    --sleep_between "${PIPELINE_SLEEP_BETWEEN_FILES}" \
    --org_model "${PIPELINE_ORG_MODEL}" \
    --category_model "${PIPELINE_CATEGORY_MODEL}" \
    --openai_refine_model "${PIPELINE_OPENAI_REFINE_MODEL}" \
    --gemini_refine_model "${PIPELINE_GEMINI_REFINE_MODEL}" \
    --dry_run 2>&1
)"
DRY_RC=$?
set -e
printf '%s\n' "${DRY_OUT}"
if (( DRY_RC != 0 )); then
  echo "[midway] dry-run failed, not resubmitting."
  exit "${RUN_RC}"
fi

TARGETS="-1"
while IFS= read -r line; do
  if [[ "${line}" =~ ^targets=([0-9]+) ]]; then
    TARGETS="${BASH_REMATCH[1]}"
    break
  fi
done <<< "${DRY_OUT}"

if [[ "${TARGETS}" == "-1" ]]; then
  echo "[midway] cannot parse targets from dry-run output, not resubmitting."
  exit "${RUN_RC}"
fi

if (( TARGETS > 0 )); then
  NEXT_DEPTH=$((CHAIN_DEPTH + 1))
  echo "[midway] remaining targets=${TARGETS}, resubmitting next job (depth=${NEXT_DEPTH})."
  sbatch --export=ALL,CHAIN_DEPTH="${NEXT_DEPTH}" "$0"
else
  echo "[midway] all done, no remaining targets."
fi

exit "${RUN_RC}"
