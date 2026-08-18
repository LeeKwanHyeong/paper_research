#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/leekwanhyeong/workspace/paper_research}"
PYTHON_BIN="${PYTHON_BIN:-/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
SOURCE_REVISION="${SOURCE_REVISION:?SOURCE_REVISION is required}"
IMPLEMENTATION_LABEL="${IMPLEMENTATION_LABEL:?IMPLEMENTATION_LABEL is required}"
BASELINE_TIMINGS="${BASELINE_TIMINGS:-}"
WARMUP="${WARMUP:-5}"
ITERATIONS="${ITERATIONS:-20}"
PROFILE_ITERATIONS="${PROFILE_ITERATIONS:-3}"
SHAPES="${SHAPES:-intermittent:128:16,instacart:128:64,long:32:256}"

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/manifest"

SOURCE_FILES=(
  "models/TPPs/CountAwareFactory.py"
  "models/TPPs/CountAwareTPP.py"
  "models/Titan/common/memory.py"
  "paper/scripts/count_aware_tpp_backbone/core.py"
  "paper/scripts/profile_titantpp_surprise_memory.py"
  "simple_lab_test/search/scripts/run_titantpp_surprise_memory_profile_20260818.sh"
)

: > "${OUTPUT_ROOT}/manifest/source_manifest.sha256"
for relative_path in "${SOURCE_FILES[@]}"; do
  sha256sum "${PROJECT_ROOT}/${relative_path}" >> \
    "${OUTPUT_ROOT}/manifest/source_manifest.sha256"
done

COMMAND=(
  "${PYTHON_BIN}"
  "${PROJECT_ROOT}/paper/scripts/profile_titantpp_surprise_memory.py"
  --output-dir "${OUTPUT_ROOT}"
  --implementation-label "${IMPLEMENTATION_LABEL}"
  --device cuda
  --warmup "${WARMUP}"
  --iterations "${ITERATIONS}"
  --profile-iterations "${PROFILE_ITERATIONS}"
  --shapes "${SHAPES}"
)
if [[ -n "${BASELINE_TIMINGS}" ]]; then
  COMMAND+=(--baseline-timings "${BASELINE_TIMINGS}")
fi

printf 'source_revision=%s\n' "${SOURCE_REVISION}" | tee \
  "${OUTPUT_ROOT}/logs/run.log"
printf 'implementation_label=%s\n' "${IMPLEMENTATION_LABEL}" | tee -a \
  "${OUTPUT_ROOT}/logs/run.log"
printf 'command=' | tee -a "${OUTPUT_ROOT}/logs/run.log"
printf '%q ' "${COMMAND[@]}" | tee -a "${OUTPUT_ROOT}/logs/run.log"
printf '\n' | tee -a "${OUTPUT_ROOT}/logs/run.log"

SOURCE_REVISION="${SOURCE_REVISION}" "${COMMAND[@]}" 2>&1 | tee -a \
  "${OUTPUT_ROOT}/logs/run.log"
