#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/leekwanhyeong/workspace/paper_research}"
PYTHON_BIN="${PYTHON_BIN:-/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/search_artifacts/count_aware_final_time_t1_integration_e300_20260820}"
AUDIT_ROOT="${AUDIT_ROOT:-${PROJECT_ROOT}/search_artifacts/count_aware_time_quantity_gradient_audit_20260820}"
SOURCE_REVISION="${SOURCE_REVISION:?SOURCE_REVISION must be the frozen 40-character Git revision}"
EXECUTION_ROLE="${EXECUTION_ROLE:-primary_5080_final_time_t1_seed42_validation_only}"
LAMBDA_TAIL="${LAMBDA_TAIL:-0.09111380335463036}"

DATA="${PROJECT_ROOT}/sample_data/intermittent_v2/intermittent_frozen_5000_with_split.parquet"
SPLIT_MANIFEST="${PROJECT_ROOT}/sample_data/intermittent_v2/intermittent_frozen_5000_split_manifest.json"
H0_ROOT="${OUTPUT_ROOT}/h0_scaled_exact_tail_shared"
H3_ROOT="${OUTPUT_ROOT}/h3_lognormal_tail_shared"
COMPARISON_ROOT="${OUTPUT_ROOT}/comparison"
SOURCE_FILES=(
  "${PROJECT_ROOT}/models/TPPs/CountAwareFactory.py"
  "${PROJECT_ROOT}/models/TPPs/CountAwareTPP.py"
  "${PROJECT_ROOT}/paper/scripts/compare_count_aware_final_time_t1_integration.py"
  "${PROJECT_ROOT}/paper/scripts/count_aware_tpp_backbone/constants.py"
  "${PROJECT_ROOT}/paper/scripts/count_aware_tpp_backbone/core.py"
  "${PROJECT_ROOT}/paper/scripts/count_aware_tpp_backbone/reporting.py"
  "${PROJECT_ROOT}/paper/scripts/count_aware_tpp_backbone/training.py"
  "${PROJECT_ROOT}/paper/scripts/run_count_aware_tpp_backbone_control.py"
  "${PROJECT_ROOT}/simple_lab_test/search/scripts/run_count_aware_final_time_t1_integration_e300_20260820.sh"
  "${PROJECT_ROOT}/simple_lab_test/search/tests/test_count_aware_lognormal_time_head.py"
  "${PROJECT_ROOT}/simple_lab_test/search/tests/test_count_aware_final_time_t1_comparator.py"
)

[[ -x "${PYTHON_BIN}" ]]
[[ -f "${DATA}" ]]
[[ -f "${SPLIT_MANIFEST}" ]]
[[ -f "${AUDIT_ROOT}/decision.json" ]]
[[ "${SOURCE_REVISION}" =~ ^[0-9a-f]{40}$ ]]
for source_file in "${SOURCE_FILES[@]}"; do
  [[ -f "${source_file}" ]]
done

"${PYTHON_BIN}" - "${AUDIT_ROOT}/decision.json" <<'PY'
import json
import sys

decision = json.load(open(sys.argv[1], encoding="utf-8"))
expected = "replace_slope_family_keep_shared_gradient"
if decision.get("recommendation") != expected:
    raise SystemExit(
        f"audit recommendation must be {expected}: {decision.get('recommendation')}"
    )
if decision.get("validation_evaluated") or decision.get("held_out_test_evaluated"):
    raise SystemExit("audit must be train-only with held-out lock")
PY

mkdir -p "${OUTPUT_ROOT}/logs"
{
  printf 'source_revision=%s\n' "${SOURCE_REVISION}"
  printf 'audit_root=%s\n' "${AUDIT_ROOT}"
  printf 'reference=H0_scaled_exact+Hard-LMM+T1\n'
  printf 'candidate=H3_lognormal+Hard-LMM+T1\n'
  printf 'seed=42\n'
  printf 'lambda_tail=%s\n' "${LAMBDA_TAIL}"
  printf 'generated_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sha256sum "${SOURCE_FILES[@]}"
} > "${OUTPUT_ROOT}/source_manifest.txt"

COMMON_ARGS=(
  --data "${DATA}"
  --split-manifest "${SPLIT_MANIFEST}"
  --source-revision "${SOURCE_REVISION}"
  --execution-role "${EXECUTION_ROLE}"
  --dataset-contract intermittent_frozen_5000
  --device cuda
  --epochs 300
  --batch-size 128
  --lr 1e-3
  --lookback-weeks 520
  --max-seq-len 256
  --hidden-dim 64
  --lambda-log-qty 1
  --lambda-tail "${LAMBDA_TAIL}"
  --tail-threshold 46
  --tail-normalization-scale 46
  --tail-clip-cap 187
  --tail-huber-delta 1
  --quantity-variants tail_shared
  --time-scale 3
  --time-head-lr-multiplier 1
  --grad-clip 1
  --min-epochs 40
  --early-stopping-patience 40
  --backbones titantpp
  --seeds 42
  --allow-partial-contract
)

export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_final_time_t1_integration}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_final_time_t1_integration}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_final_time_t1_integration}"

exec > >(tee -a "${OUTPUT_ROOT}/logs/launcher.log") 2>&1
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/paper/scripts/run_count_aware_tpp_backbone_control.py" \
  "${COMMON_ARGS[@]}" \
  --output-dir "${H0_ROOT}" \
  --time-head-mode scaled_exact_rmtpp \
  --time-w-max 3.3333333333333335 \
  --time-intercept-limit 30 \
  --time-wd-safety-limit 40

"${PYTHON_BIN}" "${PROJECT_ROOT}/paper/scripts/run_count_aware_tpp_backbone_control.py" \
  "${COMMON_ARGS[@]}" \
  --output-dir "${H3_ROOT}" \
  --time-head-mode lognormal_duration \
  --time-sigma-floor 0.001

"${PYTHON_BIN}" "${PROJECT_ROOT}/paper/scripts/compare_count_aware_final_time_t1_integration.py" \
  --reference-artifact "${H0_ROOT}" \
  --candidate-artifact "${H3_ROOT}" \
  --output-dir "${COMPARISON_ROOT}"

printf '{"status":"complete","source_revision":"%s","held_out_test_evaluated":false}\n' \
  "${SOURCE_REVISION}" > "${OUTPUT_ROOT}/status.json"
