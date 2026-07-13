#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/leekwanhyeong/workspace/paper_research}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ai_env/bin/python}"
ENTRYPOINT="$PROJECT_ROOT/simple_lab_test/search/tpp_experiment.py"
REFERENCE_ENTRYPOINT="$PROJECT_ROOT/simple_lab_test/search/reevaluate_titantpp_validation.py"
ARTIFACT_ROOT="$PROJECT_ROOT/search_artifacts"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}"
mkdir -p "$MPLCONFIGDIR"

V2_ROOT="$ARTIFACT_ROOT/model_enhancement_v2_inter_short_e50_0710"
V2_RUN_REL="runs/intermittent/titantpp/lossmode_hybrid/split_fixed/value_identity/valueinput_residual/valueemb_8/trainscope_target_only/profile_dataset_best/base_2p0/small_lmm/epochs_50/seed_42"
V2_CHECKPOINT="$V2_ROOT/$V2_RUN_REL/checkpoints/best_val_nll_model.pt"
V2_MARKED="$V2_ROOT/cache/intermittent/fixed_split/marked_fixed_base_2p0.parquet"
V2_REFERENCE="$ARTIFACT_ROOT/model_enhancement_v2_inter_validation_reference_v5a_0712"
V5_RUN="$ARTIFACT_ROOT/model_enhancement_v5a_inter_short_e50_0712"

mkdir -p "$V2_REFERENCE/logs" "$V5_RUN/logs"

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Freeze V2 validation-only RPS reference"
"$PYTHON_BIN" "$REFERENCE_ENTRYPOINT" \
  --checkpoint "$V2_CHECKPOINT" \
  --marked-parquet "$V2_MARKED" \
  --output-dir "$V2_REFERENCE" \
  --device cuda \
  2>&1 | tee -a "$V2_REFERENCE/logs/run.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Start V5a Intermittent seed-42 e50"
"$PYTHON_BIN" "$ENTRYPOINT" long-epoch \
  --base-dir "$V5_RUN" \
  --datasets intermittent \
  --models titantpp \
  --titan-candidates small_lmm \
  --epochs 50 \
  --seeds 42 \
  --lr 1e-3 \
  --batch-size 128 \
  --lookback-weeks 52 \
  --max-seq-len 16 \
  --split-mode fixed \
  --value-head-activation identity \
  --value-head-mode shared \
  --qty-mark-gradient-mode coupled \
  --value-encoder-gradient-mode coupled \
  --value-input-mode residual \
  --train-loss-scope target_only \
  --loss-mode hybrid \
  --marker-loss-mode ce_rps \
  --lambda-ordinal 0.10 \
  --eval-selections best_val_nll,best_score,final \
  --device cuda \
  --force-rerun \
  --stop-on-error \
  2>&1 | tee -a "$V5_RUN/logs/run.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] V5a Intermittent seed-42 e50 sequence completed"
