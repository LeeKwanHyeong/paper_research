#!/usr/bin/env bash
set -euo pipefail

# Instacart RMTPP LR sensitivity screening.
# This run isolates whether high-LR instability is specific to TitanTPP's
# encoder/memory path or already present in the value-conditioned TPP objective.
#
# Expected environment:
#   conda activate ai_env
#   bash simple_lab_test/search/scripts/run_insta_rmtpp_lr_sensitivity_e50.sh
#
# Common overrides:
#   PROJECT_ROOT=~/workspace/paper_research
#   BASE_ROOT=~/workspace/paper_research/search_artifacts/insta_rmtpp_lr_sensitivity_e50
#   LRS="1e-3 5e-3 1e-2"
#   BATCH_SIZE=512
#   RMTTP_RNN_TYPE=gru

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/workspace/paper_research}"
BASE_ROOT="${BASE_ROOT:-$PROJECT_ROOT/search_artifacts/insta_rmtpp_lr_sensitivity_e50}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATASET="${DATASET:-insta_market_basket}"
LRS="${LRS:-1e-3 5e-3 1e-2}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-512}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-64}"
SEEDS="${SEEDS:-42}"
DEVICE="${DEVICE:-cuda}"
RMTTP_RNN_TYPE="${RMTTP_RNN_TYPE:-gru}"
FORCE_RERUN="${FORCE_RERUN:-1}"
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"

mkdir -p "$BASE_ROOT/logs"

echo "[Instacart RMTPP LR sensitivity] project_root=$PROJECT_ROOT"
echo "[Instacart RMTPP LR sensitivity] base_root=$BASE_ROOT"
echo "[Instacart RMTPP LR sensitivity] dataset=$DATASET"
echo "[Instacart RMTPP LR sensitivity] lrs=$LRS"
echo "[Instacart RMTPP LR sensitivity] epochs=$EPOCHS batch_size=$BATCH_SIZE max_seq_len=$MAX_SEQ_LEN"
echo "[Instacart RMTPP LR sensitivity] rmtpp_rnn_type=$RMTTP_RNN_TYPE"
echo "[Instacart RMTPP LR sensitivity] force_rerun=$FORCE_RERUN stop_on_error=$STOP_ON_ERROR"
echo "[Instacart RMTPP LR sensitivity] start_time=$(date '+%Y-%m-%d %H:%M:%S %Z')"

for lr in $LRS; do
  case "$lr" in
    1e-3) lr_label="1em03" ;;
    5e-3) lr_label="5em03" ;;
    1e-2) lr_label="1em02" ;;
    *)
      lr_label="${lr//./p}"
      lr_label="${lr_label//-m/em}"
      lr_label="${lr_label//e/em}"
      ;;
  esac

  run_dir="$BASE_ROOT/value_conditioned_lr_${lr_label}"
  log_path="$BASE_ROOT/logs/value_conditioned_lr_${lr_label}.log"
  extra_flags=()
  if [[ "$FORCE_RERUN" == "1" || "$FORCE_RERUN" == "true" ]]; then
    extra_flags+=(--force-rerun)
  fi
  if [[ "$STOP_ON_ERROR" == "1" || "$STOP_ON_ERROR" == "true" ]]; then
    extra_flags+=(--stop-on-error)
  fi

  echo
  echo "[Instacart RMTPP LR sensitivity] lr=$lr run_dir=$run_dir"
  echo "[Instacart RMTPP LR sensitivity] log_path=$log_path"

  "$PYTHON_BIN" "$PROJECT_ROOT/simple_lab_test/search/tpp_experiment.py" long-epoch \
    --base-dir "$run_dir" \
    --datasets "$DATASET" \
    --models rmtpp \
    --epochs "$EPOCHS" \
    --seeds "$SEEDS" \
    --lr "$lr" \
    --batch-size "$BATCH_SIZE" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --eval-selections best_val_nll,best_score,final \
    --split-mode fixed \
    --value-head-activation identity \
    --value-input-mode residual \
    --train-loss-scope target_only \
    --loss-mode residual_only \
    --rmtpp-rnn-type "$RMTTP_RNN_TYPE" \
    --device "$DEVICE" \
    "${extra_flags[@]}" \
    2>&1 | tee -a "$log_path"
done

echo
echo "[Instacart RMTPP LR sensitivity] complete_time=$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "[Instacart RMTPP LR sensitivity] artifacts=$BASE_ROOT"
