#!/usr/bin/env bash
set -euo pipefail

# Re-run the Instacart value-conditioned LR sensitivity screening with a larger
# batch size.  The goal is to check whether a smoother/larger-batch gradient
# improves high-LR stability compared with the previous batch_size=512 run.
#
# Expected environment:
#   conda activate ai_env
#   bash simple_lab_test/search/scripts/run_insta_lr_sensitivity_b1024_e50.sh
#
# Common overrides:
#   PROJECT_ROOT=~/workspace/paper_research
#   BASE_ROOT=~/workspace/paper_research/search_artifacts/insta_lr_sensitivity_b1024_e50
#   BATCH_SIZE=1024
#   LRS="1e-3 5e-3 1e-2"
#   TITAN_CANDIDATES="small_lmm,mid_lmm"
#   FORCE_RERUN=0

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/workspace/paper_research}"
BASE_ROOT="${BASE_ROOT:-$PROJECT_ROOT/search_artifacts/insta_lr_sensitivity_b1024_e50}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATASET="${DATASET:-insta_market_basket}"
LRS="${LRS:-1e-3 5e-3 1e-2}"
VARIANTS="${VARIANTS:-value_conditioned value_conditioned_hybrid}"

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-64}"
SEEDS="${SEEDS:-42}"
DEVICE="${DEVICE:-cuda}"
TITAN_CANDIDATES="${TITAN_CANDIDATES:-small_lmm,mid_lmm}"
FORCE_RERUN="${FORCE_RERUN:-1}"
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"

mkdir -p "$BASE_ROOT/logs"

echo "[Instacart LR sensitivity b1024] project_root=$PROJECT_ROOT"
echo "[Instacart LR sensitivity b1024] base_root=$BASE_ROOT"
echo "[Instacart LR sensitivity b1024] dataset=$DATASET"
echo "[Instacart LR sensitivity b1024] lrs=$LRS"
echo "[Instacart LR sensitivity b1024] variants=$VARIANTS"
echo "[Instacart LR sensitivity b1024] titan_candidates=$TITAN_CANDIDATES"
echo "[Instacart LR sensitivity b1024] epochs=$EPOCHS batch_size=$BATCH_SIZE max_seq_len=$MAX_SEQ_LEN"
echo "[Instacart LR sensitivity b1024] force_rerun=$FORCE_RERUN stop_on_error=$STOP_ON_ERROR"
echo "[Instacart LR sensitivity b1024] start_time=$(date '+%Y-%m-%d %H:%M:%S %Z')"

for variant in $VARIANTS; do
  case "$variant" in
    value_conditioned)
      value_input_mode="residual"
      train_loss_scope="target_only"
      loss_mode="residual_only"
      ;;
    value_conditioned_hybrid)
      value_input_mode="residual"
      train_loss_scope="target_only"
      loss_mode="hybrid"
      ;;
    *)
      echo "[Instacart LR sensitivity b1024][ERROR] Unknown variant: $variant" >&2
      exit 2
      ;;
  esac

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

    run_dir="$BASE_ROOT/${variant}_lr_${lr_label}"
    log_path="$BASE_ROOT/logs/${variant}_lr_${lr_label}.log"
    extra_flags=()
    if [[ "$FORCE_RERUN" == "1" || "$FORCE_RERUN" == "true" ]]; then
      extra_flags+=(--force-rerun)
    fi
    if [[ "$STOP_ON_ERROR" == "1" || "$STOP_ON_ERROR" == "true" ]]; then
      extra_flags+=(--stop-on-error)
    fi

    echo
    echo "[Instacart LR sensitivity b1024] variant=$variant lr=$lr run_dir=$run_dir"
    echo "[Instacart LR sensitivity b1024] log_path=$log_path"

    "$PYTHON_BIN" "$PROJECT_ROOT/simple_lab_test/search/tpp_experiment.py" long-epoch \
      --base-dir "$run_dir" \
      --datasets "$DATASET" \
      --models titantpp \
      --titan-candidates "$TITAN_CANDIDATES" \
      --epochs "$EPOCHS" \
      --seeds "$SEEDS" \
      --lr "$lr" \
      --batch-size "$BATCH_SIZE" \
      --max-seq-len "$MAX_SEQ_LEN" \
      --eval-selections best_val_nll,best_score,final \
      --split-mode fixed \
      --value-head-activation identity \
      --value-input-mode "$value_input_mode" \
      --train-loss-scope "$train_loss_scope" \
      --loss-mode "$loss_mode" \
      --device "$DEVICE" \
      "${extra_flags[@]}" \
      2>&1 | tee -a "$log_path"
  done
done

echo
echo "[Instacart LR sensitivity b1024] complete_time=$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "[Instacart LR sensitivity b1024] artifacts=$BASE_ROOT"
