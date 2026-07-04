#!/usr/bin/env bash
set -euo pipefail

# Run the professor-feedback LR sensitivity screening on the two non-Instacart
# datasets.  This mirrors the Instacart LR experiment, but keeps each dataset,
# LR, and value-objective variant in its own artifact directory so later result
# analysis can be done without manually separating mixed runs.
#
# Expected environment:
#   conda activate ai_env
#   bash simple_lab_test/search/scripts/run_inter_yellow_lr_sensitivity_e50.sh
#
# Common overrides:
#   PROJECT_ROOT=~/workspace/paper_research
#   BASE_ROOT=~/workspace/paper_research/search_artifacts/inter_yellow_lr_sensitivity_e50
#   DATASETS="intermittent yellow_trip_hourly"
#   LRS="1e-3 5e-3 1e-2"
#   TITAN_CANDIDATES="small_lmm,mid_lmm"
#   EPOCHS=50
#   BATCH_SIZE=512
#   FORCE_RERUN=0   # reuse existing completed runs when possible

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/workspace/paper_research}"
BASE_ROOT="${BASE_ROOT:-$PROJECT_ROOT/search_artifacts/inter_yellow_lr_sensitivity_e50}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Keep datasets as a shell list because each dataset gets its own output tree.
DATASETS="${DATASETS:-intermittent yellow_trip_hourly}"
LRS="${LRS:-1e-3 5e-3 1e-2}"

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-512}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-64}"
SEEDS="${SEEDS:-42}"
DEVICE="${DEVICE:-cuda}"
TITAN_CANDIDATES="${TITAN_CANDIDATES:-small_lmm,mid_lmm}"
FORCE_RERUN="${FORCE_RERUN:-1}"
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"

# Current ablation variants:
# - value_conditioned: past scale_residual is used as causal input, residual loss only
# - value_conditioned_hybrid: same causal input, residual + direct quantity loss
VARIANTS="${VARIANTS:-value_conditioned value_conditioned_hybrid}"

mkdir -p "$BASE_ROOT/logs"

echo "[LR sensitivity] project_root=$PROJECT_ROOT"
echo "[LR sensitivity] base_root=$BASE_ROOT"
echo "[LR sensitivity] datasets=$DATASETS"
echo "[LR sensitivity] lrs=$LRS"
echo "[LR sensitivity] variants=$VARIANTS"
echo "[LR sensitivity] titan_candidates=$TITAN_CANDIDATES"
echo "[LR sensitivity] force_rerun=$FORCE_RERUN stop_on_error=$STOP_ON_ERROR"
echo "[LR sensitivity] start_time=$(date '+%Y-%m-%d %H:%M:%S %Z')"

for dataset in $DATASETS; do
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
      baseline)
        value_input_mode="none"
        train_loss_scope="all"
        loss_mode="residual_only"
        ;;
      target_only)
        value_input_mode="none"
        train_loss_scope="target_only"
        loss_mode="residual_only"
        ;;
      *)
        echo "[LR sensitivity][ERROR] Unknown variant: $variant" >&2
        exit 2
        ;;
    esac

    for lr in $LRS; do
      # Use filesystem-safe LR labels that match previous Instacart artifacts.
      lr_label="${lr//./p}"
      lr_label="${lr_label//-m/em}"
      lr_label="${lr_label//e/em}"
      case "$lr" in
        1e-3) lr_label="1em03" ;;
        5e-3) lr_label="5em03" ;;
        1e-2) lr_label="1em02" ;;
      esac

      run_dir="$BASE_ROOT/${dataset}_${variant}_lr_${lr_label}"
      log_path="$BASE_ROOT/logs/${dataset}_${variant}_lr_${lr_label}.log"
      extra_flags=()
      if [[ "$FORCE_RERUN" == "1" || "$FORCE_RERUN" == "true" ]]; then
        extra_flags+=(--force-rerun)
      fi
      if [[ "$STOP_ON_ERROR" == "1" || "$STOP_ON_ERROR" == "true" ]]; then
        extra_flags+=(--stop-on-error)
      fi

      echo
      echo "[LR sensitivity] dataset=$dataset variant=$variant lr=$lr run_dir=$run_dir"
      echo "[LR sensitivity] log_path=$log_path"

      "$PYTHON_BIN" "$PROJECT_ROOT/simple_lab_test/search/tpp_experiment.py" long-epoch \
        --base-dir "$run_dir" \
        --datasets "$dataset" \
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
done

echo
echo "[LR sensitivity] complete_time=$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "[LR sensitivity] artifacts=$BASE_ROOT"
