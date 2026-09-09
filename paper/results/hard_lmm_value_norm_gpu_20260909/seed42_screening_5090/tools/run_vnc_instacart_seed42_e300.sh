#!/usr/bin/env bash
set -euo pipefail
SRC=/home/leekwanhyeong/workspace/paper_research_vnc_956603f_5090
ART=/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/hard_lmm_value_norm_956603f_5090_20260909
OUT="$ART/jobs/seed42_screening_insta_market_basket"
LOG="$ART/logs/seed42_screening_insta_market_basket.log"
STATUS="$ART/logs/seed42_screening_insta_market_basket.status"
CONTRACT="$ART/contracts/hard_lmm_value_norm_screening_5090_v1.json"
test ! -e "$OUT"
test ! -e "$LOG"
test ! -e "$STATUS"
test "$(sha256sum "$CONTRACT" | cut -d ' ' -f 1)" = 515412b7750f8e864872ea37c7a50d4af57a78407a8794604b1e12301b26bc9c
test "$(sha256sum /home/leekwanhyeong/workspace/tmp/paper_research_vnc_956603f.tar.gz | cut -d ' ' -f 1)" = effaa12f18b7e6ef126218c8b2e6cbeee91a8a1c2159df677955ccc8e6ee404d
test "$(sha256sum "$SRC/sample_data/insta_market_basket/instacart_marked_target_with_split.parquet" | cut -d ' ' -f 1)" = 06296e48f5ca6c7e0c849f4b4a3c6d54a968ef892754f59369caf1d378424ef2
test "$(sha256sum "$SRC/sample_data/insta_market_basket/instacart_marked_target_split_manifest.json" | cut -d ' ' -f 1)" = 6c6cdd41f847878fbb405b73dfa038fbb7a88ad53df6843b0cc9e64531a8b71d
mkdir -p "$ART/logs" "$ART/matplotlib" "$ART/torch_kernel_cache"
nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits > "$ART/logs/seed42_screening_insta_market_basket_preflight_gpu.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits > "$ART/logs/seed42_screening_insta_market_basket_preflight_processes.txt"
test ! -s "$ART/logs/seed42_screening_insta_market_basket_preflight_processes.txt"
export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export LD_LIBRARY_PATH=/opt/miniconda3/envs/ai_env/lib/python3.12/site-packages/nvidia/cu13/lib
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export POLARS_MAX_THREADS=4
export MPLCONFIGDIR="$ART/matplotlib"
export TORCH_KERNEL_CACHE_PATH="$ART/torch_kernel_cache"
export PYTORCH_KERNEL_CACHE_PATH="$ART/torch_kernel_cache"
printf 'running %s\n' "$(date -Is)" > "$STATUS"
trap 'code=$?; printf "failed exit=%s %s\n" "$code" "$(date -Is)" > "$STATUS"; exit "$code"' ERR
cd "$SRC"
date -Is | tee "$LOG"
set -o pipefail
/opt/miniconda3/envs/ai_env/bin/python -s -u "$SRC/paper/scripts/run_count_aware_tpp_backbone_control.py" \
  --data "$SRC/sample_data/insta_market_basket/instacart_marked_target_with_split.parquet" \
  --split-manifest "$SRC/sample_data/insta_market_basket/instacart_marked_target_split_manifest.json" \
  --output-dir "$OUT" \
  --source-revision 956603f16ca3540e2012a461a494ea1ec905102d \
  --execution-role hard_lmm_backbone_5090_titantpp_hard_memory_value_norm_insta_market_basket_seed42_e300 \
  --dataset-contract insta_market_basket \
  --model-role hard_lmm_value_norm_candidate \
  --device cuda \
  --epochs 300 \
  --min-epochs 40 \
  --early-stopping-patience 40 \
  --batch-size 128 \
  --lr 0.001 \
  --lookback-weeks 52 \
  --max-seq-len 64 \
  --hidden-dim 64 \
  --lambda-log-qty 1 \
  --lambda-tail 0 \
  --grad-clip 1 \
  --backbones titantpp_hard_memory_value_norm \
  --seeds 42 \
  --quantity-variants log_mse \
  --checkpoint-monitor validation_raw_quantity_rmse \
  --quantile-adaptive-strength 0 \
  --time-head-mode legacy_clamped_rmtpp \
  --time-scale 3 \
  --time-w-max 3.3333333333333335 \
  --time-intercept-limit 300 \
  --allow-partial-contract 2>&1 | tee -a "$LOG"
date -Is | tee -a "$LOG"
printf 'complete %s\n' "$(date -Is)" > "$STATUS"
