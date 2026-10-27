#!/usr/bin/env bash
set -euo pipefail

model="${1:?model is required}"
shift
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "${1:-}" == "--help" || "$#" -eq 0 ]]; then
  printf 'Usage: SEEDS="42 52 62" bash scripts/phen/run_%s.sh DATASET [options]\n' "$model"
  printf 'Options: --data PATH --device cpu|cuda:0 --dry-run --mode preflight\n'
  printf 'Evaluation: --mode evaluate --checkpoint PATH [--split test --allow-held-out]\n'
  exit 0
fi
dataset="$1"
shift
config="$root/phen/configs/$dataset.json"
if [[ ! -f "$config" ]]; then
  printf 'Unknown dataset configuration: %s\n' "$dataset" >&2
  exit 2
fi
python="${PYTHON_BIN:-python}"
if [[ "${SEEDS+x}" == x ]]; then
  seed_list="$SEEDS"
else
  seed_list="$("$python" -c 'import json, sys; print(" ".join(str(s) for s in json.load(open(sys.argv[1]))["seeds"]))' "$config")"
fi
read -r -a seeds <<< "$seed_list"
if [[ "${#seeds[@]}" -eq 0 ]]; then
  printf 'SEEDS must not be empty\n' >&2
  exit 2
fi
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
cd "$root"
for seed in "${seeds[@]}"; do
  PYTHONHASHSEED="$seed" "$python" -m phen.run \
    --config "$config" --model "$model" --seed "$seed" "$@"
done
