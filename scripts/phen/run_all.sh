#!/usr/bin/env bash
set -euo pipefail
directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for model in phen rmtpp thp nhp sahp s2p2_matched_head attnhp_matched_head; do
  bash "$directory/run_$model.sh" "$@"
done
