#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
PYTHON_BIN="${PYTHON_BIN:?PYTHON_BIN is required}"
ORCHESTRATION_REVISION="${ORCHESTRATION_REVISION:?ORCHESTRATION_REVISION is required}"
ACTIVE_CHILD_PID="${ACTIVE_CHILD_PID:?ACTIVE_CHILD_PID is required}"
HELD_PARENT_PID="${HELD_PARENT_PID:?HELD_PARENT_PID is required}"
POLL_SECONDS="${POLL_SECONDS:-60}"

VALIDATOR="${PROJECT_ROOT}/paper/scripts/validate_count_aware_titantpp_mac_three_seed_validation.py"
CONTRACT="${PROJECT_ROOT}/paper/contracts/count_aware_titantpp_mac_three_seed_validation_v1.json"
SHARD_RUNNER="${PROJECT_ROOT}/simple_lab_test/search/scripts/run_count_aware_titantpp_mac_validation_shard_20260830.sh"
RAF_RUN_ROOT="${OUTPUT_ROOT}/shards/raf_spare_parts/seed_52"

[[ "${ACTIVE_CHILD_PID}" =~ ^[0-9]+$ ]]
[[ "${HELD_PARENT_PID}" =~ ^[0-9]+$ ]]
[[ "${POLL_SECONDS}" =~ ^[0-9]+$ ]]
(( POLL_SECONDS > 0 ))
[[ "${ORCHESTRATION_REVISION}" =~ ^[0-9a-f]{40}$ ]]
[[ -x "${PYTHON_BIN}" ]]
[[ -f "${VALIDATOR}" ]]
[[ -f "${CONTRACT}" ]]
[[ -f "${SHARD_RUNNER}" ]]

parent_state="$(ps -o stat= -p "${HELD_PARENT_PID}" 2>/dev/null | tr -d ' ' || true)"
if [[ "${parent_state}" != T* ]]; then
  printf 'Expected held parent %s, observed state=%s\n' \
    "${HELD_PARENT_PID}" "${parent_state:-missing}" >&2
  exit 1
fi

printf 'Waiting for RAF seed 52 child pid=%s under held parent pid=%s.\n' \
  "${ACTIVE_CHILD_PID}" "${HELD_PARENT_PID}"
while [[ -r "/proc/${ACTIVE_CHILD_PID}/stat" ]]; do
  child_state="$(ps -o stat= -p "${ACTIVE_CHILD_PID}" 2>/dev/null | tr -d ' ' || true)"
  if [[ -z "${child_state}" || "${child_state}" == Z* ]]; then
    break
  fi
  if ! child_command="$(tr '\0' ' ' < "/proc/${ACTIVE_CHILD_PID}/cmdline" 2>/dev/null)"; then
    break
  fi
  if [[ "${child_command}" != *"raf_spare_parts"* ]] \
    || [[ "${child_command}" != *"--seeds 52"* ]] \
    || [[ "${child_command}" != *"titantpp_titans_mac"* ]]; then
    printf 'Active child identity changed; refusing automatic handoff.\n' >&2
    exit 1
  fi
  sleep "${POLL_SECONDS}"
done

"${PYTHON_BIN}" "${VALIDATOR}" validate-run \
  --run-root "${RAF_RUN_ROOT}" \
  --dataset raf_spare_parts \
  --seed 52 \
  --contract "${CONTRACT}"

printf 'RAF seed 52 validated; starting the 5090 split shard.\n'
export PROJECT_ROOT OUTPUT_ROOT PYTHON_BIN ORCHESTRATION_REVISION
export SHARD_ID=seed52_5090
exec bash "${SHARD_RUNNER}"
