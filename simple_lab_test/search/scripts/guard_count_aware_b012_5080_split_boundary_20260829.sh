#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/leekwanhyeong/workspace/paper_research}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/search_artifacts/count_aware_b012_seed42_screening_e300_20260828_recovery1}"
SOURCE_REVISION="${SOURCE_REVISION:-08e59880cd61cbd27cec40aa04636452b87bebfc}"
RECOVERY_REVISION="${RECOVERY_REVISION:-75b185f3d5fe9fd0f146b367519e2c031b50cf38}"
PARENT_PID="${PARENT_PID:-}"
POLL_SECONDS="${POLL_SECONDS:-15}"
MARKER="${OUTPUT_ROOT}/split_boundary_guard.json"

write_marker() {
  local status="$1"
  local message="$2"
  local child_pid="${3:-}"
  local temporary="${MARKER}.tmp"
  mkdir -p "${OUTPUT_ROOT}"
  printf '{\n' > "${temporary}"
  printf '  "status": "%s",\n' "${status}" >> "${temporary}"
  printf '  "message": "%s",\n' "${message}" >> "${temporary}"
  printf '  "training_source_revision": "%s",\n' "${SOURCE_REVISION}" >> "${temporary}"
  printf '  "recovery_orchestration_revision": "%s",\n' "${RECOVERY_REVISION}" >> "${temporary}"
  printf '  "launcher_parent_pid": %s,\n' "${PARENT_PID}" >> "${temporary}"
  if [[ -n "${child_pid}" ]]; then
    printf '  "intermittent_b2_pid": %s,\n' "${child_pid}" >> "${temporary}"
  else
    printf '  "intermittent_b2_pid": null,\n' >> "${temporary}"
  fi
  printf '  "updated_at_utc": "%s"\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${temporary}"
  printf '}\n' >> "${temporary}"
  mv "${temporary}" "${MARKER}"
}

if [[ -z "${PARENT_PID}" ]]; then
  mapfile -t candidates < <(
    pgrep -f \
      "bash ${PROJECT_ROOT}/simple_lab_test/search/scripts/run_count_aware_b012_seed42_screening_recovery1_20260828.sh" \
      || true
  )
  [[ "${#candidates[@]}" == "1" ]]
  PARENT_PID="${candidates[0]}"
fi

[[ "${PARENT_PID}" =~ ^[0-9]+$ ]]
launcher_command="$(tr '\0' ' ' < "/proc/${PARENT_PID}/cmdline")"
[[ "${launcher_command}" == *"run_count_aware_b012_seed42_screening_recovery1_20260828.sh"* ]]
write_marker armed "Waiting for Intermittent B2 to start before pausing the launcher."

while kill -0 "${PARENT_PID}" 2>/dev/null; do
  while read -r child_pid; do
    [[ -n "${child_pid}" ]] || continue
    child_command="$(tr '\0' ' ' < "/proc/${child_pid}/cmdline" 2>/dev/null || true)"
    if [[ \
      "${child_command}" == *"--dataset-contract intermittent_frozen_5000"* \
      && "${child_command}" == *"--backbones titantpp_tpp_gated_memory"* \
    ]]; then
      kill -STOP "${PARENT_PID}"
      launcher_state="$(ps -o stat= -p "${PARENT_PID}" | tr -d ' ')"
      [[ "${launcher_state}" == *T* ]]
      write_marker \
        boundary_held \
        "Launcher paused while Intermittent B2 continues; Taxi and RAF are reserved for 5090." \
        "${child_pid}"
      exit 0
    fi
  done < <(pgrep -P "${PARENT_PID}" || true)
  sleep "${POLL_SECONDS}"
done

write_marker failed "Recovery launcher exited before the Intermittent B2 boundary was held."
exit 1
