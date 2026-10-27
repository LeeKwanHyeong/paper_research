#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
for file in scripts/phen/*.sh; do
  bash -n "$file"
done
"${PYTHON_BIN:-python}" -m pytest -p no:cacheprovider phen/tests -q "$@"
