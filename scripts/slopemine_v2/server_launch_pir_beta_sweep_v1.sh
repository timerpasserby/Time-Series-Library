#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
PATIENCE="${PATIENCE:-2}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
mkdir -p "${MPLCONFIGDIR}" logs

echo "Using python: ${PYTHON_BIN}"
echo "Repo root: ${REPO_ROOT}"
echo "MAX_EPOCHS=${MAX_EPOCHS} PATIENCE=${PATIENCE}"

"${PYTHON_BIN}" scripts/slopemine_v2/run_pir_beta_sweep_v1.py \
  --config scripts/slopemine_v2/config_v2.toml \
  --max-epochs "${MAX_EPOCHS}" \
  --patience "${PATIENCE}"
