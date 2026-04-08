#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-root@connect.cqa1.seetacloud.com}"
PORT="${PORT:-43537}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/autodl-tmp/Time-Series-Library}"
LOCAL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

ssh -p "${PORT}" "${HOST}" "mkdir -p '${REMOTE_ROOT}'"

rsync -av --progress \
  -e "ssh -p ${PORT}" \
  "${LOCAL_ROOT}/data_provider" \
  "${LOCAL_ROOT}/layers" \
  "${LOCAL_ROOT}/models" \
  "${LOCAL_ROOT}/scripts/slopemine_v2" \
  "${LOCAL_ROOT}/dataset/slopemine_v2" \
  "${LOCAL_ROOT}/outputs/slopemine_v2/experiment_plan_v1" \
  "${LOCAL_ROOT}/outputs/slopemine_v2/ms_timefilter_v1" \
  "${HOST}:${REMOTE_ROOT}/"

echo "Synced to ${HOST}:${REMOTE_ROOT}"
