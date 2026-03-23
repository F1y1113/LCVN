#!/usr/bin/env bash
  set -euo pipefail

  NPROC=${NPROC:-4}
  MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
  MASTER_PORT=${MASTER_PORT:-23456}

  if [ "${NPROC}" -eq 1 ]; then
    export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
    python3 scripts/train_dfot.py "$@"
  else
    python3 -m torch.distributed.run \
      --nproc_per_node="${NPROC}" \
      --master_addr="${MASTER_ADDR}" \
      --master_port="${MASTER_PORT}" \
      scripts/train_dfot.py "$@"
  fi