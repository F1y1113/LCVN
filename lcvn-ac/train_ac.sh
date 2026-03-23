#!/usr/bin/env bash
  set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

  NPROC=4 bash scripts/run_train_dfot.sh \
    "behavior_model.lcvn_worldmodel.dfot_checkpoint_path=../outputs/social_dit_xl.ckpt" \
    "behavior_model.lcvn_worldmodel.dfot_vae_checkpoint_path=../outputs/vae.ckpt" \
    "$@"
