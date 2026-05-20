#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

TRAIN_SPLIT_NAME="${LCVN_TRAIN_SPLIT_NAME:-training}"
VAL_SPLIT_NAME="${LCVN_VAL_SPLIT_NAME:-validation_seen}"
NPROC="${NPROC:-4}"
TRAINER_DEVICES="${TRAINER_DEVICES:-$NPROC}"

EXTRA_ARGS=(
  "datamodule.train_split_name=${TRAIN_SPLIT_NAME}"
  "datamodule.val_split_name=${VAL_SPLIT_NAME}"
  "behavior_model.lcvn_worldmodel.dfot_checkpoint_path=../outputs/social_dit_xl.ckpt"
  "behavior_model.lcvn_worldmodel.dfot_vae_checkpoint_path=../outputs/vae.ckpt"
  "trainer.devices=${TRAINER_DEVICES}"
)


NPROC="$NPROC" bash scripts/run_train_dfot.sh \
  "${EXTRA_ARGS[@]}" \
  "$@"
