#!/usr/bin/env bash
# =============================================================================
# LCVN Full Data Pipeline
#
# Input layout (under LAN_ROOT/data/):
#   data/lcvn/                     - flat directory of trajectory folders
#                                    each folder: 0.jpg, 1.jpg, ..., traj_data.pkl
#                                    traj_data.pkl keys: position, yaw, delta, instruction
#
# Output layout:
#   lcvn-wm/data/lcvn/
#     metadata/{training,validation}.pt
#     latents/{training,validation}/  *.pt
#     {training,validation}/{cached_feats.pkl, ep_start_end_ids.npy, auto_lang_ann.npy}
#
# Steps:
#   1. Build metadata.pt from raw trajectory folders
#   2. Encode frames → latents using SD VAE (stabilityai/sd-vae-ft-ema)
#   3. Build initial LCVN cache (so WM can train)
#   4. Train custom VAE on this initial dataset
#   5. Re-encode frames with trained VAE → new latents
#   6. Rebuild LCVN cache using the new latents
# =============================================================================

set -euo pipefail

# --- Paths ---
WM_DIR="$(cd "$(dirname "$0")" && pwd)"
LAN_ROOT="$(cd "$WM_DIR/.." && pwd)"
AC_DIR="${LAN_ROOT}/lcvn-ac"

RAW_DATA="${LAN_ROOT}/data/lcvn"
DATA_DIR="${WM_DIR}/data/lcvn"
OUTPUTS_DIR="${LAN_ROOT}/outputs"

TRAIN_RATIO="${TRAIN_RATIO:-0.9}"
VAL_RATIO="${VAL_RATIO:-0.1}"
RESOLUTION="${RESOLUTION:-256}"

# --- Which steps to run (set to 0 to skip) ---
RUN_STEP1="${RUN_STEP1:-1}"   # build metadata
RUN_STEP2="${RUN_STEP2:-1}"   # encode with SD VAE
RUN_STEP3="${RUN_STEP3:-1}"   # build initial cache
RUN_STEP4="${RUN_STEP4:-1}"   # train VAE
RUN_STEP5="${RUN_STEP5:-1}"   # re-encode with trained VAE
RUN_STEP6="${RUN_STEP6:-1}"   # rebuild cache

log() { echo -e "\n\033[1;34m[$(date '+%H:%M:%S')] $*\033[0m"; }
die() { echo -e "\033[1;31m[ERROR] $*\033[0m" >&2; exit 1; }

# --- Sanity checks ---
[[ -d "$RAW_DATA" ]] || die "Raw data not found: $RAW_DATA"

mkdir -p "$DATA_DIR/metadata" "$DATA_DIR/latents" "$DATA_DIR/training" "$DATA_DIR/validation" "$OUTPUTS_DIR"

cd "$WM_DIR"

# =============================================================================
# Step 1: Build metadata.pt from raw trajectory folders
# =============================================================================
if [[ "$RUN_STEP1" == "1" ]]; then
    log "Step 1: Building metadata from raw trajectory folders..."
    python convert_VAE.py \
        --build_metadata_from_recon \
        --recon_root "$RAW_DATA" \
        --metadata_dir "$DATA_DIR/metadata" \
        --train_ratio "$TRAIN_RATIO" \
        --val_ratio "$VAL_RATIO"
    log "Step 1a: Adding pose conditions from traj_data.pkl..."
    python convert_VAE.py \
        --add_conditions_from_traj \
        --metadata_dir "$DATA_DIR/metadata" \
        --splits "training,validation"
    log "Step 1 done."
fi

# =============================================================================
# Step 2: Encode frames → latents using SD VAE (stabilityai/sd-vae-ft-ema)
# =============================================================================
if [[ "$RUN_STEP2" == "1" ]]; then
    log "Step 2: Encoding with SD VAE..."
    for split in training validation; do
        log "  Encoding split: $split"
        python convert_VAE.py \
            --vae_ckpt "stabilityai/sd-vae-ft-ema" \
            --social_mode \
            --from_metadata \
            --split "$split" \
            --metadata_dir "$DATA_DIR/metadata" \
            --social_save_dir "$DATA_DIR" \
            --resolution "$RESOLUTION" \
            --fp16 \
            --skip_existing
    done
    log "Step 2 done."
fi

# =============================================================================
# Step 3: Build initial LCVN cache
# =============================================================================
if [[ "$RUN_STEP3" == "1" ]]; then
    log "Step 3: Building initial LCVN cache..."
    cd "$AC_DIR"
    for split in training validation; do
        log "  Caching split: $split"
        python dataset/build_cache.py \
            --dataset_root "$DATA_DIR" \
            --split "$split" \
            --output_root "$DATA_DIR"
    done
    cd "$WM_DIR"
    log "Step 3 done. Cache at $DATA_DIR/{training,validation}/"
fi

# =============================================================================
# Step 4: Train custom VAE
# =============================================================================
if [[ "$RUN_STEP4" == "1" ]]; then
    log "Step 4: Training custom VAE..."
    cd "$WM_DIR"
    WANDB_MODE="${WANDB_MODE:-offline}" python -m main \
        +name=ImageVAE_Social \
        dataset=lcvn \
        algorithm=image_vae \
        experiment=video_latent_learning \
        "wandb.entity=${WANDB_ENTITY:-lcvn}"
    log "Step 4 done. VAE checkpoint: $OUTPUTS_DIR/vae.ckpt"
fi

# =============================================================================
# Step 5: Re-encode with trained VAE
# =============================================================================
if [[ "$RUN_STEP5" == "1" ]]; then
    VAE_CKPT="${OUTPUTS_DIR}/vae.ckpt"
    [[ -f "$VAE_CKPT" ]] || die "Trained VAE checkpoint not found: $VAE_CKPT (run Step 4 first)"
    log "Step 5: Re-encoding with trained VAE: $VAE_CKPT"
    cd "$WM_DIR"
    for split in training validation; do
        log "  Re-encoding split: $split"
        python convert_VAE.py \
            --vae_ckpt "$VAE_CKPT" \
            --social_mode \
            --from_metadata \
            --split "$split" \
            --metadata_dir "$DATA_DIR/metadata" \
            --social_save_dir "$DATA_DIR" \
            --resolution "$RESOLUTION" \
            --fp16
    done
    log "Step 5 done."
fi

# =============================================================================
# Step 6: Rebuild LCVN cache with trained VAE latents
# =============================================================================
if [[ "$RUN_STEP6" == "1" ]]; then
    log "Step 6: Rebuilding LCVN cache with trained VAE latents..."
    cd "$AC_DIR"
    for split in training validation; do
        log "  Caching split: $split"
        python dataset/build_cache.py \
            --dataset_root "$DATA_DIR" \
            --split "$split" \
            --output_root "$DATA_DIR"
    done
    log "Step 6 done."
fi

log "Pipeline complete!"
echo ""
echo "Training data:    $DATA_DIR/training/"
echo "Validation data:  $DATA_DIR/validation/"
echo ""
echo "Next steps:"
echo "  WM training:   cd $WM_DIR && python -m main algorithm=ldit_video_social ..."
echo "  AC training:   cd $AC_DIR && ./train_ac.sh"
