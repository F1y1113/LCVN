#!/usr/bin/env python3
import flash_attn.flash_attn_interface as _fa
for _old, _new in [("_wrapped_flash_attn_forward", "_flash_attn_forward"),
                   ("_wrapped_flash_attn_backward", "_flash_attn_backward")]:
    if not hasattr(_fa, _old) and hasattr(_fa, _new):
        setattr(_fa, _old, getattr(_fa, _new))
"""
Using real data, strictly follow the Lcvn-AC + DFoT pipeline to run a complete trajectory forward pass and record the full data flow.

- Depend on Hydra configuration to load the real data module (default social.yaml, override via CLI)
- Enable DFoT pure inference: LUMOS uses DFoT to predict the next state, NWM disabled
- Use TrajectoryLogger: record data loading, model forward stages, action and latent statistics

Run example:
  python lcvn_ac/scripts/test_single_trajectory_real_dfot.py \
    datamodule=social \
    behavior_model=lumos-ac \
    behavior_model/lcvn_worldmodel.enabled=true \
    behavior_model/plan_recognition=transformers \
    +behavior_model.dfot_checkpoint_path=/your/dfot/weights.pt \
    +behavior_model.dfot_vae_checkpoint_path=/your/vae/ckpt.ckpt

Note: ensure `root_data_dir` points to the cached social dataset directory and `cached_feats.pkl` exists.
"""

import os
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
from typing import Dict, Any, Optional
from PIL import Image

# Add project root to path for direct script execution
import sys
sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())

from lcvn_ac.utils.trajectory_logger import TrajectoryLogger


@hydra.main(version_base="1.3", config_path="../../config", config_name="train_dfot")
def main(cfg: DictConfig) -> None:
    # Briefly print config
    print("==== Using configuration ====")
    print(OmegaConf.to_yaml(cfg))

    # Initialize data module (default social.yaml, load_feats=true)
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    # Take one batch (language modality)
    loaders = datamodule.train_dataloader()
    assert isinstance(loaders, dict), "Expected dataloaders to be a dict"
    # Select a key that contains language (social config uses 'lang'/'vis')
    key = None
    for k in loaders.keys():
        if "lang" in k:
            key = k
            break
    if key is None:
        # Fallback: choose any key
        key = list(loaders.keys())[0]
    loader = loaders[key]
    batch = next(iter(loader))

    # Use TrajectoryLogger to record data loading (invoke proper start method)
    traj_logger = TrajectoryLogger()
    traj_logger.start_new_trajectory({
        "experiment": "lcvn-ac_dfot_single_trajectory",
        "description": "Real data, single forward pass via Lcvn-AC + DFoT"
    })

    # Choose batch index for detailed printing/saving (default 0)
    batch_idx = int(os.environ.get("LCVN_AC_TEST_BATCH_INDEX", "0"))
    # Output image directory
    log_dir = Path(os.environ.get("LCVN_AC_TRAJECTORY_LOG_DIR", "trajectory_logs"))
    img_out_dir = log_dir / "latent_images"
    img_out_dir.mkdir(parents=True, exist_ok=True)

    # Utility: save 4-channel latent as 4 grayscale images
    def save_latent_channels(latent_4chw: torch.Tensor, out_dir: Path, prefix: str):
        try:
            latent_np = latent_4chw.detach().cpu().numpy()  # [4, H, W]
            for c in range(latent_np.shape[0]):
                arr = latent_np[c]
                # Normalize to [0,255]
                arr_min, arr_max = float(arr.min()), float(arr.max())
                if arr_max > arr_min:
                    norm = (arr - arr_min) / (arr_max - arr_min)
                else:
                    norm = np.zeros_like(arr)
                img = (norm * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(img).save((out_dir / f"{prefix}_ch{c}.png").as_posix())
        except Exception as e:
            print(f"Failed to save latent images: {e}")

    # Utility: safely get source_meta and print trajectory provenance
    def extract_source_meta(batch_dict: Dict[str, Any], bidx: int) -> Optional[Dict[str, Any]]:
        sm = batch_dict.get("source_meta")
        if not isinstance(sm, dict):
            return None
        # default_collate batches each field; take single sample by bidx here
        def take_idx(v):
            try:
                if isinstance(v, torch.Tensor):
                    return v[bidx].item() if v.dim() == 1 else v[bidx]
                elif isinstance(v, (list, tuple)):
                    return v[bidx]
                else:
                    return v
            except Exception:
                return v
        out = {}
        for k, v in sm.items():
            out[k] = take_idx(v)
        return out

    # Organize key fields by dataset structure (social uses cached_feats)
    # Expected fields: vae_latent, rel_actions/discrete_action, robot_obs, lang or lang_raw
    vae_latent = batch.get("vae_latent")
    robot_obs = batch.get("state_info", {}).get("robot_obs") if isinstance(batch.get("state_info"), dict) else batch.get("robot_obs")
    rel_actions = batch.get("rel_actions")
    discrete_action = batch.get("discrete_action")
    lang = batch.get("lang")
    lang_raw = batch.get("lang_raw")

    traj_logger.log_data_loading("BATCH_PREPARED", {
        "vae_latent": vae_latent,
        "robot_obs": robot_obs,
        "rel_actions": rel_actions,
        "discrete_action": discrete_action,
        "lang": lang,
        "lang_raw": lang_raw,
    })

    # Print and record trajectory provenance (which specific part it belongs to)
    source_meta_single = extract_source_meta(batch, batch_idx)
    if source_meta_single is not None:
        # Human-friendly printing
        print("==== Trajectory provenance (single sample) ====")
        print(f"batch_idx: {batch_idx}")
        for key in [
            "dataset_dir", "dataset_class", "idx", "window_size",
            "raw_traj_index", "start_idx_zero_based", "end_idx_zero_based",
            "raw_traj_start_end_0b", "frames", "naming_pattern", "n_digits"
        ]:
            if key in source_meta_single:
                print(f"{key}: {source_meta_single[key]}")
        traj_logger.log_event("TRAJECTORY_PART", "Print single-sample source_meta", source_meta_single)
    else:
        print("source_meta not found (dataset may not attach or collate changed)")

    # Explain windowing and padding: T is final padded length; original length is source_meta.window_size
    try:
        T_final = int(vae_latent.shape[0])
        win_orig = int(source_meta_single.get("window_size", T_final)) if source_meta_single else T_final
        pad_size = max(0, T_final - win_orig)
        traj_logger.log_window_processing(
            original_data={"window_size": win_orig},
            windowed_data={"vae_latent_shape": list(vae_latent.shape), "pad_mode": "repeat_last_frame"},
            window_size=win_orig,
            pad_size=pad_size,
        )
        print("==== Windowing and padding explanation ====")
        print(f"Original window length: {win_orig}, final sequence length: {T_final}, padding frames: {pad_size} (strategy: repeat last frame)")
    except Exception as e:
        print(f"Failed to log window and padding info: {e}")

    # Instantiate Lcvn-AC (DFoT enabled, internal world_model=None)
    print("==== Initialize Lcvn-AC (Lcvn WorldModel) ====")
    bm_cfg = OmegaConf.merge(cfg.behavior_model, {})
    # Enable Lcvn WorldModel
    if not bm_cfg.get("lcvn_worldmodel"):
        bm_cfg.lcvn_worldmodel = OmegaConf.create({"enabled": True})

    # Load trained AC checkpoint if provided, otherwise instantiate fresh
    ckpt_path = getattr(cfg, "checkpoint", None) or getattr(cfg, "ckpt_path", None)
    if ckpt_path and Path(ckpt_path).exists():
        print(f"==== Loading AC checkpoint: {ckpt_path} ====")
        from lcvn_ac.behavior_models.lumos import LUMOS
        try:
            model = LUMOS.load_from_checkpoint(str(ckpt_path))
            print("==== AC checkpoint loaded successfully ====")
        except Exception as e:
            print(f"==== AC checkpoint load failed: {e}, falling back to fresh model ====")
            model = hydra.utils.instantiate(bm_cfg)
    else:
        if ckpt_path:
            print(f"WARNING: checkpoint path not found: {ckpt_path}, using fresh model")
        model = hydra.utils.instantiate(bm_cfg)

    # Device
    device = torch.device(cfg.device if hasattr(cfg, "device") else "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Move batch to device and ensure latent is 32x32 (DFoT expects)
    def to_device(x):
        return x.to(device) if isinstance(x, torch.Tensor) else x
    vae_latent = to_device(vae_latent)
    robot_obs = to_device(robot_obs)
    rel_actions = to_device(rel_actions)
    discrete_action = to_device(discrete_action)
    lang = to_device(lang)

    # Log per-step action conditions (for comparison with cached data)
    try:
        if isinstance(rel_actions, torch.Tensor):
            T, B = rel_actions.shape[0], rel_actions.shape[1]
            print("==== Action conditions (rel_actions & discrete_action) ====")
            for t in range(T):
                rel_t = rel_actions[t, batch_idx] if B > batch_idx else rel_actions[t, 0]
                disc_t = None
                if isinstance(discrete_action, torch.Tensor):
                    # Discrete actions may be [T, B] or [T, B, K] logits; print label or first 3 values
                    if discrete_action.dim() == 2:
                        disc_t = int(discrete_action[t, batch_idx].item()) if discrete_action.shape[1] > batch_idx else int(discrete_action[t, 0].item())
                    else:
                        vec = discrete_action[t, batch_idx] if discrete_action.shape[1] > batch_idx else discrete_action[t, 0]
                        disc_t = vec.detach().cpu().flatten()[:3].numpy().tolist()
                traj_logger.log_event(
                    "ACTION_CONDITION",
                    f"t={t}, batch={batch_idx}",
                    {"rel_actions": rel_t.detach().cpu(), "discrete_action": disc_t}
                )
                # Console-friendly printing
                try:
                    print(f"t={t}: rel_actions[{batch_idx}] = {rel_t.detach().cpu().numpy().tolist()} | discrete={disc_t}")
                except Exception:
                    pass
    except Exception as e:
        print(f"Failed to log action conditions: {e}")

    # Trigger DFoT inferencer initialization (JIT style)
    if getattr(model, "lwm_enabled", False):
        model._ensure_dfot_inferencer()

    # Verify DFoT history initialization: first 4-to-1 repeats the first frame
    try:
        if getattr(model, "lwm_enabled", False):
            # Take the first frame latent of current sample as current_vae_latent
            cur_latent = vae_latent[0, batch_idx]  # [4, 32, 32]
            # Expand 4D relative action to 6D and use zero action under pure inference
            # Note: continuous actions are zeros as requested
            act_6 = torch.zeros((6,), device=device, dtype=torch.float32)

            # Use batch size 1 for a single call
            pred1 = model._predict_next_state_with_dfot(cur_latent.unsqueeze(0), act_6.unsqueeze(0))
            traj_logger.log_event("DFOT_INIT_CHECK", "After first call, check if history repeats the first frame")
            # Read internal history and verify 4 frames are consistent
            hist_latents = getattr(model, "_dfot_history_latents", None)
            hist_actions = getattr(model, "_dfot_history_actions", None)
            if isinstance(hist_latents, torch.Tensor):
                # Shape [B=1, 4, 4, 32, 32]
                diffs = []
                for k in range(1, 4):
                    diffs.append(torch.max(torch.abs(hist_latents[0, k] - hist_latents[0, 0])).item())
                print("==== DFoT history initialization check ====")
                print(f"History latent shape: {list(hist_latents.shape)}, max diff vs first frame: {diffs}")
                if isinstance(hist_actions, torch.Tensor):
                    zeros_ok = torch.allclose(hist_actions[0, :, :3], torch.zeros_like(hist_actions[0, :, :3]))
                    dt_vals = hist_actions[0, :, 3].detach().cpu().numpy().tolist()
                    print(f"History actions first 3 dims all zero: {bool(zeros_ok)}, dt timeline: {dt_vals}")
                # Zero continuous action and dt timeline event (first call)
                input_is_zero = bool(torch.allclose(act_6[:3], torch.zeros_like(act_6[:3])))
                print(f"DFoT first call input continuous action all zero: {input_is_zero}, first 3 dims: {act_6[:3].detach().cpu().numpy().tolist()}")
                traj_logger.log_event(
                    "DFOT_ACTION_ZERO_CHECK",
                    "DFoT first-call input zero action and dt timeline check",
                    {
                        "input_action_first_call": act_6[:3].detach().cpu().numpy().tolist(),
                        "input_is_all_zero": input_is_zero,
                        "hist_actions_zero_check": bool(zeros_ok) if 'zeros_ok' in locals() else None,
                        "hist_dt_timeline": dt_vals if 'dt_vals' in locals() else None,
                    }
                )
                traj_logger.log_event("DFOT_HISTORY_INIT", "First-step history latents max diff vs first frame", {"diffs": diffs})
                # Save history 4-frame images
                for k in range(4):
                    save_latent_channels(hist_latents[0, k], img_out_dir, prefix=f"history_step0_frame{k}")
                # Save first prediction result
                save_latent_channels(pred1[0], img_out_dir, prefix="predicted_step0")

                # === Sliding window check: use previous predicted latent to infer again (action still zero) ===
                prev_hist = hist_latents.clone()
                act_6b = torch.zeros((6,), device=device, dtype=torch.float32)  # still zero action
                act_6b[:3] = 0.0
                _ = model._predict_next_state_with_dfot(pred1[0].unsqueeze(0), act_6b.unsqueeze(0))
                hist2 = getattr(model, "_dfot_history_latents", None)
                actions2 = getattr(model, "_dfot_history_actions", None)
                if isinstance(hist2, torch.Tensor):
                    # Verify slide: hist2[0,0]≈prev_hist[0,1], hist2[0,1]≈prev_hist[0,2], hist2[0,2]≈prev_hist[0,3], hist2[0,3]≈pred1[0]
                    shift_diffs = [
                        torch.max(torch.abs(hist2[0, 0] - prev_hist[0, 1])).item(),
                        torch.max(torch.abs(hist2[0, 1] - prev_hist[0, 2])).item(),
                        torch.max(torch.abs(hist2[0, 2] - prev_hist[0, 3])).item(),
                        torch.max(torch.abs(hist2[0, 3] - pred1[0])).item(),
                    ]
                    print("==== DFoT sliding-window check ====")
                    print(f"Max diff for slide (0←1,1←2,2←3,3←pred1): {shift_diffs}")
                    if isinstance(actions2, torch.Tensor):
                        zeros_ok2 = torch.allclose(actions2[0, :, :3], torch.zeros_like(actions2[0, :, :3]))
                        dt_vals2 = actions2[0, :, 3].detach().cpu().numpy().tolist()
                        print(f"After slide, actions first 3 dims all zero: {bool(zeros_ok2)}, dt timeline: {dt_vals2}")
                    # Sliding-window event: action and timeline
                    traj_logger.log_event(
                        "DFOT_SLIDING_CHECK",
                        "DFoT sliding-window update and action/timeline consistency",
                        {
                            "shift_diffs_max": shift_diffs,
                            "post_window_actions_zero_check": bool(zeros_ok2) if 'zeros_ok2' in locals() else None,
                            "post_window_dt_timeline": dt_vals2 if 'dt_vals2' in locals() else None,
                        }
                    )
            else:
                print("Model does not expose _dfot_history_latents; cannot directly verify repeated first frame.")
    except Exception as e:
        print(f"DFoT history initialization verification failed: {e}")

    # Forward preparation: perceptual encoding etc. is done in training_step/forward path
    # Directly call inputs needed by training forward in LUMOS.train_step (forward_train)
    # As per lumos.py, forward_train expects: codes, latent_goal, gt_acts, gt_discrete_acts, robot_obs, vae_latent, raw_instructions
    # Follow lumos.py training_step: run perceptual encoding then call forward_train.

    # Perceptual encoding: latent shape check and flatten
    assert vae_latent.shape[-2:] == (32, 32), f"Training expects 32x32 latent, current {vae_latent.shape}"
    latents = torch.flatten(vae_latent, start_dim=2)  # [T, B, 4*32*32] or [B, T, 4, 32, 32] depending on organization

    # Across datasets dims may be [T, B, C, H, W] or [B, T, C, H, W]; normalize simply here:
    if latents.dim() == 3:
        # Might be [T, B, 4096]; keep as is
        pass
    elif latents.dim() == 4:
        # Might be [B, T, C, H*W] flattened -> [B, T, 4096]; transpose to [T, B, 4096]
        latents = latents.permute(1, 0, 2)

    codes = model.perceptual_encoder(latents)

    # Goal encoding: prioritize language, else visual
    if lang is not None and isinstance(lang, torch.Tensor) and lang.numel() > 0 and lang.shape[-1] > 0:
        latent_goal = model.language_goal(lang)
        goal_type = "language"
    else:
        latent_goal = model.visual_goal(codes[-1])
        goal_type = "visual"

    traj_logger.log_model_forward("perceptual_goal_encoding", {
        "vae_latent_shape": vae_latent.shape if isinstance(vae_latent, torch.Tensor) else str(type(vae_latent)),
        "latents_shape": latents.shape,
        "codes_shape": codes.shape,
        "latent_goal_shape": latent_goal.shape,
        "goal_type": goal_type,
    })

    # Run one training forward (complete trajectory)
    raw_instructions_batch = None
    if isinstance(lang_raw, list):
        raw_instructions_batch = lang_raw
    # Construct robust fallbacks for gt_acts and gt_discrete_acts to ensure shape/type matches
    gt_acts = rel_actions
    if gt_acts is None:
        # Expected shape [T, B, act_dim]; infer T, B from codes and vae_latent
        # codes shape is [T, B, ...]
        T = codes.shape[0]
        B = codes.shape[1]
        act_dim = 4  # DFoT pure inference path uses 4-dim relative actions
        gt_acts = torch.zeros((T, B, act_dim), device=device, dtype=torch.float32)

    gt_discrete_acts = discrete_action
    if gt_discrete_acts is None:
        # Training-time discrete actions usually [T, B] or [T, B, K] logits; use labels placeholder [T, B]
        T = codes.shape[0]
        B = codes.shape[1]
        gt_discrete_acts = torch.zeros((T, B), device=device, dtype=torch.long)

    # Before forward_train, wrap nwm_step counter to verify "1-step per step"
    nwm_call_counter = {"count": 0}
    original_nwm_step = model.nwm_step
    def counted_nwm_step(current_vae_latent, action_batch, raw_instructions_batch=None):
        nwm_call_counter["count"] += 1
        return original_nwm_step(current_vae_latent, action_batch, raw_instructions_batch)
    model.nwm_step = counted_nwm_step

    losses, rewards, returns, ac_actions, pp_state, pr_state, seq_feat = model.forward_train(
        codes=codes,
        latent_goal=latent_goal,
        gt_acts=gt_acts,
        gt_discrete_acts=gt_discrete_acts,
        robot_obs=robot_obs,
        vae_latent=vae_latent,
        raw_instructions=raw_instructions_batch,
    )

    # Verify 1-step-per-step: nwm_step call count should equal sequence length (T or self.seq_len)
    try:
        T = codes.shape[0]
        print("==== LUMOS 1-step-per-step verification ====")
        print(f"nwm_step call count: {nwm_call_counter['count']}, expected ≈ sequence length T={T}")
    except Exception as e:
        print(f"Failed to count nwm_step calls: {e}")

    traj_logger.log_model_forward("forward_train_done", {
        "loss_keys": list(losses.keys()) if isinstance(losses, dict) else str(type(losses)),
        "returns_mean": float(torch.stack(returns).mean().item()) if isinstance(returns, list) and len(returns) > 0 else 0.0,
        "rewards_mean": float(torch.stack(rewards).mean().item()) if isinstance(rewards, list) and len(rewards) > 0 else 0.0,
        "ac_actions_shape": ac_actions.shape if hasattr(ac_actions, "shape") else str(type(ac_actions)),
    })

    # Finish trajectory
    traj_logger.finish_trajectory()

    print("==== Done: full forward data flow recorded ====")
    print(f"Log directory: {log_dir}")
    # List the latest log files
    if log_dir.exists():
        files = sorted(log_dir.glob("trajectory_log_*.json"))
        if files:
            print(f"JSON log: {files[-1]}")
        files_txt = sorted(log_dir.glob("trajectory_log_*.txt"))
        if files_txt:
            print(f"Text log: {files_txt[-1]}")
        img_files = sorted(img_out_dir.glob("*.png"))
        if img_files:
            print(f"DFoT history/predicted latent image samples: {img_files[:4]}")


if __name__ == "__main__":
    main()
