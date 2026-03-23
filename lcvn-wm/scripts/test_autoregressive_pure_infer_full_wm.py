import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import torch

wm_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(wm_root))

from pure_infer import LDiTPureInference


def quaternion_to_yaw(quats: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    return torch.atan2(siny_cosp, cosy_cosp)


def load_social_sample(data_dir: Path, split: str, sample_idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    metadata_path = data_dir / "metadata" / f"{split}.pt"
    latent_dir = data_dir / "latents" / split
    metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)
    vm = {k: metadata[k][sample_idx] for k in metadata.keys()}
    scene = vm["scene"]
    traj_id = vm["trajectory_id"]
    conditions = vm["conditions"]
    latent_path = latent_dir / f"{scene}_{traj_id}_latent.pt"
    latents_all = torch.load(latent_path, map_location="cpu", weights_only=False)
    history_latents = latents_all[:4].unsqueeze(0)
    conds_4 = torch.as_tensor(conditions[:4], dtype=torch.float32)
    pos = conds_4[:, :3]
    quats = conds_4[:, 3:]
    dx = torch.zeros(4, dtype=torch.float32)
    dy = torch.zeros(4, dtype=torch.float32)
    if pos.shape[0] > 1:
        dx[1:] = pos[1:, 0] - pos[:-1, 0]
        dy[1:] = pos[1:, 1] - pos[:-1, 1]
    yaw = quaternion_to_yaw(quats)
    dyaw = torch.zeros(4, dtype=torch.float32)
    if yaw.shape[0] > 1:
        d = yaw[1:] - yaw[:-1]
        dyaw[1:] = torch.atan2(torch.sin(d), torch.cos(d))
    dt = torch.full((4,), 0.1, dtype=torch.float32)
    actions = torch.stack([dx, dy, dyaw, dt], dim=-1).unsqueeze(0)
    meta = {"scene": scene, "traj_id": traj_id, "latents_all": latents_all, "conditions_all": torch.as_tensor(conditions, dtype=torch.float32)}
    return history_latents, actions, meta


def save_image_png(img: torch.Tensor, path: Path) -> None:
    try:
        from torchvision.utils import save_image
        if img.ndim == 4 and img.shape[0] == 1:
            img = img[0]
        save_image(img.clamp(0, 1), str(path))
    except Exception:
        from PIL import Image
        if img.ndim == 4 and img.shape[0] == 1:
            img = img[0]
        img = img.clamp(0, 1)
        img = (img * 255.0).byte().permute(1, 2, 0).cpu().numpy()
        Image.fromarray(img).save(str(path))


def compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    x = pred.clamp(0, 1).detach().cpu()
    y = gt.clamp(0, 1).detach().cpu()
    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]
    if y.ndim == 4 and y.shape[0] == 1:
        y = y[0]
    mse = torch.mean((x - y) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    import math
    return float(20.0 * math.log10(1.0) - 10.0 * math.log10(mse))


def autoregressive_rollout(inferencer, history_latents: torch.Tensor, actions: torch.Tensor, meta: Dict[str, Any], device: str, output_dir: Path, horizon: int, force_sampling_steps: int) -> Dict[str, Any]:
    try:
        if hasattr(inferencer.model, "sampling_timesteps"):
            inferencer.model.sampling_timesteps = force_sampling_steps
        if hasattr(inferencer.model, "diffusion_model") and hasattr(inferencer.model.diffusion_model, "sampling_timesteps"):
            inferencer.model.diffusion_model.sampling_timesteps = force_sampling_steps
    except Exception:
        pass
    latents_all: torch.Tensor = meta["latents_all"]
    conditions_all: torch.Tensor = meta["conditions_all"].to(device)
    hist = history_latents.clone().to(device)
    acts = actions.clone().to(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_latent_dir = output_dir / "gt_latent"
    gt_rgb_dir = output_dir / "gt_rgb"
    pred_latent_dir = output_dir / "pred_latent"
    pred_rgb_dir = output_dir / "pred_rgb"
    metrics_txt = output_dir / "metrics.txt"
    gt_latent_dir.mkdir(exist_ok=True)
    gt_rgb_dir.mkdir(exist_ok=True)
    pred_latent_dir.mkdir(exist_ok=True)
    pred_rgb_dir.mkdir(exist_ok=True)
    T_total = latents_all.shape[0]
    with metrics_txt.open("w") as f:
        for k in range(1, horizon + 1):
            gt_index = 4 + k - 1
            gt_available = gt_index < T_total
            gt_latent_k = latents_all[gt_index].unsqueeze(0).to(device) if gt_available else None
            pred_latent, _ = inferencer.predict(hist, acts, return_rgb=True)
            
            # Decode latents
            # Since we have regenerated latents with the correct VAE checkpoint,
            # no scaling correction is needed.
            scale_factor = 1.0
            pred_rgb = inferencer._decode_latents(pred_latent * scale_factor)

            torch.save(pred_latent.detach().cpu(), pred_latent_dir / f"step_{k:02d}.pt")
            if pred_rgb is not None:
                save_image_png(pred_rgb, pred_rgb_dir / f"step_{k:02d}.png")
            if gt_latent_k is not None:
                torch.save(gt_latent_k.detach().cpu(), gt_latent_dir / f"step_{k:02d}.pt")
                try:
                    # GT latents from dataset are raw VAE latents (not normalized by data stats),
                    # so we can directly decode them.
                    gt_rgb = inferencer._decode_latents(gt_latent_k * scale_factor)
                    save_image_png(gt_rgb, gt_rgb_dir / f"step_{k:02d}.png")
                    if pred_rgb is not None:
                        psnr = compute_psnr(pred_rgb, gt_rgb)
                        f.write(f"step={k}, psnr={psnr:.3f}\n")
                except Exception as e:
                    print(f"Error decoding GT latent: {e}")
                    pass
            hist = torch.cat([hist[:, 1:], pred_latent.unsqueeze(1)], dim=1)
            idx_new = 4 + k - 1
            if idx_new < conditions_all.shape[0]:
                pos_prev = conditions_all[idx_new - 1, :3]
                pos_cur = conditions_all[idx_new, :3]
                dx_new = pos_cur[0] - pos_prev[0]
                dy_new = pos_cur[1] - pos_prev[1]
                quat_prev = conditions_all[idx_new - 1, 3:]
                quat_cur = conditions_all[idx_new, 3:]
                yaw_prev = quaternion_to_yaw(quat_prev.unsqueeze(0))[0]
                yaw_cur = quaternion_to_yaw(quat_cur.unsqueeze(0))[0]
                dyaw_new = torch.atan2(torch.sin(yaw_cur - yaw_prev), torch.cos(yaw_cur - yaw_prev))
                dt_new = torch.tensor(0.1, device=device)
            else:
                dx_new = torch.tensor(0.0, device=device)
                dy_new = torch.tensor(0.0, device=device)
                dyaw_new = torch.tensor(0.0, device=device)
                dt_new = torch.tensor(0.1, device=device)
            zero_first = acts[:, :1, :]
            keep_last_two = acts[:, 2:, :]
            new_action = torch.stack([dx_new, dy_new, dyaw_new, dt_new]).view(1, 1, 4)
            acts = torch.cat([zero_first, keep_last_two, new_action], dim=1)
    return {"saved_steps": horizon}


def find_valid_samples(data_dir: Path, split: str, min_frames: int, max_count: int) -> List[int]:
    metadata_path = data_dir / "metadata" / f"{split}.pt"
    latent_dir = data_dir / "latents" / split
    md = torch.load(metadata_path, map_location="cpu", weights_only=False)
    total = len(md["scene"])
    res = []
    for idx in range(total):
        scene = md["scene"][idx]
        traj_id = md["trajectory_id"][idx]
        lp = latent_dir / f"{scene}_{traj_id}_latent.pt"
        if not lp.exists():
            continue
        try:
            lat = torch.load(lp, map_location="cpu", weights_only=False)
            if lat.shape[0] >= min_frames:
                res.append(idx)
                if len(res) >= max_count:
                    break
        except Exception:
            continue
    return res


def gather_targets(data_dir: Path, targets: List[Tuple[str, str]], split: str) -> List[int]:
    metadata_path = data_dir / "metadata" / f"{split}.pt"
    md = torch.load(metadata_path, map_location="cpu", weights_only=False)
    total = len(md["scene"])
    idxs = []
    for idx in range(total):
        s = md["scene"][idx]
        t = md["trajectory_id"][idx]
        for a, b in targets:
            if s == a and t == b:
                idxs.append(idx)
                break
    return idxs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="lcvn-wm/data/lcvn")
    parser.add_argument("--dfot-ckpt", type=str, default="lcvn-wm/outputs/2025-12-16/04-30-11/checkpoints/epoch=epoch=20-loss=training/loss=0.1421.ckpt")
    parser.add_argument("--vae-ckpt", type=str, default="weights/vae/vae_epoch3_step50000.ckpt")
    parser.add_argument("--out-dir", type=str, default="lcvn-wm/autoreg_vis")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--num-trajectories", type=int, default=3)
    parser.add_argument("--targets", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--force-sampling-steps", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=-1)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir_root = Path(args.out_dir)
    device = args.device

    inferencer = LDiTPureInference(dfot_checkpoint_path=args.dfot_ckpt, vae_checkpoint_path=args.vae_ckpt, device=device)

    if args.targets.strip():
        pairs = []
        for it in [x.strip() for x in args.targets.split(",") if x.strip()]:
            if ":" in it:
                s, t = it.split(":", 1)
                pairs.append((s, t))
        idxs = gather_targets(data_dir, pairs, args.split)
        if len(idxs) == 0:
            print("no targets matched")
            return
    else:
        min_frames = 5 if args.horizon == -1 else (4 + args.horizon)
        idxs = find_valid_samples(data_dir, args.split, min_frames, args.num_trajectories)
        if len(idxs) < args.num_trajectories:
            idxs = idxs

    for sample_idx in idxs[: args.num_trajectories]:
        hist, acts, meta = load_social_sample(data_dir, args.split, sample_idx)
        H = args.horizon if args.horizon != -1 else int(meta["latents_all"].shape[0]) - 4
        if H <= 0:
            print(f"skip {sample_idx}: insufficient frames")
            continue
        hist = hist.to(device)
        acts = acts.to(device)
        out_dir = out_dir_root / f"{args.split}_sample_{sample_idx:03d}_{meta['scene']}_{meta['traj_id']}"
        autoregressive_rollout(inferencer, hist, acts, meta, device, out_dir, horizon=H, force_sampling_steps=args.force_sampling_steps)
        print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()

