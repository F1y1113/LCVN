try:
    import flash_attn.flash_attn_interface as _fa
    for _old, _new in [("_wrapped_flash_attn_forward", "_flash_attn_forward"),
                       ("_wrapped_flash_attn_backward", "_flash_attn_backward")]:
        if not hasattr(_fa, _old) and hasattr(_fa, _new):
            setattr(_fa, _old, getattr(_fa, _new))
except ImportError:
    pass

import argparse
import pickle
import sys
from pathlib import Path
import torch
from tqdm import tqdm
import cv2
import numpy as np
from algorithms.vae.image_vae.trainer import ImageVAE
from utils.storage_utils import safe_torch_save
from datasets.video.utils.transform import VideoTransform
import random

"""
def load_vae(ckpt_path: str, device: torch.device, use_fp16: bool) -> ImageVAE:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    vae = ImageVAE(checkpoint["cfg"])
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    for k in list(state_dict.keys()):
        if k.startswith("loss"):
            del state_dict[k]
    vae.load_state_dict(state_dict)
    vae = vae.to(device)
    if use_fp16 and device.type == "cuda":
        vae = vae.half()
    vae = vae.eval()
    return vae
"""

def load_vae(ckpt_path: str, device: torch.device, use_fp16: bool) -> ImageVAE:
    # Case A: local .ckpt file (PyTorch Lightning)
    if str(ckpt_path).endswith(".ckpt"):
        print(f"Loading VAE from local checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Try to initialize from config in checkpoint
        if "cfg" in checkpoint:
            vae = ImageVAE(checkpoint["cfg"])
        else:
            # No cfg found, fall back to default config
            from omegaconf import OmegaConf
            # Default z_channels=4, resolution=256; adjust if needed
            vae = ImageVAE(OmegaConf.create({"z_channels": 4, "resolution": 256}))

        # Load weights
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

        # Filter out loss-related keys
        for k in list(state_dict.keys()):
            if k.startswith("loss"):
                del state_dict[k]

        # Use strict=False to tolerate minor key mismatches
        msg = vae.load_state_dict(state_dict, strict=False)
        print(f"Local VAE weights loaded. Msg: {msg}")

    # Case B: HuggingFace ID or Diffusers format directory
    else:
        print(f"Loading VAE from HF/Diffusers: {ckpt_path}")
        dtype = torch.float16 if (use_fp16 and device.type == "cuda") else torch.float32
        try:
            # Try ImageVAE.from_pretrained first
            vae = ImageVAE.from_pretrained(ckpt_path, torch_dtype=dtype)
        except Exception as e:
            print(f"ImageVAE load failed ({e}), trying Diffusers AutoencoderKL...")
            try:
                from diffusers import AutoencoderKL
                vae = AutoencoderKL.from_pretrained(ckpt_path, torch_dtype=dtype)
            except ImportError:
                raise ImportError("Please install diffusers: pip install diffusers")
    vae = vae.to(device)
    if use_fp16 and device.type == "cuda":
        vae = vae.half()
    vae = vae.eval()
    return vae

def read_frames_jpg(folder: Path) -> torch.Tensor:
    """Read sorted jpg frames from a trajectory folder."""
    jpgs = sorted(folder.glob("*.jpg"), key=lambda p: int(p.stem))
    if not jpgs:
        return torch.empty(0, 3, 1, 1)
    frames = []
    for jpg in jpgs:
        img = cv2.imread(str(jpg))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        arr = torch.from_numpy(img).float() / 255.0
        arr = arr.permute(2, 0, 1)
        frames.append(arr)
    if not frames:
        return torch.empty(0, 3, 1, 1)
    return torch.stack(frames, dim=0)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_ckpt", type=str, required=False)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--suffix", type=str, default="latent")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=None)
    parser.add_argument("--shard_idx", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true", default=False)
    parser.add_argument("--min_frames", type=int, default=1)
    parser.add_argument("--probe", type=int, default=0)
    # Social dataset-compatible output and naming
    parser.add_argument("--social_mode", action="store_true", default=False)
    parser.add_argument("--social_save_dir", type=str, default="data/lcvn")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--metadata_dir", type=str, default=None)
    parser.add_argument("--from_metadata", action="store_true", default=False)
    parser.add_argument("--build_metadata_from_recon", action="store_true", default=False)
    parser.add_argument("--recon_root", type=str, default=None,
                        help="Comma-separated list of dataset root directories (e.g. recon,go_stanford,sacson,scand)")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--reorganize_latents_from_recon", action="store_true", default=False)
    parser.add_argument("--latents_root", type=str, default=None)
    parser.add_argument("--latents_recon_dir", type=str, default=None)
    parser.add_argument("--add_conditions_from_traj", action="store_true", default=False)
    parser.add_argument("--splits", type=str, default="training,validation,test")
    parser.add_argument("--normalize_traj_id", action="store_true", default=False)
    parser.add_argument("--compute_normalization_stats", action="store_true", default=False)
    args = parser.parse_args()

    if args.add_conditions_from_traj:
        if args.metadata_dir is None:
            print("Missing --metadata_dir for adding conditions")
            sys.exit(1)
        metadata_dir = Path(args.metadata_dir)
        splits = [s.strip() for s in args.splits.split(",") if s.strip()]
        updated_counts = {}
        for split in splits:
            pt_path = metadata_dir / f"{split}.pt"
            if not pt_path.exists():
                print(f"Skip split={split}, metadata not found: {pt_path}")
                continue
            md = torch.load(pt_path, map_location="cpu", weights_only=False)
            video_paths = md.get("video_path", md.get("video_paths"))
            traj_paths = md.get("traj_path", None)
            scenes = md.get("scene", None)
            tids = md.get("trajectory_id", None)
            num_frames_list = md.get("num_frames", None)
            if video_paths is None or num_frames_list is None:
                print(f"Invalid metadata format for split={split}: missing video_path(s) or num_frames")
                continue
            conditions = []
            ok = 0
            for i in tqdm(range(len(video_paths)), desc=f"Conditions[{split}]"):
                vp = Path(str(video_paths[i]))
                num_frames = int(num_frames_list[i])
                if traj_paths is not None:
                    tp = Path(str(traj_paths[i])) if traj_paths[i] else (vp.parent / "traj_data.pkl")
                else:
                    tp = vp.parent / "traj_data.pkl"
                try:
                    with open(tp, "rb") as f:
                        d = pickle.load(f)
                    pos = np.asarray(d.get("position"))
                    yaw = np.asarray(d.get("yaw"))
                    if pos.ndim != 2 or pos.shape[1] < 2 or yaw.ndim != 1:
                        raise ValueError("Invalid position/yaw shape")
                    n = min(len(yaw), len(pos), num_frames)
                    pos = pos[:n]
                    yaw = yaw[:n]
                    z = np.zeros((n,), dtype=np.float32)
                    half = 0.5 * yaw.astype(np.float32)
                    qw = np.cos(half)
                    qx = np.zeros_like(qw)
                    qy = np.zeros_like(qw)
                    qz = np.sin(half)
                    cond = np.stack([pos[:,0].astype(np.float32), pos[:,1].astype(np.float32), z, qw, qx, qy, qz], axis=1)
                    conditions.append(torch.from_numpy(cond))
                    ok += 1
                except Exception:
                    conditions.append(None)
                    continue
            md["conditions"] = conditions
            torch.save(md, pt_path)
            updated_counts[split] = ok
        print(f"Added conditions for splits: {updated_counts}")
        sys.exit(0)
    if args.compute_normalization_stats:
        if args.metadata_dir is None:
            print("Missing --metadata_dir for compute_normalization_stats")
            sys.exit(1)
        metadata_dir = Path(args.metadata_dir)
        train_path = metadata_dir / "training.pt"
        if not train_path.exists():
            print(f"Training metadata not found at {train_path}")
            sys.exit(1)
        md = torch.load(train_path, map_location="cpu", weights_only=False)
        conditions_list = md.get("conditions", [])
        def quat_to_yaw(qw, qx, qy, qz):
            siny_cosp = 2.0 * (qw * qz + qx * qy)
            cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
            return np.arctan2(siny_cosp, cosy_cosp).astype(np.float32)
        dx_all = []
        dy_all = []
        dyaw_all = []
        dt_all = []
        for cond in conditions_list:
            if cond is None:
                continue
            arr = cond.numpy() if isinstance(cond, torch.Tensor) else np.asarray(cond)
            if arr.ndim != 2 or arr.shape[1] < 7:
                continue
            x = arr[:, 0].astype(np.float32)
            y = arr[:, 1].astype(np.float32)
            qw = arr[:, 3].astype(np.float32)
            qx = arr[:, 4].astype(np.float32)
            qy = arr[:, 5].astype(np.float32)
            qz = arr[:, 6].astype(np.float32)
            yaw = quat_to_yaw(qw, qx, qy, qz)
            if len(x) >= 2:
                dx = x[1:] - x[:-1]
                dy = y[1:] - y[:-1]
                dyaw = yaw[1:] - yaw[:-1]
                dyaw = np.arctan2(np.sin(dyaw), np.cos(dyaw)).astype(np.float32)
                dt = np.full_like(dx, 0.1, dtype=np.float32)
                dx_all.append(dx)
                dy_all.append(dy)
                dyaw_all.append(dyaw)
                dt_all.append(dt)
        if len(dx_all) == 0:
            print("No valid conditions found to compute normalization stats")
            sys.exit(1)
        dx_cat = np.concatenate(dx_all, axis=0)
        dy_cat = np.concatenate(dy_all, axis=0)
        dyaw_cat = np.concatenate(dyaw_all, axis=0)
        dt_cat = np.concatenate(dt_all, axis=0)
        delta_min = torch.tensor([dx_cat.min(), dy_cat.min(), dyaw_cat.min()], dtype=torch.float32)
        delta_max = torch.tensor([dx_cat.max(), dy_cat.max(), dyaw_cat.max()], dtype=torch.float32)
        dt_min = torch.tensor([dt_cat.min()], dtype=torch.float32)
        dt_max = torch.tensor([dt_cat.max()], dtype=torch.float32)
        stats = {
            "delta": {"min": delta_min, "max": delta_max},
            "dt": {"min": dt_min, "max": dt_max},
        }
        out_path = metadata_dir / "normalization_stats.pt"
        torch.save(stats, out_path)
        print(f"Saved normalization stats to {out_path}")
        print(f"delta min={delta_min.tolist()} max={delta_max.tolist()} dt min={dt_min.item():.4f} max={dt_max.item():.4f}")
        sys.exit(0)
    if args.normalize_traj_id:
        if args.metadata_dir is None:
            print("Missing --metadata_dir for normalize_traj_id")
            sys.exit(1)
        metadata_dir = Path(args.metadata_dir)
        splits = [s.strip() for s in args.splits.split(",") if s.strip()]
        normalized = {}
        for split in splits:
            pt_path = metadata_dir / f"{split}.pt"
            if not pt_path.exists():
                continue
            md = torch.load(pt_path, map_location="cpu", weights_only=False)
            scenes = md.get("scene", [])
            tids = md.get("trajectory_id", [])
            for i in range(len(tids)):
                scene = scenes[i] if i < len(scenes) else None
                tid = str(tids[i])
                if scene and tid.startswith(scene + "_"):
                    tids[i] = tid[len(scene) + 1 :]
            md["trajectory_id"] = tids
            torch.save(md, pt_path)
            normalized[split] = len(tids)
        print(f"Normalized trajectory_id for splits: {normalized}")
        sys.exit(0)
    if args.build_metadata_from_recon:
        if args.recon_root is None or args.metadata_dir is None:
            print("Missing --recon_root or --metadata_dir for building metadata")
            sys.exit(1)
        recon_roots = [Path(r.strip()) for r in args.recon_root.split(",") if r.strip()]
        metadata_dir = Path(args.metadata_dir)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        items = []
        for recon_root in recon_roots:
            if not recon_root.exists():
                print(f"Warning: dataset root not found, skipping: {recon_root}")
                continue
            for folder in sorted(recon_root.iterdir()):
                if not folder.is_dir():
                    continue
                traj_path = folder / "traj_data.pkl"
                if not traj_path.exists():
                    continue
                jpgs = sorted(folder.glob("*.jpg"))
                if not jpgs:
                    continue
                video_path = folder  # directory of jpgs
                num_frames = len(jpgs)
                # Load instruction from traj_data.pkl; fall back to nav_map
                try:
                    with open(traj_path, "rb") as f:
                        traj_data = pickle.load(f)
                    instruction = traj_data.get("instruction", "") or ""
                except Exception:
                    instruction = ""
                name = folder.name
                parts = name.split("_")
                scene = parts[0] if len(parts) > 0 else recon_root.name
                trajectory_id = name
                items.append(
                    dict(
                        video_path=video_path,
                        traj_path=traj_path,
                        num_frames=num_frames,
                        instruction=instruction,
                        scene=scene,
                        trajectory_id=trajectory_id,
                        conditions=None,
                    )
                )
        total = len(items)
        if total == 0:
            print("No trajectories found in any dataset root")
            sys.exit(1)
        random.seed(42)
        random.shuffle(items)
        n_train = int(total * args.train_ratio)
        train_items = items[:n_train]
        val_items = items[n_train:]
        def to_columnar(lst):
            return {
                "video_path": [x["video_path"] for x in lst],
                "traj_path": [x["traj_path"] for x in lst],
                "num_frames": [x["num_frames"] for x in lst],
                "instruction": [x["instruction"] for x in lst],
                "scene": [x["scene"] for x in lst],
                "trajectory_id": [x["trajectory_id"] for x in lst],
                "conditions": [x["conditions"] for x in lst],
            }
        torch.save(to_columnar(train_items), metadata_dir / "training.pt")
        torch.save(to_columnar(val_items), metadata_dir / "validation.pt")
        print(f"Built metadata at {metadata_dir} with counts: train={len(train_items)}, val={len(val_items)} (total={total})")
        sys.exit(0)
    if args.reorganize_latents_from_recon:
        if args.metadata_dir is None or args.latents_root is None or args.latents_recon_dir is None:
            print("Missing --metadata_dir, --latents_root or --latents_recon_dir for reorganization")
            sys.exit(1)
        metadata_dir = Path(args.metadata_dir)
        latents_root = Path(args.latents_root)
        recon_dir = Path(args.latents_recon_dir)
        if not recon_dir.exists():
            print(f"Recon latents not found: {recon_dir}")
            sys.exit(1)
        split_map = {}
        for split in ["training", "validation"]:
            m = torch.load(metadata_dir / f"{split}.pt", map_location="cpu", weights_only=False)
            tids = set([str(x) for x in m["trajectory_id"]])
            split_map[split] = tids
        moved = 0
        for p in sorted(recon_dir.glob("*.pt")):
            stem = p.stem
            parts = stem.split("_")
            if len(parts) < 2:
                continue
            scene = parts[0]
            trajectory_id = "_".join(parts[:-2]) if stem.endswith("_video_latent") else "_".join(parts[:-1])
            traj_only = trajectory_id[len(scene) + 1 :] if trajectory_id.startswith(scene + "_") else trajectory_id
            target_split = None
            for s in ["training", "validation"]:
                if trajectory_id in split_map[s]:
                    target_split = s
                    break
            if target_split is None:
                continue
            new_name = f"{scene}_{traj_only}_latent.pt"
            out_dir = latents_root / target_split
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / new_name
            if not out_path.exists():
                p.rename(out_path)
                moved += 1
        print(f"Reorganized latents from {recon_dir} into splits at {latents_root}, moved={moved}")
        sys.exit(0)
    if args.social_mode and args.from_metadata:
        if not args.split or not args.metadata_dir:
            print("In --social_mode with --from_metadata you must provide --split and --metadata_dir")
            sys.exit(1)
        output_dir = Path(args.social_save_dir) / "latents" / args.split
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata = torch.load(Path(args.metadata_dir) / f"{args.split}.pt", map_location="cpu", weights_only=False)
        video_paths = metadata["video_path"]
        scenes = metadata["scene"]
        tids = metadata["trajectory_id"]
        indices = list(range(len(video_paths)))
        if args.num_shards and args.shard_idx is not None:
            n = len(indices)
            k = max(1, args.num_shards)
            size = (n + k - 1) // k
            start = size * args.shard_idx
            end = min(n, start + size)
            indices = indices[start:end]
        if args.limit is not None:
            indices = indices[: args.limit]
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        if args.vae_ckpt is None:
            print("Missing --vae_ckpt")
            sys.exit(1)
        vae = load_vae(args.vae_ckpt, device, args.fp16)
        transform = VideoTransform((args.resolution, args.resolution))
        encoded_count = 0
        skipped_existing = 0
        empty_count = 0
        short_count = 0
        error_count = 0
        for idx in tqdm(indices, desc="Encoding"):
            vp = Path(video_paths[idx])
            scene = scenes[idx]
            traj = str(tids[idx])
            save_path = Path(args.social_save_dir) / "latents" / args.split / f"{scene}_{traj}_{args.suffix}.pt"
            if args.skip_existing and save_path.exists():
                skipped_existing += 1
                continue
            video = read_frames_jpg(vp)
            num_frames = video.shape[0] if video.ndim == 4 else 0
            if args.probe and encoded_count < args.probe:
                print(f"[probe] {vp.name}: frames={num_frames}")
            if num_frames == 0:
                empty_count += 1
                continue
            if num_frames < args.min_frames:
                short_count += 1
                continue
            try:
                video = transform(video)
                video = video.to(device)
                video = 2.0 * video - 1.0
                if args.fp16 and device.type == "cuda":
                    video = video.half()
                with torch.no_grad():
                    enc = vae.encode(video)
                    latent = (enc.latent_dist.sample() if hasattr(enc, "latent_dist") else enc.sample()).to(torch.float16)
                safe_torch_save(latent.cpu(), save_path)
                encoded_count += 1
            except Exception as e:
                print(f"Error encoding {vp}: {e}")
                error_count += 1
                continue
        print(f"Saved latents to {Path(args.social_save_dir) / 'latents' / args.split}")
        print(f"Summary: encoded={encoded_count}, skipped_existing={skipped_existing}, empty={empty_count}, short={short_count}, errors={error_count}")
        sys.exit(0)


if __name__ == "__main__":
    main()
