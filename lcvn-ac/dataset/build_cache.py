import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPTokenizer, CLIPTextModel


def build_cache(dataset_root: Path, split: str, lang_dim: int, output_root: Path) -> None:
    """
    Reads actions and instructions from traj_data.pkl.
    """
    meta_path = dataset_root / f"metadata/{split}.pt"
    latent_dir = dataset_root / f"latents/{split}"

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    if not latent_dir.exists():
        raise FileNotFoundError(f"Latent directory not found: {latent_dir}")

    out_split_dir = output_root / split
    out_split_dir.mkdir(parents=True, exist_ok=True)

    metadata = torch.load(meta_path, map_location="cpu", weights_only=False)

    # Initialize CLIP text encoder
    _clip_model_id = "openai/clip-vit-base-patch32"
    clip_tokenizer = CLIPTokenizer.from_pretrained(_clip_model_id)
    clip_model = CLIPTextModel.from_pretrained(_clip_model_id, use_safetensors=True)
    clip_model.eval()

    features: dict[int, dict] = {}
    ep_bounds: list[list[int]] = []
    lang_raw: list[str] = []
    lang_emb: list[np.ndarray] = []

    # Extract from dict-of-lists metadata
    scenes = metadata["scene"]
    tids = metadata["trajectory_id"]
    frames = metadata["num_frames"]
    instrs = metadata.get("instruction", [None] * len(scenes))
    traj_paths = metadata.get("traj_path", [None] * len(scenes))

    N = len(scenes)
    g: int = 0  # global frame index
    skipped = 0

    for i in range(N):
        scene = scenes[i]
        tid = tids[i]
        T = int(frames[i])
        instr = instrs[i] if instrs[i] is not None else ""
        traj_path = traj_paths[i]

        lat_file = latent_dir / f"{scene}_{tid}_latent.pt"
        if not lat_file.exists():
            print(f"Warning: Latent file not found, skipping: {lat_file}")
            skipped += 1
            continue

        # Load latents
        data = torch.load(lat_file, map_location="cpu", weights_only=False)
        latents_obj = data["latents"] if isinstance(data, dict) else data
        latents = np.asarray(latents_obj, dtype=np.float32)  # (T, 4, 32, 32)

        # Load actions from traj_data.pkl
        if traj_path is None:
            print(f"Warning: No traj_path for {scene}_{tid}, skipping")
            skipped += 1
            continue

        traj_path = Path(traj_path)
        if not traj_path.exists():
            print(f"Warning: traj_data not found: {traj_path}, skipping")
            skipped += 1
            continue

        with open(traj_path, "rb") as f:
            traj_data = pickle.load(f)

        delta = traj_data["delta"]  # (T-1, 3) - dx, dy, dyaw
        lang_action = traj_data.get("language_action", None)

        # Verify lengths match
        if latents.shape[0] != T:
            print(f"Warning: Latent length mismatch for {scene}_{tid}: meta={T}, latent={latents.shape[0]}, skipping")
            skipped += 1
            continue

        if delta.shape[0] != T - 1:
            print(f"Warning: Delta length mismatch for {scene}_{tid}: expected={T-1}, got={delta.shape[0]}, skipping")
            skipped += 1
            continue

        # Build actions: add dt=0.1 as 4th column
        actions = []
        for t in range(T - 1):
            dx, dy, dyaw = delta[t]
            dt = 0.1
            actions.append(np.array([dx, dy, dyaw, dt], dtype=np.float32))
        # Pad last frame with zeros
        actions.append(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))

        start_g = g
        latent_index = lat_file.stem.replace("_latent", "")

        for t in range(T):
            frame = {
                "vae_latent": latents[t],  # (4, 32, 32)
                "rel_actions": actions[t],  # (4,)
                "reset": np.array([t == 0], dtype=bool),
                "discrete_action": np.array([0], dtype=np.int64),  # placeholder
                "source_meta": {
                    "latent_index": latent_index,
                    "traj_path": str(traj_path),
                },
            }
            features[g] = frame
            g += 1

        ep_bounds.append([start_g, g])

        # Build instruction: prefer metadata instruction, fall back to language_action
        if not instr and lang_action is not None:
            actions_flat = lang_action.flatten().tolist()
            deduped = []
            prev = None
            for a in actions_flat:
                if a != prev:
                    deduped.append(a)
                    prev = a
            instr = ", ".join(str(a) for a in deduped)

        lang_raw.append(instr)

        # Encode language with CLIP
        if not instr or (isinstance(instr, str) and len(instr.strip()) == 0):
            lang_emb.append(np.zeros((lang_dim,), dtype=np.float32))
        else:
            tokens = clip_tokenizer([instr], padding=True, truncation=True, max_length=77, return_tensors="pt")
            with torch.no_grad():
                emb = clip_model(**tokens).pooler_output[0].float().numpy()
            lang_emb.append(np.asarray(emb, dtype=np.float32))

        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{N} episodes, {g} frames")

    print(f"Total: {N} episodes, skipped {skipped}, processed {N - skipped}")
    print(f"Total frames: {g}")

    if not features:
        print("No data collected; not saving.")
        return

    # Write cached_feats.pkl
    with open(out_split_dir / "cached_feats.pkl", "wb") as f:
        pickle.dump(features, f)

    # Write ep_start_end_ids.npy
    ep_bounds_arr = np.array(ep_bounds, dtype=np.int32)
    np.save(out_split_dir / "ep_start_end_ids.npy", ep_bounds_arr)

    # Write auto_lang_ann.npy
    lang_data = {
        "language": {
            "emb": np.stack(lang_emb, axis=0),
            "raw": np.array(lang_raw, dtype=object),
        },
        "info": {
            "indx": ep_bounds_arr[:, 0],
        },
    }
    np.save(out_split_dir / "auto_lang_ann.npy", lang_data, allow_pickle=True)

    print(
        f"Saved cache for split '{split}':\n"
        f"  {out_split_dir / 'cached_feats.pkl'}\n"
        f"  {out_split_dir / 'ep_start_end_ids.npy'}\n"
        f"  {out_split_dir / 'auto_lang_ann.npy'}"
    )


def main():
    parser = argparse.ArgumentParser(description="Build social cache from traj_data.pkl")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root dir of social dataset")
    parser.add_argument("--split", type=str, default="training", help="Processed split name, e.g. training or validation_seen")
    parser.add_argument("--lang_dim", type=int, default=512)
    parser.add_argument("--output_root", type=str, default="data/social_cached")
    args = parser.parse_args()

    build_cache(Path(args.dataset_root), args.split, args.lang_dim, Path(args.output_root))


if __name__ == "__main__":
    main()
