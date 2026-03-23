"""
Generate episode-level instruction strings from per-frame language_action in trajectory data.
Strategy: deduplicate consecutive repeated actions while preserving order, join with ", ".

Output:
  - updates auto_lang_ann.npy['language']['raw'] for each split
  - writes instruction_map.json: {latent_index -> instruction} for use by the dataset loader
"""

import json
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm


def dedup_ordered(seq):
    """Remove consecutive duplicates, preserving order."""
    result = []
    prev = None
    for x in seq:
        if x != prev:
            result.append(x)
            prev = x
    return result


def generate_instructions(split_dir: Path) -> dict:
    """Returns {latent_index: instruction} mapping for this split."""
    ep_path = split_dir / "ep_start_end_ids.npy"
    cache_path = split_dir / "cached_feats.pkl"
    ann_path = split_dir / "auto_lang_ann.npy"

    ep = np.load(ep_path)
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    ann = np.load(ann_path, allow_pickle=True).item()

    n_episodes = len(ep)
    instructions = []
    latent_map = {}
    errors = 0

    for i in tqdm(range(n_episodes), desc=split_dir.name):
        start = ep[i][0]
        meta = data[start]["source_meta"]
        traj_path = meta["traj_path"]
        latent_index = meta["latent_index"]
        try:
            with open(traj_path, "rb") as f:
                traj = pickle.load(f)
            lang = traj["language_action"].flatten().tolist()
            deduped = dedup_ordered(lang)
            instruction = ", ".join(deduped)
        except Exception as e:
            instruction = ""
            errors += 1

        instructions.append(instruction)
        latent_map[latent_index] = instruction

    ann["language"]["raw"] = np.array(instructions)
    np.save(ann_path, ann)

    non_empty = sum(1 for s in instructions if s)
    print(f"{split_dir.name}: {non_empty}/{n_episodes} non-empty, {errors} errors")
    print(f"  Sample [0]: {instructions[0]}")
    print(f"  Sample [1]: {instructions[1]}")

    return latent_map


if __name__ == "__main__":
    data_root = Path(__file__).parent.parent / "data" / "social"
    combined_map = {}
    for split in ["training", "validation"]:
        split_dir = data_root / split
        if split_dir.exists():
            combined_map.update(generate_instructions(split_dir))
        else:
            print(f"Skipping {split} (not found)")

    out_path = data_root / "instruction_map.json"
    with open(out_path, "w") as f:
        json.dump(combined_map, f)
    print(f"\nSaved instruction_map.json: {len(combined_map)} entries → {out_path}")
