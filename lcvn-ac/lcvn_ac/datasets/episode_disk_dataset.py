from itertools import chain
import logging
from pathlib import Path
import pickle
from typing import Any, Dict, List, Tuple

from lcvn_ac.datasets.base_dataset import BaseDataset
from lcvn_ac.datasets.utils.episode_utils import lookup_naming_pattern
from lcvn_ac.utils.trajectory_logger import TrajectoryLogger
import numpy as np
from tqdm import tqdm
import re

logger = logging.getLogger(__name__)


def load_pkl(filename: Path) -> Dict[str, np.ndarray]:
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_npz(filename: Path) -> Dict[str, np.ndarray]:
    return np.load(filename.as_posix())


class EpisodeDiskDataset(BaseDataset):
    """
    Dataset that loads episodes as individual files from disk.

    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        *args: Any,
        skip_frames: int = 1,
        save_format: str = "npz",
        pretrain: bool = False,
        use_cached_data: bool = False,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.save_format = save_format
        if self.save_format == "pkl":
            self.load_file = load_pkl
        elif self.save_format == "npz":
            self.load_file = load_npz
        else:
            raise NotImplementedError

        self.pretrain = pretrain
        self.skip_frames = skip_frames

        self.load_info, self.lang_lookup, self.lang_ann = self._build_file_indices_lang(self.abs_datasets_dir)

    def _get_episode_name(self, file_idx: int) -> Path:
        """
        Convert file idx to file path.

        Args:
            file_idx: index of starting frame.

        Returns:
            Path to file.
        """
        return Path(f"{self.naming_pattern[0]}{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}")

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.

        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.

        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        # When using language-based data indexing, load according to window_size passed by __getitem__
        start_idx, valid_len = self.load_info[idx]
        if start_idx < 0:
            # Defensive fix: if language index is 1-based causing start -1, clamp to 0
            start_idx = 0
        
        # Ensure we don't load past the episode boundary (valid_len)
        load_len = min(valid_len, window_size)
        end_idx = start_idx + load_len

        # Log raw data loading (only when trajectory logging is active)
        traj_logger = TrajectoryLogger()
        do_log = (
            getattr(traj_logger, "current_trajectory_id", None) is not None
            and getattr(traj_logger, "should_log_trajectory")()
        )
        if do_log:
            # Use log_data_loading signature correctly by passing a data dict
            traj_logger.log_data_loading(
                "LOAD_EPISODE_RANGE",
                {
                    "idx": idx,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "window_size": window_size,
                    "valid_len": valid_len,
                },
            )

        episodes = [self.features[file_idx] for file_idx in range(start_idx, end_idx)]
        
        # Pad with the last frame if the loaded sequence is shorter than window_size
        if len(episodes) < window_size:
            pad_len = window_size - len(episodes)
            if episodes:
                last_frame = episodes[-1]
                for _ in range(pad_len):
                    # Shallow copy the dict to avoid modifying original cache
                    padded_frame = last_frame.copy()
                    
                    # Set actions to zero to avoid "action causes no change" conflict in World Model
                    if 'rel_actions' in padded_frame:
                        shape = padded_frame['rel_actions'].shape
                        padded_frame['rel_actions'] = np.zeros(shape, dtype=padded_frame['rel_actions'].dtype)
                    
                    if 'discrete_action' in padded_frame:
                        shape = padded_frame['discrete_action'].shape
                        padded_frame['discrete_action'] = np.zeros(shape, dtype=padded_frame['discrete_action'].dtype)
                        
                    episodes.append(padded_frame)
            else:
                # Should not happen if min_window_size >= 1
                pass
        
        # Log raw data dimensions and sample values
        if do_log and episodes:
            sample_episode = episodes[0]
            for key, value in sample_episode.items():
                if isinstance(value, np.ndarray):
                    # Pass actual data for logger to auto-extract shape/dtype
                    traj_logger.log_data_loading(
                        f"RAW_DATA_{key}",
                        {key: value},
                    )

        # Handle missing rel_actions in last frames (common issue: N frames but N-1 rel_actions)
        # First, collect all unique keys across all frames
        all_keys = set()
        for ep in episodes:
            all_keys.update(ep.keys())
        
        # Check if rel_actions is expected but missing in some frames
        if 'rel_actions' in all_keys:
            for t, ep in enumerate(episodes):
                if 'rel_actions' not in ep:
                    # Add zero rel_actions for missing frames (typically the last frame)
                    # Infer the shape from other frames that have rel_actions
                    rel_actions_shape = None
                    for other_ep in episodes:
                        if 'rel_actions' in other_ep:
                            rel_actions_shape = other_ep['rel_actions'].shape
                            break
                    
                    if rel_actions_shape is not None:
                        logger.warning(f"Adding zero rel_actions for frame {start_idx + t} (shape: {rel_actions_shape})")
                        ep['rel_actions'] = np.zeros(rel_actions_shape, dtype=np.float32)
                    else:
                        # Fallback: use default action dimension (4 for [dx, dy, dz, dyaw])
                        logger.warning(f"Adding default zero rel_actions for frame {start_idx + t}")
                        ep['rel_actions'] = np.zeros(4, dtype=np.float32)

        # Strict stacking: require all keys present in every frame after handling missing rel_actions
        episode = {}
        base_keys = list(episodes[0].keys())
        for key in base_keys:
            values = []
            for t, ep in enumerate(episodes):
                if key not in ep:
                    raise KeyError(f"Missing key '{key}' at frame index {start_idx + t}. Data is inconsistent.")
                values.append(ep[key])
            episode[key] = np.stack(values)

        if self.with_lang:
            sub_ep_lang_idx = self.lang_lookup[idx]
            # Use the corresponding embedding vector directly; current emb shape is (N, D), no need to index [0]
            episode["language"] = self.lang_ann[sub_ep_lang_idx]
            episode["language_raw"] = self.lang_ann_raw[sub_ep_lang_idx]
        return episode

    def _build_file_indices_lang(self, abs_datasets_dir: Path) -> Tuple[List[Tuple[int, int]], List[int], np.ndarray]:
        load_info = []
        lang_lookup = []

        # Load language annotation file
        try:
            lang_data = np.load(abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy", allow_pickle=True).item()
        except Exception:
            lang_data = np.load(abs_datasets_dir / "auto_lang_ann.npy", allow_pickle=True).item()

        # Compatible with original language text field names: prefer 'ann', fallback to 'raw'
        language_dict = lang_data.get("language", {})
        lang_ann = language_dict.get("emb")
        self.lang_ann_raw = language_dict.get("ann", language_dict.get("raw", None))

        # Read language indices; can be [start, end] pairs or just start
        ep_indices = lang_data.get("info", {}).get("indx")
        if ep_indices is None:
            raise KeyError("'info[\"indx\"]' missing in auto_lang_ann.npy")

        # Try to load trajectory start-end mapping, to fill end when ep_indices only provides start
        ep_bounds_map = None
        try:
            ep_bounds = np.load(abs_datasets_dir / "ep_start_end_ids.npy")
            # Build {start -> end} mapping
            ep_bounds_map = {int(s): int(e) for s, e in ep_bounds}
        except Exception:
            ep_bounds_map = None

        # Normalize to (start, end) pairs
        normalized_pairs: List[Tuple[int, int]] = []
        # Convert ep_indices to list to handle numpy object arrays
        ep_list = list(ep_indices)
        for item in ep_list:
            # item may be a scalar (start) or a length-2 sequence (start, end)
            if isinstance(item, (list, tuple, np.ndarray)) and len(item) == 2:
                start, end = int(item[0]), int(item[1])
            else:
                start = int(item)
                if ep_bounds_map is not None and start in ep_bounds_map:
                    end = ep_bounds_map[start]
                else:
                    # End not found: degrade to single-point window; later clipped by window size
                    end = start
            normalized_pairs.append((start, end))

        # Save original language indices (start-end pairs) for source metadata logging
        self.ep_start_end_ids = normalized_pairs

        step_size = self.skip_frames

        # Build load_info: (start_frame 0-based, length_to_load)
        # Ensure each window has at least min_window_size to avoid 1-length windows being padded to max_window_size
        for i, (ep_start, ep_end) in enumerate(normalized_pairs):
            # Language index start is 0-based; end is typically next start (exclusive), so use end-1 as inclusive last frame
            start_conv = ep_start
            end_conv = ep_end - 1 if ep_end > 0 else ep_end

            # Compute valid start range such that [start_idx, end_conv] length >= min_window_size
            last_valid_start = end_conv - self.min_window_size + 1
            if last_valid_start < start_conv:
                # This segment is too short to form a minimum window; skip
                continue

            for start_idx in range(start_conv, last_valid_start + 1, step_size):
                available_length = end_conv - start_idx + 1
                if available_length < self.min_window_size:
                    # Protective check: should not happen due to range limit above
                    continue
                size_to_load = min(self.max_window_size, available_length)

                load_info.append((start_idx, size_to_load))
                lang_lookup.append(i)

        return load_info, lang_lookup, lang_ann



    def _build_file_indices(self, abs_datasets_dir: Path) -> np.ndarray:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the non language
        dataset.

        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.

        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []

        ep_start_end_ids = np.load(abs_datasets_dir / "ep_start_end_ids.npy")
        logger.info(f'Found "ep_start_end_ids.npy" with {len(ep_start_end_ids)} episodes.')
        for start_idx, end_idx in ep_start_end_ids:
            assert end_idx > self.max_window_size
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                episode_lookup.append(idx)
        return np.array(episode_lookup)

    def extract_episode_number(self, file_path):
        # Regular expression to find the episode number pattern
        match = re.search(r"episode_(\d+)\.npz", str(file_path))
        if match:
            # Return the episode number as an integer
            return int(match.group(1))
        else:
            raise ValueError(f"Episode number not found in {file_path}")

    def setup_features(self, features):
        self.features = features
