import logging
from pathlib import Path
from typing import Dict, Tuple, Union, Any, List

from lcvn_ac.datasets.utils.episode_utils import (
    get_state_info_dict,
    process_actions,
    process_depth,
    process_language,
    process_rgb,
    process_state,
)
from lcvn_ac.utils.trajectory_logger import TrajectoryLogger
import numpy as np
from omegaconf import DictConfig
import pyhash
import torch
from torch.utils.data import Dataset
import copy

hasher = pyhash.fnv1_32()
logger = logging.getLogger(__name__)


def get_validation_window_size(idx: int, min_window_size: int, max_window_size: int) -> int:
    """
    In validation step, use hash function instead of random sampling for consistent window sizes across epochs.

    Args:
        idx: Sequence index.
        min_window_size: Minimum window size.
        max_window_size: Maximum window size.

    Returns:
        Window size computed with hash function.
    """
    window_range = max_window_size - min_window_size + 1
    return min_window_size + hasher(str(idx)) % window_range


class BaseDataset(Dataset):
    """
    Abstract dataset base class.

    Args:
        datasets_dir: Path of folder containing episode files (string must contain 'validation' or 'training').
        obs_space: DictConfig of observation space.
        proprio_state: DictConfig with shape of prioprioceptive state.
        key: 'vis' or 'lang'.
        lang_folder: Name of the subdirectory of the dataset containing the language annotations.
        num_workers: Number of dataloading workers for this dataset.
        transforms: Dict with pytorch data transforms.
        batch_size: Batch size.
        min_window_size: Minimum window length of loaded sequences.
        max_window_size: Maximum window length of loaded sequences.
        pad: If True, repeat last frame such that all sequences have length 'max_window_size'.
        aux_lang_loss_window: How many sliding windows to consider for auxiliary language losses, counted from the end
            of an annotated language episode.
    """

    def __init__(
        self,
        datasets_dir: Path,
        obs_space: DictConfig,
        proprio_state: DictConfig,
        key: str,
        lang_folder: str,
        num_workers: int,
        transforms: Dict = {},
        batch_size: int = 32,
        min_window_size: int = 16,
        max_window_size: int = 32,
        pad: bool = True,
        aux_lang_loss_window: int = 1,
        for_wm: bool = False,
    ):
        self.observation_space = obs_space
        self.proprio_state = proprio_state
        self.transforms = transforms
        self.with_lang = key == "lang"
        self.relative_actions = "rel_actions" in self.observation_space["actions"]

        self.pad = pad
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.abs_datasets_dir = datasets_dir
        self.lang_folder = lang_folder  # if self.with_lang else None
        self.aux_lang_loss_window = aux_lang_loss_window
        self.for_wm = for_wm
        assert "validation" in self.abs_datasets_dir.as_posix() or "training" in self.abs_datasets_dir.as_posix()
        self.validation = "validation" in self.abs_datasets_dir.as_posix()
        assert self.abs_datasets_dir.is_dir()
        logger.info(f"loading dataset at {self.abs_datasets_dir}")
        logger.info("finished loading dataset")
        if self.for_wm:
            assert self.min_window_size == self.max_window_size
            self.with_lang = False


    def __getitem__(self, idx: Union[int, Tuple[int, int]]) -> Dict:
        traj_logger = TrajectoryLogger()
        # Log only when a trajectory has explicitly started and within allowed count to avoid oversized logs
        do_log = (
            getattr(traj_logger, "current_trajectory_id", None) is not None
            and getattr(traj_logger, "should_log_trajectory")()
        )
        
        if isinstance(idx, int):
            if idx < 0 or idx >= len(self.load_info):
                logger.error(
                    f"Index {idx} out of bounds for self.load_info with length {len(self.load_info)}."
                )
                raise IndexError
            _, window_size = self.load_info[idx]
        else:
            raw_idx, window_size = idx
            if raw_idx < 0 or raw_idx >= len(self.load_info):
                logger.error(
                    f"Index {raw_idx} out of bounds for self.load_info with length {len(self.load_info)}."
                )
                raise IndexError
            idx = raw_idx

        if do_log:
            traj_logger.log_event(
                "WINDOW_PROCESSING",
                "Processing window",
                {
                    "idx": idx,
                    "window_size": window_size,
                    "min_window_size": self.min_window_size,
                    "max_window_size": self.max_window_size,
                },
            )

        sequence = self._get_sequences(idx, window_size)

        if self.pad:
            pad_size = self._get_pad_size(sequence)
            if pad_size > 0:
                original_sequence = sequence
                padded_sequence = self._pad_sequence(sequence, pad_size)
                if do_log:
                    traj_logger.log_window_processing(
                        original_sequence,
                        padded_sequence,
                        window_size,
                        pad_size,
                    )
                sequence = padded_sequence
            else:
                if do_log:
                    traj_logger.log_event(
                        "WINDOW_PROCESSING",
                        "No padding needed",
                        {"window_size": window_size},
                    )
        else:
            if do_log:
                traj_logger.log_event(
                    "WINDOW_PROCESSING",
                    "Padding disabled",
                    {"window_size": window_size},
                )
            
        return sequence

    def _get_sequences(self, idx: int, window_size: int) -> Dict:
        """
        Load sequence of length window_size.

        Args:
            idx: Index of starting frame.
            window_size: Length of sampled episode.

        Returns:
            dict: Dictionary of tensors of loaded sequence with different input modalities and actions.
        """
        traj_logger = TrajectoryLogger()
        # Log only when trajectory is active and within allowed range to avoid excessive logs during large scans
        do_log = (
            getattr(traj_logger, "current_trajectory_id", None) is not None
            and getattr(traj_logger, "should_log_trajectory")()
        )
        if do_log:
            traj_logger.log_event(
                "SEQUENCE_START",
                "Starting sequence processing",
                {"idx": idx, "window_size": window_size},
            )

        episode = self._load_episode(idx, window_size)
        
        # Log episode data after loading
        if do_log:
            traj_logger.log_data_loading("EPISODE_LOADED", episode)

        # Log source metadata to enable mapping trajectory -> raw data
        try:
            source_meta: Dict[str, Any] = {
                "dataset_dir": str(self.abs_datasets_dir),
                "dataset_class": type(self).__name__,
                "idx": int(idx),
                "window_size": int(window_size),
                "validation": bool(self.validation),
            }
            # When dataset uses language-based load_info mapping, add original trajectory/frame index info
            if hasattr(self, "load_info") and isinstance(self.load_info, list) and idx < len(self.load_info):
                try:
                    start_idx_zero, size_to_load = self.load_info[idx]
                    end_idx_zero = int(start_idx_zero) + int(window_size) - 1
                    source_meta["start_idx_zero_based"] = int(start_idx_zero)
                    source_meta["end_idx_zero_based"] = int(end_idx_zero)
                    if hasattr(self, "lang_lookup") and idx < len(self.lang_lookup):
                        raw_traj_idx = int(self.lang_lookup[idx])
                        source_meta["raw_traj_index"] = raw_traj_idx
                        if hasattr(self, "ep_start_end_ids"):
                            try:
                                raw_start, raw_end = self.ep_start_end_ids[raw_traj_idx]
                                # ep_start_end_ids are 0-based with end as exclusive.
                                # For readability, expose inclusive first/last frame indices:
                                start0b_incl = int(raw_start)
                                end0b_incl = int(raw_end) - 1
                                source_meta["raw_traj_start_end_0b"] = [start0b_incl, end0b_incl]
                                source_meta["raw_traj_start_end_1b"] = [start0b_incl + 1, end0b_incl + 1]
                            except Exception:
                                pass
                except Exception:
                    pass
            if hasattr(self, "episode_lookup") and isinstance(self.episode_lookup, np.ndarray):
                start_idx = int(self.episode_lookup[idx])
                source_meta["start_idx"] = start_idx
                source_meta["end_idx"] = start_idx + int(window_size) - 1
            if "frame" in episode and isinstance(episode["frame"], np.ndarray):
                try:
                    frames_list = episode["frame"].squeeze().tolist()
                except Exception:
                    frames_list = None
                source_meta["frames"] = frames_list
            if hasattr(self, "naming_pattern"):
                try:
                    prefix = str(self.naming_pattern[0])
                    suffix = self.naming_pattern[1]
                    source_meta["naming_pattern"] = [prefix, suffix]
                except Exception:
                    pass
            if hasattr(self, "n_digits"):
                try:
                    source_meta["n_digits"] = int(self.n_digits)
                except Exception:
                    pass
            traj_logger.log_event(
                "SEQUENCE_SOURCE",
                "Source metadata for mapping to raw files",
                source_meta,
            )
        except Exception as e:
            logger.debug(f"Failed to log source metadata: {e}")
        if self.for_wm:
            seq_state_obs = process_state(episode, self.observation_space, self.transforms, self.proprio_state)
            seq_rgb_obs = process_rgb(episode, self.observation_space, self.transforms)
            seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)
            action_keys = copy.deepcopy(self.observation_space["actions"])
            action_keys.append("pre_actions")
            seq_acts = process_actions(episode, action_keys, self.transforms)
            info = get_state_info_dict(episode, self.for_wm, self.proprio_state)
            seq_lang = process_language(episode, self.transforms, self.with_lang)
            info = self._add_language_info(info, idx, window_size)

            seq_reset = {"reset": torch.from_numpy(episode["reset"]).bool()}
            seq_frames = {"frame": torch.from_numpy(episode["frame"])}

            ####
            seq_lang_raw = {"lang_raw": ""}
            if self.with_lang and "language_raw" in episode:
                seq_lang_raw["lang_raw"] = episode["language_raw"]

            seq_dict = {
                **seq_state_obs,
                **seq_rgb_obs,
                **seq_depth_obs,
                **seq_acts,
                **info,
                **seq_lang,
                **seq_reset,
                **seq_frames,
                **seq_lang_raw,
            }  # type:ignore

        else:
            seq_vae_latent = {"vae_latent": torch.from_numpy(episode["vae_latent"]).float()} # T, 4, 28, 28
            seq_acts = {"rel_actions": process_actions(episode, self.observation_space["actions"], self.transforms)["actions"]["rel_actions"]}
            seq_reset = {"reset": torch.from_numpy(episode["reset"]).bool()}
            info = get_state_info_dict(episode, self.for_wm, self.proprio_state)
            seq_lang = process_language(episode, self.transforms, self.with_lang)
            info = self._add_language_info(info, idx, window_size)
            seq_discrete = {"discrete_action": torch.from_numpy(episode["discrete_action"])}
            ####
            seq_lang_raw = {"lang_raw": ""}
            if self.with_lang and "language_raw" in episode:
                seq_lang_raw["lang_raw"] = episode["language_raw"]
            seq_dict = {
                **seq_vae_latent,
                **seq_acts,
                **seq_reset,
                **info,
                **seq_lang,
                **seq_lang_raw,
                **seq_discrete,
            }  # type:ignore

        seq_dict["idx"] = idx  # type:ignore

        # Attach lightweight source metadata for later logging in training_step
        try:
            source_meta: Dict[str, Any] = {
                "dataset_dir": str(self.abs_datasets_dir),
                "dataset_class": type(self).__name__,
                "idx": int(idx),
                "window_size": int(window_size),
                "validation": bool(self.validation),
            }
            if hasattr(self, "load_info") and isinstance(self.load_info, list) and idx < len(self.load_info):
                try:
                    start_idx_zero, size_to_load = self.load_info[idx]
                    end_idx_zero = int(start_idx_zero) + int(window_size) - 1
                    source_meta["start_idx_zero_based"] = int(start_idx_zero)
                    source_meta["end_idx_zero_based"] = int(end_idx_zero)
                    if hasattr(self, "lang_lookup") and idx < len(self.lang_lookup):
                        raw_traj_idx = int(self.lang_lookup[idx])
                        source_meta["raw_traj_index"] = raw_traj_idx
                        if hasattr(self, "ep_start_end_ids"):
                            try:
                                raw_start, raw_end = self.ep_start_end_ids[raw_traj_idx]
                                start0b_incl = int(raw_start)
                                end0b_incl = int(raw_end) - 1
                                source_meta["raw_traj_start_end_0b"] = [start0b_incl, end0b_incl]
                                source_meta["raw_traj_start_end_1b"] = [start0b_incl + 1, end0b_incl + 1]
                            except Exception:
                                pass
                except Exception:
                    pass
            if hasattr(self, "episode_lookup") and isinstance(self.episode_lookup, np.ndarray):
                start_idx = int(self.episode_lookup[idx])
                source_meta["start_idx"] = start_idx
                source_meta["end_idx"] = start_idx + int(window_size) - 1
            if "frame" in episode and isinstance(episode["frame"], np.ndarray):
                try:
                    source_meta["frames"] = episode["frame"].squeeze().tolist()
                except Exception:
                    pass
            if hasattr(self, "naming_pattern"):
                try:
                    prefix = str(self.naming_pattern[0])
                    suffix = self.naming_pattern[1]
                    source_meta["naming_pattern"] = [prefix, suffix]
                except Exception:
                    pass
            if hasattr(self, "n_digits"):
                try:
                    source_meta["n_digits"] = int(self.n_digits)
                except Exception:
                    pass
            seq_dict["source_meta"] = source_meta
        except Exception as e:
            logger.debug(f"Failed to attach source_meta to sequence dict: {e}")

        # Log final sequence dictionary
        traj_logger.log_data_loading("FINAL_SEQUENCE", seq_dict)

        return seq_dict

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def _get_window_size(self, idx: int) -> int:
        """
        Sample a window size taking into account the episode limits.

        Args:
            idx: Index of the sequence to load.

        Returns:
            Window size.
        """
        window_diff = self.max_window_size - self.min_window_size
        if len(self.episode_lookup) <= idx + window_diff:
            # last episode
            max_window = self.min_window_size + len(self.episode_lookup) - idx - 1
        elif self.episode_lookup[idx + window_diff] != self.episode_lookup[idx] + window_diff:
            # less than max_episode steps until next episode
            steps_to_next_episode = int(
                np.nonzero(
                    self.episode_lookup[idx : idx + window_diff + 1]
                    - (self.episode_lookup[idx] + np.arange(window_diff + 1))
                )[0][0]
            )
            max_window = min(self.max_window_size, (self.min_window_size + steps_to_next_episode - 1))
        else:
            max_window = self.max_window_size

        if self.validation:
            # in validation step, repeat the window sizes for each epoch.
            return get_validation_window_size(idx, self.min_window_size, max_window)
        else:
            return np.random.randint(self.min_window_size, max_window + 1)

    def __len__(self) -> int:
        """
        Returns:
            Size of the dataset.
        """
        return len(self.load_info)

    def _get_pad_size(self, sequence: Dict) -> int:
        """
        Determine how many frames to append to end of the sequence

        Args:
            sequence: Loaded sequence.

        Returns:
            Number of frames to pad.
        """
        return self.max_window_size - len(sequence["discrete_action"])

    def _pad_sequence(self, seq: Dict, pad_size: int) -> Dict:
        """
        Pad a sequence by repeating the last frame.

        Args:
            seq: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded sequence.
        """
        seq.update({"vae_latent": self._pad_with_repetition(seq["vae_latent"], pad_size)})
        seq.update({"discrete_action": self._pad_with_repetition(seq["discrete_action"], pad_size)})
        seq.update({"reset": self._pad_with_repetition(seq["reset"], pad_size)})
        seq.update({"rel_actions": self._pad_with_repetition(seq["rel_actions"], pad_size)})
        
        seq.update({"state_info": {k: self._pad_with_repetition(v, pad_size) for k, v in seq["state_info"].items()}})
        return seq

    @staticmethod
    def _pad_with_repetition(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
        """
        Pad a sequence Tensor by repeating last element pad_size times.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        last_repeated = torch.repeat_interleave(torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0)
        padded = torch.vstack((input_tensor, last_repeated))
        return padded

    @staticmethod
    def _pad_with_zeros(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
        """
        Pad a Tensor with zeros.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        zeros_repeated = torch.repeat_interleave(
            torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0), repeats=pad_size, dim=0
        )
        padded = torch.vstack((input_tensor, zeros_repeated))
        return padded

    def _add_language_info(self, info: Dict, idx: int, window_size: int) -> Dict:
        if not self.with_lang:
            return info
    
        if idx + 1 >= len(self.lang_lookup):
            use_for_aux_lang_loss = True
        else:
            use_for_aux_lang_loss = self.lang_lookup[idx] < self.lang_lookup[idx + self.aux_lang_loss_window]
        
        info["use_for_aux_lang_loss"] = use_for_aux_lang_loss
        return info
