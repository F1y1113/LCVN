# datasets/video/social.py
"""
Social Navigation dataset for DFoT
Supports sliding-window sampling with delta condition encoding

Notes:
- Removed normalization in load_data()
- Kept data_mean and data_std in config (for model usage)
- All other functionality remains unchanged
"""

from typing import Dict, Any, Optional
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from omegaconf import DictConfig
from .base_video import BaseAdvancedVideoDataset


class SocialAdvancedVideoDataset(BaseAdvancedVideoDataset):
    """
    Social navigation dataset with sliding window sampling
    
    Key features:
    - Sliding window: each trajectory produces multiple 5-frame samples
    - Condition padding: add zero-padding at the first frame
    - Delta conditions: [dx, dy, dyaw, dt] computed from raw trajectory
    """
    
    _ALL_SPLITS = ["training", "validation", "test"]
    
    def __init__(
        self,
        cfg: DictConfig,
        split: str = "training",
        current_epoch: Optional[int] = None,
    ):
        # Set instance variables
        self.latent_suffix = cfg.latent.suffix
        self.window_size = cfg.n_frames
        self.data_mean = None
        self.data_std = None
        
        if cfg.data_mean is not None:
            # Add batch dimension: [4, 1, 1] -> [1, 4, 1, 1]
            data_mean_tensor = torch.tensor(cfg.data_mean)
            data_std_tensor = torch.tensor(cfg.data_std)
            self.data_mean = data_mean_tensor.unsqueeze(0)
            self.data_std = data_std_tensor.unsqueeze(0)
        
        # Initialize window_indices as empty list
        self.window_indices = []
        
        # Call parent class initialization
        super().__init__(cfg, split, current_epoch)

        # Build sliding window index
        self._build_sliding_window_index()

    '''
    def _build_sliding_window_index(self):
        """Build sliding-window indices for all trajectories"""
        self.window_indices = []
        
        for traj_idx in range(len(self.metadata)):
            num_frames = self.metadata[traj_idx]["num_frames"]
            num_windows = num_frames - self.window_size + 1
            
            if num_windows > 0:
                for start_frame in range(num_windows):
                    self.window_indices.append({
                        "traj_idx": traj_idx,
                        "start_frame": start_frame,
                        "end_frame": start_frame + self.window_size
                    })
        
        print(f"[{self.split}] Total trajectories: {len(self.metadata)}")
        print(f"[{self.split}] Sliding window samples: {len(self.window_indices)}")
    '''

    def _build_sliding_window_index(self):
        """Build sliding-window indices for all trajectories"""
        self.window_indices = []

        for traj_idx in range(len(self.metadata)):
            item = self.metadata[traj_idx]

            # Check if it is a plain string (path from brute-force scan)
            if isinstance(item, (str, Path)):
                # Data is known to be [12, 4, 32, 32], fixed 12 frames
                num_frames = 12
            else:
                # Original logic: if dict, read from metadata
                num_frames = item["num_frames"]

            num_windows = num_frames - self.window_size + 1

            if num_windows > 0:
                for start_frame in range(num_windows):
                    self.window_indices.append({
                        "traj_idx": traj_idx,
                        "start_frame": start_frame,
                        "end_frame": start_frame + self.window_size
                    })

        print(f"[{self.split}] Total trajectories: {len(self.metadata)}")
        print(f"[{self.split}] Sliding window samples: {len(self.window_indices)}")

    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        result = super().__getitem__(idx)
        video_idx, _ = self.get_clip_location(idx)
        video_metadata = self.metadata[video_idx]
        if isinstance(video_metadata, dict) and "instruction" in video_metadata:
            result["instruction"] = video_metadata["instruction"]
        else:
            result["instruction"] = ""
        return result
    
    def _get_video_metadata(self, traj_idx: int) -> Dict[str, Any]:
        """Get metadata for a specific trajectory"""
        return self.metadata[traj_idx]
    
    def download_dataset(self):
        """Data already exists; no download needed"""
        pass
    
    def load_metadata(self):
        """
        Load metadata and normalize keys to match BaseVideoDataset expectations.
        Supports both base format with 'video_paths'/'video_pts' and
        social format with columnar fields like 'video_path', 'num_frames', 'conditions'.
        """
        metadata_path = self.metadata_dir / f"{self.split}.pt"
        metadata_dict = torch.load(metadata_path, weights_only=False)
        if "video_paths" in metadata_dict:
            # Base format: delegate to BaseVideoDataset behavior
            return [
                {key: metadata_dict[key][i] for key in metadata_dict.keys()}
                for i in range(len(metadata_dict["video_paths"]))
            ]
        elif "video_path" in metadata_dict:
            # Social format: normalize to expected keys
            num_samples = len(metadata_dict["video_path"])
            out = []
            for i in range(num_samples):
                item = {
                    "video_paths": Path(metadata_dict["video_path"][i]),
                    "num_frames": int(metadata_dict["num_frames"][i]),
                }
                # Optional fields
                if "conditions" in metadata_dict:
                    item["conditions"] = metadata_dict["conditions"][i]
                if "scene" in metadata_dict:
                    item["scene"] = metadata_dict["scene"][i]
                if "trajectory_id" in metadata_dict:
                    item["trajectory_id"] = metadata_dict["trajectory_id"][i]
                if "instruction" in metadata_dict:
                    item["instruction"] = metadata_dict["instruction"][i]
                out.append(item)
            return out
        else:
            # Fallback: empty
            return []

    '''
    def video_length(self, video_metadata: Dict[str, Any]) -> int:
        return video_metadata["num_frames"]
    '''

    def video_length(self, video_metadata: Dict[str, Any]) -> int:
        if isinstance(video_metadata, (str, Path)):
            return 12  # Fixed: return 12 frames
        return video_metadata["num_frames"]

    '''
    def video_metadata_to_latent_path(self, video_metadata: Dict[str, Any]) -> Path:
        scene = video_metadata.get("scene", None)
        traj_id = video_metadata.get("trajectory_id", None)
        if scene is None or traj_id is None:
            raise ValueError("Missing 'scene' or 'trajectory_id' in metadata for latent path resolution")
        return (self.save_dir / "latents" / self.split / f"{scene}_{traj_id}_{self.latent_suffix}.pt")
    '''
    def video_metadata_to_latent_path(self, video_metadata: Dict[str, Any]) -> Path:
        if isinstance(video_metadata, (str, Path)):
            return Path(video_metadata)

        scene = video_metadata.get("scene", None)
        traj_id = video_metadata.get("trajectory_id", None)
        if scene is None or traj_id is None:
            raise ValueError("Missing 'scene' or 'trajectory_id' in metadata for latent path resolution")
        return (self.save_dir / "latents" / self.split / f"{scene}_{traj_id}_{self.latent_suffix}.pt")

    #def load_video(
    #    self,
    #    video_metadata: Dict[str, Any],
    #    start_frame: int,
    #    end_frame: Optional[int] = None,
    #) -> torch.Tensor:
    #    """
    #    Override base load_video to read frames by index without pts metadata.
    #    Returns (T, C, H, W) in [0, 1].
    #    """
    #    if end_frame is None:
    #        end_frame = self.video_length(video_metadata)
    #    """video_path = video_metadata.get("video_paths", video_metadata.get("video_path"))
    #    cap = cv2.VideoCapture(str(video_path))"""
    #    if isinstance(video_metadata, (str, Path)):
    #        video_path = video_metadata
    #    else:
    #        video_path = video_metadata.get("video_paths", video_metadata.get("video_path"))
        #    cap = cv2.VideoCapture(str(video_path))
        #
        #total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #if end_frame > total_frames:
        #    end_frame = total_frames
        #frames = []
        #for i in range(start_frame, end_frame):
        #    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        #    ok, frame = cap.read()
        #    if not ok:
        #        break
        #    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #    arr = frame.astype(np.float32) / 255.0
        #    arr = np.transpose(arr, (2, 0, 1))
        #    frames.append(torch.from_numpy(arr))
        #cap.release()
        #if len(frames) == 0:
        #    return torch.empty(0, 3, self.cfg.resolution, self.cfg.resolution)
        #return torch.stack(frames, dim=0)

    def load_video(
            self,
            video_metadata: Dict[str, Any],
            start_frame: int,
            end_frame: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Override base load_video to read frames by index without pts metadata.
        Returns (T, C, H, W) in [0, 1].
        """
        if end_frame is None:
            end_frame = self.video_length(video_metadata)

        if isinstance(video_metadata, (str, Path)):
            video_path = video_metadata
        else:
            video_path = video_metadata.get("video_paths", video_metadata.get("video_path"))

        cap = cv2.VideoCapture(str(video_path))
        # Note: do not rely on CAP_PROP_FRAME_COUNT as it can be inaccurate
        frames = []
        for i in range(start_frame, end_frame):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok:
                break  # Stop if can't read, pad later
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            arr = frame.astype(np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))
            frames.append(torch.from_numpy(arr))
        cap.release()

        target_length = end_frame - start_frame
        current_length = len(frames)

        if current_length == 0:
            # If no frames read, return all-black video (avoid tensor stack error)
            return torch.zeros(target_length, 3, self.cfg.resolution, self.cfg.resolution)

        if current_length < target_length:
            # If frames missing, repeat last frame to pad
            missing_count = target_length - current_length
            last_frame = frames[-1]
            for _ in range(missing_count):
                frames.append(last_frame.clone())

        return torch.stack(frames, dim=0)

    def load_data(
        self, video_metadata: Dict[str, Any], start_frame: int, end_frame: int
    ) -> torch.Tensor:
        # Use base implementation which maps from video_paths to latents path
        return super().load_latent(video_metadata, start_frame, end_frame)

    '''
    def load_cond(
        self, video_metadata: Dict[str, Any], start_frame: int, end_frame: int
    ) -> torch.Tensor:
        """Load and process conditions with zero-padding"""
        original_conditions = video_metadata["conditions"]
        
        if end_frame - 1 > len(original_conditions):
            raise IndexError(
                f"Condition index out of bounds: end_frame-1={end_frame-1}, "
                f"len(conditions)={len(original_conditions)}"
            )
        
        selected_conditions = original_conditions[start_frame:end_frame-1]
        delta_cond = self._compute_delta_conditions(selected_conditions)
        
        # Add zero-padding at front
        zero_padding = torch.zeros(1, 4, dtype=delta_cond.dtype)
        delta_cond = torch.cat([zero_padding, delta_cond], dim=0)
        
        return delta_cond
    '''

    def load_cond(
            self, video_metadata: Dict[str, Any], start_frame: int, end_frame: int
    ) -> torch.Tensor:
        """Load and process conditions with zero-padding"""

        if isinstance(video_metadata, (str, Path)):
            # Return shape: [timesteps, 4 dims (dx, dy, dyaw, dt)]
            steps = end_frame - start_frame
            return torch.zeros(steps, 4)

        original_conditions = video_metadata["conditions"]

        if end_frame - 1 > len(original_conditions):
            raise IndexError(
                f"Condition index out of bounds: end_frame-1={end_frame - 1}, "
                f"len(conditions)={len(original_conditions)}"
            )

        selected_conditions = original_conditions[start_frame:end_frame - 1]
        delta_cond = self._compute_delta_conditions(selected_conditions)

        # Add zero-padding at front
        zero_padding = torch.zeros(1, 4, dtype=delta_cond.dtype)
        delta_cond = torch.cat([zero_padding, delta_cond], dim=0)

        return delta_cond


    def _compute_delta_conditions(self, conditions: torch.Tensor) -> torch.Tensor:
        """Compute delta conditions: [dx, dy, dyaw, dt]"""
        positions = conditions[:, :3]
        quaternions = conditions[:, 3:]
        
        # Position deltas
        if len(positions) > 1:
            dx = positions[1:, 0] - positions[:-1, 0]
            dy = positions[1:, 1] - positions[:-1, 1]
            dx = torch.cat([torch.zeros(1, dtype=dx.dtype), dx])
            dy = torch.cat([torch.zeros(1, dtype=dy.dtype), dy])
        else:
            dx = torch.zeros(1, dtype=positions.dtype)
            dy = torch.zeros(1, dtype=positions.dtype)
        
        # Yaw deltas
        yaw = self._quaternion_to_yaw(quaternions)
        if len(yaw) > 1:
            dyaw = yaw[1:] - yaw[:-1]
            dyaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw))
            dyaw = torch.cat([torch.zeros(1, dtype=dyaw.dtype), dyaw])
        else:
            dyaw = torch.zeros(1, dtype=yaw.dtype)
        
        # Fixed dt
        dt = torch.full_like(dx, 0.1)
        
        return torch.stack([dx, dy, dyaw, dt], dim=-1)
    
    def _quaternion_to_yaw(self, quaternions: torch.Tensor) -> torch.Tensor:
        """Convert quaternions to yaw angle"""
        qw, qx, qy, qz = (
            quaternions[:, 0], quaternions[:, 1], 
            quaternions[:, 2], quaternions[:, 3]
        )
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        return torch.atan2(siny_cosp, cosy_cosp)
